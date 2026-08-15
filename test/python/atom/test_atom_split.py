# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python -m pytest %s -v

"""Off-device unit tests for unified-operation logical-kernel splitting.

These drive ``split_function_body`` directly on small ASTs, so they need
neither ttnn nor a device and run anywhere ttl imports. They lock in the
split-time error paths for ambiguous kernel ownership and the basic
compute/data-movement routing."""

import ast
import copy
import textwrap

import pytest
import ttl
import ttl.kernel as kernel_module

from ttl._src.atom_split import split_function_body
from ttl.atom import (
    _assign_backend_kernel_slots,
    _backend_kernel_bodies,
    _backend_kernel_capacities,
    _build_atom_spec,
    _lift_setup,
)
from ttl.kernel import Kernel, KernelKind, _operation_identity


def _fn(src: str) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(src)).body[0]


def _kernel_src(result, kernel) -> str:
    return "\n".join(ast.unparse(statement) for statement in result.body_for(kernel))


def _kind_src(result, kind: KernelKind, index: int = 0) -> str:
    kernels = [
        kernel
        for kernel in result.kernels
        if kernel == kind or isinstance(kernel, Kernel) and kernel.kind == kind
    ]
    if index >= len(kernels):
        return _kernel_src(result, kind)
    return _kernel_src(result, kernels[index])


def _logical_kernel(kind: KernelKind, name: str) -> Kernel:
    kernel = Kernel(kind)
    kernel._bind(name, "test.operation")
    return kernel


def test_kernel_resource_is_lifted_before_logical_split():
    """A top-level Kernel declaration receives its operation-local name."""
    function = _fn(
        """
        def k():
            reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
            ttl.call_extern_func("reader.hpp", "reader", kernel=reader)
        """
    )

    stripped, dfbs, nets, logical_kernels = _lift_setup(
        function,
        {"ttl": ttl},
        "test.operation",
    )

    assert not dfbs
    assert not nets
    assert tuple(logical_kernels) == ("reader",)
    reader = logical_kernels["reader"]
    assert reader.identity == "reader"
    assert "Kernel(" not in ast.unparse(stripped)

    result = split_function_body(
        stripped,
        dfb_param_names=set(),
        logical_kernels=logical_kernels,
        kernel_capacities=_backend_kernel_capacities(),
    )
    assignments = _assign_backend_kernel_slots(result)
    assert tuple(assignments.values()) == (reader,)


def test_empty_backend_kernels_retain_logical_selectors():
    """Empty backend fillers retain stable target-independent selectors."""
    reader = _logical_kernel(KernelKind.DATA_MOVEMENT, "reader")
    function = _fn(
        """
        def k():
            ttl.call_extern_func("reader.hpp", "reader", kernel=reader)
        """
    )
    result = split_function_body(
        function,
        dfb_param_names=set(),
        logical_kernels={"reader": reader},
        kernel_capacities=_backend_kernel_capacities(),
    )
    assignments = _assign_backend_kernel_slots(result)

    first = tuple(_backend_kernel_bodies(result, assignments, None))
    second = tuple(_backend_kernel_bodies(result, assignments, None))
    selectors = tuple(selector for _, selector, _ in first)

    assert selectors == tuple(selector for _, selector, _ in second)
    assert selectors[0] is KernelKind.COMPUTE
    assert selectors[1] is reader
    assert isinstance(selectors[2], Kernel)
    assert selectors[2].kind is KernelKind.DATA_MOVEMENT
    assert selectors[2].identity == "<pipe_source>"
    assert selectors[2]._implicit_role == "pipe_source"
    assert isinstance(first[0][2][0], ast.Pass)
    assert isinstance(first[2][2][0], ast.Pass)


def test_captured_kernel_is_bound_for_final_operation():
    """Registration binds the captured source handle in place."""
    reader = Kernel(KernelKind.DATA_MOVEMENT)

    def operation():
        ttl.call_extern_func("reader.hpp", "reader", kernel=reader)

    spec = _build_atom_spec(operation)

    assert spec.logical_kernels["reader"] is reader
    assert reader.identity == "reader"
    assert reader._operation_identity == spec.operation_identity

    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        kernel_capacities=_backend_kernel_capacities(),
    )
    assert result.kernels == (reader,)


def test_captured_kernel_cannot_bind_to_two_operations():
    """One public handle has one operation-local binding."""
    sender = Kernel(KernelKind.DATA_MOVEMENT)

    def first_operation():
        ttl.call_extern_func("first.hpp", "first", kernel=sender)

    first_spec = _build_atom_spec(first_operation)
    assert sender._operation_identity == first_spec.operation_identity

    def second_operation():
        ttl.call_extern_func("second.hpp", "second", kernel=sender)

    with pytest.raises(ValueError, match="already bound"):
        _build_atom_spec(second_operation)


def test_captured_kernel_cannot_have_two_names():
    """Alias validation completes before the handle is bound."""
    reader = Kernel(KernelKind.DATA_MOVEMENT)
    reader_alias = reader

    def operation():
        ttl.call_extern_func("reader.hpp", "reader", kernel=reader)
        ttl.call_extern_func("reader.hpp", "reader", kernel=reader_alias)

    with pytest.raises(ValueError, match="multiple names"):
        _build_atom_spec(operation)
    with pytest.raises(ValueError, match="no operation-local identity"):
        reader.identity


def test_composition_preserves_captured_kernel_handle():
    """Composition retains the callee-owned handle used by child resources."""
    reader = Kernel(KernelKind.DATA_MOVEMENT)

    @ttl.operation()
    def selected_callee():
        ttl.call_extern_func("reader.hpp", "reader", kernel=reader)

    @ttl.operation(grid=(1, 1))
    def selected_caller():
        selected_callee()

    spec = selected_caller._spec
    assert len(spec.logical_kernels) == 1
    inlined_reader = next(iter(spec.logical_kernels.values()))
    assert inlined_reader is reader
    assert inlined_reader.identity == "reader"
    assert (
        inlined_reader._operation_identity == selected_callee._spec.operation_identity
    )

    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        kernel_capacities=_backend_kernel_capacities(),
    )
    assert result.kernels == (reader,)


def test_repeated_composition_reuses_callee_logical_kernel():
    """Sequential calls to one helper share its declared logical kernel."""
    reader = Kernel(KernelKind.DATA_MOVEMENT)

    @ttl.operation()
    def selected_callee():
        ttl.call_extern_func("reader.hpp", "reader", kernel=reader)

    @ttl.operation(grid=(1, 1))
    def selected_caller():
        selected_callee()
        selected_callee()

    spec = selected_caller._spec
    assert tuple(spec.logical_kernels.values()) == (reader,)

    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        kernel_capacities=_backend_kernel_capacities(),
    )
    assert result.kernels == (reader,)


def test_factory_instances_with_different_captures_keep_distinct_kernels():
    """Different immutable captures distinguish factory-created operations."""

    def make_helper(entry):
        reader = Kernel(KernelKind.DATA_MOVEMENT)

        @ttl.operation()
        def selected_helper():
            ttl.call_extern_func("reader.hpp", entry, kernel=reader)

        return selected_helper, reader

    first_helper, first_reader = make_helper("first_entry")
    second_helper, second_reader = make_helper("second_entry")

    @ttl.operation(grid=(1, 1))
    def selected_caller():
        first_helper()
        second_helper()

    assert (
        first_helper._spec.operation_identity != second_helper._spec.operation_identity
    )
    assert first_reader != second_reader
    assert set(selected_caller._spec.logical_kernels.values()) == {
        first_reader,
        second_reader,
    }


def test_factory_instances_with_equal_captures_share_logical_identity():
    """Equivalent immutable captures retain deterministic identity."""

    def make_helper(entry):
        reader = Kernel(KernelKind.DATA_MOVEMENT)

        @ttl.operation()
        def selected_helper():
            ttl.call_extern_func("reader.hpp", entry, kernel=reader)

        return selected_helper, reader

    first_helper, first_reader = make_helper("shared_entry")
    second_helper, second_reader = make_helper("shared_entry")

    assert (
        first_helper._spec.operation_identity == second_helper._spec.operation_identity
    )
    assert first_reader == second_reader


def test_factory_instances_with_different_callees_keep_distinct_kernels():
    """Composed operation identity distinguishes generated parent kernels."""

    def make_callee(entry):
        @ttl.operation()
        def selected_callee():
            ttl.call_extern_func(
                "callee.hpp",
                entry,
                kernel=KernelKind.DATA_MOVEMENT,
            )

        return selected_callee

    def make_parent(selected_callee):
        reader = Kernel(KernelKind.DATA_MOVEMENT)

        @ttl.operation()
        def selected_parent():
            selected_callee()
            ttl.call_extern_func("reader.hpp", "reader", kernel=reader)

        return selected_parent, reader

    first_parent, first_reader = make_parent(make_callee("first_entry"))
    second_parent, second_reader = make_parent(make_callee("second_entry"))

    assert (
        first_parent._spec.operation_identity != second_parent._spec.operation_identity
    )
    assert first_reader != second_reader


@pytest.mark.parametrize("capture_kind", ["pipe", "pipenet"])
def test_operation_identity_encodes_pipe_topology(capture_kind):
    """Pipe and PipeNet topology distinguish factory-created operations."""

    def identity_for(destination):
        pipe = ttl.Pipe(src=(0, 0), dst=destination)
        capture = pipe if capture_kind == "pipe" else ttl.PipeNet([pipe])

        def selected_operation():
            return capture

        return _operation_identity(selected_operation)

    assert identity_for((1, 0)) == identity_for((1, 0))
    assert identity_for((1, 0)) != identity_for((2, 0))


def test_operation_identity_encodes_global_semaphore_address(monkeypatch):
    """Global semaphore addresses distinguish compiled template identities."""

    class GlobalSemaphore:
        def __init__(self, address):
            self.address = address

    monkeypatch.setattr(
        kernel_module,
        "is_ttnn_global_semaphore",
        lambda value: isinstance(value, GlobalSemaphore),
    )
    monkeypatch.setattr(
        kernel_module,
        "get_ttnn_global_semaphore_address",
        lambda value: value.address,
    )

    def identity_for(address):
        global_semaphore = GlobalSemaphore(address)

        def selected_operation():
            return global_semaphore

        return _operation_identity(selected_operation)

    assert identity_for(0x1000) == identity_for(0x1000)
    assert identity_for(0x1000) != identity_for(0x2000)


def test_operation_identity_rejects_unsupported_nonlocal_capture():
    """Unsupported captures cannot silently collapse distinct operations."""

    class UnsupportedCapture:
        pass

    unsupported_capture = UnsupportedCapture()

    def selected_operation():
        return unsupported_capture

    with pytest.raises(
        TypeError,
        match=(
            "operation identity cannot encode nonlocal capture "
            "'unsupported_capture' of type UnsupportedCapture"
        ),
    ):
        _operation_identity(selected_operation)


def test_composition_preserves_body_local_kernel_declaration():
    """An inlined local declaration is renamed and rebound to the caller."""

    @ttl.operation()
    def selected_callee():
        local_reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
        ttl.call_extern_func("reader.hpp", "reader", kernel=local_reader)

    @ttl.operation(grid=(1, 1))
    def selected_caller():
        selected_callee()

    spec = selected_caller._spec
    stripped, _, _, logical_kernels = _lift_setup(
        copy.deepcopy(spec.fn_ast),
        spec.frozen_scope,
        spec.operation_identity,
    )
    assert len(logical_kernels) == 1
    local_name, local_reader = next(iter(logical_kernels.items()))
    assert local_name.startswith("local_reader__selected_callee_inl_")
    assert local_reader._operation_identity == spec.operation_identity

    result = split_function_body(
        stripped,
        dfb_param_names=set(),
        logical_kernels=logical_kernels,
        kernel_capacities=_backend_kernel_capacities(),
    )
    assert result.kernels == (local_reader,)


def test_bound_kernel_equality_uses_logical_identity():
    """Equivalent bindings compare equally without object-address identity."""
    first = Kernel(KernelKind.DATA_MOVEMENT)
    second = Kernel(KernelKind.DATA_MOVEMENT)
    first._bind("reader", "operation")
    second._bind("reader", "operation")

    assert first == second
    assert hash(first) == hash(second)


def test_split_analysis_does_not_mutate_input_ast():
    """Planning and application leave the source AST unchanged."""
    function = _fn(
        """
        def k():
            ttl.call_extern_func(
                "shared.hpp",
                "shared",
                kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
            )
        """
    )
    before = ast.dump(function, include_attributes=True)

    result = split_function_body(function, dfb_param_names=set())

    assert ast.dump(function, include_attributes=True) == before
    assert result.plan.statements[0].source_line == 3
    assert result.plan.kernel_requirements == (
        (KernelKind.COMPUTE, 1),
        (KernelKind.DATA_MOVEMENT, 1),
    )


def test_unknown_ttl_op_is_rejected():
    fn = _fn(
        """
        def k(a):
            ttl.frobnicate(a)
        """
    )
    with pytest.raises(ValueError, match="unknown ttl.frobnicate"):
        split_function_body(fn, dfb_param_names=set())


def test_raw_addr_is_a_kernel_neutral_scalar_producer():
    fn = _fn(
        """
        def k(inp):
            call_extern_func(
                "header.hpp",
                "kernel",
                func_args=[ttl.raw_addr(inp)],
                kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
            )
        """
    )

    result = split_function_body(fn, dfb_param_names=set())

    for kind in (KernelKind.COMPUTE, KernelKind.DATA_MOVEMENT):
        kernel_source = _kind_src(result, kind)
        assert "call_extern_func" in kernel_source
        assert "ttl.raw_addr(inp)" in kernel_source
        assert "kernel=" not in kernel_source


def test_tensor_backed_dfb_factory_and_publish_are_split_by_kernel():
    fn = _fn(
        """
        def k(inp):
            inp_dfb = ttl.make_tensor_backed_dfb(inp, shape=(1, 1))
            value = ttl.exp(inp)
            inp_dfb.publish()
        """
    )

    result = split_function_body(
        fn,
        dfb_param_names=set(),
        local_dfb_names={"inp_dfb"},
    )

    compute = _kind_src(result, KernelKind.COMPUTE)
    data_movement = _kind_src(result, KernelKind.DATA_MOVEMENT)
    assert "make_tensor_backed_dfb" in compute
    assert "make_tensor_backed_dfb" in data_movement
    assert "inp_dfb.publish()" not in compute
    assert "inp_dfb.publish()" in data_movement


def test_producer_with_no_uses_is_rejected():
    fn = _fn(
        """
        def k():
            blk = a_cb.wait()
        """
    )
    with pytest.raises(ValueError, match="has no uses"):
        split_function_body(fn, dfb_param_names=set(), local_dfb_names={"a_cb"})


def test_producer_split_across_data_movement_kernels_is_rejected():
    """One reserve cannot feed two distinct data-movement callbacks."""
    fn = _fn(
        """
        def k(net):
            shared = a_cb.reserve()

            def send(pipe):
                ttl.copy(shared, pipe)

            net.if_src(send)

            def recv(pipe):
                ttl.copy(pipe, shared)

            net.if_dst(recv)
        """
    )
    with pytest.raises(ValueError, match="multiple logical kernels .*data_movement"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            local_dfb_names={"a_cb"},
        )


def test_statement_mixing_compute_and_data_movement_is_rejected():
    """One statement cannot contain compute and data-movement work."""
    fn = _fn(
        """
        def k(x, out):
            dst = out_cb.reserve()
            ttl.copy(ttl.exp(x), dst)
        """
    )

    with pytest.raises(ValueError, match="statement is assigned to multiple logical"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            local_dfb_names={"out_cb"},
        )


def test_acquired_block_used_by_compute_and_data_movement_is_rejected():
    """A DFB acquire cannot be cloned onto compute and data movement."""
    fn = _fn(
        """
        def k(x, out):
            shared = out_cb.reserve()
            shared.store(x)
            ttl.copy(shared, out)
        """
    )

    with pytest.raises(ValueError, match="multiple logical kernels .*compute"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            local_dfb_names={"out_cb"},
        )


def test_store_assigns_source_and_destination_blocks_to_compute():
    """A direct block store owns both DFB transactions on compute."""
    fn = _fn(
        """
        def k():
            source = source_dfb.wait()
            destination = destination_dfb.reserve()
            destination.store(source)
            source.pop()
            destination.push()
        """
    )

    result = split_function_body(
        fn,
        dfb_param_names={"source_dfb", "destination_dfb"},
    )

    compute = _kind_src(result, KernelKind.COMPUTE)
    data_movement = _kind_src(result, KernelKind.DATA_MOVEMENT)
    assert "source_dfb.wait()" in compute
    assert "destination_dfb.reserve()" in compute
    assert "destination.store(source)" in compute
    assert "source.pop()" in compute
    assert "destination.push()" in compute
    assert "source_dfb.wait()" not in data_movement


def test_with_acquired_block_used_by_multiple_kernels_is_rejected():
    """The scoped DFB acquire form has the same single-kernel requirement."""
    fn = _fn(
        """
        def k(x, out):
            with out_cb.reserve() as shared:
                shared.store(x)
                ttl.copy(shared, out)
        """
    )

    with pytest.raises(ValueError, match="DFB block.*multiple logical kernels"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            local_dfb_names={"out_cb"},
        )


def test_assigned_copy_transfer_handle_is_rejected():
    """Assigned transfer handles remain unsupported until alias tracking lands."""
    fn = _fn(
        """
        def k(x, out):
            tx = ttl.copy(x, out)
            tx.wait()
        """
    )

    with pytest.raises(ValueError, match="assigned transfer handle"):
        split_function_body(
            fn,
            dfb_param_names=set(),
        )


def test_chained_copy_wait_routes_to_data_movement():
    """The supported non-assigned transfer wait remains on data movement."""
    fn = _fn(
        """
        def k(x, out):
            ttl.copy(x, out).wait()
        """
    )

    result = split_function_body(
        fn,
        dfb_param_names=set(),
    )

    assert "ttl.copy" not in _kind_src(result, KernelKind.COMPUTE)
    assert "ttl.copy" in _kind_src(result, KernelKind.DATA_MOVEMENT)


def test_read_index_routes_to_data_movement():
    """Tensor-provided indices remain with their dataflow buffer acquire."""
    fn = _fn(
        """
        def k(weights, output):
            index_block = index_dfb.wait()
            slot = ttl.read_index(index_block, 0, 0)
            ttl.copy(weights[slot], output)
        """
    )

    result = split_function_body(
        fn,
        dfb_param_names=set(),
        local_dfb_names={"index_dfb"},
    )

    assert "ttl.read_index" not in _kind_src(result, KernelKind.COMPUTE)
    assert "ttl.read_index" in _kind_src(result, KernelKind.DATA_MOVEMENT)


def test_compute_and_data_movement_route_to_separate_logical_kernels():
    """Copies and compute operations receive distinct logical kernels."""
    fn = _fn(
        """
        def k(a, out):
            a_blk = a_cb.reserve()
            ttl.copy(a, a_blk)
            s = out_cb.reserve()
            x = a_cb.wait()
            s.store(ttl.exp(x))
            done = out_cb.wait()
            ttl.copy(done, out)
        """
    )
    result = split_function_body(
        fn,
        dfb_param_names=set(),
        local_dfb_names={"a_cb", "out_cb"},
    )

    compute = _kind_src(result, KernelKind.COMPUTE)
    data_movement = _kind_src(result, KernelKind.DATA_MOVEMENT)

    assert "ttl.exp" in compute
    assert "ttl.copy" not in compute
    assert "ttl.copy" in data_movement
    assert "ttl.exp" not in data_movement


def test_external_call_selects_canonical_compute_kernel():
    """A kind selector emits an opaque call only in the canonical kernel."""
    fn = _fn(
        """
        def k():
            ttl.call_extern_func("compute.hpp", "compute", kernel=ttl.KernelKind.COMPUTE)
        """
    )

    result = split_function_body(fn, dfb_param_names=set())

    compute = _kernel_src(result, KernelKind.COMPUTE)
    data_movement = _kernel_src(result, KernelKind.DATA_MOVEMENT)
    assert "call_extern_func" in compute
    assert "kernel=" not in compute
    assert "call_extern_func" not in data_movement


def test_direct_external_call_selects_canonical_data_movement_kernel():
    """The directly imported call form accepts the same kind selector."""
    fn = _fn(
        """
        def k():
            call_extern_func(
                "reader.hpp", "reader", kernel=KernelKind.DATA_MOVEMENT
            )
        """
    )

    result = split_function_body(fn, dfb_param_names=set())

    compute = _kernel_src(result, KernelKind.COMPUTE)
    data_movement = _kernel_src(result, KernelKind.DATA_MOVEMENT)
    assert "call_extern_func" not in compute
    assert "call_extern_func" in data_movement
    assert "kernel=" not in data_movement


def test_external_call_selects_named_logical_kernel():
    """A logical handle distinguishes a noncanonical kernel of the same kind."""
    reader = _logical_kernel(KernelKind.DATA_MOVEMENT, "reader")
    fn = _fn(
        """
        def k():
            ttl.call_extern_func("reader.hpp", "reader", kernel=reader)
        """
    )

    result = split_function_body(
        fn,
        dfb_param_names=set(),
        logical_kernels={"reader": reader},
    )

    assert "call_extern_func" in _kernel_src(result, reader)
    assert "call_extern_func" not in _kernel_src(result, KernelKind.DATA_MOVEMENT)


def test_two_named_data_movement_kernels_remain_distinct():
    """Two handles of one kind receive only their selected statements."""
    reader = _logical_kernel(KernelKind.DATA_MOVEMENT, "reader")
    writer = _logical_kernel(KernelKind.DATA_MOVEMENT, "writer")
    fn = _fn(
        """
        def k():
            ttl.call_extern_func("reader.hpp", "reader", kernel=reader)
            ttl.call_extern_func("writer.hpp", "writer", kernel=writer)
        """
    )

    result = split_function_body(
        fn,
        dfb_param_names=set(),
        logical_kernels={"reader": reader, "writer": writer},
    )

    reader_source = _kernel_src(result, reader)
    writer_source = _kernel_src(result, writer)
    assert "'reader'" in reader_source
    assert "'writer'" not in reader_source
    assert "'writer'" in writer_source
    assert "'reader'" not in writer_source


def test_external_call_tuple_selects_multiple_logical_kernels():
    """An external tuple emits one stripped call in every selected kernel."""
    reader = _logical_kernel(KernelKind.DATA_MOVEMENT, "reader")
    fn = _fn(
        """
        def k():
            ttl.call_extern_func(
                "shared.hpp",
                "shared",
                kernel=(ttl.KernelKind.COMPUTE, reader),
            )
        """
    )

    result = split_function_body(
        fn,
        dfb_param_names=set(),
        logical_kernels={"reader": reader},
    )

    for kernel in (KernelKind.COMPUTE, reader):
        source = _kernel_src(result, kernel)
        assert source.count("call_extern_func") == 1
        assert "kernel=" not in source


def test_external_call_kind_union_selects_multiple_logical_kernels():
    """A kind union selects both canonical logical kernels."""
    fn = _fn(
        """
        def k():
            ttl.call_extern_func(
                "shared.hpp",
                "shared",
                kernel=ttl.KernelKind.COMPUTE | ttl.KernelKind.DATA_MOVEMENT,
            )
        """
    )

    result = split_function_body(fn, dfb_param_names=set())

    assert result.kernels == (
        KernelKind.COMPUTE,
        KernelKind.DATA_MOVEMENT,
    )
    for kernel in result.kernels:
        source = _kernel_src(result, kernel)
        assert source.count("call_extern_func") == 1
        assert "kernel=" not in source


def test_kernel_kind_union_builds_tuple_selection():
    """The public union expression evaluates to an accepted selector tuple."""
    assert KernelKind.COMPUTE | KernelKind.DATA_MOVEMENT == (
        KernelKind.COMPUTE,
        KernelKind.DATA_MOVEMENT,
    )


def test_external_call_kind_union_rejects_named_kernel():
    """Named logical kernels remain explicit tuple elements."""
    reader = _logical_kernel(KernelKind.DATA_MOVEMENT, "reader")
    fn = _fn(
        """
        def k():
            ttl.call_extern_func(
                "shared.hpp",
                "shared",
                kernel=ttl.KernelKind.COMPUTE | reader,
            )
        """
    )

    with pytest.raises(ValueError, match="kind union operands must be KernelKind"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            logical_kernels={"reader": reader},
        )


@pytest.mark.parametrize(
    "selector, message",
    [
        ("()", "nonempty tuple"),
        (
            "(ttl.KernelKind.COMPUTE, ttl.KernelKind.COMPUTE)",
            "duplicate kernel selector",
        ),
    ],
)
def test_external_call_rejects_invalid_selector_tuple(selector, message):
    """External tuples must be nonempty and unique after canonicalization."""
    fn = _fn(
        f"""
        def k():
            ttl.call_extern_func("shared.hpp", "shared", kernel={selector})
        """
    )

    with pytest.raises(ValueError, match=message):
        split_function_body(fn, dfb_param_names=set())


@pytest.mark.parametrize("method", ["push", "pop"])
def test_release_rejects_tuple_selector(method):
    """A DFB release executes once and therefore rejects tuple selection."""
    acquire = "reserve" if method == "push" else "wait"
    fn = _fn(
        f"""
        def k():
            block = buffer.{acquire}()
            block.{method}(kernel=(ttl.KernelKind.DATA_MOVEMENT,))
        """
    )

    with pytest.raises(ValueError, match="accepts one kernel selector"):
        split_function_body(
            fn,
            dfb_param_names={"buffer"},
        )


@pytest.mark.parametrize("acquire", ["reserve", "wait"])
@pytest.mark.parametrize("form", ["assign", "with"])
def test_acquire_rejects_kernel_selector(acquire, form):
    """DFB acquisition ownership cannot be selected directly."""
    if form == "assign":
        source = f"""
            def k():
                block = buffer.{acquire}(kernel=ttl.KernelKind.DATA_MOVEMENT)
        """
    else:
        source = f"""
            def k():
                with buffer.{acquire}(
                    kernel=ttl.KernelKind.DATA_MOVEMENT
                ) as block:
                    block.pop()
        """
    fn = _fn(source)

    with pytest.raises(
        ValueError,
        match=rf"kernel= is not supported on DFB {acquire}\(\)",
    ):
        split_function_body(fn, dfb_param_names={"buffer"})


@pytest.mark.parametrize("method", ["push", "pop"])
def test_release_rejects_kind_union(method):
    """A DFB release executes in exactly one logical kernel."""
    fn = _fn(
        f"""
        def k():
            block = buffer.reserve()
            block.{method}(
                kernel=ttl.KernelKind.COMPUTE | ttl.KernelKind.DATA_MOVEMENT
            )
        """
    )

    with pytest.raises(
        ValueError, match="accepts one kernel selector, not a kind union"
    ):
        split_function_body(fn, dfb_param_names={"buffer"})


@pytest.mark.parametrize("acquire", ["reserve", "wait"])
@pytest.mark.parametrize(
    "selector",
    [
        "ttl.KernelKind.COMPUTE",
        "named_compute",
        "(ttl.KernelKind.DATA_MOVEMENT, ttl.KernelKind.COMPUTE)",
    ],
)
def test_acquire_block_rejects_nested_selection_outside_owner(acquire, selector):
    """A selected nested statement must execute with its DFB acquire."""
    named_compute = _logical_kernel(KernelKind.COMPUTE, "named_compute")
    fn = _fn(
        f"""
        def k(inp):
            with buffer.{acquire}() as block:
                ttl.copy(inp, block).wait()
                ttl.call_extern_func("compute.hpp", "compute", kernel={selector})
        """
    )

    with pytest.raises(ValueError, match="outside its enclosing DFB acquire owner"):
        split_function_body(
            fn,
            dfb_param_names={"buffer"},
            logical_kernels={"named_compute": named_compute},
        )


@pytest.mark.parametrize(
    "acquire, release",
    [("reserve", "push"), ("wait", "pop")],
)
def test_release_without_selector_retains_inferred_ownership(acquire, release):
    """Argument-free releases retain ownership inferred from block uses."""
    fn = _fn(
        f"""
        def k(source):
            block = buffer.{acquire}()
            ttl.copy(source, block).wait()
            block.{release}()
        """
    )

    result = split_function_body(fn, dfb_param_names={"buffer"})
    data_movement = _kernel_src(result, KernelKind.DATA_MOVEMENT)
    compute = _kernel_src(result, KernelKind.COMPUTE)
    assert f"buffer.{acquire}()" in data_movement
    assert f"block.{release}()" in data_movement
    assert f"block.{release}()" not in compute


@pytest.mark.parametrize(
    "acquire, release",
    [("reserve", "push"), ("wait", "pop")],
)
def test_otherwise_unused_block_accepts_explicit_release_owner(acquire, release):
    """An explicit kind assigns an otherwise unreferenced DFB transaction."""
    fn = _fn(
        f"""
        def k():
            block = buffer.{acquire}()
            block.{release}(kernel=ttl.KernelKind.DATA_MOVEMENT)
        """
    )

    result = split_function_body(fn, dfb_param_names={"buffer"})
    data_movement = _kernel_src(result, KernelKind.DATA_MOVEMENT)
    compute = _kernel_src(result, KernelKind.COMPUTE)
    assert f"buffer.{acquire}()" in data_movement
    assert f"block.{release}()" in data_movement
    assert "kernel=" not in data_movement
    assert f"buffer.{acquire}()" not in compute


def test_explicit_release_conflicting_with_inferred_owner_is_rejected():
    """Explicit and inferred DFB release ownership must agree."""
    fn = _fn(
        """
        def k(value):
            block = buffer.reserve()
            block.store(value)
            block.push(kernel=ttl.KernelKind.DATA_MOVEMENT)
        """
    )

    with pytest.raises(ValueError, match="explicit.*data_movement.*inferred.*compute"):
        split_function_body(fn, dfb_param_names={"buffer"})


def test_external_call_requires_inferred_or_explicit_kernel():
    """An unselected top-level opaque call is never cloned speculatively."""
    fn = _fn(
        """
        def k():
            ttl.call_extern_func("opaque.hpp", "opaque")
        """
    )

    with pytest.raises(ValueError, match="call_extern_func.*kernel selector"):
        split_function_body(fn, dfb_param_names=set())


@pytest.mark.parametrize("selector", ["42", "selected_kernel()"])
def test_external_call_rejects_nonconstant_selector(selector):
    """Only enum members and lifted logical handles are valid selectors."""
    fn = _fn(
        f"""
        def k():
            ttl.call_extern_func("opaque.hpp", "opaque", kernel={selector})
        """
    )

    with pytest.raises(ValueError, match="KernelKind or Kernel"):
        split_function_body(fn, dfb_param_names=set())


def test_external_call_accepts_kernel_kind_import_alias():
    """Selector resolution uses the frozen value rather than its spelling."""
    fn = _fn(
        """
        def k():
            ttl.call_extern_func("compute.hpp", "compute", kernel=KK.COMPUTE)
        """
    )

    result = split_function_body(
        fn,
        dfb_param_names=set(),
        selector_scope={"KK": KernelKind},
    )

    assert result.kernels == (KernelKind.COMPUTE,)


def test_external_call_respects_rebound_kernel_kind_name():
    """A rebound name is not interpreted as the public selector enum."""

    class ReboundKernelKind:
        COMPUTE = 42

    fn = _fn(
        """
        def k():
            ttl.call_extern_func(
                "compute.hpp", "compute", kernel=KernelKind.COMPUTE
            )
        """
    )

    with pytest.raises(ValueError, match="KernelKind or Kernel.*got int"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            selector_scope={"KernelKind": ReboundKernelKind},
        )


def test_selector_tuple_order_does_not_change_kernel_order():
    """Canonical kernel ordering is independent of selector tuple order."""
    reader = _logical_kernel(KernelKind.DATA_MOVEMENT, "reader")
    writer = _logical_kernel(KernelKind.DATA_MOVEMENT, "writer")
    logical_kernels = {"reader": reader, "writer": writer}
    first = _fn(
        """
        def k():
            ttl.call_extern_func("shared.hpp", "shared", kernel=(reader, writer))
        """
    )
    second = _fn(
        """
        def k():
            ttl.call_extern_func("shared.hpp", "shared", kernel=(writer, reader))
        """
    )

    first_result = split_function_body(
        first, dfb_param_names=set(), logical_kernels=logical_kernels
    )
    second_result = split_function_body(
        second, dfb_param_names=set(), logical_kernels=logical_kernels
    )

    assert first_result.kernels == second_result.kernels
    assert [_kernel_src(first_result, kernel) for kernel in first_result.kernels] == [
        _kernel_src(second_result, kernel) for kernel in second_result.kernels
    ]


def test_named_kernel_order_uses_operation_identity_to_break_name_ties():
    """Distinct operations provide a total order for same-named kernels."""
    operation_b_reader = Kernel(KernelKind.DATA_MOVEMENT)
    operation_b_reader._bind("reader", "operation.b")
    operation_a_reader = Kernel(KernelKind.DATA_MOVEMENT)
    operation_a_reader._bind("reader", "operation.a")
    function = _fn(
        """
        def k():
            ttl.call_extern_func("reader.hpp", "b", kernel=operation_b_reader)
            ttl.call_extern_func("reader.hpp", "a", kernel=operation_a_reader)
        """
    )

    result = split_function_body(
        function,
        dfb_param_names=set(),
        logical_kernels={
            "operation_b_reader": operation_b_reader,
            "operation_a_reader": operation_a_reader,
        },
    )

    assert result.kernels == (operation_a_reader, operation_b_reader)
    assert tuple(_assign_backend_kernel_slots(result).values()) == result.kernels


def test_logical_kernel_capacity_uses_supplied_target_limit():
    """Capacity diagnostics use backend-provided limits for each kernel kind."""
    readers = {
        name: _logical_kernel(KernelKind.DATA_MOVEMENT, name)
        for name in ("first", "second")
    }
    fn = _fn(
        """
        def k():
            ttl.call_extern_func("first.hpp", "first", kernel=first)
            ttl.call_extern_func("second.hpp", "second", kernel=second)
        """
    )

    with pytest.raises(
        ValueError,
        match="2 data_movement kernels.*target supports 1",
    ):
        split_function_body(
            fn,
            dfb_param_names=set(),
            logical_kernels=readers,
            kernel_capacities={
                KernelKind.COMPUTE: 1,
                KernelKind.DATA_MOVEMENT: 1,
            },
        )


def test_kernel_capacity_diagnostic_names_conflicts_at_last_introduction():
    """Capacity failures identify the selected kernels and later statement."""
    extra_compute = _logical_kernel(KernelKind.COMPUTE, "extra_compute")
    fn = _fn(
        """
        def k(value):
            ttl.call_extern_func("compute.hpp", "compute", kernel=extra_compute)
            ttl.fill(value)
        """
    )

    with pytest.raises(ValueError) as error:
        split_function_body(
            fn,
            dfb_param_names=set(),
            logical_kernels={"extra_compute": extra_compute},
            kernel_capacities={
                KernelKind.COMPUTE: 1,
                KernelKind.DATA_MOVEMENT: 2,
            },
        )

    message = str(error.value)
    assert "selected kernels: compute, compute kernel 'extra_compute'" in message
    assert "(line 4)" in message
