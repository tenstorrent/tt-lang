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
import textwrap

import pytest
import ttl

from ttl._src.atom_split import split_function_body
from ttl.atom import (
    _assign_backend_kernel_slots,
    _backend_kernel_capacities,
    _bind_logical_kernels,
    _build_atom_spec,
    _lift_setup,
)
from ttl.kernel import Kernel, KernelKind


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
    return Kernel(kind)._bind(name, "test.operation")


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


def test_captured_kernel_is_bound_only_for_final_operation():
    sender = Kernel(KernelKind.DATA_MOVEMENT)

    def operation():
        ttl.call_extern_func("sender.hpp", "sender", kernel=sender)

    spec = _build_atom_spec(operation)

    assert spec.logical_kernels == {"sender": sender}
    with pytest.raises(ValueError, match="no operation-local identity"):
        sender.identity

    bound_kernels = _bind_logical_kernels(
        spec.logical_kernels,
        spec.operation_identity,
    )
    bound_sender = bound_kernels["sender"]
    assert bound_sender is not sender
    assert bound_sender.identity == "sender"
    assert bound_sender._operation_identity == spec.operation_identity


def test_composed_operations_share_one_factory_owned_kernel():
    sender = Kernel(KernelKind.DATA_MOVEMENT)

    @ttl.operation()
    def first_operation():
        ttl.call_extern_func("sender.hpp", "first", kernel=sender)

    @ttl.operation()
    def second_operation():
        ttl.call_extern_func("sender.hpp", "second", kernel=sender)

    def composed_operation():
        first_operation()
        second_operation()

    spec = _build_atom_spec(composed_operation)
    assert tuple(spec.logical_kernels.values()) == (sender,)

    bound_kernels = _bind_logical_kernels(
        spec.logical_kernels,
        spec.operation_identity,
    )
    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=bound_kernels,
        kernel_capacities=_backend_kernel_capacities(),
    )

    assert result.kernels == tuple(bound_kernels.values())
    kernel_source = _kernel_src(result, result.kernels[0])
    assert "'first'" in kernel_source
    assert "'second'" in kernel_source
    with pytest.raises(ValueError, match="no operation-local identity"):
        sender.identity


def test_bound_kernel_equality_includes_operation_identity():
    first = Kernel(KernelKind.DATA_MOVEMENT)._bind("sender", "first.operation")
    same = Kernel(KernelKind.DATA_MOVEMENT)._bind("sender", "first.operation")
    second = Kernel(KernelKind.DATA_MOVEMENT)._bind("sender", "second.operation")

    assert first == same
    assert hash(first) == hash(same)
    assert first != second


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
