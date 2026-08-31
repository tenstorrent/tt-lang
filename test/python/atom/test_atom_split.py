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
import ttl.atom as atom_module
import ttl.kernel as kernel_module

from ttl._src import atom_rules
from ttl._src.atom_split import split_function_body
from ttl.atom import (
    _assign_backend_kernel_slots,
    _backend_kernel_bodies,
    _backend_kernel_capacities,
    _build_atom_spec,
    _lift_setup,
)
from ttl.compiler_options import CompilerOptions
from ttl.dfb_allocation_group import (
    _bind_dfb_allocation_groups,
    _dfb_allocation_group_binding_scope,
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


def test_captured_fabric_manager_claim_binds_to_selected_kernel():
    """A claim and its selected logical kernel share operation ownership."""
    reader = Kernel(KernelKind.DATA_MOVEMENT)
    manager = ttl.FabricManagerClaim("external", kernel=reader)

    def operation():
        ttl.call_extern_func(
            "reader.hpp",
            "open",
            kernel=reader,
            fabric_manager_effects=(manager.acquire(),),
        )

    spec = _build_atom_spec(operation)

    assert spec.logical_kernels["reader"] is reader
    assert spec.fabric_manager_claims["manager"] is manager
    assert manager.operation_identity == spec.operation_identity
    assert manager.kernel is reader


def test_composition_binds_fabric_manager_claim_to_final_operation():
    """Separate lifetime helpers forward one claim to the final operation."""
    manager = ttl.FabricManagerClaim("external", kernel=ttl.PIPE_SOURCE_KERNEL)

    @ttl.operation()
    def open_manager():
        ttl.call_extern_func(
            "reader.hpp",
            "open",
            kernel=ttl.PIPE_SOURCE_KERNEL,
            fabric_manager_effects=(manager.acquire(),),
        )

    @ttl.operation()
    def use_manager():
        ttl.call_extern_func(
            "reader.hpp",
            "use",
            kernel=ttl.PIPE_SOURCE_KERNEL,
            fabric_manager_effects=(manager.use(),),
        )

    @ttl.operation()
    def close_manager():
        ttl.call_extern_func(
            "reader.hpp",
            "close",
            kernel=ttl.PIPE_SOURCE_KERNEL,
            fabric_manager_effects=(manager.release(),),
        )

    with pytest.raises(ValueError, match="has no operation identity"):
        manager.operation_identity

    @ttl.operation(grid=(1, 1))
    def composed_manager():
        open_manager()
        use_manager()
        close_manager()

    assert tuple(composed_manager._spec.fabric_manager_claims.values()) == (manager,)
    assert manager.operation_identity == composed_manager._spec.operation_identity


def test_composed_claim_can_select_callee_owned_kernel():
    """A final claim may select a logical kernel retained from its helper."""
    reader = Kernel(KernelKind.DATA_MOVEMENT)
    manager = ttl.FabricManagerClaim("external", kernel=reader)

    @ttl.operation()
    def selected_helper():
        ttl.call_extern_func(
            "reader.hpp",
            "open_and_close",
            kernel=reader,
            fabric_manager_effects=(manager.scoped(),),
        )

    @ttl.operation(grid=(1, 1))
    def selected_parent():
        selected_helper()

    assert reader._operation_identity == selected_helper._spec.operation_identity
    assert manager.operation_identity == selected_parent._spec.operation_identity
    assert selected_parent._spec.fabric_manager_claims


def test_composed_claim_cannot_bind_to_two_final_operations():
    """One claim remains local to one executable operation."""
    manager = ttl.FabricManagerClaim("external", kernel=ttl.PIPE_SOURCE_KERNEL)

    @ttl.operation()
    def manager_helper():
        ttl.call_extern_func(
            "reader.hpp",
            "open_and_close",
            kernel=ttl.PIPE_SOURCE_KERNEL,
            fabric_manager_effects=(manager.scoped(),),
        )

    @ttl.operation(grid=(1, 1))
    def first_parent():
        manager_helper()

    assert manager.operation_identity == first_parent._spec.operation_identity

    with pytest.raises(ValueError, match="already bound to operation"):

        @ttl.operation(grid=(1, 1))
        def second_parent():
            manager_helper()


def test_composed_claim_names_are_unique_in_final_operation():
    """Distinct composed claims cannot share one final operation identity."""
    first_manager = ttl.FabricManagerClaim("external", kernel=ttl.PIPE_SOURCE_KERNEL)
    second_manager = ttl.FabricManagerClaim("external", kernel=ttl.PIPE_SOURCE_KERNEL)

    @ttl.operation()
    def first_helper():
        ttl.call_extern_func(
            "reader.hpp",
            "first",
            kernel=ttl.PIPE_SOURCE_KERNEL,
            fabric_manager_effects=(first_manager.scoped(),),
        )

    @ttl.operation()
    def second_helper():
        ttl.call_extern_func(
            "reader.hpp",
            "second",
            kernel=ttl.PIPE_SOURCE_KERNEL,
            fabric_manager_effects=(second_manager.scoped(),),
        )

    with pytest.raises(ValueError, match="identity 'external' is declared by"):

        @ttl.operation(grid=(1, 1))
        def conflicting_parent():
            first_helper()
            second_helper()


def test_captured_fabric_manager_claim_cannot_have_two_names():
    """One claim identity has one source name in its operation."""
    reader = Kernel(KernelKind.DATA_MOVEMENT)
    manager = ttl.FabricManagerClaim("external", kernel=reader)
    manager_alias = manager

    def operation():
        ttl.call_extern_func(
            "reader.hpp",
            "open",
            kernel=reader,
            fabric_manager_effects=(manager.acquire(), manager_alias.use()),
        )

    with pytest.raises(ValueError, match="multiple names"):
        _build_atom_spec(operation)
    with pytest.raises(ValueError, match="has no operation identity"):
        manager.operation_identity


def test_expand_only_fabric_manager_claim_cannot_have_two_names():
    """Expand-only registration rejects claim aliases before composition."""
    manager = ttl.FabricManagerClaim("external", kernel=ttl.PIPE_SOURCE_KERNEL)
    manager_alias = manager

    with pytest.raises(ValueError, match="multiple names"):

        @ttl.operation()
        def manager_helper():
            ttl.call_extern_func(
                "reader.hpp",
                "open",
                kernel=ttl.PIPE_SOURCE_KERNEL,
                fabric_manager_effects=(
                    manager.acquire(),
                    manager_alias.use(),
                ),
            )


def test_external_fabric_manager_effect_must_select_claim_kernel():
    """A manager effect and its external call select the same kernel."""
    reader = _logical_kernel(KernelKind.DATA_MOVEMENT, "reader")
    writer = _logical_kernel(KernelKind.DATA_MOVEMENT, "writer")
    manager = ttl.FabricManagerClaim("external", kernel=reader)
    function = _fn(
        """
        def operation():
            ttl.call_extern_func(
                "writer.hpp",
                "open",
                kernel=writer,
                fabric_manager_effects=(manager.acquire(),),
            )
        """
    )
    with pytest.raises(
        ValueError,
        match=(
            "fabric manager claim 'external' selects data_movement kernel "
            "'reader', but the external call selects data_movement kernel "
            "'writer'"
        ),
    ):
        split_function_body(
            function,
            dfb_param_names=set(),
            logical_kernels={"reader": reader, "writer": writer},
            selector_scope={
                "reader": reader,
                "writer": writer,
                "manager": manager,
            },
            kernel_capacities=_backend_kernel_capacities(),
        )


def test_inferred_external_fabric_effect_must_select_claim_kernel():
    """Inferred external-call placement also validates manager ownership."""
    reader = _logical_kernel(KernelKind.DATA_MOVEMENT, "reader")
    manager = ttl.FabricManagerClaim("external", kernel=reader)
    function = _fn(
        """
        def operation():
            def receive(pipe):
                ttl.call_extern_func(
                    "receiver.hpp",
                    "open",
                    fabric_manager_effects=(manager.acquire(),),
                )
            exchange_net.if_dst(receive)
        """
    )

    with pytest.raises(
        ValueError,
        match=(
            "fabric manager claim 'external' selects data_movement kernel "
            "'reader', but the external call selects data_movement"
        ),
    ):
        split_function_body(
            function,
            dfb_param_names=set(),
            logical_kernels={"reader": reader},
            selector_scope={"reader": reader, "manager": manager},
            kernel_capacities=_backend_kernel_capacities(),
        )


def test_unified_operation_propagates_runtime_resource_factory(monkeypatch):
    observed = {}

    def make_resources(**_kwargs):
        return None

    def fake_compile_atom(*_args, **kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(atom_module, "_compile_atom", fake_compile_atom)
    result = atom_module._compile_unified_operation(
        object(),
        {
            "num_outs": 1,
            "memory_space": "L1",
            "tiled": True,
            "fp32_dest_acc_en": None,
            "dst_full_sync_en": None,
            "math_fidelity": None,
            "device_domain": None,
            "runtime_resource_factory": make_resources,
        },
        (),
        {},
        (1, 1),
        1,
        None,
        CompilerOptions(),
        0,
    )

    assert result is not None
    assert observed["runtime_resource_factory"] is make_resources


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


def test_reset_owned_kernel_cannot_bind_to_two_operations():
    """Reset metadata cannot transfer a bound kernel to another operation."""
    compute_kernel = Kernel(KernelKind.COMPUTE)
    reader_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    writer_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))

    @ttl.operation()
    def first_operation():
        ttl.reset_all_dfbs(reset)

    with pytest.raises(TypeError, match="already bound to operation"):

        @ttl.operation()
        def second_operation():
            ttl.reset_all_dfbs(reset)


def test_reset_participant_membership_changes_operation_identity():
    """Anonymous reset participant membership contributes to identity."""

    def make_reset_operation(shared_participants):
        first_participants = (
            Kernel(KernelKind.COMPUTE),
            Kernel(KernelKind.DATA_MOVEMENT),
            Kernel(KernelKind.DATA_MOVEMENT),
        )
        second_participants = (
            first_participants
            if shared_participants
            else (
                Kernel(KernelKind.COMPUTE),
                Kernel(KernelKind.DATA_MOVEMENT),
                Kernel(KernelKind.DATA_MOVEMENT),
            )
        )
        first_reset = ttl.DFBReset(participants=first_participants)
        second_reset = ttl.DFBReset(participants=second_participants)

        @ttl.operation()
        def reset_operation():
            ttl.reset_all_dfbs(first_reset)
            ttl.reset_all_dfbs(second_reset)

        return reset_operation

    shared_operation = make_reset_operation(shared_participants=True)
    independent_operation = make_reset_operation(shared_participants=False)

    assert (
        shared_operation._spec.operation_identity
        != independent_operation._spec.operation_identity
    )


def test_reset_only_kernel_names_ignore_participant_order():
    """Reset participant tuples represent sets in operation identity."""

    def make_reset_operation(reverse_participants):
        compute_participant = Kernel(KernelKind.COMPUTE)
        reader_participant = Kernel(KernelKind.DATA_MOVEMENT)
        writer_participant = Kernel(KernelKind.DATA_MOVEMENT)
        participants = (
            compute_participant,
            reader_participant,
            writer_participant,
        )
        if reverse_participants:
            participants = tuple(reversed(participants))
        first_reset = ttl.DFBReset(participants=participants)

        @ttl.operation()
        def reset_operation():
            ttl.reset_all_dfbs(first_reset)

        return reset_operation

    forward_operation = make_reset_operation(False)
    reversed_operation = make_reset_operation(True)

    assert (
        forward_operation._spec.operation_identity
        == reversed_operation._spec.operation_identity
    )


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


def test_operation_identity_encodes_graph_pipenet_topology():
    """Graph PipeNet identity includes its domain and transfer relation."""

    def identity_for(destination):
        domain = ttl.DeviceDomain((1, 3))
        graph = ttl.TransferGraph.edges(
            domain,
            edges=[((0, 0), destination)],
        )
        pipe_net = ttl.PipeNet(graph=graph)

        def selected_operation():
            return pipe_net

        return _operation_identity(selected_operation)

    assert identity_for((0, 1)) == identity_for((0, 1))
    assert identity_for((0, 1)) != identity_for((0, 2))


def test_operation_identity_encodes_device_domain():
    """Device-domain components distinguish factory-created operations."""

    def identity_for(domain):
        def selected_operation():
            return domain

        return _operation_identity(selected_operation)

    regular = ttl.DeviceDomain((2, 3), name="worker")
    same_regular = ttl.DeviceDomain((2, 3), name="worker")
    different_extent = ttl.DeviceDomain((2, 4), name="worker")
    product = ttl.DeviceDomain.product(
        rack=ttl.DeviceDomain((2,)),
        worker=ttl.DeviceDomain((3,)),
    )

    assert identity_for(regular) == identity_for(same_regular)
    assert identity_for(regular) != identity_for(different_extent)
    assert identity_for(regular) != identity_for(product)


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


def test_composed_scalar_result_is_replicated_with_selected_logical_kernels():
    """Composition retains a typed result in each selected logical kernel."""
    result_type = ttl.ScalarType.I64
    writer = Kernel(KernelKind.DATA_MOVEMENT)

    @ttl.operation()
    def selected_predicate():
        active = ttl.call_extern_func(
            "role.hpp",
            "active",
            result_type=result_type,
            kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
        )
        if active:
            ttl.call_extern_func("work.hpp", "compute", kernel=ttl.KernelKind.COMPUTE)
            ttl.call_extern_func(
                "work.hpp",
                "data_movement",
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    @ttl.operation()
    def composed_predicate():
        selected_predicate()
        ttl.call_extern_func("work.hpp", "writer", kernel=writer)

    spec = composed_predicate._spec
    result_type_names = [
        keyword.value.id
        for node in ast.walk(spec.fn_ast)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "result_type" and isinstance(keyword.value, ast.Name)
    ]
    assert result_type_names
    assert all(
        spec.frozen_scope[name] is ttl.ScalarType.I64 for name in result_type_names
    )
    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )

    compute = _kind_src(result, KernelKind.COMPUTE)
    data_movement = _kind_src(result, KernelKind.DATA_MOVEMENT)
    for source in (compute, data_movement):
        predicate_assignment = next(
            line for line in source.splitlines() if "'role.hpp', 'active'" in line
        )
        predicate_name = predicate_assignment.split(" = ", maxsplit=1)[0]
        assert " = ttl.call_extern_func" in predicate_assignment
        assert "result_type=" in source
        assert f"if {predicate_name}:" in source
        assert "kernel=" not in source
    assert "'compute'" in compute
    assert "'data_movement'" not in compute
    assert "'data_movement'" in data_movement
    assert "'compute'" not in data_movement

    writer_source = _kernel_src(result, writer)
    assert "'role.hpp', 'active'" not in writer_source
    assert "'writer'" in writer_source


def test_scalar_type_capture_changes_operation_identity():
    """Factory-selected scalar widths distinguish compiled operations."""

    def make_operation(result_type):
        @ttl.operation()
        def scalar_result():
            ttl.call_extern_func(
                "result.hpp",
                "result",
                result_type=result_type,
                kernel=ttl.KernelKind.COMPUTE,
            )

        return scalar_result

    i32_operation = make_operation(ttl.ScalarType.I32)
    i64_operation = make_operation(ttl.ScalarType.I64)

    assert (
        i32_operation._spec.operation_identity != i64_operation._spec.operation_identity
    )


def test_composition_preserves_one_dispatch_condition_identity():
    """Inlining preserves one captured condition across logical kernels."""
    condition = ttl.DispatchCondition(ttl.ScalarType.I64)

    @ttl.operation()
    def conditional_helper():
        active = ttl.call_extern_func(
            "condition.hpp",
            "active",
            condition_result=condition,
            kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
        )
        if active:
            ttl.call_extern_func("work.hpp", "work", kernel=ttl.KernelKind.COMPUTE)

    @ttl.operation()
    def composed_condition():
        conditional_helper()
        ttl.call_extern_func("work.hpp", "read", kernel=ttl.KernelKind.DATA_MOVEMENT)

    spec = composed_condition._spec
    assert tuple(spec.dispatch_conditions.values()) == (condition,)

    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    for source in (
        _kind_src(result, KernelKind.COMPUTE),
        _kind_src(result, KernelKind.DATA_MOVEMENT),
    ):
        assert "condition_result=" in source
        condition_name = next(iter(spec.dispatch_conditions))
        assert f"condition_result={condition_name}" in source


def test_composition_does_not_invent_missing_dispatch_condition_identity():
    """A partially annotated composition retains its untyped evaluation."""
    condition = ttl.DispatchCondition(ttl.ScalarType.I64)

    @ttl.operation()
    def typed_evaluation():
        ttl.call_extern_func(
            "condition.hpp",
            "typed",
            condition_result=condition,
            kernel=ttl.KernelKind.COMPUTE,
        )

    @ttl.operation()
    def untyped_evaluation():
        ttl.call_extern_func(
            "condition.hpp",
            "untyped",
            result_type=ttl.ScalarType.I64,
            kernel=ttl.KernelKind.COMPUTE,
        )

    @ttl.operation()
    def partial_condition():
        typed_evaluation()
        untyped_evaluation()

    spec = partial_condition._spec
    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    source = _kind_src(result, KernelKind.COMPUTE)
    assert source.count("condition_result=") == 1
    assert source.count("result_type=") == 1


def test_dispatch_condition_alias_topology_changes_operation_identity():
    """The cache identity distinguishes shared and independent declarations."""

    def make_operation(shared_identity):
        first_condition = ttl.DispatchCondition(ttl.ScalarType.I32)
        second_condition = (
            first_condition
            if shared_identity
            else ttl.DispatchCondition(ttl.ScalarType.I32)
        )

        @ttl.operation()
        def conditional_operation():
            ttl.call_extern_func(
                "condition.hpp",
                "first",
                condition_result=first_condition,
                kernel=ttl.KernelKind.COMPUTE,
            )
            ttl.call_extern_func(
                "condition.hpp",
                "second",
                condition_result=second_condition,
                kernel=ttl.KernelKind.COMPUTE,
            )

        return conditional_operation

    shared = make_operation(shared_identity=True)
    independent = make_operation(shared_identity=False)

    assert _operation_identity(shared._spec.fn) != _operation_identity(
        independent._spec.fn
    )
    assert shared._spec.operation_identity != independent._spec.operation_identity


def test_inlined_dispatch_condition_topology_changes_operation_identity():
    """The parent identity includes aliasing across composed operations."""

    def make_operation(shared_identity):
        first_condition = ttl.DispatchCondition(ttl.ScalarType.I32)
        second_condition = (
            first_condition
            if shared_identity
            else ttl.DispatchCondition(ttl.ScalarType.I32)
        )

        def make_helper(condition):
            @ttl.operation()
            def conditional_helper():
                ttl.call_extern_func(
                    "condition.hpp",
                    "active",
                    condition_result=condition,
                    kernel=ttl.KernelKind.COMPUTE,
                )

            return conditional_helper

        first_helper = make_helper(first_condition)
        second_helper = make_helper(second_condition)

        @ttl.operation()
        def composed_condition():
            first_helper()
            second_helper()

        return composed_condition

    shared = make_operation(shared_identity=True)
    independent = make_operation(shared_identity=False)

    assert _operation_identity(shared._spec.fn) == _operation_identity(
        independent._spec.fn
    )
    assert shared._spec.operation_identity != independent._spec.operation_identity


shadowed_dispatch_condition = ttl.DispatchCondition(ttl.ScalarType.I32)


def test_global_dispatch_condition_name_can_be_shadowed_by_parameter():
    """Global-capture validation respects Python lexical name resolution."""

    @ttl.operation()
    def shadowed_condition_operation(shadowed_dispatch_condition):
        pass

    assert shadowed_condition_operation._spec.params[0].name == (
        "shadowed_dispatch_condition"
    )


shadowed_reset_compute = ttl.Kernel(ttl.KernelKind.COMPUTE)
shadowed_reset_reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
shadowed_reset_writer = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
shadowed_dfb_reset = ttl.DFBReset(
    participants=(
        shadowed_reset_compute,
        shadowed_reset_reader,
        shadowed_reset_writer,
    )
)


def test_global_dfb_reset_name_can_be_shadowed_by_parameter():
    """Global-capture validation respects Python lexical name resolution."""

    @ttl.operation()
    def shadowed_reset_operation(shadowed_dfb_reset):
        pass

    assert shadowed_reset_operation._spec.params[0].name == "shadowed_dfb_reset"


def test_composition_hoists_resources_from_control_flow():
    """Composed static resources remain operation-level declarations."""

    @ttl.operation()
    def resource_helper():
        first_dfb = ttl.make_dfb("bf16", shape=(1, 1))
        second_dfb = ttl.make_dfb("bf16", shape=(1, 2))

    @ttl.operation()
    def composed_operation(enabled):
        for iteration in range(2):
            if enabled:
                resource_helper()

    spec = composed_operation._spec
    resource_names = [
        atom_rules.setup_assign_target(statement)
        for statement in spec.fn_ast.body
        if atom_rules.setup_assign_target(statement) is not None
    ]
    assert resource_names[0].startswith("first_dfb__")
    assert resource_names[1].startswith("second_dfb__")

    loop = next(
        statement for statement in spec.fn_ast.body if isinstance(statement, ast.For)
    )
    assert not any(
        isinstance(node, ast.Call)
        and atom_rules.call_name(node) in atom_rules.SETUP_FACTORY_NAMES
        for node in ast.walk(loop)
    )
    assert any(isinstance(node, ast.Pass) for node in ast.walk(loop))
    ast.parse(ast.unparse(spec.fn_ast))

    _, dfbs, _, _ = _lift_setup(
        copy.deepcopy(spec.fn_ast),
        dict(spec.frozen_scope),
        spec.operation_identity,
    )
    assert tuple(dfbs) == tuple(resource_names)


def test_composition_rejects_hoisted_resource_with_local_dependency():
    """A resource cannot move outside the scope of a required local value."""

    @ttl.operation()
    def resource_helper(blocks):
        helper_dfb = ttl.make_dfb("bf16", shape=(1, blocks))

    with pytest.raises(
        ValueError,
        match=(
            "composed resource declaration cannot be hoisted because it depends "
            "on operation-local values .*iteration"
        ),
    ):

        @ttl.operation()
        def composed_operation():
            for iteration in range(2):
                resource_helper(iteration)


def test_composition_rejects_hoisted_resource_with_shadowed_builtin():
    @ttl.operation()
    def resource_helper(blocks):
        helper_dfb = ttl.make_dfb("bf16", shape=(1, blocks))

    with pytest.raises(
        ValueError,
        match=(
            "composed resource declaration cannot be hoisted because it depends "
            "on operation-local values .*max"
        ),
    ):

        @ttl.operation()
        def composed_operation():
            max = 2
            for iteration in range(2):
                resource_helper(max)


def test_composition_does_not_hoist_resources_from_nested_scope():
    @ttl.operation()
    def resource_helper():
        helper_dfb = ttl.make_dfb("bf16", shape=(1, 1))

    with pytest.raises(
        ValueError,
        match="resource declaration 'make_dfb' must be a simple top-level assignment",
    ):

        @ttl.operation()
        def composed_operation():
            def callback():
                resource_helper()


def test_resource_name_collection_excludes_nested_scopes():
    function = _fn(
        """
        def operation():
            operation_dfb = ttl.make_dfb("bf16", shape=(1, 1))

            def callback():
                callback_dfb = ttl.make_dfb("bf16", shape=(1, 1))
        """
    )

    assert atom_module._operation_resource_names(function) == {"operation_dfb"}


def test_composition_preserves_one_dfb_allocation_group_identity():
    """Inlining preserves one captured allocation identity across declarations."""

    def make_operation():
        shared_allocation = ttl.make_dfb_allocation_group()

        @ttl.operation()
        def allocation_helper():
            helper_dfb = ttl.make_dfb(
                "bf16",
                shape=(1, 1),
                block_count=2,
                allocation_group=shared_allocation,
            )

        @ttl.operation()
        def composed_allocation():
            allocation_helper()
            caller_dfb = ttl.make_dfb(
                "bf16",
                shape=(1, 1),
                block_count=4,
                allocation_group=shared_allocation,
            )

        return composed_allocation, shared_allocation

    composed_allocation, shared_allocation = make_operation()
    spec = composed_allocation._spec

    captured_groups = tuple(spec.allocation_groups.values())
    assert len(captured_groups) == 2
    assert all(group is shared_allocation for group in captured_groups)
    with _dfb_allocation_group_binding_scope():
        _, dfbs, _, _ = _lift_setup(
            copy.deepcopy(spec.fn_ast),
            dict(spec.frozen_scope),
            spec.operation_identity,
        )
    assert len(dfbs) == 2
    helper_dfb, caller_dfb = dfbs.values()
    assert helper_dfb.allocation_group is caller_dfb.allocation_group


def test_composition_hoists_allocation_groups_from_control_flow():
    """Generated group tokens and members remain operation-level resources."""

    def make_operation():
        shared_allocation = ttl.make_dfb_allocation_group()

        @ttl.operation()
        def allocation_helper():
            local_allocation = ttl.make_dfb_allocation_group()
            helper_shared = ttl.make_dfb(
                "bf16", shape=(1, 1), allocation_group=shared_allocation
            )
            helper_local_first = ttl.make_dfb(
                "bf16", shape=(1, 1), allocation_group=local_allocation
            )
            helper_local_second = ttl.make_dfb(
                "bf16", shape=(1, 2), allocation_group=local_allocation
            )

        @ttl.operation()
        def composed_allocation(enabled):
            caller_shared = ttl.make_dfb(
                "bf16", shape=(1, 2), allocation_group=shared_allocation
            )
            for iteration in range(2):
                if enabled:
                    allocation_helper()

        return composed_allocation

    spec = make_operation()._spec
    resource_statements = [
        statement
        for statement in spec.fn_ast.body
        if atom_rules.setup_assign_target(statement) is not None
    ]
    assert len(resource_statements) == 5

    eval_scope = dict(spec.frozen_scope)
    eval_scope.update(_bind_dfb_allocation_groups(spec.allocation_groups))
    with _dfb_allocation_group_binding_scope(spec.allocation_groups.values()):
        _, dfbs, _, _ = _lift_setup(
            copy.deepcopy(spec.fn_ast), eval_scope, spec.operation_identity
        )

    caller_shared = dfbs["caller_shared"]
    helper_shared = next(
        dfb for name, dfb in dfbs.items() if name.startswith("helper_shared__")
    )
    helper_local = [
        dfb for name, dfb in dfbs.items() if name.startswith("helper_local_")
    ]
    assert helper_shared.allocation_group is caller_shared.allocation_group
    assert len(helper_local) == 2
    assert helper_local[0].allocation_group is helper_local[1].allocation_group
    assert helper_local[0].allocation_group is not caller_shared.allocation_group


def test_dfb_allocation_group_alias_topology_changes_operation_identity():
    """The cache identity distinguishes shared and independent groups."""

    def make_operation(shared_identity):
        first_group = ttl.make_dfb_allocation_group()
        second_group = (
            first_group if shared_identity else ttl.make_dfb_allocation_group()
        )

        @ttl.operation()
        def grouped_operation():
            first_dfb = ttl.make_dfb(
                "bf16",
                shape=(1, 1),
                allocation_group=first_group,
            )
            second_dfb = ttl.make_dfb(
                "bf16",
                shape=(1, 1),
                allocation_group=second_group,
            )

        return grouped_operation

    shared = make_operation(shared_identity=True)
    independent = make_operation(shared_identity=False)

    assert shared._spec.operation_identity != independent._spec.operation_identity


def test_captured_and_local_dfb_allocation_groups_receive_distinct_ordinals():
    """One binding context covers inlined captures and caller declarations."""

    def make_operation():
        captured_group = ttl.make_dfb_allocation_group()

        @ttl.operation()
        def allocation_helper():
            helper_dfb = ttl.make_dfb(
                "bf16", shape=(1, 1), allocation_group=captured_group
            )

        @ttl.operation()
        def composed_allocation():
            allocation_helper()
            local_group = ttl.make_dfb_allocation_group()
            caller_dfb = ttl.make_dfb(
                "bf16", shape=(1, 1), allocation_group=local_group
            )

        return composed_allocation

    spec = make_operation()._spec
    eval_scope = dict(spec.frozen_scope)
    eval_scope.update(_bind_dfb_allocation_groups(spec.allocation_groups))
    with _dfb_allocation_group_binding_scope(spec.allocation_groups.values()):
        _, dfbs, _, _ = _lift_setup(
            copy.deepcopy(spec.fn_ast), eval_scope, spec.operation_identity
        )

    helper_dfb, caller_dfb = dfbs.values()
    assert helper_dfb.allocation_group.ordinal == 0
    assert caller_dfb.allocation_group.ordinal == 1


def test_composition_preserves_one_synchronized_dfb_reset_identity():
    """Inlining and splitting preserve one reset across all participants."""
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))

    @ttl.operation()
    def reset_helper(target: ttl.DFB):
        ttl.reset_dfbs(reset, dfbs=[target])

    @ttl.operation()
    def composed_reset(target: ttl.DFB):
        reset_helper(target)

    spec = composed_reset._spec
    assert len(spec.dfb_resets) == 1
    assert set(spec.logical_kernels.values()) == {
        compute_kernel,
        reader_kernel,
        writer_kernel,
    }
    composed_reset_identity = next(iter(spec.dfb_resets.values()))
    assert composed_reset_identity is not reset
    assert composed_reset_identity.participants == reset.participants
    result = split_function_body(
        spec.fn_ast,
        dfb_param_names={"target"},
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    reset_name = next(iter(spec.dfb_resets))
    participant_sources = [
        _kind_src(result, KernelKind.COMPUTE),
        _kind_src(result, KernelKind.DATA_MOVEMENT, 0),
        _kind_src(result, KernelKind.DATA_MOVEMENT, 1),
    ]
    for source in participant_sources:
        assert source.count("ttl.reset_dfbs(") == 1
        assert f"ttl.reset_dfbs({reset_name}, dfbs=[target])" in source


def test_composition_preserves_inspect_dfb_access():
    """Inlining and logical-kernel replication retain the typed access."""

    @ttl.operation()
    def descriptor_helper(descriptor: ttl.DFB):
        ttl.call_extern_func(
            "descriptor.hpp",
            "inspect",
            template_args=[ttl.dfb_descriptor(descriptor)],
            dfb_accesses=[ttl.DFBAccess.inspect(descriptor)],
            kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
        )

    @ttl.operation()
    def composed_descriptor_access(descriptor: ttl.DFB):
        descriptor_helper(descriptor)

    spec = composed_descriptor_access._spec
    result = split_function_body(
        spec.fn_ast,
        dfb_param_names={"descriptor"},
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    for source in (
        _kind_src(result, KernelKind.COMPUTE),
        _kind_src(result, KernelKind.DATA_MOVEMENT),
    ):
        assert source.count("ttl.DFBAccess.inspect(descriptor)") == 1
        assert source.count("dfb_accesses=") == 1


def test_composition_instantiates_reset_identity_per_call_site():
    """Repeated helper calls denote distinct dynamic reset instances."""
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))

    @ttl.operation()
    def reset_helper(target: ttl.DFB):
        ttl.reset_dfbs(reset, dfbs=[target])

    @ttl.operation()
    def repeated_reset(first: ttl.DFB, second: ttl.DFB):
        reset_helper(first)
        reset_helper(second)

    spec = repeated_reset._spec
    reset_identities = tuple(spec.dfb_resets.values())
    assert len(reset_identities) == 2
    assert reset_identities[0] is not reset_identities[1]

    result = split_function_body(
        spec.fn_ast,
        dfb_param_names={"first", "second"},
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    compute_source = _kind_src(result, KernelKind.COMPUTE)
    for reset_name in spec.dfb_resets:
        assert f"ttl.reset_dfbs({reset_name}, dfbs=" in compute_source


def test_composition_remaps_equivalent_reset_participants():
    """Equivalent composed kernels use the caller's selected handles."""

    def make_reset_helper():
        compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
        reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
        writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
        reset = ttl.DFBReset(
            participants=(compute_kernel, reader_kernel, writer_kernel)
        )

        @ttl.operation()
        def reset_helper(target: ttl.DFB):
            ttl.reset_dfbs(reset, dfbs=[target])

        return reset_helper

    first_helper = make_reset_helper()
    second_helper = make_reset_helper()

    @ttl.operation()
    def composed_reset(first: ttl.DFB, second: ttl.DFB):
        first_helper(first)
        second_helper(second)

    spec = composed_reset._spec
    assert len(spec.logical_kernels) == 3
    logical_kernels = tuple(spec.logical_kernels.values())
    for reset in spec.dfb_resets.values():
        assert all(
            any(participant is kernel for kernel in logical_kernels)
            for participant in reset.participants
        )

    result = split_function_body(
        spec.fn_ast,
        dfb_param_names={"first", "second"},
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    assert _kind_src(result, KernelKind.COMPUTE).count("ttl.reset_dfbs(") == 2


def test_synchronized_dfb_reset_requires_positional_boundary():
    """The public API matches the frontend's positional boundary syntax."""
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))

    with pytest.raises(TypeError, match="positional-only"):
        ttl.reset_all_dfbs(reset=reset)


def test_direct_kernel_capture_names_precede_reset_participant_names():
    """Reset capture order does not replace direct logical-kernel identities."""
    z_compute = ttl.Kernel(ttl.KernelKind.COMPUTE)
    z_reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    z_writer = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    a_reset = ttl.DFBReset(participants=(z_compute, z_reader, z_writer))

    @ttl.operation()
    def reset_with_direct_participants():
        ttl.call_extern_func("compute.hpp", "compute", kernel=z_compute)
        ttl.call_extern_func("reader.hpp", "reader", kernel=z_reader)
        ttl.call_extern_func("writer.hpp", "writer", kernel=z_writer)
        ttl.reset_all_dfbs(a_reset)

    spec = reset_with_direct_participants._spec
    assert tuple(spec.logical_kernels) == ("z_compute", "z_reader", "z_writer")
    bound_reset = spec.dfb_resets["a_reset"]
    assert bound_reset.participants == tuple(spec.logical_kernels.values())


def test_synchronized_dfb_reset_alias_topology_changes_operation_identity():
    """The cache identity distinguishes shared and independent reset instances."""

    def make_operation(shared_identity):
        compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
        reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
        writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
        participants = (compute_kernel, reader_kernel, writer_kernel)
        first_reset = ttl.DFBReset(participants=participants)
        second_reset = (
            first_reset if shared_identity else ttl.DFBReset(participants=participants)
        )

        @ttl.operation()
        def reset_operation(first: ttl.DFB, second: ttl.DFB):
            ttl.reset_dfbs(first_reset, dfbs=[first])
            ttl.reset_dfbs(second_reset, dfbs=[second])

        return reset_operation

    shared = make_operation(shared_identity=True)
    independent = make_operation(shared_identity=False)

    assert shared._spec.operation_identity != independent._spec.operation_identity


def test_synchronized_dfb_reset_is_replicated_to_every_participant():
    """One reset call creates one operation in each declared logical kernel."""
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))

    @ttl.operation()
    def reset_participants(target: ttl.DFB):
        ttl.reset_dfbs(reset, dfbs=[target])

    spec = reset_participants._spec
    assert len(spec.logical_kernels) == 3
    result = split_function_body(
        spec.fn_ast,
        dfb_param_names={"target"},
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    assert _kind_src(result, KernelKind.COMPUTE).count("ttl.reset_dfbs(") == 1
    assert _kind_src(result, KernelKind.DATA_MOVEMENT, 0).count("ttl.reset_dfbs(") == 1
    assert _kind_src(result, KernelKind.DATA_MOVEMENT, 1).count("ttl.reset_dfbs(") == 1


def test_dfb_reconfiguration_requires_complete_distinct_participants():
    """A boundary names one compute kernel and both data-movement kernels."""
    with pytest.raises(TypeError, match="nonempty tuple"):
        ttl.DFBReconfiguration(participants=[ttl.KernelKind.COMPUTE])
    with pytest.raises(ValueError, match="one compute and two data movement"):
        ttl.DFBReconfiguration(
            participants=(
                ttl.KernelKind.COMPUTE,
                ttl.KernelKind.DATA_MOVEMENT,
            )
        )
    with pytest.raises(ValueError, match="participants must be distinct"):
        ttl.DFBReconfiguration(
            participants=(
                ttl.KernelKind.COMPUTE,
                ttl.KernelKind.DATA_MOVEMENT,
                ttl.KernelKind.DATA_MOVEMENT,
            )
        )


def test_dfb_reconfiguration_routes_to_every_participant():
    """One boundary declaration is retained by all three kernel bodies."""
    compute_kernel = Kernel(KernelKind.COMPUTE)
    reader_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    writer_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation()
    def reconfiguration_operation():
        ttl.call_extern_func(
            "boundary.hpp",
            "before_boundary",
            kernel=(compute_kernel, reader_kernel, writer_kernel),
        )
        ttl.reconfigure_dfbs(boundary)

    spec = reconfiguration_operation._spec
    assert len(spec.dfb_reconfigurations) == 1
    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    boundary_name = next(iter(spec.dfb_reconfigurations))
    for participant in (compute_kernel, reader_kernel, writer_kernel):
        source = _kernel_src(result, participant)
        assert source.count("reconfigure_dfbs") == 1
        assert f"reconfigure_dfbs({boundary_name})" in source


def test_dfb_reconfiguration_materializes_participant_only_kernels():
    """A boundary retains participants with no other operation reference."""
    compute_kernel = Kernel(KernelKind.COMPUTE)
    reader_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    writer_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation()
    def reconfiguration_operation():
        ttl.reconfigure_dfbs(boundary)

    spec = reconfiguration_operation._spec
    assert set(spec.logical_kernels.values()) == {
        compute_kernel,
        reader_kernel,
        writer_kernel,
    }
    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    for participant in boundary.participants:
        assert _kernel_src(result, participant).count("reconfigure_dfbs") == 1


def test_composition_instantiates_reconfiguration_per_call_site():
    """Repeated helper calls declare distinct ordered boundary sites."""
    compute_kernel = Kernel(KernelKind.COMPUTE)
    reader_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    writer_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation()
    def reconfiguration_helper():
        ttl.reconfigure_dfbs(boundary)

    @ttl.operation()
    def repeated_reconfiguration():
        reconfiguration_helper()
        reconfiguration_helper()

    spec = repeated_reconfiguration._spec
    boundary_instances = tuple(spec.dfb_reconfigurations.values())
    assert len(boundary_instances) == 2
    assert boundary_instances[0] is not boundary_instances[1]

    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    for participant in (compute_kernel, reader_kernel, writer_kernel):
        source = _kernel_src(result, participant)
        for boundary_name in spec.dfb_reconfigurations:
            assert f"reconfigure_dfbs({boundary_name})" in source


def test_composition_remaps_equivalent_reconfiguration_participants():
    """Equivalent composed kernels use the caller's selected handles."""

    def make_reconfiguration_helper():
        compute_kernel = Kernel(KernelKind.COMPUTE)
        reader_kernel = Kernel(KernelKind.DATA_MOVEMENT)
        writer_kernel = Kernel(KernelKind.DATA_MOVEMENT)
        boundary = ttl.DFBReconfiguration(
            participants=(compute_kernel, reader_kernel, writer_kernel)
        )

        @ttl.operation()
        def reconfiguration_helper():
            ttl.reconfigure_dfbs(boundary)

        return reconfiguration_helper

    first_helper = make_reconfiguration_helper()
    second_helper = make_reconfiguration_helper()

    @ttl.operation()
    def composed_reconfiguration():
        first_helper()
        second_helper()

    spec = composed_reconfiguration._spec
    assert len(spec.logical_kernels) == 3
    logical_kernels = tuple(spec.logical_kernels.values())
    for boundary in spec.dfb_reconfigurations.values():
        assert all(
            any(participant is kernel for kernel in logical_kernels)
            for participant in boundary.participants
        )

    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )
    assert _kind_src(result, KernelKind.COMPUTE).count("ttl.reconfigure_dfbs(") == 2


def test_control_header_anchor_is_retained_only_in_selected_logical_kernel():
    """Control selection includes logical-kernel anchors in the condition."""
    writer = Kernel(KernelKind.DATA_MOVEMENT)

    @ttl.operation()
    def selected_control_header():
        if ttl.call_extern_func(
            "role.hpp",
            "writer_active",
            result_type=ttl.ScalarType.I32,
            kernel=writer,
        ):
            scalar_value = 1
        ttl.call_extern_func("work.hpp", "compute", kernel=ttl.KernelKind.COMPUTE)

    spec = selected_control_header._spec
    result = split_function_body(
        spec.fn_ast,
        dfb_param_names=set(),
        logical_kernels=spec.logical_kernels,
        selector_scope=spec.frozen_scope,
    )

    writer_source = _kernel_src(result, writer)
    assert "'writer_active'" in writer_source
    assert "scalar_value = 1" in writer_source
    assert "'compute'" not in writer_source

    compute_source = _kind_src(result, KernelKind.COMPUTE)
    assert "'writer_active'" not in compute_source
    assert "scalar_value = 1" not in compute_source
    assert "'compute'" in compute_source


def test_selected_scalar_producer_covers_consumer_kernels():
    reader = _logical_kernel(KernelKind.DATA_MOVEMENT, "reader")
    writer = _logical_kernel(KernelKind.DATA_MOVEMENT, "writer")

    function = _fn(
        """
        def k():
            active = ttl.call_extern_func(
                "role.hpp", "active", result_type=RESULT_TYPE,
                kernel=(reader, writer),
            )
            ttl.call_extern_func(
                "role.hpp", "consume", func_args=[active], kernel=writer,
            )
        """
    )
    result = split_function_body(
        function,
        dfb_param_names=set(),
        logical_kernels={"reader": reader, "writer": writer},
        selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
    )

    assert "'active'" in _kernel_src(result, reader)
    writer_source = _kernel_src(result, writer)
    assert "'active'" in writer_source
    assert "'consume'" in writer_source


def test_definite_branch_assignments_end_an_earlier_selected_value_lifetime():
    function = _fn(
        """
        def k():
            value = ttl.call_extern_func(
                "role.hpp", "compute_only", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            if selector:
                value = 1
            else:
                value = 2
            ttl.call_extern_func(
                "role.hpp", "consume", func_args=[value],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )
        """
    )
    result = split_function_body(
        function,
        dfb_param_names=set(),
        selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
    )

    data_movement = _kernel_src(result, KernelKind.DATA_MOVEMENT)
    assert "'compute_only'" not in data_movement
    assert "value = 1" in data_movement
    assert "value = 2" in data_movement
    assert "'consume'" in data_movement


@pytest.mark.parametrize(
    "body",
    [
        """
        active = ttl.call_extern_func(
            "role.hpp", "active", result_type=RESULT_TYPE, kernel=reader,
        )
        ttl.call_extern_func(
            "role.hpp", "consume", func_args=[active], kernel=writer,
        )
        """,
        """
        active = ttl.call_extern_func(
            "role.hpp", "active", result_type=RESULT_TYPE, kernel=reader,
        )
        if active:
            ttl.call_extern_func("role.hpp", "consume", kernel=writer)
        """,
        """
        active = ttl.call_extern_func(
            "role.hpp", "active", result_type=RESULT_TYPE,
            kernel=(reader, writer),
        )
        ttl.call_extern_func(
            "role.hpp", "consume", func_args=[active],
            kernel=(writer, ttl.KernelKind.COMPUTE),
        )
        """,
    ],
    ids=["direct", "condition", "partial-overlap"],
)
def test_selected_scalar_producer_rejects_excluded_consumers(body):
    reader = _logical_kernel(KernelKind.DATA_MOVEMENT, "reader")
    writer = _logical_kernel(KernelKind.DATA_MOVEMENT, "writer")
    function = _fn("def k():\n" + textwrap.indent(textwrap.dedent(body), "    "))

    with pytest.raises(ValueError, match="produced for.*consumed by excluded"):
        split_function_body(
            function,
            dfb_param_names=set(),
            logical_kernels={"reader": reader, "writer": writer},
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_type_only_annotation_does_not_kill_scalar_liveness():
    function = _fn(
        """
        def k():
            active = ttl.call_extern_func(
                "role.hpp", "active", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            active: int
            ttl.call_extern_func(
                "role.hpp", "consume", func_args=[active],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )
        """
    )

    with pytest.raises(ValueError, match="produced for.*consumed by excluded"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


@pytest.mark.parametrize(
    "statement",
    [
        """ttl.call_extern_func(
            "outer.hpp", "outer",
            func_args=[ttl.call_extern_func(
                "inner.hpp", "inner", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )],
            kernel=ttl.KernelKind.COMPUTE,
        )""",
        """ttl.call_extern_func(
            "outer.hpp", "outer",
            func_args=[ttl.call_extern_func(
                "inner.hpp", "inner", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )],
            kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
        )""",
        """if ttl.call_extern_func(
            "outer.hpp", "outer", result_type=RESULT_TYPE,
            func_args=[ttl.call_extern_func(
                "inner.hpp", "inner", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )],
            kernel=ttl.KernelKind.COMPUTE,
        ):\n    pass""",
    ],
    ids=["disjoint", "partial-overlap", "control-header"],
)
def test_indivisible_expression_rejects_different_kernel_selections(statement):
    function = _fn("def k():\n" + textwrap.indent(statement, "    "))
    with pytest.raises(ValueError, match="indivisible expression select different"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_indivisible_expression_accepts_identical_kernel_selections():
    function = _fn(
        """
        def k():
            ttl.call_extern_func(
                "outer.hpp", "outer",
                func_args=[ttl.call_extern_func(
                    "inner.hpp", "inner", result_type=RESULT_TYPE,
                    kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
                )],
                kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
            )
        """
    )
    result = split_function_body(
        function,
        dfb_param_names=set(),
        selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
    )

    for kernel in (KernelKind.COMPUTE, KernelKind.DATA_MOVEMENT):
        source = _kernel_src(result, kernel)
        assert "'outer'" in source
        assert "'inner'" in source
        assert "kernel=" not in source


def test_selected_control_expression_rejects_excluded_body_kernel():
    function = _fn(
        """
        def k():
            if ttl.call_extern_func(
                "role.hpp", "active", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            ):
                ttl.call_extern_func(
                    "work.hpp", "compute", kernel=ttl.KernelKind.COMPUTE,
                )
        """
    )

    with pytest.raises(ValueError, match="control expression excludes"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_selected_control_expression_rejects_excluded_live_out_consumer():
    function = _fn(
        """
        def k():
            if ttl.call_extern_func(
                "role.hpp", "active", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            ):
                value = 1
            ttl.call_extern_func(
                "work.hpp", "writer", func_args=[value],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )
        """
    )

    with pytest.raises(ValueError, match="consume values defined by its body"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_nested_function_free_scalar_requires_its_consumer_kernel():
    function = _fn(
        """
        def k():
            active = ttl.call_extern_func(
                "role.hpp", "active", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            def writer_callback():
                ttl.call_extern_func(
                    "work.hpp", "writer", func_args=[active],
                    kernel=ttl.KernelKind.DATA_MOVEMENT,
                )
        """
    )

    with pytest.raises(ValueError, match="produced for.*consumed by excluded"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_nested_function_local_scalar_does_not_escape_its_scope():
    function = _fn(
        """
        def k():
            value = ttl.call_extern_func(
                "role.hpp", "compute_value", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            def writer_callback():
                value = 1
                ttl.call_extern_func(
                    "work.hpp", "writer", func_args=[value],
                    kernel=ttl.KernelKind.DATA_MOVEMENT,
                )
        """
    )

    result = split_function_body(
        function,
        dfb_param_names=set(),
        selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
    )
    assert "'compute_value'" not in _kernel_src(result, KernelKind.DATA_MOVEMENT)


def test_nested_function_default_requires_its_definition_kernels():
    function = _fn(
        """
        def k():
            active = ttl.call_extern_func(
                "role.hpp", "active", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            def writer_callback(enabled=active):
                ttl.call_extern_func(
                    "work.hpp", "writer", func_args=[enabled],
                )
            net.if_dst(writer_callback)
        """
    )

    with pytest.raises(ValueError, match="produced for.*consumed by excluded"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_lambda_free_scalar_requires_its_callback_kernel():
    function = _fn(
        """
        def k():
            active = ttl.call_extern_func(
                "role.hpp", "active", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            net.if_dst(lambda pipe: ttl.call_extern_func(
                "work.hpp", "writer", func_args=[active],
            ))
        """
    )

    with pytest.raises(ValueError, match="produced for.*consumed by excluded"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_lambda_parameter_does_not_resolve_to_an_outer_scalar():
    function = _fn(
        """
        def k():
            active = ttl.call_extern_func(
                "role.hpp", "compute_value", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            net.if_dst(lambda active: ttl.call_extern_func(
                "work.hpp", "writer", func_args=[active],
            ))
        """
    )

    result = split_function_body(
        function,
        dfb_param_names=set(),
        selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
    )
    assert "'compute_value'" not in _kernel_src(result, KernelKind.DATA_MOVEMENT)


def test_lambda_default_requires_its_evaluation_kernel():
    function = _fn(
        """
        def k():
            active = ttl.call_extern_func(
                "role.hpp", "active", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            net.if_dst(lambda pipe, enabled=active: ttl.call_extern_func(
                "work.hpp", "writer", func_args=[enabled],
            ))
        """
    )

    with pytest.raises(ValueError, match="produced for.*consumed by excluded"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_selected_control_preserves_unrelated_live_values():
    function = _fn(
        """
        def k():
            active = ttl.call_extern_func(
                "role.hpp", "active", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            if ttl.call_extern_func(
                "role.hpp", "compute_condition", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            ):
                pass
            ttl.call_extern_func(
                "work.hpp", "writer", func_args=[active],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )
        """
    )

    with pytest.raises(ValueError, match="produced for.*consumed by excluded"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_loop_backedge_rejects_cross_kernel_scalar_value():
    function = _fn(
        """
        def k():
            value = 0
            for index in range(4):
                ttl.call_extern_func(
                    "work.hpp", "writer", func_args=[value],
                    kernel=ttl.KernelKind.DATA_MOVEMENT,
                )
                value = ttl.call_extern_func(
                    "work.hpp", "compute_value", result_type=RESULT_TYPE,
                    kernel=ttl.KernelKind.COMPUTE,
                )
        """
    )

    with pytest.raises(ValueError, match="produced for.*consumed by excluded"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_loop_else_rejects_cross_kernel_scalar_value():
    function = _fn(
        """
        def k():
            value = 0
            for index in range(4):
                value = ttl.call_extern_func(
                    "work.hpp", "compute_value", result_type=RESULT_TYPE,
                    kernel=ttl.KernelKind.COMPUTE,
                )
            else:
                ttl.call_extern_func(
                    "work.hpp", "writer", func_args=[value],
                    kernel=ttl.KernelKind.DATA_MOVEMENT,
                )
        """
    )

    with pytest.raises(ValueError, match="produced for.*consumed by excluded"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_while_backedge_rejects_cross_kernel_condition_value():
    function = _fn(
        """
        def k():
            active = 1
            while active:
                ttl.call_extern_func(
                    "work.hpp", "writer", kernel=ttl.KernelKind.DATA_MOVEMENT,
                )
                active = ttl.call_extern_func(
                    "work.hpp", "compute_value", result_type=RESULT_TYPE,
                    kernel=ttl.KernelKind.COMPUTE,
                )
        """
    )

    with pytest.raises(ValueError, match="produced for.*consumed by excluded"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_with_body_assignment_kills_previous_value():
    function = _fn(
        """
        def k():
            value = ttl.call_extern_func(
                "work.hpp", "compute_value", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            with ttl.signpost("scope"):
                value = 1
            ttl.call_extern_func(
                "work.hpp", "writer", func_args=[value],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )
        """
    )

    result = split_function_body(
        function,
        dfb_param_names=set(),
        selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
    )
    data_movement = _kernel_src(result, KernelKind.DATA_MOVEMENT)
    assert "'compute_value'" not in data_movement
    assert "value = 1" in data_movement
    assert "'writer'" in data_movement


def test_unselected_with_preserves_selected_body_kernels():
    function = _fn(
        """
        def k():
            with ttl.signpost("scope"):
                value = 1
                ttl.call_extern_func(
                    "work.hpp", "compute", kernel=ttl.KernelKind.COMPUTE,
                )
            ttl.call_extern_func(
                "work.hpp", "writer", func_args=[value],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )
        """
    )

    result = split_function_body(function, dfb_param_names=set())
    compute = _kernel_src(result, KernelKind.COMPUTE)
    data_movement = _kernel_src(result, KernelKind.DATA_MOVEMENT)
    assert "with ttl.signpost('scope'):" in compute
    assert "'compute'" in compute
    assert "value = 1" in data_movement
    assert "'writer'" in data_movement


def test_selected_control_rejects_nested_generic_with_body_kernel():
    function = _fn(
        """
        def k():
            if ttl.call_extern_func(
                "work.hpp", "predicate", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            ):
                with ttl.signpost("scope"):
                    ttl.call_extern_func(
                        "work.hpp", "writer",
                        kernel=ttl.KernelKind.DATA_MOVEMENT,
                    )
        """
    )

    with pytest.raises(ValueError, match="control expression excludes"):
        split_function_body(
            function,
            dfb_param_names=set(),
            selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
        )


def test_with_bindings_kill_values_before_later_context_expressions():
    function = _fn(
        """
        def k():
            value = ttl.call_extern_func(
                "work.hpp", "compute_value", result_type=RESULT_TYPE,
                kernel=ttl.KernelKind.COMPUTE,
            )
            with first() as value, second(value):
                pass
            ttl.call_extern_func(
                "work.hpp", "writer", kernel=ttl.KernelKind.DATA_MOVEMENT,
            )
        """
    )

    result = split_function_body(
        function,
        dfb_param_names=set(),
        selector_scope={"RESULT_TYPE": ttl.ScalarType.I64},
    )
    data_movement = _kernel_src(result, KernelKind.DATA_MOVEMENT)
    assert "'compute_value'" not in data_movement
    assert "second(value)" in data_movement


def test_unanchored_scalar_in_control_survives_in_consumer_kernel():
    function = _fn(
        """
        def k():
            accumulator = 0
            for index in range(4):
                accumulator = accumulator + 1
                ttl.call_extern_func(
                    "work.hpp", "compute", kernel=ttl.KernelKind.COMPUTE,
                )
            ttl.call_extern_func(
                "work.hpp", "writer", func_args=[accumulator],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )
        """
    )
    result = split_function_body(function, dfb_param_names=set())

    data_movement = _kernel_src(result, KernelKind.DATA_MOVEMENT)
    assert "accumulator = accumulator + 1" in data_movement
    assert "'compute'" not in data_movement


def test_nested_branch_scalar_live_out_survives_in_consumer_kernel():
    function = _fn(
        """
        def k():
            value = 0
            for index in range(4):
                if index:
                    value = index + 1
                    ttl.call_extern_func(
                        "work.hpp", "compute", kernel=ttl.KernelKind.COMPUTE,
                    )
                else:
                    value = index + 2
            ttl.call_extern_func(
                "work.hpp", "writer", func_args=[value],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )
        """
    )
    result = split_function_body(function, dfb_param_names=set())

    data_movement = _kernel_src(result, KernelKind.DATA_MOVEMENT)
    assert "if index:" in data_movement
    assert "value = index + 1" in data_movement
    assert "value = index + 2" in data_movement
    assert "'compute'" not in data_movement


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

    with pytest.raises(ValueError, match="indivisible expression select different"):
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


def test_public_pipe_source_selector_shares_pipenet_source_kernel():
    """External transport work can share the compiler-owned PipeNet source."""
    fn = _fn(
        """
        def k(net):
            net.if_src(lambda pipe: None)
            ttl.call_extern_func(
                "transport.hpp",
                "transport",
                kernel=ttl.PIPE_SOURCE_KERNEL,
            )
        """
    )

    result = split_function_body(fn, dfb_param_names=set())

    data_movement_kernels = tuple(
        kernel
        for kernel in result.kernels
        if kernel is KernelKind.DATA_MOVEMENT
        or isinstance(kernel, Kernel)
        and kernel.kind is KernelKind.DATA_MOVEMENT
    )
    assert data_movement_kernels == (ttl.PIPE_SOURCE_KERNEL,)
    source = _kernel_src(result, ttl.PIPE_SOURCE_KERNEL)
    assert "net.if_src" in source
    assert "'transport'" in source
    assert "kernel=" not in source

    assignments = _assign_backend_kernel_slots(result)
    brisc_selector = next(
        selector
        for slot, selector in assignments.items()
        if slot.source_name == "brisc"
    )
    assert brisc_selector is ttl.PIPE_SOURCE_KERNEL


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
