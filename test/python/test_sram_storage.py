# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent SRAM ownership and cross-operation device correctness."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch
import ttnn
import ttl

from ttl.sram import SRAMStorage, _storage_spec
from ttl._sram_requirements import (
    PreparedSRAMOperation,
    SRAMArenaBinding,
    SRAMAddressDomain,
    SRAMAddressing,
    SRAMLocation,
    SRAMLifetime,
    SRAMOwner,
    SRAMOwnerKind,
    SRAMOwnership,
    SRAMStorageRequirement,
)
from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose


DEVICE_ADDRESSING_MODES = (
    "uniform",
    pytest.param("per-core", marks=pytest.mark.hybrid_allocator),
)


class Device:
    def __init__(self):
        self.open = True
        self.manager = 1
        self.device_id = 17

    def is_initialized(self):
        return self.open

    def id(self):
        return self.device_id

    def num_hw_cqs(self):
        return 2

    def get_sub_device_ids(self):
        return (ttnn.SubDeviceId(0), ttnn.SubDeviceId(1))

    def get_active_sub_device_manager_id(self):
        return self.manager

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=8, y=8)


class Resource:
    def __init__(self, shape, options, address):
        self.shape = tuple(shape)
        self.padded_shape = tuple(shape)
        self.options = options
        self.dtype = options["dtype"]
        self.layout = options["layout"]
        self.tile = (32, 32)
        self.address = address
        self.allocated = True
        self.per_core = False
        self.device_coordinates = (ttnn.MeshCoordinate(0, 0),)
        self.core_addresses = {}
        self.value = None

    def is_allocated(self):
        return self.allocated

    def device(self):
        return self.options["device"]

    def memory_config(self):
        return self.options["memory_config"]

    def is_per_core_allocated(self):
        return self.per_core

    def buffer_address(self):
        return self.address

    def device_coords(self):
        return self.device_coordinates

    def experimental_per_core_buffer_address(self, device_coordinate, core):
        return self.core_addresses[(tuple(device_coordinate), core.x, core.y)]

    def buffer_aligned_page_size(self):
        shard_shape = self.memory_config().shard_spec.shape
        return int(shard_shape[0]) * int(shard_shape[1]) * 4


def make_per_core(resource):
    resource.per_core = True
    cores = ttnn.corerange_to_cores(resource.memory_config().shard_spec.grid)
    for device_coordinate in resource.device_coordinates:
        for core_index, core in enumerate(cores):
            resource.core_addresses[(tuple(device_coordinate), core.x, core.y)] = (
                resource.address + core_index * 0x1000
            )


@pytest.fixture
def runtime(monkeypatch):
    events = []
    allocations = []
    failure = {"stage": None, "at": 0}

    def check(stage, index):
        if failure["stage"] == stage and failure["at"] == index:
            raise RuntimeError(f"injected {stage} failure")

    def empty(shape, **options):
        check("allocation", len(allocations))
        resource = Resource(shape, options, address=0x10000 + len(allocations) * 0x1000)
        allocations.append(resource)
        if "per_core_allocation=1" in str(options["memory_config"]):
            make_per_core(resource)
        events.append(("allocate", resource))
        return resource

    def create_view(owner, spec, shard_offset):
        resource = Resource(
            spec.shape,
            {
                "dtype": spec.dtype,
                "layout": spec.layout,
                "device": owner.device(),
                "memory_config": spec.memory_config,
            },
            address=owner.address + shard_offset,
        )
        allocations.append(resource)
        if "per_core_allocation=1" in str(spec.memory_config):
            make_per_core(resource)
        events.append(("view", owner, resource, shard_offset))
        return resource

    def full(shape, value, **options):
        resource = options["optional_tensor"]
        check("initialize", allocations.index(resource))
        resource.value = value
        events.append(("initialize", resource))
        return resource

    def record_event(device, **options):
        check("record", options["cq_id"])
        record = ("record", device, options)
        events.append(record)
        return record

    def wait(event):
        check("wait", 0)
        events.append(("wait", event))

    def recover(device, **options):
        check("recover", options["cq_id"])
        events.append(("recover", device, options))

    def release(resource):
        assert resource.allocated
        check("release", allocations.index(resource))
        resource.allocated = False
        events.append(("release", resource))

    api = SimpleNamespace(
        **{
            name: getattr(ttnn, name)
            for name in (
                "bfloat16",
                "float32",
                "TILE_LAYOUT",
                "ROW_MAJOR_LAYOUT",
                "Shape",
                "TensorSpec",
                "TensorMemoryLayout",
                "BufferType",
                "CoreCoord",
                "CoreRange",
                "CoreRangeSet",
                "ShardSpec",
                "ShardOrientation",
                "SubDeviceId",
                "StorageType",
                "corerange_to_cores",
            )
        }
    )
    api.empty = empty
    api.full = full
    api.get_l1_alignment = lambda: 16
    api.get_allocator_base_address = lambda device, buffer_type: 0
    api._ttnn = SimpleNamespace(
        reports=SimpleNamespace(
            get_device_info=lambda device: SimpleNamespace(cb_limit=1 << 20),
            get_buffer_pages=lambda device: (),
        )
    )
    api.experimental_create_sharded_tensor_view = create_view
    api.record_event = record_event
    api.wait_for_event = lambda **options: events.append(("order", options))
    api.event_synchronize = wait
    api.synchronize_device = recover
    api.deallocate = release
    monkeypatch.setitem(sys.modules, "ttnn", api)
    return SimpleNamespace(
        device=Device(),
        api=api,
        events=events,
        allocations=allocations,
        failure=failure,
    )


def declare(storage, **options):
    arguments = dict(
        shape=(64, 32),
        shard_shape=(32, 32),
        cores=((0, 0), (1, 0)),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    arguments.update(options)
    return storage.tensor(**arguments)


def compiler_arena_requirements(name, extent_bytes=2048):
    locations = (
        SRAMLocation((), (0, 0)),
        SRAMLocation((), (1, 0)),
    )
    requirement = SRAMStorageRequirement(
        owner=SRAMOwner(SRAMOwnerKind.COMPILER_ARENA, 0),
        extent_bytes=extent_bytes,
        alignment_bytes=16,
        address_domains=(SRAMAddressDomain(locations),),
        ownership=SRAMOwnership.MOVABLE,
        lifetime=SRAMLifetime.INVOCATION,
        addressing=SRAMAddressing.UNIFORM,
    )
    return PreparedSRAMOperation(
        name,
        (requirement,),
        (),
        (SRAMArenaBinding(0, ((0, 0), (1, 0))),),
    )


def preparable_operation(name, extent_bytes=2048, prepare_program=None):
    requirements = compiler_arena_requirements(name, extent_bytes)

    class CompiledKernel:
        cb_configs = ()

        def prepare_sram_requirements(self, prepared_state):
            del prepared_state
            return requirements

        def prepare_device_program(self, prepared_state, *, prepared_sram_resources):
            if prepare_program is not None:
                return prepare_program(prepared_state, prepared_sram_resources)
            return SimpleNamespace(
                max_program_config_size_bytes=4096,
                max_kernel_binary_size_bytes=8192,
            )

    compiled_kernel = CompiledKernel()

    def operation(*args, **kwargs):
        raise AssertionError("preparation must not execute the operation")

    operation._ttlang_prepare_operation = lambda prepared_state: (
        compiled_kernel,
        (prepared_state,),
    )
    return operation, compiled_kernel


def measure_separate_reservation(device, declaration_options, prepared_operations):
    with SRAMStorage(device=device) as separate_storage:
        separate_state = separate_storage.tensor(**declaration_options)
        separate_storage.allocate()
        persistent_bytes = separate_storage.submit(
            lambda tensor: tensor.buffer_aligned_page_size(), separate_state
        ) * len(declaration_options["cores"])

    maximum_arena_bytes = 0
    for prepared in prepared_operations:
        operation_arena_bytes = 0
        for requirement in prepared.requirements.requirements:
            if requirement.owner.kind is not SRAMOwnerKind.COMPILER_ARENA:
                continue
            cores = tuple(
                sorted(
                    {
                        location.core
                        for domain in requirement.address_domains
                        for location in domain.locations
                    }
                )
            )
            spec = _storage_spec(
                ttnn,
                cores,
                requirement.extent_bytes,
                requirement.addressing,
            )
            arena = ttnn.empty(
                spec.shape,
                dtype=spec.dtype,
                layout=spec.layout,
                device=device,
                memory_config=spec.memory_config,
            )
            operation_arena_bytes += arena.buffer_aligned_page_size() * len(cores)
            ttnn.deallocate(arena)
        maximum_arena_bytes = max(maximum_arena_bytes, operation_arena_bytes)
    return persistent_bytes + maximum_arena_bytes


def test_prepare_operation_uses_declaration_metadata_before_allocation(runtime):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage, addressing="per-core")
    marker = object()

    class CompiledKernel:
        def prepare_sram_requirements(self, prepared_state):
            assert prepared_state.shape == ttnn.Shape((64, 32))
            assert prepared_state.dtype == ttnn.float32
            assert prepared_state.layout == ttnn.TILE_LAYOUT
            assert prepared_state.device() is runtime.device
            assert prepared_state.is_per_core_allocated()
            assert not prepared_state.is_allocated()
            return marker

    compiled_kernel = CompiledKernel()

    def operation(*args, **kwargs):
        raise AssertionError("preparation must not execute the operation")

    def prepare(prepared_state):
        return compiled_kernel, (prepared_state,)

    operation._ttlang_prepare_operation = prepare
    prepared = storage.prepare_operation(operation, state)

    assert prepared.operation is operation
    assert prepared.compiled_kernel is compiled_kernel
    assert prepared.requirements is marker
    assert runtime.allocations == []
    storage.close()


def test_prepare_operation_rejects_duplicate_specialization(runtime):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)

    class CompiledKernel:
        def prepare_sram_requirements(self, prepared_state):
            del prepared_state
            return object()

    compiled_kernel = CompiledKernel()

    def operation(*args, **kwargs):
        raise AssertionError("preparation must not execute the operation")

    operation._ttlang_prepare_operation = lambda prepared_state: (
        compiled_kernel,
        (prepared_state,),
    )
    storage.prepare_operation(operation, state)
    with pytest.raises(ValueError, match="already prepared"):
        storage.prepare_operation(operation, state)
    storage.close()


def test_prepare_operation_requires_owned_tensor(runtime):
    storage = SRAMStorage(device=runtime.device)
    declare(storage)

    def operation(*args, **kwargs):
        raise AssertionError("preparation must not execute the operation")

    operation._ttlang_prepare_operation = lambda value: (object(), (value,))
    with pytest.raises(ValueError, match="must borrow a tensor"):
        storage.prepare_operation(operation, object())
    storage.close()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("dfb_reconfiguration_plan", object()),
        ("runtime_resource_factory", lambda: None),
        ("mesh_program_placements", [object()]),
        ("device_domain", object()),
        ("kernel_fabric_routes", [[object()]]),
        ("num_pipe_sync_semaphores", 1),
        ("pipe_sram_scratch_bytes", 16),
        ("num_pipe_global_semaphores", 1),
        ("num_dfb_resets", 1),
    ],
)
def test_prepare_operation_rejects_unmodeled_resources(runtime, attribute, value):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    compiled_kernel = SimpleNamespace(**{attribute: value})
    compiled_kernel.prepare_sram_requirements = lambda prepared_state: object()

    def operation(*args, **kwargs):
        raise AssertionError("preparation must not execute the operation")

    operation._ttlang_prepare_operation = lambda prepared_state: (
        compiled_kernel,
        (prepared_state,),
    )
    with pytest.raises(ValueError, match=attribute):
        storage.prepare_operation(operation, state)
    assert runtime.allocations == []
    storage.close()


def test_joint_allocation_publishes_views_and_operation_binding(runtime):
    from ttl._persistent_storage import (
        current_operation_binding,
        with_persistent_storage,
    )

    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    locations = (
        SRAMLocation((), (0, 0)),
        SRAMLocation((), (1, 0)),
    )
    requirement = SRAMStorageRequirement(
        owner=SRAMOwner(SRAMOwnerKind.COMPILER_ARENA, 0),
        extent_bytes=2048,
        alignment_bytes=16,
        address_domains=(SRAMAddressDomain(locations),),
        ownership=SRAMOwnership.MOVABLE,
        lifetime=SRAMLifetime.INVOCATION,
        addressing=SRAMAddressing.UNIFORM,
    )
    operation_requirements = PreparedSRAMOperation(
        "prepared",
        (requirement,),
        (),
        (SRAMArenaBinding(0, ((0, 0), (1, 0))),),
    )

    class CompiledKernel:
        cb_configs = ()

        def prepare_sram_requirements(self, prepared_state):
            del prepared_state
            return operation_requirements

        def prepare_device_program(self, prepared_state, *, prepared_sram_resources):
            assert prepared_state.buffer_address() == 0x10000
            assert prepared_sram_resources.uniform_arena.buffer_address() == 0x11000
            return SimpleNamespace(
                max_program_config_size_bytes=43264,
                max_kernel_binary_size_bytes=28672,
            )

    compiled_kernel = CompiledKernel()

    def operation(*args, **kwargs):
        raise AssertionError("preparation must not execute the operation")

    operation._ttlang_prepare_operation = lambda prepared_state: (
        compiled_kernel,
        (prepared_state,),
    )
    storage.prepare_operation(operation, state)
    storage.allocate()

    binding = with_persistent_storage(
        lambda value: current_operation_binding(compiled_kernel)
    )(state)
    assert binding.requirements is operation_requirements
    assert binding.uniform_arena.buffer_address() == 0x11000
    assert binding.core_arenas == ()
    assert binding.control_tensors == ()
    assert storage.submit(lambda value: value, state) is state
    metrics = storage.allocation_metrics()
    assert metrics["required_peak_bytes"] == 12288
    assert metrics["actual_reservation_bytes"] == 12288
    assert metrics["fragmentation_bytes"] == 0
    assert metrics["programs"] == (
        {
            "max_program_config_size_bytes": 43264,
            "max_kernel_binary_size_bytes": 28672,
        },
    )
    storage.close()


def test_joint_allocation_rolls_back_pool_and_views(runtime):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    locations = (
        SRAMLocation((), (0, 0)),
        SRAMLocation((), (1, 0)),
    )
    requirement = SRAMStorageRequirement(
        owner=SRAMOwner(SRAMOwnerKind.COMPILER_ARENA, 0),
        extent_bytes=2048,
        alignment_bytes=16,
        address_domains=(SRAMAddressDomain(locations),),
        ownership=SRAMOwnership.MOVABLE,
        lifetime=SRAMLifetime.INVOCATION,
    )
    prepared_requirements = PreparedSRAMOperation(
        "rollback",
        (requirement,),
        (),
        (SRAMArenaBinding(0, ((0, 0), (1, 0))),),
    )

    class CompiledKernel:
        cb_configs = ()

        def prepare_sram_requirements(self, prepared_state):
            del prepared_state
            return prepared_requirements

        def prepare_device_program(self, prepared_state, *, prepared_sram_resources):
            del prepared_state, prepared_sram_resources
            return SimpleNamespace(
                max_program_config_size_bytes=4096,
                max_kernel_binary_size_bytes=8192,
            )

    compiled_kernel = CompiledKernel()

    def operation(*args, **kwargs):
        raise AssertionError("preparation must not execute the operation")

    operation._ttlang_prepare_operation = lambda prepared_state: (
        compiled_kernel,
        (prepared_state,),
    )
    storage.prepare_operation(operation, state)
    runtime.failure.update(stage="initialize", at=1)
    with pytest.raises(RuntimeError, match="injected initialize failure"):
        storage.allocate()
    assert runtime.allocations
    assert all(not resource.allocated for resource in runtime.allocations)

    runtime.failure.update(stage=None, at=0)
    storage.allocate()
    assert storage.allocation_metrics()["actual_reservation_bytes"] == 12288
    storage.close()


def test_joint_allocation_rolls_back_when_program_preparation_fails(runtime):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    locations = (
        SRAMLocation((), (0, 0)),
        SRAMLocation((), (1, 0)),
    )
    requirement = SRAMStorageRequirement(
        owner=SRAMOwner(SRAMOwnerKind.COMPILER_ARENA, 0),
        extent_bytes=2048,
        alignment_bytes=16,
        address_domains=(SRAMAddressDomain(locations),),
        ownership=SRAMOwnership.MOVABLE,
        lifetime=SRAMLifetime.INVOCATION,
    )
    prepared_requirements = PreparedSRAMOperation(
        "program-failure",
        (requirement,),
        (),
        (SRAMArenaBinding(0, ((0, 0), (1, 0))),),
    )

    class CompiledKernel:
        cb_configs = ()

        def prepare_sram_requirements(self, prepared_state):
            del prepared_state
            return prepared_requirements

        def prepare_device_program(self, prepared_state, *, prepared_sram_resources):
            del prepared_state, prepared_sram_resources
            raise RuntimeError("program capacity exceeded")

    compiled_kernel = CompiledKernel()

    def operation(*args, **kwargs):
        raise AssertionError("preparation must not execute the operation")

    operation._ttlang_prepare_operation = lambda prepared_state: (
        compiled_kernel,
        (prepared_state,),
    )
    storage.prepare_operation(operation, state)
    with pytest.raises(RuntimeError, match="program capacity exceeded"):
        storage.allocate()

    assert runtime.allocations
    assert all(not resource.allocated for resource in runtime.allocations)
    assert not any(event[0] == "initialize" for event in runtime.events)
    storage.close()


def test_joint_allocation_prepares_distinct_operations_with_shared_scratch(runtime):
    from ttl._persistent_storage import (
        current_operation_binding,
        with_persistent_storage,
    )

    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    observed_bindings = []

    def observe_program(_state, resources):
        observed_bindings.append(resources)
        return SimpleNamespace(
            max_program_config_size_bytes=4096,
            max_kernel_binary_size_bytes=8192,
        )

    first, first_kernel = preparable_operation("first", 1024, observe_program)
    second, second_kernel = preparable_operation("second", 2048, observe_program)
    storage.prepare_operation(first, state)
    storage.prepare_operation(second, state)
    storage.allocate()

    assert len(observed_bindings) == 2
    assert (
        observed_bindings[0].uniform_arena.buffer_address()
        == observed_bindings[1].uniform_arena.buffer_address()
    )
    assert (
        with_persistent_storage(lambda value: current_operation_binding(first_kernel))(
            state
        )
        is observed_bindings[0]
    )
    assert (
        with_persistent_storage(lambda value: current_operation_binding(second_kernel))(
            state
        )
        is observed_bindings[1]
    )
    assert storage.allocation_metrics()["required_peak_bytes"] == 12288
    storage.close()


@pytest.mark.parametrize("program_would_overflow", [False, True])
def test_joint_data_overflow_precedes_program_preparation(
    runtime, program_would_overflow
):
    runtime.api._ttnn.reports.get_device_info = lambda device: SimpleNamespace(
        cb_limit=6143
    )
    program_calls = []

    def prepare_program(_state, _resources):
        program_calls.append(True)
        if program_would_overflow:
            raise RuntimeError("program capacity exceeded")
        return SimpleNamespace(
            max_program_config_size_bytes=4096,
            max_kernel_binary_size_bytes=8192,
        )

    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    operation, _ = preparable_operation("overflow", 2048, prepare_program)
    storage.prepare_operation(operation, state)
    with pytest.raises(ValueError, match="exceeds SRAM budget"):
        storage.allocate()

    assert program_calls == []
    assert runtime.allocations == []
    storage.close()


def test_joint_allocation_failure_rolls_back_and_retries(runtime):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    operation, _ = preparable_operation("allocation-failure")
    storage.prepare_operation(operation, state)
    runtime.failure.update(stage="allocation", at=0)

    with pytest.raises(RuntimeError, match="injected allocation failure"):
        storage.allocate()
    assert runtime.allocations == []

    runtime.failure.update(stage=None, at=0)
    storage.allocate()
    storage.close()


def test_joint_allocation_rolls_back_invalid_measured_capacity(runtime, monkeypatch):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    operation, _ = preparable_operation("invalid-measured-capacity")
    storage.prepare_operation(operation, state)
    original_page_size = Resource.buffer_aligned_page_size
    monkeypatch.setattr(Resource, "buffer_aligned_page_size", lambda resource: 1)

    with pytest.raises(RuntimeError, match="smaller than the required live data"):
        storage.allocate()
    assert runtime.allocations
    assert all(not resource.allocated for resource in runtime.allocations)

    monkeypatch.setattr(Resource, "buffer_aligned_page_size", original_page_size)
    storage.allocate()
    storage.close()


@pytest.mark.parametrize("configuration_bytes", [4096, 4097])
def test_joint_program_capacity_boundary(runtime, configuration_bytes):
    def prepare_program(_state, _resources):
        if configuration_bytes > 4096:
            raise RuntimeError("program capacity exceeded")
        return SimpleNamespace(
            max_program_config_size_bytes=configuration_bytes,
            max_kernel_binary_size_bytes=8192,
        )

    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    operation, _ = preparable_operation("program-boundary", 2048, prepare_program)
    storage.prepare_operation(operation, state)

    if configuration_bytes == 4096:
        storage.allocate()
        assert storage.allocation_metrics()["programs"] == (
            {
                "max_program_config_size_bytes": 4096,
                "max_kernel_binary_size_bytes": 8192,
            },
        )
    else:
        with pytest.raises(RuntimeError, match="program capacity exceeded"):
            storage.allocate()
        assert not any(event[0] == "initialize" for event in runtime.events)
        assert all(not resource.allocated for resource in runtime.allocations)
    storage.close()


@pytest.mark.parametrize("addressing", ["uniform", "per-core"])
def test_persistent_declarations_export_movable_requirements(runtime, addressing):
    storage = SRAMStorage(device=runtime.device)
    declare(storage, addressing=addressing)

    prepared = storage.requirements()
    requirement = prepared.requirements[0]
    assert requirement.owner == SRAMOwner(SRAMOwnerKind.PERSISTENT_DECLARATION, 0)
    assert requirement.extent_bytes == 4096
    assert requirement.ownership is SRAMOwnership.MOVABLE
    assert requirement.lifetime is SRAMLifetime.PERSISTENT
    expected_domain_count = 1 if addressing == "uniform" else 2
    assert len(requirement.address_domains) == expected_domain_count
    storage.close()


def _make_persistent_increment(grid, increment):
    @ttl.operation(grid=grid)
    def update_persistent_state(state):
        state_dfb = ttl.make_tensor_backed_dfb(state, shape=(1, 1), block_count=1)

        @ttl.compute()
        def update():
            with state_dfb.wait() as state_block:
                value = ttl.block.fill(
                    increment,
                    shape=state_block.shape,
                    dtype=state_block.dtype,
                )
                state_block.store(state_block + value)

        @ttl.datamovement()
        def publish():
            state_dfb.publish()

        @ttl.datamovement()
        def unused():
            pass

    return update_persistent_state


def _make_accumulate_with_temporary_storage(grid):
    @ttl.operation(grid=grid)
    def accumulate_with_temporary_storage(input_tensor, state, output_tensor):
        state_dfb = ttl.make_tensor_backed_dfb(state, shape=(1, 1), block_count=1)
        input_dfb = ttl.make_dataflow_buffer_like(state, shape=(1, 1), block_count=2)
        output_dfb = ttl.make_dataflow_buffer_like(state, shape=(1, 1), block_count=2)

        @ttl.compute()
        def accumulate():
            with (
                input_dfb.wait() as input_block,
                state_dfb.wait() as state_block,
                output_dfb.reserve() as output_block,
            ):
                updated = state_block + input_block
                state_block.store(updated)
                output_block.store(updated)

        @ttl.datamovement()
        def read_input():
            column, row = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[row, column], input_block).wait()
            state_dfb.publish()

        @ttl.datamovement()
        def write_output():
            column, row = ttl.node(dims=2)
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output_tensor[row, column]).wait()

    return accumulate_with_temporary_storage


@pytest.mark.requires_device
@pytest.mark.parametrize(
    ("torch_dtype", "ttnn_dtype"),
    [(torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32)],
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("addressing", DEVICE_ADDRESSING_MODES)
@pytest.mark.parametrize("sharding_name", ["height", "width", "block"])
def test_persistent_state_shared_by_distinct_operations(
    device, torch_dtype, ttnn_dtype, addressing, sharding_name
):
    configurations = {
        "height": (
            (64, 32),
            ((0, 0), (1, 0)),
            (2, 1),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        "width": (
            (32, 64),
            ((0, 0), (1, 0)),
            (2, 1),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        "block": (
            (64, 64),
            ((0, 0), (1, 0), (0, 1), (1, 1)),
            (2, 2),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
    }
    shape, cores, grid, sharding = configurations[sharding_name]
    add_one = _make_persistent_increment(grid, 1)
    add_two = _make_persistent_increment(grid, 2)
    declaration_options = {
        "shape": shape,
        "shard_shape": (32, 32),
        "cores": cores,
        "dtype": ttnn_dtype,
        "layout": ttnn.TILE_LAYOUT,
        "addressing": addressing,
        "sharding": sharding,
    }
    with SRAMStorage(device=device) as storage:
        state = storage.tensor(**declaration_options)
        first_prepared = storage.prepare_operation(
            add_one, state, options="--ttl-memory-model=compiler-l1"
        )
        second_prepared = storage.prepare_operation(
            add_two, state, options="--ttl-memory-model=compiler-l1"
        )
        separate_reservation_bytes = measure_separate_reservation(
            device,
            declaration_options,
            (first_prepared, second_prepared),
        )
        storage.allocate()
        metrics = storage.allocation_metrics()
        assert metrics["actual_reservation_bytes"] >= metrics["required_peak_bytes"]
        assert metrics["fragmentation_bytes"] == (
            metrics["actual_reservation_bytes"] - metrics["required_peak_bytes"]
        )
        assert len(metrics["programs"]) == 2
        assert separate_reservation_bytes >= metrics["required_peak_bytes"]
        print(
            "joint SRAM metrics:",
            {
                **metrics,
                "separate_actual_reservation_bytes": separate_reservation_bytes,
            },
        )
        returned_state = storage.submit(
            ttnn.full,
            shape,
            1.0,
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            optional_tensor=state,
        )
        assert returned_state is state
        for iteration_index in range(2):
            add_one(state, options="--ttl-memory-model=compiler-l1")
            add_two(state, options="--ttl-memory-model=compiler-l1")
        actual = storage.submit(ttnn.to_torch, state).float()
    expected = torch.full(shape, 7, dtype=torch_dtype).float()
    if torch_dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    ("torch_dtype", "ttnn_dtype"),
    [(torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32)],
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("addressing", DEVICE_ADDRESSING_MODES)
@pytest.mark.parametrize("sharding_name", ["height", "width", "block"])
def test_persistent_state_with_compiler_managed_temporary_storage(
    device, torch_dtype, ttnn_dtype, addressing, sharding_name
):
    configurations = {
        "height": (
            (64, 32),
            ((0, 0), (0, 1)),
            (1, 2),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        "width": (
            (32, 64),
            ((0, 0), (1, 0)),
            (2, 1),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        "block": (
            (64, 64),
            ((0, 0), (1, 0), (0, 1), (1, 1)),
            (2, 2),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
    }
    shape, cores, grid, sharding = configurations[sharding_name]
    accumulate = _make_accumulate_with_temporary_storage(grid)
    input_tensor = to_dram(torch.ones(shape, dtype=torch_dtype), device)
    output_tensor = to_dram(torch.zeros(shape, dtype=torch_dtype), device)
    declaration_options = {
        "shape": shape,
        "shard_shape": (32, 32),
        "cores": cores,
        "dtype": ttnn_dtype,
        "layout": ttnn.TILE_LAYOUT,
        "addressing": addressing,
        "sharding": sharding,
    }
    with SRAMStorage(device=device) as storage:
        state = storage.tensor(**declaration_options)
        prepared = storage.prepare_operation(
            accumulate,
            input_tensor,
            state,
            output_tensor,
            options="--ttl-memory-model=compiler-l1",
        )
        separate_reservation_bytes = measure_separate_reservation(
            device, declaration_options, (prepared,)
        )
        storage.allocate()
        metrics = storage.allocation_metrics()
        assert metrics["actual_reservation_bytes"] >= metrics["required_peak_bytes"]
        assert len(metrics["programs"]) == 1
        assert separate_reservation_bytes >= metrics["required_peak_bytes"]
        print(
            "joint SRAM metrics:",
            {
                **metrics,
                "separate_actual_reservation_bytes": separate_reservation_bytes,
            },
        )
        for iteration_index in range(2):
            accumulate(
                input_tensor,
                state,
                output_tensor,
                options="--ttl-memory-model=compiler-l1",
            )
        actual_state = storage.submit(ttnn.to_torch, state).float()
        actual_output = ttnn.to_torch(output_tensor).float()
    expected = torch.full(shape, 2, dtype=torch_dtype).float()
    if torch_dtype == torch.bfloat16:
        assert_allclose(actual_state, expected, rtol=0.05, atol=1.0)
        assert_allclose(actual_output, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual_state, expected, rtol=1e-5, atol=1e-6)
        assert_allclose(actual_output, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("addressing", ["uniform", "per-core"])
@pytest.mark.parametrize("layout", ["height", "width", "block"])
def test_allocate_initialize_once_and_repeated_external_updates(
    runtime, dtype, addressing, layout
):
    geometries = {
        "height": ((64, 32), ((0, 0), (1, 0)), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        "width": ((32, 64), ((0, 0), (1, 0)), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        "block": (
            (64, 64),
            ((0, 0), (1, 0), (0, 1), (1, 1)),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
    }
    shape, cores, sharding = geometries[layout]
    with SRAMStorage(device=runtime.device) as storage:
        state = declare(
            storage,
            shape=shape,
            cores=cores,
            sharding=sharding,
            dtype=dtype,
            addressing=addressing,
        )
        assert runtime.allocations == []
        with pytest.raises(RuntimeError, match="not been allocated"):
            storage.submit(lambda value: None, state)
        storage.allocate()
        resource = runtime.allocations[0]
        assert resource.value == 0
        assert resource.shape == shape
        assert resource.options["dtype"] == dtype
        config = resource.options["memory_config"]
        assert config.memory_layout == sharding
        expected = ttnn.MemoryConfig(sharding, ttnn.BufferType.L1, config.shard_spec)
        expected.experimental_set_per_core_allocation(addressing == "per-core")
        assert config == expected

        def update(value, amount):
            value.value += amount
            return value

        assert storage.submit(update, state, 2) is state
        assert storage.submit(update, state, 3) is state
        assert resource.value == 5
        assert len([event for event in runtime.events if event[0] == "initialize"]) == 1
        with pytest.raises(RuntimeError, match="unallocated declarations"):
            storage.allocate()
    assert not resource.allocated
    with pytest.raises(RuntimeError, match="closing or closed"):
        storage.submit(update, state, 1)


def test_completion_selection_ignores_mutable_stall_group(runtime):
    selected = [ttnn.SubDeviceId(1)]
    storage = SRAMStorage(
        device=runtime.device, queue_ids=(0, 1), sub_device_ids=selected
    )
    state = declare(storage)
    selected.clear()
    storage.allocate()
    storage.submit(lambda value: None, state)
    storage.submit(lambda value: None, state)
    records = [event for event in runtime.events if event[0] == "record"]
    assert [event[2]["cq_id"] for event in records] == [0, 1, 0, 1, 0, 1]
    assert all(event[2]["sub_device_ids"] == [ttnn.SubDeviceId(1)] for event in records)
    orders = [event for event in runtime.events if event[0] == "order"]
    assert len(orders) == 4
    storage.close()


@pytest.mark.parametrize("addressing", ["uniform", "per-core"])
def test_external_tensor_alias_wrapper_restores_owned_reference(runtime, addressing):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage, addressing=addressing)
    storage.allocate()
    if addressing == "per-core":
        make_per_core(runtime.allocations[0])
    alias_wrapper = copy.copy(runtime.allocations[0])
    assert storage.submit(lambda value: alias_wrapper, state) is state
    storage.close()


@pytest.mark.parametrize(
    "difference",
    [
        "address",
        "shape",
        "padded_shape",
        "dtype",
        "layout",
        "tile",
        "memory_config",
        "allocation_state",
        "device",
        "addressing",
        "per_core_address",
    ],
)
def test_external_nonalias_tensor_wrapper_is_not_restored(runtime, difference):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    storage.allocate()
    if difference == "per_core_address":
        make_per_core(runtime.allocations[0])
    candidate = copy.copy(runtime.allocations[0])
    candidate.core_addresses = dict(candidate.core_addresses)
    if difference == "address":
        candidate.address += 0x1000
    elif difference == "shape":
        candidate.shape = (32, 32)
    elif difference == "padded_shape":
        candidate.padded_shape = (96, 32)
    elif difference == "dtype":
        candidate.dtype = ttnn.bfloat16
    elif difference == "layout":
        candidate.layout = ttnn.ROW_MAJOR_LAYOUT
    elif difference == "tile":
        candidate.tile = (16, 32)
    elif difference == "memory_config":
        candidate.options = {
            **candidate.options,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }
    elif difference == "allocation_state":
        candidate.allocated = False
    elif difference == "device":
        other_device = Device()
        other_device.device_id += 1
        candidate.options = {**candidate.options, "device": other_device}
    elif difference == "addressing":
        candidate.is_per_core_allocated = lambda: True
    elif difference == "per_core_address":
        first_address = next(iter(candidate.core_addresses))
        candidate.core_addresses[first_address] += 0x1000
    assert storage.submit(lambda value: candidate, state) is candidate
    storage.close()


@pytest.mark.parametrize("stage", ["allocation", "initialize", "record", "wait"])
def test_failed_allocation_rolls_back_without_publishing_references(runtime, stage):
    storage = SRAMStorage(device=runtime.device)
    first, second = declare(storage), declare(storage)
    runtime.failure.update(
        stage=stage, at=1 if stage in ("allocation", "initialize") else 0
    )
    with pytest.raises(RuntimeError, match="injected"):
        storage.allocate()
    assert all(not resource.allocated for resource in runtime.allocations)
    with pytest.raises(RuntimeError, match="not been allocated"):
        storage.submit(lambda value: None, first)
    runtime.failure["stage"] = None
    storage.allocate()
    storage.submit(lambda *values: None, first, second)
    storage.close()
    assert all(not resource.allocated for resource in runtime.allocations)


def test_failed_recovery_retains_allocation_and_close_can_retry(runtime):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    storage.allocate()
    runtime.failure.update(stage="record", at=0)
    with pytest.raises(RuntimeError, match="record failure"):
        storage.submit(lambda value: None, state)
    runtime.failure["stage"] = "recover"
    with pytest.raises(RuntimeError, match="recover failure"):
        storage.close()
    assert runtime.allocations[0].allocated
    runtime.failure["stage"] = None
    storage.close()
    assert not runtime.allocations[0].allocated


def test_failed_allocation_recovery_prevents_reallocation(runtime):
    storage = SRAMStorage(device=runtime.device)
    declare(storage)
    runtime.failure.update(stage="initialize", at=0)
    original_recover = runtime.api.synchronize_device

    def fail_recovery(*args, **kwargs):
        raise RuntimeError("injected recover failure")

    runtime.api.synchronize_device = fail_recovery
    with pytest.raises(RuntimeError, match="recover failure"):
        storage.allocate()
    assert runtime.allocations[0].allocated
    with pytest.raises(RuntimeError, match="unallocated declarations"):
        storage.allocate()
    runtime.api.synchronize_device = original_recover
    runtime.failure["stage"] = None
    storage.close()
    assert not runtime.allocations[0].allocated


def test_declaration_after_allocation_is_rejected_without_affecting_state(runtime):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    storage.allocate()
    with pytest.raises(RuntimeError, match="declarations are complete"):
        declare(storage)
    storage.submit(lambda value: None, state)
    storage.close()


def test_uninitialized_allocation_does_not_write_payload(runtime):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage, initialize="uninitialized")
    storage.allocate()
    assert not any(event[0] == "initialize" for event in runtime.events)
    storage.submit(lambda value: None, state)
    storage.close()


@pytest.mark.parametrize("change", ["close", "manager", "release"])
def test_invalid_runtime_binding_is_rejected_before_external_launch(runtime, change):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    storage.allocate()
    if change == "close":
        runtime.device.open = False
    elif change == "manager":
        runtime.device.manager = 2
    else:
        runtime.allocations[0].allocated = False
    launched = []
    with pytest.raises(RuntimeError):
        storage.submit(lambda value: launched.append(value), state)
    assert launched == []
    # Restore the simulated device solely to discharge ownership in this test.
    runtime.device.open = True
    runtime.device.manager = 1
    if change == "release":
        runtime.allocations[0].allocated = True
    storage.close()


def test_changed_device_identity_is_rejected_before_external_launch(runtime):
    storage = SRAMStorage(device=runtime.device)
    state = declare(storage)
    storage.allocate()
    runtime.device.device_id = 18
    launched = []
    with pytest.raises(RuntimeError, match="closed or replaced"):
        storage.submit(lambda value: launched.append(value), state)
    assert launched == []
    runtime.device.device_id = 17
    storage.close()


@pytest.mark.parametrize("queue_ids", [(), (1,), (0, 0), (0, 2), (False,)])
def test_invalid_completion_queues_fail_before_allocation(runtime, queue_ids):
    with pytest.raises(ValueError, match="queue_ids"):
        SRAMStorage(device=runtime.device, queue_ids=queue_ids)
    assert runtime.allocations == []


@pytest.mark.parametrize("ids", [[], [ttnn.SubDeviceId(9)], [ttnn.SubDeviceId(0)] * 2])
def test_invalid_completion_subdevices_fail_before_allocation(runtime, ids):
    with pytest.raises(ValueError, match="sub-device"):
        SRAMStorage(device=runtime.device, sub_device_ids=ids)
    assert runtime.allocations == []


@pytest.mark.parametrize(
    "options",
    [
        {"shape": (0, 32)},
        {"shape": (32, 17)},
        {"shard_shape": (32,)},
        {"cores": ()},
        {"cores": ((0, 0), (0, 0))},
        {"cores": ((8, 0),)},
        {"layout": ttnn.ROW_MAJOR_LAYOUT},
        {"addressing": "invalid"},
        {"initialize": "invalid"},
        {"sharding": ttnn.TensorMemoryLayout.INTERLEAVED},
    ],
)
def test_invalid_declarations_do_not_allocate(runtime, options):
    with SRAMStorage(device=runtime.device) as storage:
        with pytest.raises((ValueError, RuntimeError)):
            declare(storage, **options)
        assert runtime.allocations == []


def test_block_sharding_requires_complete_rectangular_core_set(runtime):
    with SRAMStorage(device=runtime.device) as storage:
        with pytest.raises(ValueError, match="complete rectangle"):
            declare(
                storage,
                shape=(64, 64),
                cores=((0, 0), (1, 0), (1, 1)),
                sharding=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            )
        assert runtime.allocations == []


def test_pinned_mesh_api_uses_explicit_default_completion_domain(runtime):
    device = Device()
    device.is_initialized = None
    device.num_hw_cqs = None
    device.get_sub_device_ids = None
    device.get_active_sub_device_manager_id = None
    storage = SRAMStorage(device=device)
    state = declare(storage)
    storage.allocate()
    storage.submit(lambda value: None, state)
    records = [event for event in runtime.events if event[0] == "record"]
    assert all(event[2]["cq_id"] == 0 for event in records)
    assert all(event[2]["sub_device_ids"] == [ttnn.SubDeviceId(0)] for event in records)
    storage.close()


def test_non_mesh_device_is_rejected_before_allocating(runtime):
    with pytest.raises(TypeError, match="mesh device"):
        SRAMStorage(device=object())
    assert runtime.allocations == []


def test_sram_storage_is_exported_from_ttl():
    assert ttl.SRAMStorage is SRAMStorage


def test_allocation_requires_at_least_one_declaration(runtime):
    with SRAMStorage(device=runtime.device) as storage:
        with pytest.raises(ValueError, match="no tensor declarations"):
            storage.allocate()


def test_closed_storage_cannot_be_reentered(runtime):
    storage = SRAMStorage(device=runtime.device)
    storage.close()
    with pytest.raises(RuntimeError, match="closing or closed"):
        with storage:
            pass


def test_device_close_before_storage_close_retains_allocation(runtime):
    storage = SRAMStorage(device=runtime.device)
    declare(storage)
    storage.allocate()
    runtime.device.open = False
    with pytest.raises(RuntimeError, match="closed or replaced"):
        storage.close()
    assert runtime.allocations[0].allocated
    runtime.device.open = True
    storage.close()
    assert not runtime.allocations[0].allocated


def test_external_release_is_rejected_during_close(runtime):
    storage = SRAMStorage(device=runtime.device)
    declare(storage)
    storage.allocate()
    runtime.allocations[0].allocated = False
    with pytest.raises(RuntimeError, match="released externally"):
        storage.close()
    runtime.allocations[0].allocated = True
    storage.close()
    assert not runtime.allocations[0].allocated


def test_storage_submit_requires_its_own_reference(runtime):
    with SRAMStorage(device=runtime.device) as storage:
        with pytest.raises(ValueError, match="borrow a tensor"):
            storage.submit(lambda value: value, object())
