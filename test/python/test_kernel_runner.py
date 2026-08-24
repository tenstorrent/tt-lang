# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only tests for ttl.kernel_runner resource allocation helpers."""

from collections import defaultdict
from dataclasses import FrozenInstanceError
import gc
import os
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace
from typing import NamedTuple
import weakref

import pytest

from ttl import (
    CoreRuntimeArgs,
    KernelDefine,
    Kernel,
    KernelKind,
    KernelRuntimeResources,
    ProgramRuntimeResources,
    kernel_runner,
)
from ttl.dataflow_buffer import DFBStorageSegment, PhysicalDFBConfig
from ttl.ttl import ProgramRuntimeResources as TTLProgramRuntimeResources


class _FakeTensor:
    def __init__(
        self,
        device,
        address=0x2000,
        dtype=None,
        tile_shape=(32, 32),
        shard_shape=(32, 64),
    ):
        self._device = device
        self._address = address
        self.dtype = dtype
        self.layout = "TILE"
        self.tile_shape = tile_shape
        self.shard_shape = shard_shape

    def device(self):
        return self._device

    def buffer_address(self):
        return self._address

    def get_tile(self):
        return _FakeTTNN.Tile(self.tile_shape)

    def memory_config(self):
        class ShardSpec:
            shape = self.shard_shape

        class MemoryConfig:
            buffer_type = "L1"
            memory_layout = "HEIGHT_SHARDED"
            shard_spec = ShardSpec()

        return MemoryConfig()


class _FakeTensorWithoutDevice:
    pass


class _FakeDevice:
    def __init__(self, device_id):
        self._device_id = device_id

    def id(self):
        return self._device_id


class _FakeGridSize:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeBoundingBox:
    def __init__(self, ranges):
        self._ranges = ranges

    def grid_size(self):
        max_x = max(core_range.end.x for core_range in self._ranges)
        max_y = max(core_range.end.y for core_range in self._ranges)
        return _FakeGridSize(max_x + 1, max_y + 1)


class _FakeCoreCoord(NamedTuple):
    x: int
    y: int


class _FakeCoreRange(NamedTuple):
    start: _FakeCoreCoord
    end: _FakeCoreCoord


class _FakeCoreRanges:
    def __init__(self, ranges=(((0, 0), (0, 0)),)):
        parsed_ranges = []
        for core_range in ranges:
            if len(core_range) == 4:
                start_x, start_y, end_x, end_y = core_range
                start = _FakeCoreCoord(start_x, start_y)
                end = _FakeCoreCoord(end_x, end_y)
            else:
                start, end = core_range
                start = _FakeCoreCoord(*start)
                end = _FakeCoreCoord(*end)
            parsed_ranges.append(_FakeCoreRange(start, end))
        self._ranges = tuple(parsed_ranges)

    def bounding_box(self):
        return _FakeBoundingBox(self._ranges)

    def ranges(self):
        return self._ranges


class _CopyingRuntimeArgumentRow(defaultdict):
    def __init__(self):
        super().__init__(list)

    def __getitem__(self, key):
        return list(super().__getitem__(key))

    def __setitem__(self, key, value):
        super().__setitem__(key, list(value))


class _EmittedCoreRangeSet:
    def __init__(self, ranges):
        self._ranges = tuple(ranges)

    def ranges(self):
        return self._ranges

    def bounding_box(self):
        return _FakeBoundingBox(self._ranges)


def _load_emitted_runner(monkeypatch, source, run_kernel):
    emitted_ttnn = SimpleNamespace(
        ComputeConfigDescriptor=type("ComputeConfigDescriptor", (), {}),
        ReaderConfigDescriptor=type("ReaderConfigDescriptor", (), {}),
        WriterConfigDescriptor=type("WriterConfigDescriptor", (), {}),
        CoreCoord=_FakeTTNN.CoreCoord,
        CoreRange=_FakeTTNN.CoreRange,
        CoreRangeSet=_EmittedCoreRangeSet,
    )
    monkeypatch.setitem(sys.modules, "ttnn", emitted_ttnn)
    monkeypatch.setattr(kernel_runner, "run_kernel_on_device", run_kernel)
    namespace = {"__name__": "emitted_runner_test"}
    exec(compile(source, "<generated-runner>", "exec"), namespace)
    return namespace


class _MalformedCoreRanges:
    def ranges(self):
        return (object(),)

    @staticmethod
    def num_cores():
        return 1


def _FakeExplicitCoreRanges(start, end):
    return _FakeTTNN.CoreRangeSet(
        (_FakeTTNN.CoreRange(_FakeTTNN.CoreCoord(*start), _FakeTTNN.CoreCoord(*end)),)
    )


class _FakeTTNN:
    def __init__(self):
        self.create_calls = []
        self.generic_op_calls = []
        self.synchronize_calls = []
        self.next_address = 0x1000

    class DataType:
        BFLOAT16 = "BFLOAT16"
        BFLOAT4_B = "BFLOAT4_B"
        BFLOAT8_B = "BFLOAT8_B"
        FLOAT32 = "FLOAT32"
        INT32 = "INT32"
        UINT32 = "UINT32"
        UINT16 = "UINT16"
        UINT8 = "UINT8"

    class TensorAccessorArgs:
        def __init__(self, tensor):
            self.tensor = tensor

        @staticmethod
        def get_compile_time_args():
            return []

    class ProgramDescriptor:
        def __init__(self, kernels, cbs, semaphores):
            self.kernels = kernels
            self.cbs = cbs
            self.semaphores = semaphores
            self.custom_program_hash = None

    class SemaphoreDescriptor:
        def __init__(
            self,
            semaphore_id,
            core_ranges,
            initial_value,
            core_type="WORKER",
        ):
            self.id = semaphore_id
            self.core_ranges = core_ranges
            self.initial_value = initial_value
            self.core_type = core_type

    class KernelDescriptor:
        def __init__(
            self,
            kernel_source,
            core_ranges,
            compile_time_args,
            common_runtime_args,
            config,
            compiler_include_paths=None,
            defines=None,
            runtime_args=None,
        ):
            self.kernel_source = kernel_source
            self.core_ranges = core_ranges
            self.compile_time_args = compile_time_args
            self.common_runtime_args = common_runtime_args
            self.config = config
            self.compiler_include_paths = compiler_include_paths or []
            self.defines = defines or []
            self.runtime_args = defaultdict(_CopyingRuntimeArgumentRow)
            for core, values in runtime_args or []:
                self.runtime_args[core.x][core.y] = values

    class Tile:
        def __init__(self, tile_shape):
            self.tile_shape = tuple(tile_shape)

        def get_tile_size(self, _data_format):
            tile_height, tile_width = self.tile_shape
            return tile_height * tile_width * 2

    class TileDescriptor:
        def __init__(self, tile):
            self.tile = tile

    class CBFormatDescriptor:
        def __init__(self, buffer_index, data_format, page_size, tile=None):
            self.buffer_index = buffer_index
            self.data_format = data_format
            self.page_size = page_size
            self.tile = tile

    class CBDescriptor:
        def __init__(self, total_size, core_ranges, format_descriptors):
            self.total_size = total_size
            self.core_ranges = core_ranges
            self.format_descriptors = format_descriptors
            self.backing_desc = None

        def set_buffer_from_cb(self, backing_desc):
            self.backing_desc = backing_desc

    class CoreCoord:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class CoreRange:
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def num_cores(self):
            return (self.end.x - self.start.x + 1) * (self.end.y - self.start.y + 1)

    class CoreRangeSet:
        def __init__(self, ranges):
            self._ranges = tuple(ranges)

        def ranges(self):
            return self._ranges

        def num_cores(self):
            return sum(core_range.num_cores() for core_range in self._ranges)

        def bounding_box(self):
            return _FakeBoundingBox(self._ranges)

    @staticmethod
    def corerange_to_cores(core_range_set, max_cores=None, row_wise=False):
        cores = []
        for core_range in core_range_set.ranges():
            xs = range(int(core_range.start.x), int(core_range.end.x) + 1)
            ys = range(int(core_range.start.y), int(core_range.end.y) + 1)
            if row_wise:
                for y in ys:
                    for x in xs:
                        cores.append(_FakeTTNN.CoreCoord(x, y))
            else:
                for x in xs:
                    for y in ys:
                        cores.append(_FakeTTNN.CoreCoord(x, y))
            if max_cores is not None and len(cores) >= max_cores:
                return cores[:max_cores]
        return cores

    @staticmethod
    def cb_descriptor_from_sharded_tensor(
        cb_index, tensor, total_size, core_ranges, address_offset=0
    ):
        return {
            "cb_index": cb_index,
            "tensor": tensor,
            "address_offset": address_offset,
            "total_size": total_size,
            "core_ranges": core_ranges,
        }

    @staticmethod
    def get_optimal_worker_cores_for_sharded_tensor(_tensor):
        return [_FakeTTNN.CoreCoord(0, 0), _FakeTTNN.CoreCoord(1, 0)]

    @staticmethod
    def generic_op(tensors, program):
        return {
            "tensors": tensors,
            "program": program,
        }

    def create_global_semaphore(self, device, core_ranges, initial_value):
        semaphore = {
            "device": device,
            "core_ranges": core_ranges,
            "initial_value": initial_value,
            "address": self.next_address,
        }
        self.next_address += 0x20
        self.create_calls.append(semaphore)
        return semaphore

    @staticmethod
    def get_global_semaphore_address(semaphore):
        return semaphore["address"]

    def synchronize_device(self, device):
        self.synchronize_calls.append(device)


class _LifetimeTrackedSemaphore:
    def __init__(self, identifier, address, events):
        self.identifier = identifier
        self.address = address
        self.events = events

    def __del__(self):
        self.events.append(("release", self.identifier))


class _LifetimeTrackingTTNN(_FakeTTNN):
    def __init__(self):
        super().__init__()
        self.events = []
        self.semaphore_refs = []

    def create_global_semaphore(self, device, core_ranges, initial_value):
        identifier = len(self.semaphore_refs)
        semaphore = _LifetimeTrackedSemaphore(
            identifier, self.next_address, self.events
        )
        self.next_address += 0x20
        self.events.append(("allocate", identifier))
        self.semaphore_refs.append(weakref.ref(semaphore))
        return semaphore

    @staticmethod
    def get_global_semaphore_address(semaphore):
        return semaphore.address

    def synchronize_device(self, device):
        self.events.append(("synchronize", device))


@pytest.mark.parametrize(
    "replace_device_wrapper", [False, True], ids=["same-wrapper", "same-device-id"]
)
def test_cached_l1_budget_excludes_only_owned_buffer_pages(
    monkeypatch, replace_device_wrapper
):
    cached_device = _FakeDevice(7)
    query_device = _FakeDevice(7) if replace_device_wrapper else cached_device
    l1_buffer_type = object()
    pages = [
        SimpleNamespace(
            address=0x1000,
            core_x=0,
            core_y=0,
            page_size=32,
            buffer_type=l1_buffer_type,
        ),
        SimpleNamespace(
            address=0x1100,
            core_x=0,
            core_y=0,
            page_size=4,
            buffer_type=l1_buffer_type,
        ),
        SimpleNamespace(
            address=0x2000,
            core_x=1,
            core_y=0,
            page_size=48,
            buffer_type=l1_buffer_type,
        ),
    ]
    reports = SimpleNamespace(
        get_device_info=lambda selected_device: SimpleNamespace(cb_limit=1024),
        get_buffer_pages=lambda selected_device: pages,
    )
    fake_ttnn = SimpleNamespace(
        BufferType=SimpleNamespace(L1=l1_buffer_type),
        _ttnn=SimpleNamespace(reports=reports),
        corerange_to_cores=lambda core_ranges, row_wise: [_FakeCoreCoord(0, 0)],
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    cache = kernel_runner.KernelRuntimeResourceCache(
        device=cached_device,
        owned_l1_buffer_addresses=frozenset((0x1000, 0x1100)),
    )

    remaining = kernel_runner.get_min_remaining_l1_excluding_cached_resources(
        cache, query_device
    )

    assert remaining == 976


def test_cached_resources_reuse_equivalent_device_wrapper(monkeypatch):
    first_device = _FakeDevice(7)
    equivalent_device = _FakeDevice(7)
    fake_ttnn = SimpleNamespace(
        corerange_to_cores=lambda core_ranges, row_wise: [_FakeCoreCoord(0, 0)]
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    allocations = []

    def build_resources(**arguments):
        allocations.append(arguments["device"])
        return kernel_runner.PipeRuntimeResources(
            scratch_tensors=[],
            global_semaphores=[],
            computed_address_dfb_tensors={},
            computed_address_base_addresses={},
            extra_common_runtime_args=[],
            expected_extra_common_runtime_args=0,
        )

    monkeypatch.setattr(kernel_runner, "build_pipe_runtime_resources", build_resources)
    cache = kernel_runner.KernelRuntimeResourceCache()
    arguments = {
        "cache": cache,
        "tensors": [],
        "cb_configs": [],
        "core_ranges": _FakeCoreRanges(),
        "pipe_sram_scratch_bytes": 16,
        "num_pipe_global_semaphores": 0,
        "pipe_computed_address_dfb_indices": (),
        "num_dfb_resets": 1,
    }

    first_resources = kernel_runner.get_cached_runtime_resources(
        device=first_device, **arguments
    )
    repeated_resources = kernel_runner.get_cached_runtime_resources(
        device=equivalent_device, **arguments
    )

    assert repeated_resources is first_resources
    assert allocations == [first_device]


@pytest.mark.parametrize("scratch_bytes", [16, 32])
def test_cached_scratch_budget_uses_reported_allocation_pages(
    monkeypatch, scratch_bytes
):
    device = object()
    l1_buffer_type = object()
    scratch_address = 0x3000
    reports = SimpleNamespace(
        get_device_info=lambda selected_device: SimpleNamespace(cb_limit=1024),
        get_buffer_pages=lambda selected_device: [
            SimpleNamespace(
                address=scratch_address,
                core_x=0,
                core_y=0,
                page_size=32,
                buffer_type=l1_buffer_type,
            )
        ],
    )
    fake_ttnn = SimpleNamespace(
        BufferType=SimpleNamespace(L1=l1_buffer_type),
        _ttnn=SimpleNamespace(reports=reports),
        corerange_to_cores=lambda core_ranges, row_wise: [_FakeCoreCoord(0, 0)],
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner,
        "build_pipe_runtime_resources",
        lambda **kwargs: kernel_runner.PipeRuntimeResources(
            scratch_tensors=[object()],
            global_semaphores=[],
            computed_address_dfb_tensors={},
            computed_address_base_addresses={},
            extra_common_runtime_args=[scratch_address],
            expected_extra_common_runtime_args=1,
            l1_buffer_addresses=frozenset((scratch_address,)),
        ),
    )
    cache = kernel_runner.KernelRuntimeResourceCache()
    kernel_runner.get_cached_runtime_resources(
        cache,
        tensors=[],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
        pipe_sram_scratch_bytes=scratch_bytes,
        num_pipe_global_semaphores=0,
        pipe_computed_address_dfb_indices=(),
        num_dfb_resets=1,
        device=device,
    )

    remaining = kernel_runner.get_min_remaining_l1_excluding_cached_resources(
        cache, device
    )

    assert remaining == 1024


def test_cached_global_semaphore_budget_uses_reported_allocation_pages(monkeypatch):
    device = object()
    l1_buffer_type = object()
    semaphore_address = 0x4000
    reports = SimpleNamespace(
        get_device_info=lambda selected_device: SimpleNamespace(cb_limit=1024),
        get_buffer_pages=lambda selected_device: [
            SimpleNamespace(
                address=semaphore_address,
                core_x=0,
                core_y=0,
                page_size=4,
                buffer_type=l1_buffer_type,
            )
        ],
    )
    fake_ttnn = SimpleNamespace(
        BufferType=SimpleNamespace(L1=l1_buffer_type),
        _ttnn=SimpleNamespace(reports=reports),
        corerange_to_cores=lambda core_ranges, row_wise: [_FakeCoreCoord(0, 0)],
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner,
        "build_pipe_runtime_resources",
        lambda **kwargs: kernel_runner.PipeRuntimeResources(
            scratch_tensors=[],
            global_semaphores=[object()],
            computed_address_dfb_tensors={},
            computed_address_base_addresses={},
            extra_common_runtime_args=[semaphore_address],
            expected_extra_common_runtime_args=1,
            l1_buffer_addresses=frozenset((semaphore_address,)),
        ),
    )
    cache = kernel_runner.KernelRuntimeResourceCache()
    kernel_runner.get_cached_runtime_resources(
        cache,
        tensors=[],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
        pipe_sram_scratch_bytes=0,
        num_pipe_global_semaphores=1,
        pipe_computed_address_dfb_indices=(),
        num_dfb_resets=0,
        device=device,
    )

    remaining = kernel_runner.get_min_remaining_l1_excluding_cached_resources(
        cache, device
    )

    assert remaining == 1024


def test_build_pipe_global_semaphores_empty_does_not_require_ttnn(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", None)

    semaphores, addresses = kernel_runner.build_pipe_global_semaphores(
        tensors=[],
        core_ranges=object(),
        count=0,
    )

    assert semaphores == []
    assert addresses == []


def test_runtime_resource_records_are_frozen_with_tuple_defaults():
    core_args = CoreRuntimeArgs(core=object(), values=(1, 2))
    define = KernelDefine(name="MODE", value="1")
    kernel_resources = KernelRuntimeResources(
        kernel=KernelKind.DATA_MOVEMENT,
        runtime_args=(core_args,),
        defines=(define,),
    )
    resources = ProgramRuntimeResources(kernel_resources=(kernel_resources,))

    assert TTLProgramRuntimeResources is ProgramRuntimeResources
    assert ProgramRuntimeResources().semaphore_descriptors == ()
    assert ProgramRuntimeResources().kernel_resources == ()
    assert ProgramRuntimeResources().lifetimes == ()
    with pytest.raises(FrozenInstanceError):
        resources.lifetimes = ()


def _kernel_spec(logical_kernel, core_ranges=None):
    return kernel_runner.KernelSpec(
        path="/tmp/kernel.cpp",
        thread_type="compute",
        tensor_indices=[],
        config=object(),
        core_ranges=core_ranges,
        logical_kernel=logical_kernel,
    )


def _plan_runtime_resources(
    resources,
    kernel_specs,
    core_ranges=None,
    first_free_id=0,
):
    return kernel_runner.plan_program_runtime_resources(
        operation_name="planned_operation",
        resources=resources,
        kernel_specs=kernel_specs,
        operation_core_ranges=core_ranges or _FakeCoreRanges((((0, 0), (1, 1)),)),
        first_free_semaphore_id=first_free_id,
    )


class _Indexable:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def test_plan_runtime_resources_normalizes_canonical_kernel_records():
    owner = object()
    resources = ProgramRuntimeResources(
        kernel_resources=(
            KernelRuntimeResources(
                kernel=KernelKind.COMPUTE,
                runtime_args=(
                    CoreRuntimeArgs(_FakeCoreCoord(1, 1), (_Indexable(5), -2)),
                    CoreRuntimeArgs(_FakeCoreCoord(0, 0), (7,)),
                ),
                defines=(KernelDefine("MODE", "test"),),
            ),
        ),
        lifetimes=(owner,),
    )

    plan = _plan_runtime_resources(resources, [_kernel_spec(KernelKind.COMPUTE)])

    assert plan.lifetimes == (owner,)
    assert plan.semaphore_descriptors == ()
    assert len(plan.kernel_descriptors) == 1
    descriptor_plan = plan.kernel_descriptors[0]
    assert descriptor_plan.logical_kernel == kernel_runner.LogicalKernelId(
        KernelKind.COMPUTE, None, None, None
    )
    assert descriptor_plan.coordinates == ((0, 0), (1, 0), (0, 1), (1, 1))
    assert [runtime_arg.coordinate for runtime_arg in descriptor_plan.runtime_args] == [
        (0, 0),
        (1, 1),
    ]
    assert [runtime_arg.values for runtime_arg in descriptor_plan.runtime_args] == [
        (7,),
        (5, -2),
    ]
    assert descriptor_plan.defines == (("MODE", "test"),)
    with pytest.raises(FrozenInstanceError):
        plan.lifetimes = ()


def test_plan_runtime_resources_resolves_explicit_kernel_identity():
    named_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    named_kernel._bind("named_kernel", "test.operation")
    resources = ProgramRuntimeResources(
        kernel_resources=(KernelRuntimeResources(kernel=named_kernel),)
    )

    plan = _plan_runtime_resources(resources, [_kernel_spec(named_kernel)])

    assert plan.kernel_descriptors[0].logical_kernel == kernel_runner.LogicalKernelId(
        KernelKind.DATA_MOVEMENT,
        "named_kernel",
        "test.operation",
        None,
    )


@pytest.mark.parametrize("kernel_kind", tuple(KernelKind))
def test_plan_runtime_resources_resolves_each_canonical_kernel_kind(kernel_kind):
    resources = ProgramRuntimeResources(
        kernel_resources=(KernelRuntimeResources(kernel=kernel_kind),)
    )

    plan = _plan_runtime_resources(resources, [_kernel_spec(kernel_kind)])

    assert plan.kernel_descriptors[0].logical_kernel == kernel_runner.LogicalKernelId(
        kernel_kind, None, None, None
    )


def test_plan_runtime_resources_targets_one_of_two_explicit_kernels():
    selected_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    selected_kernel._bind("selected", "test.operation")
    unselected_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    unselected_kernel._bind("unselected", "test.operation")
    resources = ProgramRuntimeResources(
        kernel_resources=(
            KernelRuntimeResources(
                kernel=selected_kernel,
                defines=(KernelDefine("SELECTED", "1"),),
            ),
        )
    )

    plan = _plan_runtime_resources(
        resources,
        [_kernel_spec(selected_kernel), _kernel_spec(unselected_kernel)],
    )

    assert plan.kernel_descriptors[0].defines == (("SELECTED", "1"),)
    assert plan.kernel_descriptors[1].defines == ()


def test_plan_runtime_resources_rejects_kernel_bound_to_another_operation():
    executing_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    executing_kernel._bind("named", "executing.operation")
    foreign_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    foreign_kernel._bind("named", "foreign.operation")
    resources = ProgramRuntimeResources(
        kernel_resources=(KernelRuntimeResources(kernel=foreign_kernel),)
    )

    with pytest.raises(ValueError) as exception_info:
        _plan_runtime_resources(resources, [_kernel_spec(executing_kernel)])
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': kernel resource 0 selects "
        "data_movement kernel 'named', but the operation emitted no matching "
        "kernel descriptor"
    )


def test_plan_runtime_resources_canonicalizes_range_order():
    first_ranges = _FakeCoreRanges(
        (
            ((1, 1), (1, 1)),
            ((0, 0), (1, 0)),
            ((0, 1), (0, 1)),
        )
    )
    second_ranges = _FakeCoreRanges(
        (
            ((0, 1), (1, 1)),
            ((0, 0), (1, 0)),
        )
    )

    first_plan = _plan_runtime_resources(
        ProgramRuntimeResources(),
        [_kernel_spec(KernelKind.COMPUTE)],
        first_ranges,
    )
    second_plan = _plan_runtime_resources(
        ProgramRuntimeResources(),
        [_kernel_spec(KernelKind.COMPUTE)],
        second_ranges,
    )

    assert first_plan.kernel_descriptors[0].coordinates == (
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    )
    assert (
        first_plan.kernel_descriptors[0].coordinates
        == second_plan.kernel_descriptors[0].coordinates
    )


@pytest.mark.parametrize(
    ("core_ranges", "message"),
    [
        (
            object(),
            (
                "@ttl.operation 'planned_operation': operation core_ranges "
                "must provide ranges()"
            ),
        ),
        (
            _MalformedCoreRanges(),
            (
                "@ttl.operation 'planned_operation': operation core_ranges "
                "range 0 must provide start and end coordinates"
            ),
        ),
        (
            _FakeCoreRanges((((1, 0), (0, 0)),)),
            (
                "@ttl.operation 'planned_operation': operation core_ranges "
                "range 0 has start (1, 0) after end (0, 0)"
            ),
        ),
    ],
)
def test_plan_runtime_resources_rejects_invalid_core_ranges(core_ranges, message):
    with pytest.raises((TypeError, ValueError)) as exception_info:
        _plan_runtime_resources(
            ProgramRuntimeResources(),
            [_kernel_spec(KernelKind.COMPUTE)],
            core_ranges,
        )
    assert str(exception_info.value) == message


@pytest.mark.parametrize(
    ("resources", "message"),
    [
        (
            ProgramRuntimeResources(kernel_resources=[]),
            "@ttl.operation 'planned_operation': kernel_resources must be a tuple, got list",
        ),
        (
            ProgramRuntimeResources(lifetimes=[]),
            "@ttl.operation 'planned_operation': lifetimes must be a tuple, got list",
        ),
        (
            ProgramRuntimeResources(semaphore_descriptors=[]),
            "@ttl.operation 'planned_operation': semaphore_descriptors must be a tuple, got list",
        ),
        (
            ProgramRuntimeResources(kernel_resources=(object(),)),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 must be "
                "KernelRuntimeResources, got object"
            ),
        ),
        (
            ProgramRuntimeResources(
                kernel_resources=(
                    KernelRuntimeResources(
                        kernel=KernelKind.COMPUTE,
                        runtime_args=[],
                    ),
                )
            ),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 "
                "runtime_args must be a tuple, got list"
            ),
        ),
        (
            ProgramRuntimeResources(
                kernel_resources=(
                    KernelRuntimeResources(
                        kernel=KernelKind.COMPUTE,
                        defines=[],
                    ),
                )
            ),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 defines "
                "must be a tuple, got list"
            ),
        ),
        (
            ProgramRuntimeResources(
                kernel_resources=(
                    KernelRuntimeResources(
                        kernel=KernelKind.COMPUTE,
                        runtime_args=(object(),),
                    ),
                )
            ),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 runtime "
                "argument 0 must be CoreRuntimeArgs, got object"
            ),
        ),
        (
            ProgramRuntimeResources(
                kernel_resources=(
                    KernelRuntimeResources(
                        kernel=KernelKind.COMPUTE,
                        runtime_args=(CoreRuntimeArgs(_FakeCoreCoord(0, 0), []),),
                    ),
                )
            ),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 runtime "
                "argument 0 values must be a tuple, got list"
            ),
        ),
        (
            ProgramRuntimeResources(semaphore_descriptors=(object(),)),
            (
                "@ttl.operation 'planned_operation': semaphore descriptor 0 "
                "must provide id"
            ),
        ),
    ],
)
def test_plan_runtime_resources_rejects_mutable_or_malformed_records(
    resources, message
):
    with pytest.raises((TypeError, ValueError)) as exception_info:
        _plan_runtime_resources(resources, [_kernel_spec(KernelKind.COMPUTE)])
    assert str(exception_info.value) == message


def test_plan_runtime_resources_rejects_wrong_top_level_type():
    with pytest.raises(TypeError) as exception_info:
        _plan_runtime_resources({}, [_kernel_spec(KernelKind.COMPUTE)])
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': runtime_resource_factory must "
        "return ProgramRuntimeResources, got dict"
    )


@pytest.mark.parametrize(
    ("value", "type_name"),
    [(True, "bool"), (object(), "object")],
)
def test_plan_runtime_resources_rejects_invalid_runtime_values(value, type_name):
    resources = ProgramRuntimeResources(
        kernel_resources=(
            KernelRuntimeResources(
                kernel=KernelKind.COMPUTE,
                runtime_args=(CoreRuntimeArgs(_FakeCoreCoord(0, 0), (value,)),),
            ),
        )
    )

    with pytest.raises(TypeError) as exception_info:
        _plan_runtime_resources(resources, [_kernel_spec(KernelKind.COMPUTE)])
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': kernel resource 0 runtime "
        f"argument 0 value 0 must be an integer, got {type_name}"
    )


@pytest.mark.parametrize(
    ("core", "message"),
    [
        (
            object(),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 runtime "
                "argument 0 core must provide integer x and y coordinates"
            ),
        ),
        (
            _FakeCoreCoord(-1, 0),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 runtime "
                "argument 0 core coordinate (-1, 0) must be nonnegative"
            ),
        ),
        (
            _FakeCoreCoord(True, 0),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 runtime "
                "argument 0 core.x must be an integer, got bool"
            ),
        ),
        (
            _FakeCoreCoord(2, 0),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 runtime "
                "argument 0 core (2, 0) is outside the operation core range"
            ),
        ),
    ],
)
def test_plan_runtime_resources_rejects_invalid_runtime_cores(core, message):
    resources = ProgramRuntimeResources(
        kernel_resources=(
            KernelRuntimeResources(
                kernel=KernelKind.COMPUTE,
                runtime_args=(CoreRuntimeArgs(core, (1,)),),
            ),
        )
    )

    with pytest.raises((TypeError, ValueError)) as exception_info:
        _plan_runtime_resources(
            resources,
            [_kernel_spec(KernelKind.COMPUTE)],
            _FakeCoreRanges((((0, 0), (1, 0)),)),
        )
    assert str(exception_info.value) == message


def test_plan_runtime_resources_rejects_duplicate_runtime_core():
    resources = ProgramRuntimeResources(
        kernel_resources=(
            KernelRuntimeResources(
                kernel=KernelKind.COMPUTE,
                runtime_args=(
                    CoreRuntimeArgs(_FakeCoreCoord(0, 0), (1,)),
                    CoreRuntimeArgs(_FakeCoreCoord(0, 0), (2,)),
                ),
            ),
        )
    )

    with pytest.raises(ValueError) as exception_info:
        _plan_runtime_resources(resources, [_kernel_spec(KernelKind.COMPUTE)])
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': kernel resource 0 specifies "
        "runtime arguments for core (0, 0) more than once"
    )


@pytest.mark.parametrize(
    ("defines", "message"),
    [
        (
            (object(),),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 define 0 "
                "must be a KernelDefine, got object"
            ),
        ),
        (
            (KernelDefine(1, "1"),),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 define 0 "
                "name must be a str, got int"
            ),
        ),
        (
            (KernelDefine("", "1"),),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 define 0 "
                "name must be nonempty and contain no NUL"
            ),
        ),
        (
            (KernelDefine("MODE", 1),),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 define 0 "
                "value must be a str, got int"
            ),
        ),
        (
            (KernelDefine("MODE", "1"), KernelDefine("MODE", "1")),
            (
                "@ttl.operation 'planned_operation': kernel resource 0 defines "
                "name 'MODE' more than once"
            ),
        ),
    ],
)
def test_plan_runtime_resources_rejects_invalid_defines(defines, message):
    resources = ProgramRuntimeResources(
        kernel_resources=(
            KernelRuntimeResources(
                kernel=KernelKind.COMPUTE,
                defines=defines,
            ),
        )
    )

    with pytest.raises((TypeError, ValueError)) as exception_info:
        _plan_runtime_resources(resources, [_kernel_spec(KernelKind.COMPUTE)])
    assert str(exception_info.value) == message


def test_plan_runtime_resources_rejects_unbound_kernel():
    unbound_kernel = Kernel(KernelKind.DATA_MOVEMENT)
    resources = ProgramRuntimeResources(
        kernel_resources=(KernelRuntimeResources(kernel=unbound_kernel),)
    )

    with pytest.raises(ValueError) as exception_info:
        _plan_runtime_resources(resources, [_kernel_spec(KernelKind.COMPUTE)])
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': kernel resource 0 uses an unbound Kernel"
    )


def test_plan_runtime_resources_rejects_physical_kernel_name():
    resources = ProgramRuntimeResources(
        kernel_resources=(KernelRuntimeResources(kernel="ncrisc"),)
    )

    with pytest.raises(TypeError) as exception_info:
        _plan_runtime_resources(resources, [_kernel_spec(KernelKind.COMPUTE)])
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': kernel resource 0 must select a "
        "KernelKind or Kernel, got str"
    )


def test_plan_runtime_resources_rejects_absent_logical_kernel():
    resources = ProgramRuntimeResources(
        kernel_resources=(KernelRuntimeResources(kernel=KernelKind.DATA_MOVEMENT),)
    )

    with pytest.raises(ValueError) as exception_info:
        _plan_runtime_resources(resources, [_kernel_spec(KernelKind.COMPUTE)])
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': kernel resource 0 selects canonical "
        "data_movement kernel, but the operation emitted no matching kernel descriptor"
    )


def test_plan_runtime_resources_rejects_duplicate_kernel_resource():
    resources = ProgramRuntimeResources(
        kernel_resources=(
            KernelRuntimeResources(kernel=KernelKind.COMPUTE),
            KernelRuntimeResources(kernel=KernelKind.COMPUTE),
        )
    )

    with pytest.raises(ValueError) as exception_info:
        _plan_runtime_resources(resources, [_kernel_spec(KernelKind.COMPUTE)])
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': runtime resources for canonical compute "
        "kernel were specified more than once"
    )


def test_plan_runtime_resources_rejects_negative_first_free_semaphore_id():
    with pytest.raises(ValueError) as exception_info:
        _plan_runtime_resources(
            ProgramRuntimeResources(),
            [_kernel_spec(KernelKind.COMPUTE)],
            first_free_id=-1,
        )
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': first_free_semaphore_id must be "
        "nonnegative, got -1"
    )


def test_plan_runtime_resources_partitions_specialized_runtime_args_and_defines():
    operation_ranges = _FakeCoreRanges((((0, 0), (1, 1)),))
    left_ranges = _FakeCoreRanges((((0, 0), (0, 1)),))
    right_ranges = _FakeCoreRanges((((1, 0), (1, 1)),))
    resources = ProgramRuntimeResources(
        kernel_resources=(
            KernelRuntimeResources(
                kernel=KernelKind.DATA_MOVEMENT,
                runtime_args=(
                    CoreRuntimeArgs(_FakeCoreCoord(1, 1), (11,)),
                    CoreRuntimeArgs(_FakeCoreCoord(0, 1), (10,)),
                    CoreRuntimeArgs(_FakeCoreCoord(1, 0), (9,)),
                    CoreRuntimeArgs(_FakeCoreCoord(0, 0), (8,)),
                ),
                defines=(KernelDefine("MODE", "specialized"),),
            ),
        )
    )

    plan = _plan_runtime_resources(
        resources,
        [
            _kernel_spec(KernelKind.DATA_MOVEMENT, left_ranges),
            _kernel_spec(KernelKind.DATA_MOVEMENT, right_ranges),
        ],
        operation_ranges,
    )

    assert [
        tuple(runtime_arg.coordinate for runtime_arg in descriptor.runtime_args)
        for descriptor in plan.kernel_descriptors
    ] == [((0, 0), (0, 1)), ((1, 0), (1, 1))]
    assert [
        tuple(runtime_arg.values for runtime_arg in descriptor.runtime_args)
        for descriptor in plan.kernel_descriptors
    ] == [((8,), (10,)), ((9,), (11,))]
    assert [descriptor.defines for descriptor in plan.kernel_descriptors] == [
        (("MODE", "specialized"),),
        (("MODE", "specialized"),),
    ]


def test_plan_runtime_resources_accepts_empty_specialized_record_set():
    operation_ranges = _FakeCoreRanges((((0, 0), (1, 0)),))

    plan = _plan_runtime_resources(
        ProgramRuntimeResources(
            kernel_resources=(KernelRuntimeResources(kernel=KernelKind.DATA_MOVEMENT),)
        ),
        [
            _kernel_spec(
                KernelKind.DATA_MOVEMENT,
                _FakeCoreRanges((((0, 0), (0, 0)),)),
            ),
            _kernel_spec(
                KernelKind.DATA_MOVEMENT,
                _FakeCoreRanges((((1, 0), (1, 0)),)),
            ),
        ],
        operation_ranges,
    )

    assert [descriptor.runtime_args for descriptor in plan.kernel_descriptors] == [
        (),
        (),
    ]


@pytest.mark.parametrize("include_runtime_arg", [False, True])
def test_plan_runtime_resources_rejects_overlapping_specialized_descriptors(
    include_runtime_arg,
):
    shared_ranges = _FakeCoreRanges((((0, 0), (1, 0)),))
    resources = ProgramRuntimeResources(
        kernel_resources=(
            KernelRuntimeResources(
                kernel=KernelKind.COMPUTE,
                runtime_args=(
                    (CoreRuntimeArgs(_FakeCoreCoord(1, 0), (7,)),)
                    if include_runtime_arg
                    else ()
                ),
            ),
        )
    )

    with pytest.raises(AssertionError) as exception_info:
        _plan_runtime_resources(
            resources,
            [
                _kernel_spec(KernelKind.COMPUTE, shared_ranges),
                _kernel_spec(
                    KernelKind.COMPUTE,
                    _FakeCoreRanges((((1, 0), (1, 0)),)),
                ),
            ],
            shared_ranges,
        )
    assert str(exception_info.value) == (
        "compiler emitted kernel descriptors 0 and 1 for canonical compute "
        "kernel with overlapping cores ((1, 0),)"
    )


def test_plan_runtime_resources_rejects_uncovered_specialized_runtime_arg():
    operation_ranges = _FakeCoreRanges((((0, 0), (1, 1)),))
    resources = ProgramRuntimeResources(
        kernel_resources=(
            KernelRuntimeResources(
                kernel=KernelKind.COMPUTE,
                runtime_args=(CoreRuntimeArgs(_FakeCoreCoord(0, 1), (7,)),),
            ),
        )
    )

    with pytest.raises(ValueError) as exception_info:
        _plan_runtime_resources(
            resources,
            [
                _kernel_spec(
                    KernelKind.COMPUTE,
                    _FakeCoreRanges((((0, 0), (0, 0)),)),
                ),
                _kernel_spec(
                    KernelKind.COMPUTE,
                    _FakeCoreRanges((((1, 0), (1, 0)),)),
                ),
            ],
            operation_ranges,
        )
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': kernel resource 0 runtime argument "
        "0 core (0, 1) is not covered by any descriptor for canonical compute "
        "kernel; descriptor ranges are ("
        "(0, ((0, 0),)), (1, ((1, 0),)))"
    )


def test_plan_runtime_resources_rejects_descriptor_outside_operation_range():
    with pytest.raises(ValueError) as exception_info:
        _plan_runtime_resources(
            ProgramRuntimeResources(),
            [
                _kernel_spec(
                    KernelKind.COMPUTE,
                    _FakeCoreRanges((((0, 0), (2, 0)),)),
                )
            ],
            _FakeCoreRanges((((0, 0), (1, 0)),)),
        )
    assert str(exception_info.value) == (
        "@ttl.operation 'planned_operation': kernel descriptor 0 has cores "
        "outside the operation range: ((2, 0),)"
    )


def test_plan_runtime_resources_validates_semaphores():
    core_ranges = _FakeCoreRanges((((0, 0), (1, 0)),))
    semaphore = _FakeTTNN.SemaphoreDescriptor(
        2,
        core_ranges=core_ranges,
        initial_value=_Indexable(3),
        core_type="WORKER",
    )
    resources = ProgramRuntimeResources(semaphore_descriptors=(semaphore,))

    plan = _plan_runtime_resources(
        resources,
        [_kernel_spec(KernelKind.COMPUTE)],
        core_ranges,
        first_free_id=2,
    )

    assert plan.semaphore_descriptors == (semaphore,)
    assert plan.semaphore_descriptors[0].initial_value.value == 3
    assert plan.semaphore_descriptors[0].core_type == "WORKER"


def test_plan_runtime_resources_accepts_semaphore_id_zero_and_disjoint_ranges():
    operation_ranges = _FakeCoreRanges((((0, 0), (1, 0)),))
    first_semaphore = _FakeTTNN.SemaphoreDescriptor(
        0,
        _FakeCoreRanges((((0, 0), (0, 0)),)),
        3,
        core_type="WORKER",
    )
    second_semaphore = _FakeTTNN.SemaphoreDescriptor(
        1,
        _FakeCoreRanges((((1, 0), (1, 0)),)),
        4,
        core_type="ETH",
    )

    plan = _plan_runtime_resources(
        ProgramRuntimeResources(
            semaphore_descriptors=(first_semaphore, second_semaphore)
        ),
        [_kernel_spec(KernelKind.COMPUTE)],
        operation_ranges,
        first_free_id=0,
    )

    assert plan.semaphore_descriptors == (first_semaphore, second_semaphore)
    assert [descriptor.initial_value for descriptor in plan.semaphore_descriptors] == [
        3,
        4,
    ]
    assert [descriptor.core_type for descriptor in plan.semaphore_descriptors] == [
        "WORKER",
        "ETH",
    ]


def _fingerprint_resource_plan(
    *,
    logical_kernel=KernelKind.COMPUTE,
    kernel_specs=None,
    define_name="MODE",
    define_value="base",
    runtime_core=(0, 0),
    runtime_values=(7,),
    semaphore_id=0,
    semaphore_ranges=None,
    semaphore_initial_value=0,
    semaphore_core_type="WORKER",
    lifetimes=(),
):
    operation_ranges = _FakeCoreRanges((((0, 0), (1, 0)),))
    if kernel_specs is None:
        kernel_specs = [_kernel_spec(logical_kernel)]
    if semaphore_ranges is None:
        semaphore_ranges = _FakeCoreRanges((((0, 0), (0, 0)),))
    resources = ProgramRuntimeResources(
        semaphore_descriptors=(
            _FakeTTNN.SemaphoreDescriptor(
                semaphore_id,
                semaphore_ranges,
                semaphore_initial_value,
                core_type=semaphore_core_type,
            ),
        ),
        kernel_resources=(
            KernelRuntimeResources(
                kernel=logical_kernel,
                runtime_args=(
                    CoreRuntimeArgs(_FakeCoreCoord(*runtime_core), runtime_values),
                ),
                defines=(KernelDefine(define_name, define_value),),
            ),
        ),
        lifetimes=lifetimes,
    )
    return _plan_runtime_resources(resources, kernel_specs, operation_ranges)


def test_runtime_resource_fingerprint_changes_for_every_structural_field():
    base_fingerprint = _fingerprint_resource_plan().structural_fingerprint
    named_kernel = Kernel._from_metadata(
        KernelKind.COMPUTE,
        "named_compute",
        "test.operation",
    )
    specialized_specs = [
        _kernel_spec(
            KernelKind.COMPUTE,
            _FakeCoreRanges((((0, 0), (0, 0)),)),
        ),
        _kernel_spec(
            KernelKind.COMPUTE,
            _FakeCoreRanges((((1, 0), (1, 0)),)),
        ),
    ]
    variants = (
        _fingerprint_resource_plan(
            logical_kernel=KernelKind.DATA_MOVEMENT
        ).structural_fingerprint,
        _fingerprint_resource_plan(
            logical_kernel=named_kernel,
        ).structural_fingerprint,
        _fingerprint_resource_plan(
            kernel_specs=specialized_specs,
        ).structural_fingerprint,
        _fingerprint_resource_plan(define_name="OTHER").structural_fingerprint,
        _fingerprint_resource_plan(define_value="changed").structural_fingerprint,
        _fingerprint_resource_plan(runtime_core=(1, 0)).structural_fingerprint,
        _fingerprint_resource_plan(runtime_values=(7, 8)).structural_fingerprint,
        _fingerprint_resource_plan(semaphore_id=1).structural_fingerprint,
        _fingerprint_resource_plan(
            semaphore_ranges=_FakeCoreRanges((((1, 0), (1, 0)),))
        ).structural_fingerprint,
        _fingerprint_resource_plan(semaphore_initial_value=1).structural_fingerprint,
        _fingerprint_resource_plan(semaphore_core_type="ETH").structural_fingerprint,
    )

    assert all(fingerprint != base_fingerprint for fingerprint in variants)


def test_runtime_resource_fingerprint_excludes_invocation_values_and_lifetimes():
    first_plan = _fingerprint_resource_plan(
        runtime_values=(7, 8),
        lifetimes=(object(),),
    )
    second_plan = _fingerprint_resource_plan(
        runtime_values=(17, 18),
        lifetimes=(object(), object()),
    )

    assert first_plan.structural_fingerprint == second_plan.structural_fingerprint


def test_runtime_resource_fingerprint_canonicalizes_range_order():
    forward_ranges = _FakeCoreRanges((((0, 0), (0, 0)), ((1, 0), (1, 0))))
    reverse_ranges = _FakeCoreRanges((((1, 0), (1, 0)), ((0, 0), (0, 0))))

    forward_plan = _fingerprint_resource_plan(
        kernel_specs=[_kernel_spec(KernelKind.COMPUTE, forward_ranges)]
    )
    reverse_plan = _fingerprint_resource_plan(
        kernel_specs=[_kernel_spec(KernelKind.COMPUTE, reverse_ranges)]
    )

    assert forward_plan.structural_fingerprint == reverse_plan.structural_fingerprint


def test_runtime_resource_fingerprint_orders_semaphores_by_id():
    core_ranges = _FakeCoreRanges((((0, 0), (1, 0)),))
    first_semaphore = _FakeTTNN.SemaphoreDescriptor(
        0,
        _FakeCoreRanges((((0, 0), (0, 0)),)),
        3,
    )
    second_semaphore = _FakeTTNN.SemaphoreDescriptor(
        1,
        _FakeCoreRanges((((1, 0), (1, 0)),)),
        4,
    )

    fingerprints = []
    for semaphores in (
        (first_semaphore, second_semaphore),
        (second_semaphore, first_semaphore),
    ):
        plan = _plan_runtime_resources(
            ProgramRuntimeResources(semaphore_descriptors=semaphores),
            [_kernel_spec(KernelKind.COMPUTE)],
            core_ranges,
        )
        fingerprints.append(plan.structural_fingerprint)

    assert fingerprints[0] == fingerprints[1]


def test_runtime_resource_fingerprint_is_stable_across_python_hash_seeds():
    script = textwrap.dedent(
        """
        from ttl import CoreRuntimeArgs, KernelDefine, KernelKind
        from ttl import KernelRuntimeResources, ProgramRuntimeResources
        from ttl import kernel_runner

        class Coordinate:
            def __init__(self, core_x, core_y):
                self.x = core_x
                self.y = core_y

        class CoreRange:
            def __init__(self, start, end):
                self.start = Coordinate(*start)
                self.end = Coordinate(*end)

        class CoreRanges:
            def ranges(self):
                return (CoreRange((0, 0), (0, 0)),)

        resources = ProgramRuntimeResources(kernel_resources=(
            KernelRuntimeResources(
                kernel=KernelKind.COMPUTE,
                runtime_args=(CoreRuntimeArgs(Coordinate(0, 0), (7, 8)),),
                defines=(KernelDefine("MODE", "stable"),),
            ),
        ))
        plan = kernel_runner.plan_program_runtime_resources(
            operation_name="seed_stability",
            resources=resources,
            kernel_specs=(kernel_runner.KernelSpec(
                path="/tmp/kernel.cpp",
                thread_type="compute",
                tensor_indices=[],
                config=object(),
                logical_kernel=KernelKind.COMPUTE,
            ),),
            operation_core_ranges=CoreRanges(),
            first_free_semaphore_id=0,
        )
        print(plan.structural_fingerprint)
        """
    )
    fingerprints = []
    for hash_seed in ("1", "937"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        fingerprints.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                env=environment,
                text=True,
            ).strip()
        )

    assert fingerprints[0] == fingerprints[1]


@pytest.mark.parametrize(
    ("semaphores", "first_free_id", "message"),
    [
        (
            (
                _FakeTTNN.SemaphoreDescriptor(
                    0,
                    _FakeCoreRanges(),
                    0,
                ),
            ),
            1,
            (
                "@ttl.operation 'planned_operation': semaphore descriptor 0 id 0 "
                "is below first free semaphore id 1"
            ),
        ),
        (
            (
                _FakeTTNN.SemaphoreDescriptor(1, _FakeCoreRanges(), 0),
                _FakeTTNN.SemaphoreDescriptor(1, _FakeCoreRanges(), 0),
            ),
            1,
            (
                "@ttl.operation 'planned_operation': semaphore id 1 was "
                "specified more than once"
            ),
        ),
        (
            (
                _FakeTTNN.SemaphoreDescriptor(
                    1,
                    _FakeCoreRanges((((2, 0), (2, 0)),)),
                    0,
                ),
            ),
            1,
            (
                "@ttl.operation 'planned_operation': semaphore descriptor 0 has "
                "cores outside the operation range: ((2, 0),)"
            ),
        ),
        (
            (_FakeTTNN.SemaphoreDescriptor(1, _FakeCoreRanges(), True),),
            1,
            (
                "@ttl.operation 'planned_operation': semaphore descriptor 0 "
                "initial_value must be an integer, got bool"
            ),
        ),
        (
            (
                _FakeTTNN.SemaphoreDescriptor(
                    1,
                    _FakeCoreRanges(()),
                    0,
                ),
            ),
            1,
            (
                "@ttl.operation 'planned_operation': semaphore descriptor 0 "
                "core_ranges must not be empty"
            ),
        ),
        (
            (
                _FakeTTNN.SemaphoreDescriptor(
                    1,
                    _FakeCoreRanges(),
                    0,
                    core_type=object(),
                ),
            ),
            1,
            (
                "@ttl.operation 'planned_operation': semaphore descriptor 0 "
                "core_type must be a named value, got object"
            ),
        ),
    ],
)
def test_plan_runtime_resources_rejects_invalid_semaphores(
    semaphores, first_free_id, message
):
    with pytest.raises((TypeError, ValueError)) as exception_info:
        _plan_runtime_resources(
            ProgramRuntimeResources(semaphore_descriptors=semaphores),
            [_kernel_spec(KernelKind.COMPUTE)],
            _FakeCoreRanges((((0, 0), (1, 0)),)),
            first_free_id=first_free_id,
        )
    assert str(exception_info.value) == message


def test_build_kernel_descriptors_materializes_planned_resources(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    core_ranges = _FakeCoreRanges((((0, 0), (1, 0)),))
    spec = _kernel_spec(KernelKind.COMPUTE)
    plan = _plan_runtime_resources(
        ProgramRuntimeResources(
            kernel_resources=(
                KernelRuntimeResources(
                    kernel=KernelKind.COMPUTE,
                    runtime_args=(CoreRuntimeArgs(_FakeCoreCoord(1, 0), (4, 5)),),
                    defines=(KernelDefine("MODE", "planned"),),
                ),
            )
        ),
        [spec],
        core_ranges,
    )

    descriptors = kernel_runner.build_kernel_descriptors(
        kernel_specs=[spec],
        tensors=[],
        tensor_accessor_args=[],
        core_ranges=core_ranges,
        grid_cols=2,
        grid_rows=1,
        num_cbs=0,
        descriptor_resource_plans=plan.kernel_descriptors,
    )

    assert descriptors[0].defines == [("MODE", "planned")]
    assert len(descriptors[0].runtime_args) == 1
    assert descriptors[0].runtime_args[1][0] == [4, 5]


def test_run_kernel_materializes_resources_and_synchronizes_lifetimes(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    core_ranges = _FakeCoreRanges((((0, 0), (1, 0)),))
    device = object()
    owner = object()

    def make_resources(*, tensors, core_ranges, first_free_semaphore_id):
        assert len(tensors) == 1
        assert first_free_semaphore_id == 1
        return ProgramRuntimeResources(
            semaphore_descriptors=(
                _FakeTTNN.SemaphoreDescriptor(
                    1,
                    core_ranges=core_ranges,
                    initial_value=0,
                ),
            ),
            kernel_resources=(
                KernelRuntimeResources(
                    kernel=KernelKind.COMPUTE,
                    runtime_args=(CoreRuntimeArgs(_FakeCoreCoord(1, 0), (8, 9)),),
                    defines=(KernelDefine("MODE", "runtime"),),
                ),
            ),
            lifetimes=(owner,),
        )

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[_kernel_spec(KernelKind.COMPUTE)],
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[],
        core_ranges=core_ranges,
        num_pipe_sync_semaphores=1,
        runtime_resource_factory=make_resources,
        operation_name="resource_execution",
        device=device,
    )

    program = result["program"]
    assert [semaphore.id for semaphore in program.semaphores] == [0, 1]
    assert program.kernels[0].defines == [("MODE", "runtime")]
    assert program.kernels[0].runtime_args[1][0] == [8, 9]
    assert fake_ttnn.synchronize_calls == [device]


def test_run_kernel_failure_preserves_runtime_resource_lifetimes(monkeypatch):
    fake_ttnn = _FakeTTNN()
    device = object()

    def fail_generic_op(_tensors, _program):
        raise RuntimeError("device execution failed")

    fake_ttnn.generic_op = fail_generic_op
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)

    with pytest.raises(RuntimeError, match="device execution failed"):
        kernel_runner.run_kernel_on_device(
            kernel_specs=[_kernel_spec(KernelKind.COMPUTE)],
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            runtime_resource_factory=lambda **_kwargs: ProgramRuntimeResources(
                lifetimes=(object(),)
            ),
            operation_name="failed_execution",
            device=device,
        )

    assert fake_ttnn.synchronize_calls == [device]


def test_run_kernel_plan_failure_preserves_runtime_resource_lifetimes(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())

    with pytest.raises(TypeError) as exception_info:
        kernel_runner.run_kernel_on_device(
            kernel_specs=[_kernel_spec(KernelKind.COMPUTE)],
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            runtime_resource_factory=lambda **_kwargs: ProgramRuntimeResources(
                kernel_resources=[]
            ),
            operation_name="failed_execution",
        )

    assert str(exception_info.value) == (
        "@ttl.operation 'failed_execution': kernel_resources must be a tuple, "
        "got list"
    )


def test_run_kernel_descriptor_failure_preserves_runtime_resource_lifetimes(
    monkeypatch,
):
    fake_ttnn = _FakeTTNN()

    def fail_kernel_descriptor(**_kwargs):
        raise RuntimeError("descriptor construction failed")

    fake_ttnn.KernelDescriptor = fail_kernel_descriptor
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)

    with pytest.raises(RuntimeError) as exception_info:
        kernel_runner.run_kernel_on_device(
            kernel_specs=[_kernel_spec(KernelKind.COMPUTE)],
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            runtime_resource_factory=lambda **_kwargs: ProgramRuntimeResources(
                lifetimes=(object(),)
            ),
            operation_name="failed_execution",
        )

    assert str(exception_info.value) == "descriptor construction failed"


def test_run_kernel_keeps_new_lifetimes_alive_through_execution(monkeypatch):
    class Owner:
        pass

    fake_ttnn = _FakeTTNN()
    owner_reference = []

    def make_resources(**_kwargs):
        owner = Owner()
        owner_reference.append(weakref.ref(owner))
        return ProgramRuntimeResources(lifetimes=(owner,))

    def verify_lifetime(_tensors, _program):
        assert owner_reference[0]() is not None
        return "executed"

    fake_ttnn.generic_op = verify_lifetime
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    device = object()

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[_kernel_spec(KernelKind.COMPUTE)],
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
        runtime_resource_factory=make_resources,
        operation_name="lifetime_execution",
        device=device,
    )

    assert result == "executed"
    assert fake_ttnn.synchronize_calls == [device]
    assert owner_reference[0]() is None


def test_run_kernel_cache_replaces_portable_lifetimes_after_synchronization(
    monkeypatch,
):
    events = []

    class LifetimeOwner:
        def __init__(self, identifier):
            self.identifier = identifier
            events.append(("allocate", identifier))

        def __del__(self):
            events.append(("release", self.identifier))

    fake_ttnn = _FakeTTNN()
    device = object()
    fake_ttnn.synchronize_device = lambda synchronized_device: events.append(
        ("synchronize", synchronized_device)
    )
    fake_ttnn.generic_op = lambda _tensors, _program: events.append(
        ("dispatch", device)
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    cache = kernel_runner.KernelRuntimeResourceCache()
    owner_references = []

    def make_resources(**_kwargs):
        owner = LifetimeOwner(len(owner_references))
        owner_references.append(weakref.ref(owner))
        return ProgramRuntimeResources(lifetimes=(owner,))

    for _invocation in range(2):
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            runtime_resource_factory=make_resources,
            runtime_resource_cache=cache,
            device=device,
        )

    first_synchronize = events.index(("synchronize", device))
    first_release = events.index(("release", 0))
    second_allocation = events.index(("allocate", 1))
    assert first_synchronize < first_release < second_allocation
    assert owner_references[0]() is None
    assert owner_references[1]() is cache.portable_resource_lifetimes[0]

    kernel_runner.release_cached_runtime_resources(cache)
    assert events[-2:] == [("synchronize", device), ("release", 1)]
    assert owner_references[1]() is None


def test_run_kernel_cache_discards_portable_lifetimes_after_dispatch_error(
    monkeypatch,
):
    class LifetimeOwner:
        pass

    fake_ttnn = _FakeTTNN()
    device = object()
    owner_reference = []

    def make_resources(**_kwargs):
        owner = LifetimeOwner()
        owner_reference.append(weakref.ref(owner))
        return ProgramRuntimeResources(lifetimes=(owner,))

    fake_ttnn.generic_op = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("dispatch failed")
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    cache = kernel_runner.KernelRuntimeResourceCache()

    with pytest.raises(RuntimeError, match="dispatch failed"):
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            runtime_resource_factory=make_resources,
            runtime_resource_cache=cache,
            device=device,
        )

    assert owner_reference[0]() is None
    assert cache.portable_resource_lifetimes == ()
    assert fake_ttnn.synchronize_calls == [device]
    kernel_runner.release_cached_runtime_resources(cache)
    assert fake_ttnn.synchronize_calls == [device]
    assert owner_reference[0]() is None


def test_run_kernel_checks_compiler_semaphore_ids_before_factory(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    invalid_descriptor = _FakeTTNN.SemaphoreDescriptor(
        1,
        _FakeCoreRanges(),
        0,
    )
    monkeypatch.setattr(
        kernel_runner,
        "build_pipe_sync_semaphore_descriptors",
        lambda **_kwargs: [invalid_descriptor],
    )
    factory_calls = []

    with pytest.raises(RuntimeError) as exception_info:
        kernel_runner.run_kernel_on_device(
            kernel_specs=[_kernel_spec(KernelKind.COMPUTE)],
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            num_pipe_sync_semaphores=1,
            runtime_resource_factory=lambda **_kwargs: factory_calls.append(True),
            operation_name="invalid_compiler_semaphores",
        )

    assert str(exception_info.value) == (
        "compiler-managed semaphore descriptors must use the dense ID range [0, 1)"
    )
    assert factory_calls == []


def test_build_pipe_global_semaphores_uses_explicit_device(monkeypatch):
    fake_ttnn = _FakeTTNN()
    explicit_device = object()
    core_ranges = object()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)

    semaphores, addresses = kernel_runner.build_pipe_global_semaphores(
        tensors=[],
        core_ranges=core_ranges,
        count=2,
        device=explicit_device,
    )

    assert semaphores == fake_ttnn.create_calls
    assert addresses == [0x1000, 0x1020]
    assert [call["device"] for call in fake_ttnn.create_calls] == [
        explicit_device,
        explicit_device,
    ]
    assert [call["core_ranges"] for call in fake_ttnn.create_calls] == [
        core_ranges,
        core_ranges,
    ]
    assert [call["initial_value"] for call in fake_ttnn.create_calls] == [0, 0]


def test_build_pipe_global_semaphores_uses_first_tensor_device(monkeypatch):
    fake_ttnn = _FakeTTNN()
    tensor_device = object()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)

    _semaphores, addresses = kernel_runner.build_pipe_global_semaphores(
        tensors=[None, _FakeTensor(tensor_device)],
        core_ranges=object(),
        count=1,
    )

    assert addresses == [0x1000]
    assert fake_ttnn.create_calls[0]["device"] is tensor_device


def test_build_pipe_global_semaphores_requires_device(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())

    with pytest.raises(ValueError, match="requires a device tensor"):
        kernel_runner.build_pipe_global_semaphores(
            tensors=[],
            core_ranges=object(),
            count=1,
        )


def test_build_pipe_runtime_resources_appends_global_semaphore_args(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    tensor = _FakeTensor(object())

    resources = kernel_runner.build_pipe_runtime_resources(
        tensors=[tensor],
        core_ranges=object(),
        num_pipe_global_semaphores=2,
    )

    assert resources.scratch_tensors == []
    assert resources.global_semaphores == fake_ttnn.create_calls
    assert resources.extra_common_runtime_args == [0x1000, 0x1020]
    assert resources.expected_extra_common_runtime_args == 2


def test_build_pipe_runtime_resources_zero_initializes_reset_state(monkeypatch):
    observed_allocations = []

    def allocate_scratch(core_ranges, num_bytes, device, *, zero_initialize=False):
        observed_allocations.append((core_ranges, num_bytes, device, zero_initialize))
        return _FakeTensor(device, address=0x4000)

    monkeypatch.setattr(kernel_runner, "ttnn", object())
    monkeypatch.setattr(
        kernel_runner, "_allocate_l1_sharded_storage_tensor", allocate_scratch
    )
    core_ranges = object()
    device = object()

    resources = kernel_runner.build_pipe_runtime_resources(
        tensors=[],
        core_ranges=core_ranges,
        pipe_sram_scratch_bytes=16,
        device=device,
        initialize_sram_scratch=True,
    )

    assert len(resources.scratch_tensors) == 1
    assert resources.extra_common_runtime_args == [0x4000]
    assert observed_allocations == [(core_ranges, 16, device, True)]


def test_build_kernel_descriptors_checks_pipe_runtime_arg_count(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    spec = kernel_runner.KernelSpec(
        path="/tmp/kernel.cpp",
        thread_type="noc",
        tensor_indices=[0],
        config=object(),
    )
    tensor = _FakeTensor(object(), address=0x2000)

    descriptors = kernel_runner.build_kernel_descriptors(
        kernel_specs=[spec],
        tensors=[tensor],
        tensor_accessor_args=[],
        core_ranges=object(),
        grid_cols=1,
        grid_rows=1,
        num_cbs=0,
        extra_common_runtime_args=[0x3000, 0x3020],
        expected_extra_common_runtime_args=2,
    )

    assert descriptors[0].common_runtime_args == [0x2000, 0x3000, 0x3020]
    with pytest.raises(
        RuntimeError,
        match="pipe resource plan expected 2 extra common runtime args, got 1",
    ):
        kernel_runner.build_kernel_descriptors(
            kernel_specs=[spec],
            tensors=[tensor],
            tensor_accessor_args=[],
            core_ranges=object(),
            grid_cols=1,
            grid_rows=1,
            num_cbs=0,
            extra_common_runtime_args=[0x3000],
            expected_extra_common_runtime_args=2,
        )


def test_build_kernel_descriptors_passes_computed_addresses_as_runtime_args(
    monkeypatch,
):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    spec = kernel_runner.KernelSpec(
        path="/tmp/kernel.cpp",
        thread_type="noc",
        tensor_indices=[0],
        config=object(),
        pipe_computed_address_dfb_indices=[1, 3],
    )
    tensor = _FakeTensor(object(), address=0x2000)

    descriptors = kernel_runner.build_kernel_descriptors(
        kernel_specs=[spec],
        tensors=[tensor],
        tensor_accessor_args=[0x44, 0x55],
        core_ranges=object(),
        grid_cols=1,
        grid_rows=1,
        num_cbs=2,
        pipe_computed_address_base_addresses={1: 0x8000, 3: 0x9000},
        extra_common_runtime_args=[0xA000],
        expected_extra_common_runtime_args=1,
    )

    dfb_indices = [0, 1]
    pipe_dfb_bases = [0x8000, 0x9000]
    tensor_accessor_args = [0x44, 0x55]
    assert descriptors[0].compile_time_args == dfb_indices + tensor_accessor_args
    assert descriptors[0].common_runtime_args == [0x2000] + pipe_dfb_bases + [0xA000]


def test_run_kernel_without_pipe_resources_does_not_require_device(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    tensor = _FakeTensorWithoutDevice()

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[],
        tensors=[tensor],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
    )

    assert result["tensors"] == [tensor, tensor]
    assert result["program"].kernels == []
    assert result["program"].cbs == []
    assert result["program"].semaphores == []


def test_run_kernel_invokes_empty_runtime_resource_factory(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    tensor = _FakeTensorWithoutDevice()
    observed = {}

    def make_resources(*, tensors, core_ranges, first_free_semaphore_id):
        observed["tensors"] = tensors
        observed["core_ranges"] = core_ranges
        observed["first_free_semaphore_id"] = first_free_semaphore_id
        return ProgramRuntimeResources()

    core_ranges = _FakeCoreRanges()
    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[],
        tensors=[tensor],
        cb_configs=[],
        core_ranges=core_ranges,
        num_pipe_sync_semaphores=2,
        runtime_resource_factory=make_resources,
        operation_name="empty_resources",
    )

    assert observed == {
        "tensors": (tensor,),
        "core_ranges": core_ranges,
        "first_free_semaphore_id": 2,
    }
    assert [descriptor.id for descriptor in result["program"].semaphores] == [0, 1]


def test_run_kernel_rejects_wrong_runtime_resource_factory_result(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())

    with pytest.raises(
        TypeError,
        match=(
            "@ttl.operation 'wrong_result': runtime_resource_factory must return "
            "ProgramRuntimeResources, got dict"
        ),
    ):
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            runtime_resource_factory=lambda **_kwargs: {},
            operation_name="wrong_result",
        )


def test_run_kernel_contextualizes_runtime_resource_factory_failure(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    factory_error = ValueError("factory detail")

    def fail_factory(**_kwargs):
        raise factory_error

    with pytest.raises(
        RuntimeError,
        match=(
            "@ttl.operation 'factory_failure': runtime resource factory failed: "
            "factory detail"
        ),
    ) as exception_info:
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            runtime_resource_factory=fail_factory,
            operation_name="factory_failure",
        )

    assert exception_info.value.__cause__ is factory_error


def test_run_kernel_sets_custom_program_hash(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    tensor = _FakeTensorWithoutDevice()

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[],
        tensors=[tensor],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
        program_hash=-1,
    )

    assert result["program"].custom_program_hash == (1 << 64) - 1


def test_run_kernel_leaves_custom_program_hash_unset_by_default(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    tensor = _FakeTensorWithoutDevice()

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[],
        tensors=[tensor],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
    )

    assert result["program"].custom_program_hash is None


def test_run_kernel_passes_through_in_range_program_hash(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    tensor = _FakeTensorWithoutDevice()

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[],
        tensors=[tensor],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
        program_hash=5,
    )

    assert result["program"].custom_program_hash == 5


def test_run_kernel_combines_program_hash_with_empty_resource_contract(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    core_ranges = _FakeCoreRanges()
    empty_plan = _plan_runtime_resources(
        ProgramRuntimeResources(),
        [],
        core_ranges,
    )

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[],
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[],
        core_ranges=core_ranges,
        program_hash=5,
        runtime_resource_factory=lambda **_kwargs: ProgramRuntimeResources(),
        operation_name="empty_resource_contract",
    )

    assert result["program"].custom_program_hash == (
        kernel_runner.combine_program_hash_with_runtime_resources(
            5,
            empty_plan.structural_fingerprint,
        )
    )
    assert result["program"].custom_program_hash != 5


def test_run_kernel_leaves_resource_custom_hash_unset_without_compiler_hash(
    monkeypatch,
):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[_kernel_spec(KernelKind.COMPUTE)],
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
        runtime_resource_factory=lambda **_kwargs: ProgramRuntimeResources(
            kernel_resources=(
                KernelRuntimeResources(
                    kernel=KernelKind.COMPUTE,
                    defines=(KernelDefine("MODE", "resource"),),
                ),
            )
        ),
        operation_name="resource_without_compiler_hash",
    )

    assert result["program"].custom_program_hash is None


def test_run_kernel_reuses_structural_hash_while_updating_invocation_values(
    monkeypatch,
):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    next_values = iter((7, 19))
    device = object()

    def make_resources(**_kwargs):
        return ProgramRuntimeResources(
            kernel_resources=(
                KernelRuntimeResources(
                    kernel=KernelKind.COMPUTE,
                    runtime_args=(
                        CoreRuntimeArgs(
                            _FakeCoreCoord(0, 0),
                            (next(next_values),),
                        ),
                    ),
                ),
            ),
            lifetimes=(object(),),
        )

    programs = []
    for _call_index in range(2):
        result = kernel_runner.run_kernel_on_device(
            kernel_specs=[_kernel_spec(KernelKind.COMPUTE)],
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            program_hash=23,
            runtime_resource_factory=make_resources,
            operation_name="repeated_resources",
            device=device,
        )
        programs.append(result["program"])

    assert programs[0].custom_program_hash == programs[1].custom_program_hash
    assert programs[0].kernels[0].runtime_args[0][0] == [7]
    assert programs[1].kernels[0].runtime_args[0][0] == [19]
    assert kernel_runner.ttnn.synchronize_calls == [device, device]


def test_build_generic_op_io_tensors_duplicates_single_output():
    tensor = _FakeTensorWithoutDevice()

    assert kernel_runner.build_generic_op_io_tensors([tensor], []) == [
        tensor,
        tensor,
    ]


def test_build_generic_op_io_tensors_keeps_user_output_last():
    inp = object()
    output = object()
    scratch = object()
    computed_dfb_1 = object()
    computed_dfb_3 = object()

    io_tensors = kernel_runner.build_generic_op_io_tensors(
        [inp, output],
        [scratch],
        {3: computed_dfb_3, 1: computed_dfb_1},
    )

    assert io_tensors == [scratch, computed_dfb_1, computed_dfb_3, inp, output]
    assert io_tensors[-1] is output


def test_build_generic_op_io_tensors_requires_user_output():
    with pytest.raises(ValueError, match="kernel must have at least one output tensor"):
        kernel_runner.build_generic_op_io_tensors([], [object()])


def test_run_kernel_replaces_global_semaphores_between_invocations(monkeypatch):
    fake_ttnn = _LifetimeTrackingTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    tensor = _FakeTensor(object())
    core_ranges = _FakeCoreRanges()
    cache = kernel_runner.KernelRuntimeResourceCache()

    for _ in range(2):
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[tensor],
            cb_configs=[],
            core_ranges=core_ranges,
            num_pipe_global_semaphores=2,
            runtime_resource_cache=cache,
        )

    assert fake_ttnn.events == [
        ("allocate", 0),
        ("allocate", 1),
        ("synchronize", tensor.device()),
        ("release", 1),
        ("release", 0),
        ("allocate", 2),
        ("allocate", 3),
    ]
    assert [
        semaphore.identifier for semaphore in cache.pipe_resources.global_semaphores
    ] == [2, 3]


def test_run_kernel_allows_concurrent_resource_free_invocations(monkeypatch):
    cache = kernel_runner.KernelRuntimeResourceCache()
    invocation_lock = threading.Lock()
    first_invocation_entered = threading.Event()
    second_invocation_entered = threading.Event()
    release_invocations = threading.Event()
    invocation_count = 0
    invocation_errors = []

    def run_impl(**kwargs):
        nonlocal invocation_count
        assert kwargs["runtime_resource_cache"] is None
        with invocation_lock:
            invocation_count += 1
            current_invocation = invocation_count
        if current_invocation == 1:
            first_invocation_entered.set()
        else:
            second_invocation_entered.set()
        if not release_invocations.wait(timeout=2):
            raise TimeoutError("resource-free invocations did not overlap")

    def invoke():
        try:
            kernel_runner.run_kernel_on_device(
                kernel_specs=[],
                tensors=[],
                cb_configs=[],
                core_ranges=None,
                runtime_resource_cache=cache,
            )
        except BaseException as error:
            invocation_errors.append(error)

    monkeypatch.setattr(kernel_runner, "_run_kernel_on_device_impl", run_impl)
    first_thread = threading.Thread(target=invoke)
    second_thread = threading.Thread(target=invoke)
    first_thread.start()
    assert first_invocation_entered.wait(timeout=1)
    second_thread.start()
    assert second_invocation_entered.wait(timeout=1)
    release_invocations.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert invocation_errors == []


def test_run_kernel_serializes_resource_owning_invocations(monkeypatch):
    cache = kernel_runner.KernelRuntimeResourceCache()
    invocation_lock = threading.Lock()
    first_invocation_entered = threading.Event()
    second_invocation_entered = threading.Event()
    release_first_invocation = threading.Event()
    invocation_count = 0
    invocation_errors = []

    def run_impl(**kwargs):
        nonlocal invocation_count
        assert kwargs["runtime_resource_cache"] is cache
        with invocation_lock:
            invocation_count += 1
            current_invocation = invocation_count
        if current_invocation == 1:
            first_invocation_entered.set()
            if not release_first_invocation.wait(timeout=2):
                raise TimeoutError("first resource-owning invocation was not released")
        else:
            second_invocation_entered.set()

    def invoke():
        try:
            kernel_runner.run_kernel_on_device(
                kernel_specs=[],
                tensors=[],
                cb_configs=[],
                core_ranges=None,
                pipe_sram_scratch_bytes=32,
                runtime_resource_cache=cache,
            )
        except BaseException as error:
            invocation_errors.append(error)

    monkeypatch.setattr(kernel_runner, "_run_kernel_on_device_impl", run_impl)
    first_thread = threading.Thread(target=invoke)
    second_thread = threading.Thread(target=invoke)
    first_thread.start()
    assert first_invocation_entered.wait(timeout=1)
    second_thread.start()
    assert not second_invocation_entered.wait(timeout=0.1)
    release_first_invocation.set()
    assert second_invocation_entered.wait(timeout=1)
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert invocation_errors == []


def test_run_kernel_releases_cached_resources_before_resource_free_invocation(
    monkeypatch,
):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    device = object()
    cache = kernel_runner.KernelRuntimeResourceCache(
        compatibility_key=("resource-owning",),
        device=device,
        pipe_resources=kernel_runner.PipeRuntimeResources(
            scratch_tensors=[object()],
            global_semaphores=[],
            computed_address_dfb_tensors={},
            computed_address_base_addresses={},
            extra_common_runtime_args=[],
            expected_extra_common_runtime_args=0,
        ),
    )
    dispatch_events = []

    def run_impl(**kwargs):
        assert kwargs["runtime_resource_cache"] is None
        dispatch_events.append("dispatch")

    monkeypatch.setattr(kernel_runner, "_run_kernel_on_device_impl", run_impl)
    kernel_runner.run_kernel_on_device(
        kernel_specs=[],
        tensors=[],
        cb_configs=[],
        core_ranges=None,
        runtime_resource_cache=cache,
    )

    assert fake_ttnn.synchronize_calls == [device]
    assert dispatch_events == ["dispatch"]
    assert cache.compatibility_key is None
    assert cache.pipe_resources is None


def test_cached_dispatch_failure_discards_reset_state(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    build_initialization = []
    scratch_generations = []

    def build_resources(**kwargs):
        build_initialization.append(kwargs["initialize_sram_scratch"])
        scratch = object()
        scratch_generations.append(scratch)
        return kernel_runner.PipeRuntimeResources(
            scratch_tensors=[scratch],
            global_semaphores=[],
            computed_address_dfb_tensors={},
            computed_address_base_addresses={},
            extra_common_runtime_args=[0x1000],
            expected_extra_common_runtime_args=1,
        )

    dispatch_count = 0

    def dispatch(_tensors, program):
        nonlocal dispatch_count
        dispatch_count += 1
        if dispatch_count == 1:
            raise RuntimeError("dispatch failed")
        return program

    monkeypatch.setattr(kernel_runner, "build_pipe_runtime_resources", build_resources)
    fake_ttnn.generic_op = dispatch
    device = object()
    cache = kernel_runner.KernelRuntimeResourceCache()
    arguments = {
        "kernel_specs": [],
        "tensors": [_FakeTensor(device)],
        "cb_configs": [],
        "core_ranges": _FakeCoreRanges(),
        "pipe_sram_scratch_bytes": 32,
        "num_dfb_resets": 1,
        "runtime_resource_cache": cache,
    }

    with pytest.raises(RuntimeError, match="dispatch failed"):
        kernel_runner.run_kernel_on_device(**arguments)
    assert cache.pipe_resources is None
    kernel_runner.run_kernel_on_device(**arguments)

    assert build_initialization == [True, True]
    assert cache.pipe_resources.scratch_tensors[0] is scratch_generations[1]
    assert fake_ttnn.synchronize_calls == [device]


def test_cached_dispatch_failure_retains_state_when_sync_fails(monkeypatch):
    fake_ttnn = _LifetimeTrackingTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    retained_caches = []
    monkeypatch.setattr(
        kernel_runner, "_RETAINED_RUNTIME_RESOURCE_CACHES", retained_caches
    )
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    device = object()

    def dispatch(_tensors, _program):
        raise RuntimeError("dispatch failed")

    def fail_synchronization(_device):
        raise ValueError("synchronization failed")

    fake_ttnn.generic_op = dispatch
    fake_ttnn.synchronize_device = fail_synchronization
    cache = kernel_runner.KernelRuntimeResourceCache()
    with pytest.raises(RuntimeError, match="dispatch failed") as error:
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[_FakeTensor(device)],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            num_pipe_global_semaphores=1,
            runtime_resource_cache=cache,
        )

    assert "device synchronization also failed" in str(error.value.__notes__)
    assert cache.pipe_resources is None
    assert len(retained_caches) == 1
    assert retained_caches[0].pipe_resources.global_semaphores[0] is not None


def test_cached_pipe_resources_distinguish_reset_initialization(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    build_calls = []

    def build_resources(**kwargs):
        build_calls.append(kwargs["initialize_sram_scratch"])
        return kernel_runner.PipeRuntimeResources(
            scratch_tensors=[object()],
            global_semaphores=[],
            computed_address_dfb_tensors={},
            computed_address_base_addresses={},
            extra_common_runtime_args=[0x1000],
            expected_extra_common_runtime_args=1,
        )

    monkeypatch.setattr(kernel_runner, "build_pipe_runtime_resources", build_resources)
    cache = kernel_runner.KernelRuntimeResourceCache()
    device = object()
    arguments = {
        "cache": cache,
        "tensors": [],
        "cb_configs": [],
        "core_ranges": _FakeCoreRanges(),
        "pipe_sram_scratch_bytes": 16,
        "num_pipe_global_semaphores": 0,
        "pipe_computed_address_dfb_indices": (),
        "device": device,
    }

    first_without_reset = kernel_runner.get_cached_runtime_resources(
        num_dfb_resets=0, **arguments
    )
    repeated_without_reset = kernel_runner.get_cached_runtime_resources(
        num_dfb_resets=0, **arguments
    )
    first_with_reset = kernel_runner.get_cached_runtime_resources(
        num_dfb_resets=1, **arguments
    )
    repeated_with_reset = kernel_runner.get_cached_runtime_resources(
        num_dfb_resets=1, **arguments
    )

    assert first_without_reset is repeated_without_reset
    assert first_with_reset is repeated_with_reset
    assert first_with_reset is not first_without_reset
    assert build_calls == [False, True]
    assert fake_ttnn.synchronize_calls == [device]


def test_run_kernel_synchronizes_uncached_runtime_resources(monkeypatch):
    fake_ttnn = _LifetimeTrackingTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    device = object()

    kernel_runner.run_kernel_on_device(
        kernel_specs=[],
        tensors=[_FakeTensor(device)],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
        num_pipe_global_semaphores=1,
    )

    assert fake_ttnn.semaphore_refs[0]() is None
    synchronize_index = fake_ttnn.events.index(("synchronize", device))
    release_index = fake_ttnn.events.index(("release", 0))
    assert synchronize_index < release_index


def test_run_kernel_retains_uncached_resources_when_cleanup_cannot_synchronize(
    monkeypatch,
):
    fake_ttnn = _LifetimeTrackingTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    retained_caches = []
    monkeypatch.setattr(
        kernel_runner, "_RETAINED_RUNTIME_RESOURCE_CACHES", retained_caches
    )
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    device = object()

    def failing_generic_op(_tensors, _program):
        fake_ttnn.events.append(("dispatch", device))
        raise RuntimeError("dispatch failed")

    def failing_synchronize(_device):
        fake_ttnn.events.append(("synchronize", device))
        raise ValueError("synchronization failed")

    fake_ttnn.generic_op = failing_generic_op
    fake_ttnn.synchronize_device = failing_synchronize
    with pytest.raises(RuntimeError, match="dispatch failed") as error:
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[_FakeTensor(device)],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            num_pipe_global_semaphores=1,
        )

    dispatch_index = fake_ttnn.events.index(("dispatch", device))
    synchronize_index = fake_ttnn.events.index(("synchronize", device))
    assert dispatch_index < synchronize_index
    assert "device synchronization also failed: synchronization failed" in str(
        error.value.__notes__
    )
    assert fake_ttnn.semaphore_refs[0]() is not None
    assert len(retained_caches) == 1
    error.value.__traceback__ = None
    del error

    fake_ttnn.synchronize_device = lambda cleanup_device: fake_ttnn.events.append(
        ("synchronize", cleanup_device)
    )
    kernel_runner.release_cached_runtime_resources(retained_caches.pop())
    assert fake_ttnn.semaphore_refs[0]() is None


def test_run_kernel_retains_uncached_resources_after_synchronization_error(
    monkeypatch,
):
    fake_ttnn = _LifetimeTrackingTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    retained_caches = []
    monkeypatch.setattr(
        kernel_runner, "_RETAINED_RUNTIME_RESOURCE_CACHES", retained_caches
    )
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    device = object()

    def successful_generic_op(_tensors, program):
        fake_ttnn.events.append(("dispatch", device))
        return program

    def failing_synchronize(_device):
        fake_ttnn.events.append(("synchronize", device))
        raise ValueError("synchronization failed")

    fake_ttnn.generic_op = successful_generic_op
    fake_ttnn.synchronize_device = failing_synchronize
    with pytest.raises(ValueError, match="synchronization failed"):
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[_FakeTensor(device)],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            num_pipe_global_semaphores=1,
        )

    dispatch_index = next(
        index for index, event in enumerate(fake_ttnn.events) if event[0] == "dispatch"
    )
    synchronize_index = fake_ttnn.events.index(("synchronize", device))
    assert dispatch_index < synchronize_index
    assert fake_ttnn.semaphore_refs[0]() is not None
    assert len(retained_caches) == 1

    fake_ttnn.synchronize_device = lambda cleanup_device: fake_ttnn.events.append(
        ("synchronize", cleanup_device)
    )
    kernel_runner.release_cached_runtime_resources(retained_caches.pop())
    assert fake_ttnn.semaphore_refs[0]() is None


def test_run_kernel_synchronizes_before_replacing_resource_variants(monkeypatch):
    fake_ttnn = _LifetimeTrackingTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    first_device = object()
    second_device = object()
    cache = kernel_runner.KernelRuntimeResourceCache()

    for device in (first_device, second_device, first_device):
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[_FakeTensor(device)],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            num_pipe_global_semaphores=1,
            runtime_resource_cache=cache,
        )

    synchronize_index = fake_ttnn.events.index(("synchronize", first_device))
    release_index = fake_ttnn.events.index(("release", 0))
    replacement_index = fake_ttnn.events.index(("allocate", 1))
    assert synchronize_index < release_index < replacement_index
    second_synchronize_index = fake_ttnn.events.index(("synchronize", second_device))
    second_release_index = fake_ttnn.events.index(("release", 1))
    restored_variant_index = fake_ttnn.events.index(("allocate", 2))
    assert second_synchronize_index < second_release_index < restored_variant_index


def test_build_cb_descriptors_excludes_computed_address_backing_tensors(
    monkeypatch,
):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 1024
    )

    cb_configs = [
        PhysicalDFBConfig(0, 1, "bfloat16", 1, 512, None),
        PhysicalDFBConfig(1, 1, "bfloat16", 1, 800, None),
    ]

    # DFB 1 (800 bytes) is a computed-address backing tensor, already allocated
    # separately, so it is excluded from the budget; only DFB 0 (512) counts and
    # stays under 1024.
    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensor(object())],
        cb_configs=cb_configs,
        core_ranges=_FakeCoreRanges(),
        pipe_computed_address_backing_tensors={1: object()},
    )
    assert len(descriptors) == 2

    # Without the backing exclusion the same DFBs (512 + 800) exceed 1024, so
    # non-backing DFBs are still charged.
    with pytest.raises(
        ValueError,
        match="Total circular buffer allocation \\(1312 bytes\\) exceeds L1 budget \\(1024 bytes\\)",
    ):
        kernel_runner.build_cb_descriptors(
            tensors=[_FakeTensor(object())],
            cb_configs=cb_configs,
            core_ranges=_FakeCoreRanges(),
            pipe_computed_address_backing_tensors={},
        )


def test_build_cb_descriptors_preserves_subtile_geometry(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensor(object())],
        cb_configs=[PhysicalDFBConfig(0, 1, "bfloat16", 2, 512, (16, 16))],
        core_ranges=_FakeCoreRanges(),
    )

    descriptor = descriptors[0]
    format_descriptor = descriptor.format_descriptors[0]
    assert descriptor.total_size == 1024
    assert format_descriptor.page_size == 512
    assert format_descriptor.tile.tile.tile_shape == (16, 16)


def _descriptor_cores(descriptor):
    return {
        (int(core.x), int(core.y))
        for core in kernel_runner.ttnn.corerange_to_cores(descriptor.core_ranges)
    }


def _descriptor_placement(descriptor):
    return (
        descriptor.format_descriptors[0].buffer_index,
        frozenset(_descriptor_cores(descriptor)),
    )


def _specialized_spec(core_ranges, used_dfb_indices):
    return kernel_runner.KernelSpec(
        path="/tmp/specialized.cpp",
        thread_type="compute",
        tensor_indices=[],
        config=object(),
        core_ranges=core_ranges,
        used_dfb_indices=used_dfb_indices,
    )


def test_specialized_dfb_descriptors_partition_overlapping_use(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_grid = _FakeExplicitCoreRanges((0, 0), (2, 0))
    core_0 = _FakeExplicitCoreRanges((0, 0), (0, 0))
    core_1 = _FakeExplicitCoreRanges((1, 0), (1, 0))
    core_2 = _FakeExplicitCoreRanges((2, 0), (2, 0))
    configs = [
        PhysicalDFBConfig(index, 1, "bfloat16", 1, 2048, (32, 32)) for index in range(2)
    ]

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=configs,
        core_ranges=full_grid,
        kernel_specs=[
            _specialized_spec(core_0, [0]),
            _specialized_spec(core_1, [0, 1]),
            _specialized_spec(core_2, [1]),
        ],
    )

    assert {_descriptor_placement(descriptor) for descriptor in descriptors} == {
        (0, frozenset({(0, 0)})),
        (0, frozenset({(1, 0)})),
        (1, frozenset({(1, 0)})),
        (1, frozenset({(2, 0)})),
    }


def test_specialized_dfb_descriptor_follows_active_kernel_core(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))
    active = _FakeExplicitCoreRanges((0, 0), (0, 0))
    inactive = _FakeExplicitCoreRanges((1, 0), (1, 0))

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[PhysicalDFBConfig(0, 1, "bfloat16", 1, 2048, (32, 32))],
        core_ranges=full_grid,
        kernel_specs=[
            _specialized_spec(active, [0]),
            _specialized_spec(inactive, []),
        ],
    )

    assert {_descriptor_placement(descriptor) for descriptor in descriptors} == {
        (0, frozenset({(0, 0)})),
    }


def test_unannotated_kernel_keeps_whole_grid_dfb_descriptor(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[PhysicalDFBConfig(0, 1, "bfloat16", 1, 2048, (32, 32))],
        core_ranges=full_grid,
        kernel_specs=[_specialized_spec(full_grid, None)],
    )

    assert len(descriptors) == 1
    assert descriptors[0].core_ranges is full_grid


def test_unannotated_kernel_remains_conservative_with_annotated_peer(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))
    core_0 = _FakeExplicitCoreRanges((0, 0), (0, 0))
    core_1 = _FakeExplicitCoreRanges((1, 0), (1, 0))

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[PhysicalDFBConfig(0, 1, "bfloat16", 1, 2048, (32, 32))],
        core_ranges=full_grid,
        kernel_specs=[
            _specialized_spec(core_0, []),
            _specialized_spec(core_1, None),
        ],
    )

    assert len(descriptors) == 1
    assert _descriptor_cores(descriptors[0]) == {(1, 0)}


def test_specialized_dfb_descriptors_allow_sparse_physical_ids(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_grid = _FakeExplicitCoreRanges((0, 0), (0, 0))
    configs = [
        PhysicalDFBConfig(index, 1, "bfloat16", 1, 2048, (32, 32)) for index in range(3)
    ]

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=configs,
        core_ranges=full_grid,
        kernel_specs=[_specialized_spec(full_grid, [2])],
    )

    assert [
        descriptor.format_descriptors[0].buffer_index for descriptor in descriptors
    ] == [2]


def test_specialized_dfb_budget_is_computed_per_core(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(kernel_runner, "DEFAULT_L1_CB_BUDGET_BYTES", 2048)
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))
    core_0 = _FakeExplicitCoreRanges((0, 0), (0, 0))
    core_1 = _FakeExplicitCoreRanges((1, 0), (1, 0))
    configs = [
        PhysicalDFBConfig(index, 1, "bfloat16", 1, 2048, (32, 32)) for index in range(2)
    ]

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=configs,
        core_ranges=full_grid,
        kernel_specs=[
            _specialized_spec(core_0, [0]),
            _specialized_spec(core_1, [1]),
        ],
    )

    assert {_descriptor_placement(descriptor) for descriptor in descriptors} == {
        (0, frozenset({(0, 0)})),
        (1, frozenset({(1, 0)})),
    }


def test_specialized_dfb_budget_uses_each_cores_remaining_l1(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner,
        "_get_remaining_l1_by_core_for_device",
        lambda _device, _cores: {(0, 0): 2048, (1, 0): 1024},
    )
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))
    core_0 = _FakeExplicitCoreRanges((0, 0), (0, 0))
    core_1 = _FakeExplicitCoreRanges((1, 0), (1, 0))

    with pytest.raises(
        ValueError,
        match=r"core \(1, 0\).*2048 bytes.*L1 budget \(1024 bytes\)",
    ):
        kernel_runner.build_cb_descriptors(
            tensors=[_FakeTensor(object())],
            cb_configs=[
                PhysicalDFBConfig(index, 1, "bfloat16", 1, 2048, (32, 32))
                for index in range(2)
            ],
            core_ranges=full_grid,
            kernel_specs=[
                _specialized_spec(core_0, [0]),
                _specialized_spec(core_1, [1]),
            ],
        )


def test_computed_address_backing_uses_specialized_dfb_cores(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    allocations = []
    monkeypatch.setattr(
        kernel_runner,
        "_allocate_l1_sharded_storage_tensor",
        lambda ranges, size, device: allocations.append((ranges, size, device))
        or object(),
    )
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))
    core_0 = _FakeExplicitCoreRanges((0, 0), (0, 0))
    core_1 = _FakeExplicitCoreRanges((1, 0), (1, 0))
    device = object()

    kernel_runner.build_pipe_computed_address_dfb_tensors(
        tensors=[],
        cb_configs=[PhysicalDFBConfig(0, 1, "bfloat16", 1, 2048, (32, 32))],
        core_ranges=full_grid,
        pipe_computed_address_dfb_indices=[0],
        device=device,
        kernel_specs=[
            _specialized_spec(core_0, [0]),
            _specialized_spec(core_1, []),
        ],
    )

    assert len(allocations) == 1
    assert _descriptor_cores(
        type("Allocation", (), {"core_ranges": allocations[0][0]})()
    ) == {(0, 0)}
    assert allocations[0][1:] == (2048, device)


def test_l1_sharded_storage_counts_sparse_cores(monkeypatch):
    fake_ttnn = _FakeTTNN()
    fake_ttnn.ShardSpec = lambda *args: args
    fake_ttnn.MemoryConfig = lambda *args: args
    fake_ttnn.ShardOrientation = type("ShardOrientation", (), {"ROW_MAJOR": object()})
    fake_ttnn.TensorMemoryLayout = type(
        "TensorMemoryLayout", (), {"HEIGHT_SHARDED": object()}
    )
    fake_ttnn.BufferType = type("BufferType", (), {"L1": object()})
    fake_ttnn.float32 = object()
    fake_ttnn.ROW_MAJOR_LAYOUT = object()
    empty_calls = []
    fake_ttnn.empty = (
        lambda shape, **kwargs: empty_calls.append((shape, kwargs)) or object()
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    sparse_ranges = _FakeTTNN.CoreRangeSet(
        (
            _FakeTTNN.CoreRange(_FakeTTNN.CoreCoord(0, 0), _FakeTTNN.CoreCoord(0, 0)),
            _FakeTTNN.CoreRange(_FakeTTNN.CoreCoord(2, 0), _FakeTTNN.CoreCoord(2, 0)),
        )
    )

    kernel_runner._allocate_l1_sharded_storage_tensor(
        sparse_ranges, num_bytes=2048, device=object()
    )

    assert empty_calls[0][0] == (2, 512)


def test_specialized_dfb_use_intersects_storage_segments(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))
    active = _FakeExplicitCoreRanges((0, 0), (0, 0))
    inactive = _FakeExplicitCoreRanges((1, 0), (1, 0))
    config = PhysicalDFBConfig(
        0,
        1,
        "bfloat16",
        1,
        2048,
        (32, 32),
        (DFBStorageSegment(nodes=((0, 0), (1, 0))),),
    )

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[config],
        core_ranges=full_grid,
        kernel_specs=[
            _specialized_spec(active, [0]),
            _specialized_spec(inactive, []),
        ],
    )

    assert len(descriptors) == 1
    assert _descriptor_cores(descriptors[0]) == {(0, 0)}


def test_specialized_dfb_use_requires_storage_segment_coverage(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))
    active = _FakeExplicitCoreRanges((0, 0), (0, 0))
    config = PhysicalDFBConfig(
        0,
        1,
        "bfloat16",
        1,
        2048,
        (32, 32),
        (DFBStorageSegment(nodes=((1, 0),)),),
    )

    with pytest.raises(
        ValueError,
        match=r"DFB\[0\] is used on cores \[\(0, 0\)\] that are not covered "
        r"by any storage segment",
    ):
        kernel_runner.build_cb_descriptors(
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[config],
            core_ranges=full_grid,
            kernel_specs=[_specialized_spec(active, [0])],
        )


def test_build_cb_descriptors_binds_tensor_on_exact_nodes(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    tensor = _FakeTensor(object(), dtype=expected_dtype)
    config = PhysicalDFBConfig(
        0,
        1,
        "bfloat16",
        1,
        2048,
        (32, 32),
        (
            DFBStorageSegment(
                nodes=((0, 0), (1, 0)),
                tensor_index=0,
                byte_offset=2048,
                byte_size=2048,
            ),
        ),
    )

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[tensor],
        cb_configs=[config],
        core_ranges=_FakeCoreRanges(),
    )

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor["tensor"] is tensor
    assert descriptor["address_offset"] == 2048
    assert descriptor["total_size"] == 2048
    selected = [
        (core_range.start.x, core_range.start.y)
        for core_range in descriptor["core_ranges"].ranges()
    ]
    assert selected == [(0, 0), (1, 0)]


def test_build_cb_descriptors_rejects_range_past_shard_boundary(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    tensor = _FakeTensor(object(), dtype=expected_dtype, shard_shape=(32, 32))

    with pytest.raises(ValueError, match="exceeds logical per-shard size 2048"):
        kernel_runner.build_cb_descriptors(
            tensors=[tensor],
            cb_configs=[
                _tensor_backing_config(
                    0,
                    nodes=((0, 0),),
                    byte_offset=2048,
                    byte_size=2048,
                )
            ],
            core_ranges=_FakeCoreRanges(),
        )


def test_build_cb_descriptors_rejects_node_without_tensor_shard(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        fake_ttnn,
        "get_optimal_worker_cores_for_sharded_tensor",
        lambda _tensor: [_FakeTTNN.CoreCoord(0, 0)],
    )
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    tensor = _FakeTensor(object(), dtype=expected_dtype)

    with pytest.raises(ValueError, match="no shard data on launch nodes.*1, 0"):
        kernel_runner.build_cb_descriptors(
            tensors=[tensor],
            cb_configs=[_tensor_backing_config(0, nodes=((0, 0), (1, 0)))],
            core_ranges=_FakeCoreRanges(),
        )


def test_build_cb_descriptors_rejects_equal_size_different_tile_shape(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    tensor = _FakeTensor(object(), dtype=expected_dtype, tile_shape=(16, 32))
    config = PhysicalDFBConfig(
        0,
        1,
        "bfloat16",
        1,
        1024,
        (32, 16),
        (
            DFBStorageSegment(
                nodes=((0, 0),),
                tensor_index=0,
                byte_offset=0,
                byte_size=1024,
            ),
        ),
    )

    with pytest.raises(ValueError, match="tile shape.*16, 32.*32, 16"):
        kernel_runner.build_cb_descriptors(
            tensors=[tensor],
            cb_configs=[config],
            core_ranges=_FakeCoreRanges(),
        )


def test_build_cb_descriptors_reports_unsupported_tensor_backing_format(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )
    tensor = _FakeTensor(object(), dtype=object())
    config = PhysicalDFBConfig(
        0,
        1,
        "bfp8",
        1,
        2048,
        (32, 32),
        (
            DFBStorageSegment(
                nodes=((0, 0),),
                tensor_index=0,
                byte_offset=0,
                byte_size=2048,
            ),
        ),
    )

    with pytest.raises(ValueError, match="tensor backing format bfp8 is not supported"):
        kernel_runner.build_cb_descriptors(
            tensors=[tensor],
            cb_configs=[config],
            core_ranges=_FakeCoreRanges(),
        )


def test_build_cb_descriptors_uses_current_tensor_allocation(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    config = _tensor_backing_config(0, nodes=((0, 0),))
    first_tensor = _FakeTensor(object(), address=0x2000, dtype=expected_dtype)
    second_tensor = _FakeTensor(object(), address=0x4000, dtype=expected_dtype)

    first_descriptor = kernel_runner.build_cb_descriptors(
        tensors=[first_tensor],
        cb_configs=[config],
        core_ranges=_FakeCoreRanges(),
    )[0]
    second_descriptor = kernel_runner.build_cb_descriptors(
        tensors=[second_tensor],
        cb_configs=[config],
        core_ranges=_FakeCoreRanges(),
    )[0]

    assert first_descriptor["tensor"] is first_tensor
    assert second_descriptor["tensor"] is second_tensor


def test_build_cb_descriptors_preserves_descriptor_helper_failure(monkeypatch):
    fake_ttnn = _FakeTTNN()

    def fail_descriptor_creation(*_args, **_kwargs):
        raise RuntimeError("TTNN descriptor construction failed")

    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )
    monkeypatch.setattr(
        fake_ttnn,
        "cb_descriptor_from_sharded_tensor",
        fail_descriptor_creation,
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    tensor = _FakeTensor(object(), dtype=expected_dtype)

    with pytest.raises(RuntimeError, match="TTNN descriptor construction failed"):
        kernel_runner.build_cb_descriptors(
            tensors=[tensor],
            cb_configs=[_tensor_backing_config(0, nodes=((0, 0),))],
            core_ranges=_FakeCoreRanges(),
        )


def _tensor_backing_config(
    dfb_index, *, nodes, block_count=1, byte_offset=0, byte_size=2048
):
    return PhysicalDFBConfig(
        dfb_index,
        1,
        "bfloat16",
        block_count,
        2048,
        (32, 32),
        (
            DFBStorageSegment(
                nodes=nodes,
                tensor_index=dfb_index,
                byte_offset=byte_offset,
                byte_size=byte_size,
            ),
        ),
    )


def test_build_cb_descriptors_rejects_aliased_partial_tensor_ranges(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    tensors = [
        _FakeTensor(object(), address=0x4000, dtype=expected_dtype),
        _FakeTensor(object(), address=0x4000, dtype=expected_dtype),
    ]
    configs = [
        _tensor_backing_config(0, nodes=((0, 0),), block_count=2, byte_size=4096),
        _tensor_backing_config(1, nodes=((0, 0),), byte_offset=2048, byte_size=2048),
    ]

    with pytest.raises(ValueError, match="byte ranges partially overlap"):
        kernel_runner.build_cb_descriptors(
            tensors=tensors,
            cb_configs=configs,
            core_ranges=_FakeCoreRanges(),
        )


def test_build_cb_descriptors_rejects_aliased_range_with_distinct_indices(
    monkeypatch,
):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    tensors = [
        _FakeTensor(object(), address=0x4000, dtype=expected_dtype),
        _FakeTensor(object(), address=0x4000, dtype=expected_dtype),
    ]
    configs = [
        _tensor_backing_config(0, nodes=((0, 0),)),
        _tensor_backing_config(1, nodes=((0, 0),)),
    ]

    with pytest.raises(ValueError, match="require one physical DFB index"):
        kernel_runner.build_cb_descriptors(
            tensors=tensors,
            cb_configs=configs,
            core_ranges=_FakeCoreRanges(),
        )


def test_build_cb_descriptors_allows_same_address_on_disjoint_nodes(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    tensors = [
        _FakeTensor(object(), address=0x4000, dtype=expected_dtype),
        _FakeTensor(object(), address=0x4000, dtype=expected_dtype),
    ]
    configs = [
        _tensor_backing_config(0, nodes=((0, 0),)),
        _tensor_backing_config(1, nodes=((1, 0),)),
    ]

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=tensors,
        cb_configs=configs,
        core_ranges=_FakeCoreRanges(),
    )

    assert len(descriptors) == 2


def test_build_cb_descriptors_excludes_tensor_backing_from_static_budget(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 1
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    tensor = _FakeTensor(object(), dtype=expected_dtype)

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[tensor],
        cb_configs=[_tensor_backing_config(0, nodes=((0, 0),))],
        core_ranges=_FakeCoreRanges(),
    )

    assert len(descriptors) == 1


def test_build_cb_descriptors_charges_mixed_storage_once(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 1024
    )
    expected_dtype = kernel_runner.format_name_to_ttnn_dtype("bfloat16")
    tensor = _FakeTensor(object(), dtype=expected_dtype)
    config = PhysicalDFBConfig(
        0,
        1,
        "bfloat16",
        1,
        2048,
        (32, 32),
        (
            DFBStorageSegment(
                nodes=((0, 0),),
                tensor_index=0,
                byte_offset=0,
                byte_size=2048,
            ),
            DFBStorageSegment(nodes=((1, 0),)),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Total circular buffer allocation \\(2048 bytes\\) exceeds L1 budget \\(1024 bytes\\)",
    ):
        kernel_runner.build_cb_descriptors(
            tensors=[tensor],
            cb_configs=[config],
            core_ranges=_FakeCoreRanges(),
        )


def test_emit_runner_source_preserves_subtile_geometry(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())

    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[PhysicalDFBConfig(0, 1, "bfloat16", 2, 512, (16, 16))],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "num_tiles=1" in source
    assert "data_format='bfloat16'" in source
    assert "block_count=2" in source
    assert "page_size=512" in source
    assert "tile=(16, 16)" in source


def test_emit_runner_source_preserves_tensor_backing_segments(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    config = _tensor_backing_config(
        0,
        nodes=((0, 0), (1, 0)),
        byte_offset=2048,
        byte_size=2048,
    )

    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[config],
        grid_cols=2,
        grid_rows=1,
        num_tensors=1,
    )

    assert "from ttl.dataflow_buffer import DFBStorageSegment" in source
    assert "nodes=((0, 0), (1, 0))" in source
    assert "tensor_index=0" in source
    assert "byte_offset=2048" in source
    assert "byte_size=2048" in source


def test_emit_runner_source_uses_shared_pipe_resource_helpers():
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
        program_hash=-2,
        num_pipe_global_semaphores=3,
        num_dfb_resets=2,
    )

    assert "NUM_PIPE_GLOBAL_SEMAPHORES = 3" in source
    assert "NUM_DFB_RESETS = 2" in source
    assert "PROGRAM_HASH = 18446744073709551614" in source
    assert "return run_kernel_on_device(" in source
    assert "program_hash=PROGRAM_HASH" in source
    assert "num_pipe_global_semaphores=NUM_PIPE_GLOBAL_SEMAPHORES" in source
    assert "num_dfb_resets=NUM_DFB_RESETS" in source
    assert "operation_name=OPERATION_NAME" in source
    assert "ttnn.create_global_semaphore(device, core_ranges, 0)" not in source


def test_emitted_runner_without_resources_executes_shared_runner(monkeypatch):
    calls = []

    def record_run(**kwargs):
        calls.append(kwargs)
        return "executed"

    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
        kernel_name="no_resources",
    )
    emitted_runner = _load_emitted_runner(monkeypatch, source, record_run)

    result = emitted_runner["run"]([object()], device=object())

    assert result == "executed"
    assert len(calls) == 1
    assert calls[0]["kernel_specs"] == []
    assert "runtime_resource_factory" not in calls[0]
    assert calls[0]["operation_name"] == "no_resources"


def test_emitted_runner_requires_and_applies_runtime_resource_factory(monkeypatch):
    runtime_kernel = Kernel._from_metadata(
        KernelKind.COMPUTE,
        "runtime_kernel",
        "test.emitted_operation",
    )
    left_ranges = _FakeCoreRanges((((0, 0), (0, 0)),))
    right_ranges = _FakeCoreRanges((((1, 0), (1, 0)),))
    source = kernel_runner.emit_runner_source(
        kernel_specs=[
            _kernel_spec(runtime_kernel, left_ranges),
            _kernel_spec(runtime_kernel, right_ranges),
        ],
        cb_configs=[],
        grid_cols=2,
        grid_rows=1,
        num_tensors=1,
        kernel_name="test.emitted_operation",
        requires_runtime_resource_factory=True,
    )
    plans = []

    def plan_run(**kwargs):
        resources = kwargs["runtime_resource_factory"](
            tensors=tuple(kwargs["tensors"]),
            core_ranges=kwargs["core_ranges"],
            first_free_semaphore_id=kwargs["num_pipe_sync_semaphores"],
        )
        plan = kernel_runner.plan_program_runtime_resources(
            operation_name=kwargs["operation_name"],
            resources=resources,
            kernel_specs=kwargs["kernel_specs"],
            operation_core_ranges=kwargs["core_ranges"],
            first_free_semaphore_id=kwargs["num_pipe_sync_semaphores"],
        )
        plans.append(plan)
        return plan

    emitted_runner = _load_emitted_runner(monkeypatch, source, plan_run)
    with pytest.raises(TypeError, match="runtime_resource_factory"):
        emitted_runner["run"]([object()], device=object())
    with pytest.raises(
        TypeError,
        match=(
            "emitted runner for 'test.emitted_operation' requires "
            "runtime_resource_factory"
        ),
    ):
        emitted_runner["run"](
            [object()],
            runtime_resource_factory=None,
            device=object(),
        )

    next_runtime_values = iter(((7, 8), (17, 18)))

    def make_resources(**_kwargs):
        left_value, right_value = next(next_runtime_values)
        return ProgramRuntimeResources(
            kernel_resources=(
                KernelRuntimeResources(
                    kernel=runtime_kernel,
                    runtime_args=(
                        CoreRuntimeArgs(_FakeCoreCoord(0, 0), (left_value,)),
                        CoreRuntimeArgs(_FakeCoreCoord(1, 0), (right_value,)),
                    ),
                    defines=(KernelDefine("MODE", "emitted"),),
                ),
            )
        )

    for _call_index in range(2):
        emitted_runner["run"](
            [object()],
            runtime_resource_factory=make_resources,
            device=object(),
        )

    assert len(plans) == 2
    assert [descriptor.coordinates for descriptor in plans[0].kernel_descriptors] == [
        ((0, 0),),
        ((1, 0),),
    ]
    assert [descriptor.defines for descriptor in plans[0].kernel_descriptors] == [
        (("MODE", "emitted"),),
        (("MODE", "emitted"),),
    ]
    assert [
        descriptor.runtime_args[0].values for descriptor in plans[0].kernel_descriptors
    ] == [(7,), (8,)]
    assert [
        descriptor.runtime_args[0].values for descriptor in plans[1].kernel_descriptors
    ] == [(17,), (18,)]
    assert plans[0].structural_fingerprint == plans[1].structural_fingerprint


def test_emitted_runner_replaces_global_semaphore_owners(monkeypatch):
    fake_ttnn = _LifetimeTrackingTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    monkeypatch.setitem(sys.modules, "ttnn", fake_ttnn)
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
        num_pipe_global_semaphores=1,
    )
    namespace = {"__name__": "generated_runner"}
    exec(compile(source, "<generated-runner>", "exec"), namespace)

    device = object()
    tensor = _FakeTensor(device)
    namespace["run"]([tensor], device=device)
    first_owner = namespace["_RUNTIME_RESOURCE_CACHE"].pipe_resources.global_semaphores[
        0
    ]

    namespace["run"]([tensor], device=device)
    second_owner = namespace[
        "_RUNTIME_RESOURCE_CACHE"
    ].pipe_resources.global_semaphores[0]
    assert second_owner is not first_owner
    assert fake_ttnn.events[:3] == [
        ("allocate", 0),
        ("synchronize", device),
        ("allocate", 1),
    ]

    del first_owner
    del second_owner
    namespace.clear()
    gc.collect()
    assert fake_ttnn.events == [
        ("allocate", 0),
        ("synchronize", device),
        ("allocate", 1),
        ("release", 0),
        ("synchronize", device),
        ("release", 1),
    ]


def test_emitted_runner_synchronizes_before_owner_destruction(monkeypatch):
    fake_ttnn = _LifetimeTrackingTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    monkeypatch.setitem(sys.modules, "ttnn", fake_ttnn)
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
        num_pipe_global_semaphores=1,
    )
    namespace = {"__name__": "generated_runner"}
    exec(compile(source, "<generated-runner>", "exec"), namespace)

    device = object()
    namespace["run"]([_FakeTensor(device)], device=device)
    resource_owner = namespace["_RUNTIME_RESOURCE_CACHE"].pipe_resources
    semaphore_reference = weakref.ref(resource_owner.global_semaphores[0])
    del resource_owner
    namespace.clear()
    gc.collect()

    assert fake_ttnn.events == [
        ("allocate", 0),
        ("synchronize", device),
        ("release", 0),
    ]
    assert semaphore_reference() is None


def test_emit_runner_source_accepts_physical_dfb_configs(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[
            PhysicalDFBConfig(
                dfb_index=0,
                num_tiles=3,
                data_format="bfloat16",
                block_count=2,
                page_size=32,
                tile=(1, 16),
            )
        ],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "from ttl.dataflow_buffer import PhysicalDFBConfig" in source
    assert "dfb_index=0" in source
    assert "num_tiles=3" in source
    assert "data_format='bfloat16'" in source
    assert "block_count=2" in source
    assert "page_size=32" in source
    assert "tile=(1, 16)" in source
    assert "cb_configs=CB_CONFIGS" in source
    assert "for i, (num_tiles, block_count" not in source


def test_emit_runner_source_omits_tile_descriptor_for_scalar_dfb(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[PhysicalDFBConfig(0, 128, "float32", 2, 4, None)],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "data_format='float32'" in source
    assert "tile=None" in source


def test_physical_dfb_allocation_scales_with_subtile_area(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_tile = kernel_runner._get_dfb_allocation(
        PhysicalDFBConfig(0, 1, "bfloat16", 2, 2048, (32, 32))
    )
    two_half_tiles = kernel_runner._get_dfb_allocation(
        PhysicalDFBConfig(0, 2, "bfloat16", 2, 1024, (16, 32))
    )

    assert two_half_tiles.page_size * 2 == full_tile.page_size
    assert two_half_tiles.total_size == full_tile.total_size


def test_physical_dfb_allocation_uses_complete_rank_three_shape(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    allocation = kernel_runner._get_dfb_allocation(
        PhysicalDFBConfig(0, 8, "bfloat16", 2, 2048, (32, 32))
    )

    assert allocation.total_size == 8 * 2 * 2048


def test_dfb_allocation_requires_finalized_config(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())

    with pytest.raises(
        TypeError,
        match="must be a finalized PhysicalDFBConfig",
    ):
        kernel_runner._get_dfb_allocation(object())


@pytest.mark.parametrize(
    ("cb_configs", "error_type", "message"),
    [
        (
            [None],
            TypeError,
            "must be a finalized PhysicalDFBConfig",
        ),
        (
            [PhysicalDFBConfig(1, 1, "bfloat16", 2, 2048, (32, 32))],
            ValueError,
            "DFB config at physical index 0",
        ),
    ],
    ids=["missing-config", "wrong-index"],
)
def test_emit_runner_source_rejects_invalid_physical_dfb_sequence(
    monkeypatch, cb_configs, error_type, message
):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    with pytest.raises(error_type, match=message):
        kernel_runner.emit_runner_source(
            kernel_specs=[],
            cb_configs=cb_configs,
            grid_cols=1,
            grid_rows=1,
            num_tensors=1,
        )


@pytest.mark.parametrize(
    ("data_format", "page_size"),
    [
        ("bfloat4_b", 576),
        ("bfloat8_b", 1088),
        ("uint8", 1024),
    ],
)
def test_emit_runner_source_preserves_physical_dfb_format(
    monkeypatch, data_format, page_size
):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[
            PhysicalDFBConfig(
                dfb_index=0,
                num_tiles=1,
                data_format=data_format,
                block_count=2,
                page_size=page_size,
                tile=(32, 32),
            )
        ],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert f"data_format={data_format!r}" in source
    assert f"page_size={page_size}" in source


def test_emit_runner_source_rejects_unknown_physical_dfb_format(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    with pytest.raises(ValueError, match="Unrecognized data format"):
        kernel_runner.emit_runner_source(
            kernel_specs=[],
            cb_configs=[
                PhysicalDFBConfig(
                    dfb_index=0,
                    num_tiles=1,
                    data_format="unknown",
                    block_count=2,
                    page_size=2048,
                    tile=(32, 32),
                )
            ],
            grid_cols=1,
            grid_rows=1,
            num_tensors=1,
        )


def test_emit_runner_source_omits_program_hash_by_default():
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "PROGRAM_HASH = None" in source


def test_emit_runner_source_preserves_specialized_dfb_use(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    source = kernel_runner.emit_runner_source(
        kernel_specs=[
            kernel_runner.KernelSpec(
                path="/tmp/sparse.cpp",
                thread_type="compute",
                tensor_indices=[],
                config=object(),
                used_dfb_indices=[2],
            )
        ],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "KERNEL_USED_DFB_INDICES = [" in source
    assert "    [2],  # compute" in source
    assert "used_dfb_indices=KERNEL_USED_DFB_INDICES[kernel_idx]" in source
    assert "kernel_specs=kernel_specs" in source
    compile(source, "<emitted-runner>", "exec")


def test_emit_runner_source_preserves_positional_options():
    source = kernel_runner.emit_runner_source(
        [],
        [],
        1,
        1,
        1,
        "legacy_kernel",
        2,
        64,
        3,
    )

    assert '"""Auto-generated runner for legacy_kernel."""' in source
    assert "PROGRAM_HASH = None" in source
    assert "NUM_PIPE_SYNC_SEMAPHORES = 2" in source
    assert "PIPE_SRAM_SCRATCH_BYTES = 64" in source
    assert "NUM_PIPE_GLOBAL_SEMAPHORES = 3" in source


def test_emit_runner_file_preserves_positional_options(tmp_path):
    output_path = tmp_path / "legacy_runner.py"

    result_path = kernel_runner.emit_runner_file(
        [],
        [],
        1,
        1,
        1,
        str(output_path),
        "legacy_kernel",
        2,
        64,
        3,
    )

    assert result_path == str(output_path)
    source = output_path.read_text()
    assert '"""Auto-generated runner for legacy_kernel."""' in source
    assert "PROGRAM_HASH = None" in source
    assert "NUM_PIPE_SYNC_SEMAPHORES = 2" in source
    assert "PIPE_SRAM_SCRATCH_BYTES = 64" in source
    assert "NUM_PIPE_GLOBAL_SEMAPHORES = 3" in source
