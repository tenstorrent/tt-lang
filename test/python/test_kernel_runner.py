# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only tests for ttl.kernel_runner resource allocation helpers."""

from types import SimpleNamespace

import pytest

from ttl import kernel_runner
from ttl.dataflow_buffer import DFBStorageSegment, PhysicalDFBConfig


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


class _FakeGridSize:
    x = 1
    y = 1


class _FakeBoundingBox:
    @staticmethod
    def grid_size():
        return _FakeGridSize()


class _FakeCoreRanges:
    @staticmethod
    def bounding_box():
        return _FakeBoundingBox()


class _FakeExplicitCoreRanges:
    def __init__(self, start, end):
        self.ranges = (
            _FakeTTNN.CoreRange(_FakeTTNN.CoreCoord(*start), _FakeTTNN.CoreCoord(*end)),
        )


class _FakeTTNN:
    def __init__(self):
        self.create_calls = []
        self.generic_op_calls = []
        self.next_address = 0x1000

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

    class KernelDescriptor:
        def __init__(
            self,
            kernel_source,
            core_ranges,
            compile_time_args,
            common_runtime_args,
            config,
            compiler_include_paths=None,
        ):
            self.kernel_source = kernel_source
            self.core_ranges = core_ranges
            self.compile_time_args = compile_time_args
            self.common_runtime_args = common_runtime_args
            self.config = config
            self.compiler_include_paths = compiler_include_paths or []

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

    class CoreRangeSet:
        def __init__(self, ranges):
            self.ranges = tuple(ranges)

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
    def get_dram_alignment():
        return 64

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


def test_build_pipe_global_semaphores_empty_does_not_require_ttnn(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", None)

    semaphores, addresses = kernel_runner.build_pipe_global_semaphores(
        tensors=[],
        core_ranges=object(),
        count=0,
    )

    assert semaphores == []
    assert addresses == []


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


def test_run_kernel_global_semaphore_lifetime_is_bounded(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    tensor = _FakeTensor(object())
    lifetime = []

    for _ in range(2):
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[tensor],
            cb_configs=[],
            core_ranges=_FakeCoreRanges(),
            num_pipe_global_semaphores=2,
            pipe_global_semaphore_lifetime=lifetime,
        )

    assert len(fake_ttnn.create_calls) == 4
    assert lifetime == fake_ttnn.create_calls[-2:]


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
    ranges = descriptor.core_ranges.ranges
    if callable(ranges):
        ranges = ranges()
    return {
        (core_x, core_y)
        for core_range in ranges
        for core_y in range(core_range.start.y, core_range.end.y + 1)
        for core_x in range(core_range.start.x, core_range.end.x + 1)
    }


def _specialized_spec(core_ranges, used_dfb_indices):
    return kernel_runner.KernelSpec(
        path="/tmp/specialized.cpp",
        thread_type="compute",
        tensor_indices=[],
        config=object(),
        core_ranges=core_ranges,
        used_dfb_indices=used_dfb_indices,
    )


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

    assert len(descriptors) == 1
    assert _descriptor_cores(descriptors[0]) == {(0, 0)}


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


def test_unannotated_kernel_is_conservative_with_annotated_peer(monkeypatch):
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


def test_allocation_nodes_scope_unspecialized_dfb_descriptor(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[
            PhysicalDFBConfig(
                0,
                1,
                "bfloat16",
                1,
                2048,
                (32, 32),
                allocation_nodes=((1, 0),),
            )
        ],
        core_ranges=full_grid,
        kernel_specs=[_specialized_spec(full_grid, None)],
    )

    assert len(descriptors) == 1
    assert _descriptor_cores(descriptors[0]) == {(1, 0)}


def test_empty_allocation_nodes_omit_dfb_descriptor(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[
            PhysicalDFBConfig(
                0,
                1,
                "bfloat16",
                1,
                2048,
                (32, 32),
                allocation_nodes=(),
            )
        ],
        core_ranges=full_grid,
        kernel_specs=[_specialized_spec(full_grid, None)],
    )

    assert descriptors == []


@pytest.mark.parametrize(
    ("allocation_nodes", "message"),
    [
        (((0, 0), (0, 0)), "contains duplicates"),
        (([0, 0],), "must be a nonnegative integer"),
        (((True, 0),), "must be a nonnegative integer"),
        (((-1, 0),), "must be a nonnegative integer"),
    ],
    ids=["duplicate", "non-tuple", "non-integer", "negative"],
)
def test_invalid_allocation_nodes_are_rejected(monkeypatch, allocation_nodes, message):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    config = PhysicalDFBConfig(
        0,
        1,
        "bfloat16",
        1,
        2048,
        (32, 32),
        allocation_nodes=allocation_nodes,
    )

    with pytest.raises(ValueError, match=message):
        kernel_runner.build_cb_descriptors(
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[config],
            core_ranges=_FakeExplicitCoreRanges((0, 0), (1, 0)),
        )


def test_allocation_nodes_must_cover_storage_segments(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))
    config = PhysicalDFBConfig(
        0,
        1,
        "bfloat16",
        1,
        2048,
        (32, 32),
        (DFBStorageSegment(nodes=((0, 0),)),),
        allocation_nodes=((1, 0),),
    )

    with pytest.raises(ValueError, match="must cover its exact allocation nodes"):
        kernel_runner.build_cb_descriptors(
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[config],
            core_ranges=full_grid,
        )


def test_allocation_nodes_outside_program_grid_are_rejected(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())

    with pytest.raises(ValueError, match="outside the program grid"):
        kernel_runner.build_cb_descriptors(
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[
                PhysicalDFBConfig(
                    0,
                    1,
                    "bfloat16",
                    1,
                    2048,
                    (32, 32),
                    allocation_nodes=((2, 0),),
                )
            ],
            core_ranges=_FakeExplicitCoreRanges((0, 0), (1, 0)),
        )


def test_allocation_nodes_compute_dfb_budget_per_core(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(kernel_runner, "DEFAULT_L1_CB_BUDGET_BYTES", 2048)
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[
            PhysicalDFBConfig(
                index,
                1,
                "bfloat16",
                1,
                2048,
                (32, 32),
                allocation_nodes=((index, 0),),
            )
            for index in range(2)
        ],
        core_ranges=full_grid,
    )

    assert len(descriptors) == 2
    assert {_descriptor_cores(descriptor).pop() for descriptor in descriptors} == {
        (0, 0),
        (1, 0),
    }


def test_specialized_dfb_descriptors_allow_sparse_physical_indices(monkeypatch):
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

    assert len(descriptors) == 2
    assert {_descriptor_cores(descriptor).pop() for descriptor in descriptors} == {
        (0, 0),
        (1, 0),
    }


@pytest.mark.parametrize(
    ("budget_bytes", "expected_order"),
    [(16000, [0, 2, 1]), (18000, [0, 1, 2])],
    ids=["reordered", "already-fits"],
)
def test_static_dfb_descriptors_are_ordered_to_fit_l1(
    monkeypatch, budget_bytes, expected_order
):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(kernel_runner, "DEFAULT_L1_CB_BUDGET_BYTES", budget_bytes)
    full_grid = _FakeExplicitCoreRanges((0, 0), (11, 9))
    sparse_nodes = tuple(
        (node_x, node_y)
        for node_y, row_end_x in ((0, 7), (1, 7), (4, 7), (9, 0))
        for node_x in range(row_end_x + 1)
    )
    configs = [
        PhysicalDFBConfig(
            0,
            32,
            "bfloat16",
            1,
            64,
            (1, 32),
            allocation_nodes=((0, 6),),
        ),
        PhysicalDFBConfig(
            1,
            128,
            "bfloat16",
            1,
            64,
            (1, 32),
            allocation_nodes=tuple(
                (node_x, node_y) for node_y in range(10) for node_x in range(12)
            ),
        ),
        PhysicalDFBConfig(
            2,
            112,
            "bfloat16",
            1,
            64,
            (1, 32),
            allocation_nodes=sparse_nodes,
        ),
    ]

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=configs,
        core_ranges=full_grid,
        kernel_specs=[_specialized_spec(full_grid, None)],
    )

    # The physical order requires 17,408 bytes because the full-grid DFB
    # inherits the singleton DFB frontier. Independent DFBs first need 15,360.
    assert [
        descriptor.format_descriptors[0].buffer_index for descriptor in descriptors
    ] == expected_order
    descriptors_by_index = {
        descriptor.format_descriptors[0].buffer_index: descriptor
        for descriptor in descriptors
    }
    assert _descriptor_cores(descriptors_by_index[0]) == {(0, 6)}
    assert _descriptor_cores(descriptors_by_index[1]) == {
        (node_x, node_y) for node_y in range(10) for node_x in range(12)
    }
    assert _descriptor_cores(descriptors_by_index[2]) == set(sparse_nodes)


def test_static_dfb_descriptor_exact_search_finds_nonlocal_reordering(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    plan_nodes = (
        ((1, 0),),
        ((0, 0), (3, 0)),
        ((1, 0),),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 0), (2, 0)),
    )
    plan_sizes = (96, 80, 256, 128, 24)
    descriptor_plans = [
        kernel_runner._DFBDescriptorPlan(
            descriptor=object(),
            physical_index=physical_index,
            total_size=plan_sizes[physical_index],
            nodes=plan_nodes[physical_index],
            has_static_storage=True,
        )
        for physical_index in range(len(plan_nodes))
    ]
    remaining_bytes_by_core = {
        (0, 0): 336,
        (1, 0): 544,
        (2, 0): 216,
        (3, 0): 272,
    }

    ordered_plans = kernel_runner._order_static_dfb_descriptor_plans(
        descriptor_plans, remaining_bytes_by_core
    )

    frontiers = {core: 0 for core in remaining_bytes_by_core}
    for plan in ordered_plans:
        address = kernel_runner._align_up(
            max(frontiers[core] for core in plan.nodes), 64
        )
        for core in plan.nodes:
            frontiers[core] = address + plan.total_size
    assert all(
        frontiers[core] <= remaining_bytes
        for core, remaining_bytes in remaining_bytes_by_core.items()
    )


def test_static_dfb_descriptor_order_reports_no_fitting_candidate(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(kernel_runner, "DEFAULT_L1_CB_BUDGET_BYTES", 10240)
    full_grid = _FakeExplicitCoreRanges((0, 0), (2, 0))
    configs = [
        PhysicalDFBConfig(
            physical_index,
            64,
            "bfloat16",
            1,
            64,
            (1, 32),
            allocation_nodes=allocation_nodes,
        )
        for physical_index, allocation_nodes in enumerate(
            (
                ((0, 0), (1, 0)),
                ((1, 0), (2, 0)),
                ((0, 0), (2, 0)),
            )
        )
    ]

    with pytest.raises(
        ValueError,
        match=(
            "the best candidate requires 12288 bytes on core .* where 10240 bytes "
            "remain, and exceeds the L1 budget by 2048 bytes"
        ),
    ):
        kernel_runner.build_cb_descriptors(
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=configs,
            core_ranges=full_grid,
            kernel_specs=[_specialized_spec(full_grid, None)],
        )


def test_remaining_l1_uses_lowest_live_tensor_address(monkeypatch):
    l1_buffer_type = object()
    dram_buffer_type = object()
    device_info = SimpleNamespace(
        address_at_first_l1_cb_buffer=0x1FC0,
        cb_limit=0x1000,
    )
    buffer_pages = [
        SimpleNamespace(
            buffer_type=l1_buffer_type,
            core_x=0,
            core_y=0,
            page_address=0x2900,
            page_size=0x100,
        ),
        SimpleNamespace(
            buffer_type=l1_buffer_type,
            core_x=1,
            core_y=0,
            page_address=0x2C00,
            page_size=0x400,
        ),
        SimpleNamespace(
            buffer_type=dram_buffer_type,
            core_x=0,
            core_y=0,
            page_address=0x2100,
            page_size=0x800,
        ),
    ]
    reports = SimpleNamespace(
        get_device_info=lambda _device: device_info,
        get_buffer_pages=lambda _device: buffer_pages,
    )
    monkeypatch.setattr(
        kernel_runner,
        "ttnn",
        SimpleNamespace(
            BufferType=SimpleNamespace(L1=l1_buffer_type),
            get_allocator_base_address=lambda _device, _buffer_type: 0x2000,
            _ttnn=SimpleNamespace(reports=reports),
        ),
    )

    device = SimpleNamespace(get_num_devices=lambda: 1)
    remaining_by_core = kernel_runner._get_remaining_l1_by_core_for_device(
        device,
        {(0, 0), (1, 0), (2, 0)},
    )

    assert remaining_by_core == {
        (0, 0): 0x900,
        (1, 0): 0xC00,
        (2, 0): 0x1000,
    }
    assert kernel_runner.get_min_remaining_l1_for_device(device) == 0x900


@pytest.mark.parametrize(
    "device",
    [SimpleNamespace(get_num_devices=lambda: 32), object()],
    ids=["multiple-devices", "unknown-device-type"],
)
def test_global_remaining_l1_uses_reference_allocator_floor(monkeypatch, device):
    l1_buffer_type = object()
    device_info = SimpleNamespace(cb_limit=0x1000)
    buffer_pages = [
        SimpleNamespace(
            buffer_type=l1_buffer_type,
            core_x=0,
            core_y=0,
            page_address=0x2900,
        ),
        SimpleNamespace(
            buffer_type=l1_buffer_type,
            core_x=1,
            core_y=0,
            page_address=0x2C00,
        ),
    ]
    reports = SimpleNamespace(
        get_device_info=lambda _device: device_info,
        get_buffer_pages=lambda _device: buffer_pages,
    )
    monkeypatch.setattr(
        kernel_runner,
        "ttnn",
        SimpleNamespace(
            BufferType=SimpleNamespace(L1=l1_buffer_type),
            get_allocator_base_address=lambda _device, _buffer_type: 0x2000,
            _ttnn=SimpleNamespace(reports=reports),
        ),
    )

    remaining_by_core = kernel_runner._get_remaining_l1_by_core_for_device(
        device,
        {(0, 0), (1, 0), (2, 0)},
    )

    assert remaining_by_core == {
        (0, 0): 0x900,
        (1, 0): 0x900,
        (2, 0): 0x900,
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
    allocated_ranges, allocated_bytes, allocated_device = allocations[0]
    assert _descriptor_cores(
        type("Allocation", (), {"core_ranges": allocated_ranges})()
    ) == {(0, 0)}
    assert allocated_bytes == 2048
    assert allocated_device is device


def test_computed_address_backing_resolves_each_dfb_from_full_grid(monkeypatch):
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

    kernel_runner.build_pipe_computed_address_dfb_tensors(
        tensors=[],
        cb_configs=[
            PhysicalDFBConfig(index, 1, "bfloat16", 1, 2048, (32, 32))
            for index in range(2)
        ],
        core_ranges=full_grid,
        pipe_computed_address_dfb_indices=[0, 1],
        device=object(),
        kernel_specs=[
            _specialized_spec(core_0, [0]),
            _specialized_spec(core_1, [1]),
        ],
    )

    assert [
        _descriptor_cores(type("Allocation", (), {"core_ranges": ranges})())
        for ranges, _, _ in allocations
    ] == [{(0, 0)}, {(1, 0)}]


def test_computed_address_backing_uses_allocation_nodes(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    allocations = []
    monkeypatch.setattr(
        kernel_runner,
        "_allocate_l1_sharded_storage_tensor",
        lambda ranges, size, device: allocations.append((ranges, size, device))
        or object(),
    )
    full_grid = _FakeExplicitCoreRanges((0, 0), (1, 0))

    kernel_runner.build_pipe_computed_address_dfb_tensors(
        tensors=[],
        cb_configs=[
            PhysicalDFBConfig(
                0,
                1,
                "bfloat16",
                1,
                2048,
                (32, 32),
                allocation_nodes=((1, 0),),
            )
        ],
        core_ranges=full_grid,
        pipe_computed_address_dfb_indices=[0],
        device=object(),
    )

    allocated_ranges, _, _ = allocations[0]
    assert _descriptor_cores(
        type("Allocation", (), {"core_ranges": allocated_ranges})()
    ) == {(1, 0)}


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
        lambda tensor_shape, **kwargs: empty_calls.append((tensor_shape, kwargs))
        or object()
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
        for core_range in descriptor["core_ranges"].ranges
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
    )

    assert "NUM_PIPE_GLOBAL_SEMAPHORES = 3" in source
    assert "PROGRAM_HASH = 18446744073709551614" in source
    assert "build_pipe_runtime_resources(" in source
    assert "build_kernel_descriptors(" in source
    assert "build_pipe_sync_semaphore_descriptors(" in source
    assert "build_generic_op_io_tensors(" in source
    assert "program.custom_program_hash = PROGRAM_HASH" in source
    assert "ttnn.create_global_semaphore(device, core_ranges, 0)" not in source


def test_emit_runner_source_accepts_physical_dfb_configs():
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
                allocation_nodes=((0, 0),),
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
    assert "allocation_nodes=((0, 0),)" in source
    assert "cb_descriptors = build_cb_descriptors(" in source
    assert "for i, (num_tiles, block_count" not in source


def test_emit_runner_source_omits_tile_descriptor_for_scalar_dfb():
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[PhysicalDFBConfig(0, 128, "float32", 2, 4, None)],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "data_format='float32'" in source
    assert "tile=None" in source


def test_physical_dfb_allocation_scales_with_subtile_area():
    full_tile = kernel_runner._get_dfb_allocation(
        PhysicalDFBConfig(0, 1, "bfloat16", 2, 2048, (32, 32))
    )
    two_half_tiles = kernel_runner._get_dfb_allocation(
        PhysicalDFBConfig(0, 2, "bfloat16", 2, 1024, (16, 32))
    )

    assert two_half_tiles.page_size * 2 == full_tile.page_size
    assert two_half_tiles.total_size == full_tile.total_size


def test_physical_dfb_allocation_uses_complete_rank_three_shape():
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
    cb_configs, error_type, message
):
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
def test_emit_runner_source_preserves_physical_dfb_format(data_format, page_size):
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


def test_emit_runner_source_rejects_unknown_physical_dfb_format():
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
