# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only tests for ttl.kernel_runner resource allocation helpers."""

from collections import defaultdict
from typing import NamedTuple

import pytest

from ttl import kernel_runner
from ttl.domains import DeviceDomain


class _FakeTensor:
    def __init__(self, device, address=0x2000):
        self._device = device
        self._address = address

    def device(self):
        return self._device

    def buffer_address(self):
        return self._address


class _FakeTensorWithoutDevice:
    pass


class _FakeDataFormat:
    name = "bfloat16"


class _FakeDFBConfig:
    dtype = _FakeDataFormat()
    shape = (1, 1)
    block_count = 2
    tile = (16, 16)


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


class _FakeTTNN:
    def __init__(self):
        self.create_calls = []
        self.generic_op_calls = []
        self.next_address = 0x1000
        self.fabric_setup_calls = []
        self.fabric_route_calls = []
        self.fabric_config = "linear"
        self.fabric_route_infos = {}

    class CoreCoord:
        def __init__(self, x, y):
            self.x = x
            self.y = y

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

    class MeshCoordinate:
        def __init__(self, *coords):
            if len(coords) == 1 and isinstance(coords[0], (tuple, list)):
                coords = tuple(coords[0])
            self.coords = tuple(coords)

        def __eq__(self, other):
            return (
                isinstance(other, _FakeTTNN.MeshCoordinate)
                and self.coords == other.coords
            )

    class MeshCoordinateRange:
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __eq__(self, other):
            return (
                isinstance(other, _FakeTTNN.MeshCoordinateRange)
                and self.start == other.start
                and self.end == other.end
            )

        def __hash__(self):
            return hash((self.start.coords, self.end.coords))

    class MeshProgramDescriptor:
        def __init__(self):
            self.mesh_programs = []

        def __setitem__(self, key, value):
            self.mesh_programs.append((key, value))

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
            self.runtime_args = defaultdict(lambda: defaultdict(list))

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

    @staticmethod
    def cb_descriptor_from_sharded_tensor(cb_index, tensor, total_size, core_ranges):
        return {
            "cb_index": cb_index,
            "tensor": tensor,
            "total_size": total_size,
            "core_ranges": core_ranges,
        }

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

    def setup_routing_plane_connection(
        self,
        source_node_id,
        destination_node_ids,
        link_indices,
        program_descriptor,
        kernel_index,
        worker_node,
    ):
        self.fabric_setup_calls.append(
            (
                source_node_id,
                destination_node_ids,
                link_indices,
                kernel_index,
                (worker_node.x, worker_node.y),
            )
        )
        return [0xA0, 0xB0]

    def get_fabric_route_info(
        self, source_node_id, destination_node_id, link_index=None
    ):
        self.fabric_route_calls.append(
            (source_node_id, destination_node_id, link_index)
        )
        return self.fabric_route_infos.get(
            destination_node_id,
            _FakeFabricRouteInfo(
                connection_node_id=destination_node_id,
                direction=1,
                link_index=2,
                hop_count=3,
            ),
        )

    def get_fabric_config(self):
        return self.fabric_config


class _FakeFabricNodeId(NamedTuple):
    mesh_id: int
    chip_id: int


class _FakeFabricRouteInfo(NamedTuple):
    connection_node_id: _FakeFabricNodeId
    direction: int
    link_index: int
    hop_count: int


class _FakeMeshDevice:
    shape = (1, 2)

    @staticmethod
    def get_fabric_node_id(coordinate):
        return _FakeFabricNodeId(0, coordinate.coords[-1])


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


def test_build_kernel_descriptors_appends_per_kernel_runtime_args(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    tensor = _FakeTensor(object(), address=0x2000)
    specs = [
        kernel_runner.KernelSpec(
            path="/tmp/reader.cpp",
            thread_type="noc",
            tensor_indices=[0],
            config=object(),
            extra_common_runtime_args=[0x4000, 0x4004],
        ),
        kernel_runner.KernelSpec(
            path="/tmp/compute.cpp",
            thread_type="compute",
            tensor_indices=[0],
            config=object(),
        ),
    ]

    descriptors = kernel_runner.build_kernel_descriptors(
        kernel_specs=specs,
        tensors=[tensor],
        tensor_accessor_args=[],
        core_ranges=object(),
        grid_cols=1,
        grid_rows=1,
        num_cbs=0,
        extra_common_runtime_args=[0x3000],
        expected_extra_common_runtime_args=1,
    )

    assert descriptors[0].common_runtime_args == [0x2000, 0x3000, 0x4000, 0x4004]
    assert descriptors[1].common_runtime_args == [0x2000, 0x3000]


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


def test_device_domain_builds_per_device_runtime_coordinates(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 0
    )
    tensor = _FakeTensor(object(), address=0x2000)
    spec = kernel_runner.KernelSpec(
        path="/tmp/kernel.cpp",
        thread_type="noc",
        tensor_indices=[0],
        config=object(),
    )

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[spec],
        tensors=[tensor],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
        device_domain=DeviceDomain((1, 2)),
    )

    mesh_programs = result["program"].mesh_programs
    assert len(mesh_programs) == 2
    assert mesh_programs[0][1] is not mesh_programs[1][1]
    assert mesh_programs[0][1].kernels[0].common_runtime_args == [0x2000, 0, 0]
    assert mesh_programs[1][1].kernels[0].common_runtime_args == [0x2000, 0, 1]


def test_routing_plane_runtime_args_are_dense_per_device(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    kernel = _FakeTTNN.KernelDescriptor(
        kernel_source="/tmp/kernel.cpp",
        core_ranges=object(),
        compile_time_args=[],
        common_runtime_args=[],
        config=object(),
    )
    program = _FakeTTNN.ProgramDescriptor(kernels=[kernel], cbs=[], semaphores=[])
    routes = [
        kernel_runner.FabricRouteSpec((0, 0), (0, 1), ((0, 0),)),
        kernel_runner.FabricRouteSpec((0, 0), (0, 1), ((0, 0),)),
        kernel_runner.FabricRouteSpec((0, 1), (0, 0), ((1, 0),)),
    ]

    kernel_runner.configure_routing_plane_runtime_args(
        program_descriptor=program,
        kernel_fabric_routes=[routes],
        mesh_device=_FakeMeshDevice(),
        device_coordinates=(0, 0),
        grid_cols=2,
        grid_rows=1,
    )

    assert kernel.runtime_args[0][0] == [1, 0, 0, 0, 3, 3, 0, 0xA0, 0xB0]
    assert kernel.runtime_args[1][0] == [0, 0, 0, 0, 0, 0, 0]
    assert fake_ttnn.fabric_setup_calls == [
        (_FakeFabricNodeId(0, 0), [_FakeFabricNodeId(0, 1)], [2], 0, (0, 0)),
    ]
    assert fake_ttnn.fabric_route_calls == [
        (_FakeFabricNodeId(0, 0), _FakeFabricNodeId(0, 1), None),
    ]


def test_routing_plane_reuses_connection_for_one_direction(monkeypatch):
    fake_ttnn = _FakeTTNN()
    fake_ttnn.fabric_route_infos = {
        _FakeFabricNodeId(0, 1): _FakeFabricRouteInfo(_FakeFabricNodeId(0, 1), 1, 2, 1),
        _FakeFabricNodeId(0, 2): _FakeFabricRouteInfo(_FakeFabricNodeId(0, 1), 1, 2, 3),
    }
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    kernel = _FakeTTNN.KernelDescriptor(
        kernel_source="/tmp/kernel.cpp",
        core_ranges=object(),
        compile_time_args=[],
        common_runtime_args=[],
        config=object(),
    )
    program = _FakeTTNN.ProgramDescriptor(kernels=[kernel], cbs=[], semaphores=[])
    routes = [
        kernel_runner.FabricRouteSpec((0, 0), (0, 1), ((0, 0),)),
        kernel_runner.FabricRouteSpec((0, 0), (0, 2), ((0, 0),)),
    ]

    kernel_runner.configure_routing_plane_runtime_args(
        program_descriptor=program,
        kernel_fabric_routes=[routes],
        mesh_device=_FakeMeshDevice(),
        device_coordinates=(0, 0),
        grid_cols=1,
        grid_rows=1,
    )

    assert kernel.runtime_args[0][0] == [1, 0, 0, 1, 3, 0xA0, 0xB0]
    assert fake_ttnn.fabric_setup_calls == [
        (_FakeFabricNodeId(0, 0), [_FakeFabricNodeId(0, 1)], [2], 0, (0, 0)),
    ]


def test_routing_plane_route_cache_tracks_mesh_and_fabric_config(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    route_cache = kernel_runner._FabricRouteCache()
    routes = [kernel_runner.FabricRouteSpec((0, 0), (0, 1), ((0, 0),))]

    def configure(mesh_device):
        kernel = _FakeTTNN.KernelDescriptor(
            kernel_source="/tmp/kernel.cpp",
            core_ranges=object(),
            compile_time_args=[],
            common_runtime_args=[],
            config=object(),
        )
        program = _FakeTTNN.ProgramDescriptor(kernels=[kernel], cbs=[], semaphores=[])
        kernel_runner.configure_routing_plane_runtime_args(
            program_descriptor=program,
            kernel_fabric_routes=[routes],
            mesh_device=mesh_device,
            device_coordinates=(0, 0),
            grid_cols=1,
            grid_rows=1,
            fabric_route_cache=route_cache,
        )

    first_mesh = _FakeMeshDevice()
    configure(first_mesh)
    configure(first_mesh)
    assert len(fake_ttnn.fabric_route_calls) == 1
    assert len(fake_ttnn.fabric_setup_calls) == 2

    fake_ttnn.fabric_config = "mesh"
    configure(first_mesh)
    assert len(fake_ttnn.fabric_route_calls) == 2

    configure(_FakeMeshDevice())
    assert len(fake_ttnn.fabric_route_calls) == 3
    assert len(fake_ttnn.fabric_setup_calls) == 4


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


def test_run_kernel_with_mesh_program_descriptor(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    tensor = _FakeTensorWithoutDevice()

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[],
        tensors=[tensor],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
        program_hash=5,
        mesh_program_placements=[
            (0, 0),
            kernel_runner.MeshProgramPlacement((0, 1), (0, 3)),
        ],
    )

    mesh_program = result["program"]
    assert isinstance(mesh_program, _FakeTTNN.MeshProgramDescriptor)
    assert len(mesh_program.mesh_programs) == 2
    first_range, first_program = mesh_program.mesh_programs[0]
    second_range, second_program = mesh_program.mesh_programs[1]
    assert first_range == _FakeTTNN.MeshCoordinateRange(
        _FakeTTNN.MeshCoordinate(0, 0),
        _FakeTTNN.MeshCoordinate(0, 0),
    )
    assert second_range == _FakeTTNN.MeshCoordinateRange(
        _FakeTTNN.MeshCoordinate(0, 1),
        _FakeTTNN.MeshCoordinate(0, 3),
    )
    assert first_program is second_program
    assert first_program.kernels == []
    assert first_program.custom_program_hash == 5


def test_build_mesh_program_descriptor_rejects_empty_placements(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())

    with pytest.raises(ValueError, match="mesh_program_placements must not be empty"):
        kernel_runner.build_mesh_program_descriptor(
            program_descriptor=object(),
            mesh_program_placements=[],
        )


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
        ((1, 1), 1, object(), None, 512, 512),
        ((1, 1), 1, object(), None, 800, 800),
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


def test_serialized_dfb_config_requires_current_format():
    with pytest.raises(
        ValueError,
        match="Serialized CB config 0 has 5 fields; regenerate the runner",
    ):
        kernel_runner._get_dfb_descriptor_configs(
            [((1, 1), 2, _FakeDataFormat(), (16, 16), 1024)]
        )


def test_build_cb_descriptors_preserves_subtile_geometry(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    monkeypatch.setattr(
        kernel_runner, "get_min_remaining_l1_for_device", lambda _device: 4096
    )

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensor(object())],
        cb_configs=[_FakeDFBConfig()],
        core_ranges=_FakeCoreRanges(),
    )

    descriptor = descriptors[0]
    format_descriptor = descriptor.format_descriptors[0]
    assert descriptor.total_size == 1024
    assert format_descriptor.page_size == 512
    assert format_descriptor.tile.tile.tile_shape == (16, 16)


def test_emit_runner_source_preserves_subtile_geometry(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())

    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[_FakeDFBConfig()],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "((1, 1), 2, ttnn.bfloat16, (16, 16), 512, 1024)" in source


def test_emit_runner_source_uses_shared_pipe_resource_helpers():
    source = kernel_runner.emit_runner_source(
        kernel_specs=[
            kernel_runner.KernelSpec(
                path="/tmp/reader.cpp",
                thread_type="noc",
                tensor_indices=[],
                config=object(),
                extra_common_runtime_args=[7, 9],
            )
        ],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
        program_hash=-2,
        num_pipe_global_semaphores=3,
    )

    assert "NUM_PIPE_GLOBAL_SEMAPHORES = 3" in source
    assert "PROGRAM_HASH = 18446744073709551614" in source
    assert "MESH_PROGRAM_PLACEMENTS = None" in source
    assert "build_pipe_runtime_resources(" in source
    assert "build_kernel_descriptors(" in source
    assert "build_program_descriptor(" in source
    assert "build_pipe_sync_semaphore_descriptors(" in source
    assert "build_generic_op_io_tensors(" in source
    assert "program_descriptor.custom_program_hash = PROGRAM_HASH" in source
    assert "KERNEL_EXTRA_COMMON_RUNTIME_ARGS = [" in source
    assert "    [7, 9],  # noc" in source
    assert (
        "extra_common_runtime_args=KERNEL_EXTRA_COMMON_RUNTIME_ARGS[kernel_idx]"
        in source
    )
    assert "ttnn.create_global_semaphore(device, core_ranges, 0)" not in source


def test_emit_runner_source_omits_program_hash_by_default():
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "PROGRAM_HASH = None" in source


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


def test_emit_runner_source_with_mesh_program_placements():
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
        mesh_program_placements=[
            (0, 0),
            kernel_runner.MeshProgramPlacement((0, 1), (0, 3)),
        ],
    )

    assert "MeshProgramPlacement" in source
    assert "MESH_PROGRAM_PLACEMENTS = [" in source
    assert "    (0, 0)," in source
    assert "    MeshProgramPlacement((0, 1), (0, 3))," in source
    assert "build_mesh_program_descriptor(" in source
