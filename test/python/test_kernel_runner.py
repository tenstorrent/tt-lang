# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only tests for ttl.kernel_runner resource allocation helpers."""

import pytest

from ttl import kernel_runner
from ttl.dataflow_buffer import PhysicalDFBConfig


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
