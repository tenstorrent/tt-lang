# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only tests for ttl.kernel_runner resource allocation helpers."""

from types import SimpleNamespace

import pytest

from ttl import kernel_runner
from ttl.dataflow_buffer import CompilerAllocatedDFBConfig


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


class _FakeCoreCoord:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeCoreRange:
    def __init__(self, start, end):
        self.start = _FakeCoreCoord(*start)
        self.end = _FakeCoreCoord(*end)


class _FakeExplicitCoreRanges:
    def __init__(self, *ranges):
        self._ranges = [_FakeCoreRange(start, end) for start, end in ranges]

    def ranges(self):
        return self._ranges

    def bounding_box(self):
        max_x = max(core_range.end.x for core_range in self._ranges)
        max_y = max(core_range.end.y for core_range in self._ranges)
        return SimpleNamespace(
            grid_size=lambda: SimpleNamespace(x=max_x + 1, y=max_y + 1)
        )


class _FakeSubsetCoreRanges:
    def __init__(self, members):
        self.members = set(members)

    def contains(self, core):
        return core in self.members


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

    CoreCoord = _FakeCoreCoord

    class CoreRange:
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __hash__(self):
            return hash((self.start.x, self.start.y, self.end.x, self.end.y))

    class CoreRangeSet:
        def __init__(self, ranges):
            self._ranges = list(ranges)

        def ranges(self):
            return self._ranges

    class ReaderConfigDescriptor:
        pass

    class WriterConfigDescriptor:
        pass

    class KernelDescriptor:
        def __init__(
            self,
            kernel_source,
            core_ranges,
            compile_time_args,
            defines,
            runtime_args,
            common_runtime_args,
            config,
            compiler_include_paths,
        ):
            self.kernel_source = kernel_source
            self.core_ranges = core_ranges
            self.compile_time_args = compile_time_args
            self.defines = defines
            self.runtime_args = runtime_args
            self.common_runtime_args = common_runtime_args
            self.config = config
            self.compiler_include_paths = compiler_include_paths

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


def _fake_cb_descriptor(
    cb_id,
    core_ranges,
    total_size,
    *,
    page_size=32,
    tile=(1, 32),
    data_format="bf16",
):
    return SimpleNamespace(
        total_size=total_size,
        core_ranges=core_ranges,
        format_descriptors=[
            SimpleNamespace(
                buffer_index=cb_id,
                data_format=data_format,
                page_size=page_size,
                tile=SimpleNamespace(height=tile[0], width=tile[1]),
            )
        ],
    )


def test_remaining_l1_uses_absolute_cb_and_page_addresses(monkeypatch):
    l1 = object()
    reports = SimpleNamespace(
        get_device_info=lambda _device: SimpleNamespace(
            address_at_first_l1_cb_buffer=0x4A000,
            cb_limit=0x130000,
        ),
        get_buffer_pages=lambda _device: [
            SimpleNamespace(buffer_type=l1, page_address=0xEC000),
            SimpleNamespace(buffer_type=l1, page_address=0xF0000),
            SimpleNamespace(buffer_type=object(), page_address=0x1000),
        ],
    )
    fake_ttnn = SimpleNamespace(
        BufferType=SimpleNamespace(L1=l1),
        _ttnn=SimpleNamespace(reports=reports),
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)

    assert kernel_runner.get_min_remaining_l1_for_device(object()) == (
        0xEC000 - 0x4A000
    )


def test_remaining_l1_without_tensors_uses_cb_address_interval(monkeypatch):
    reports = SimpleNamespace(
        get_device_info=lambda _device: SimpleNamespace(
            address_at_first_l1_cb_buffer=0x4A000,
            cb_limit=0x130000,
        ),
        get_buffer_pages=lambda _device: [],
    )
    fake_ttnn = SimpleNamespace(
        BufferType=SimpleNamespace(L1=object()),
        _ttnn=SimpleNamespace(reports=reports),
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)

    assert kernel_runner.get_min_remaining_l1_for_device(object()) == (
        0x130000 - 0x4A000
    )


def test_remaining_l1_by_core_does_not_conflate_tensor_placements(monkeypatch):
    l1 = object()
    reports = SimpleNamespace(
        get_device_info=lambda _device: SimpleNamespace(
            address_at_first_l1_cb_buffer=0x4A000,
            cb_limit=0x130000,
        ),
        get_buffer_pages=lambda _device: [
            SimpleNamespace(
                buffer_type=l1,
                page_address=0x6A000,
                core_x=0,
                core_y=0,
            ),
            SimpleNamespace(
                buffer_type=l1,
                page_address=0xEA000,
                core_x=1,
                core_y=0,
            ),
        ],
    )
    fake_ttnn = SimpleNamespace(
        BufferType=SimpleNamespace(L1=l1),
        _ttnn=SimpleNamespace(reports=reports),
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)

    assert kernel_runner.get_remaining_l1_by_core_for_device(
        object(), {(0, 0), (1, 0), (2, 0)}
    ) == {
        (0, 0): 0x20000,
        (1, 0): 0xA0000,
        (2, 0): 0xE6000,
    }


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


def test_build_kernel_descriptors_filters_specialized_runtime_args(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    selected_ranges = _FakeSubsetCoreRanges({"core-1"})
    spec = kernel_runner.KernelSpec(
        path="/tmp/kernel.cpp",
        thread_type="noc",
        tensor_indices=[],
        config=fake_ttnn.WriterConfigDescriptor(),
        core_ranges=selected_ranges,
    )

    descriptors = kernel_runner.build_kernel_descriptors(
        kernel_specs=[spec],
        tensors=[],
        tensor_accessor_args=[],
        core_ranges=_FakeCoreRanges(),
        grid_cols=2,
        grid_rows=1,
        num_cbs=0,
        runtime_args_by_thread={
            "brisc": [("core-0", [10]), ("core-1", [11])]
        },
    )

    assert descriptors[0].runtime_args == [("core-1", [11])]


def test_specialized_cb_descriptor_follows_active_kernel_core(monkeypatch):

    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(kernel_runner, "DEFAULT_L1_CB_BUDGET_BYTES", 128)
    full_grid = _FakeExplicitCoreRanges(((0, 0), (1, 0)))
    active_core = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    inactive_core = _FakeExplicitCoreRanges(((1, 0), (1, 0)))
    geometry = SimpleNamespace(
        data_format="bf16",
        page_size=32,
        num_pages=4,
        total_size=128,
        tile_descriptor=SimpleNamespace(height=1, width=32),
    )
    monkeypatch.setattr(
        kernel_runner, "cb_geometry", lambda _index, _cb: geometry
    )
    specs = [
        kernel_runner.KernelSpec(
            path="/tmp/active_core_uses_cb.cpp",
            thread_type="noc",
            tensor_indices=[],
            config=fake_ttnn.ReaderConfigDescriptor(),
            core_ranges=active_core,
            used_cb_indices=[0],
        ),
        kernel_runner.KernelSpec(
            path="/tmp/inactive_core_has_cb_branch_elided.cpp",
            thread_type="noc",
            tensor_indices=[],
            config=fake_ttnn.ReaderConfigDescriptor(),
            core_ranges=inactive_core,
            used_cb_indices=[],
        ),
    ]

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=specs,
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[object()],
        core_ranges=full_grid,
    )

    assert [descriptor.core_ranges for descriptor in result["program"].kernels] == [
        active_core,
        inactive_core,
    ]
    assert len(result["program"].cbs) == 1
    assert result["program"].cbs[0].total_size == 128
    assert kernel_runner._core_range_coordinates(
        result["program"].cbs[0].core_ranges, label="CB[0]"
    ) == {(0, 0)}


def test_unannotated_kernel_keeps_whole_grid_cb_descriptor(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    full_grid = _FakeExplicitCoreRanges(((0, 0), (1, 0)))
    geometry = SimpleNamespace(
        data_format="bf16",
        page_size=32,
        num_pages=1,
        total_size=32,
        tile_descriptor=None,
    )
    monkeypatch.setattr(
        kernel_runner, "cb_geometry", lambda _index, _cb: geometry
    )

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[object()],
        core_ranges=full_grid,
        kernel_specs=[
            kernel_runner.KernelSpec(
                path="/tmp/unspecialized.cpp",
                thread_type="noc",
                tensor_indices=[],
                config=fake_ttnn.ReaderConfigDescriptor(),
                core_ranges=full_grid,
            )
        ],
    )

    assert len(descriptors) == 1
    assert descriptors[0].core_ranges is full_grid


def test_specialized_cb_descriptors_allow_sparse_physical_ids(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    full_grid = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    geometry = SimpleNamespace(
        data_format="bf16",
        page_size=32,
        num_pages=1,
        total_size=32,
        tile_descriptor=None,
    )
    monkeypatch.setattr(
        kernel_runner, "cb_geometry", lambda _index, _cb: geometry
    )

    descriptors = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[object(), object(), object()],
        core_ranges=full_grid,
        kernel_specs=[
            kernel_runner.KernelSpec(
                path="/tmp/sparse_cb.cpp",
                thread_type="noc",
                tensor_indices=[],
                config=fake_ttnn.ReaderConfigDescriptor(),
                core_ranges=full_grid,
                used_cb_indices=[2],
            )
        ],
    )

    assert [
        descriptor.format_descriptors[0].buffer_index
        for descriptor in descriptors
    ] == [2]


def test_specialized_cb_use_intersects_per_role_capacities(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    full_grid = _FakeExplicitCoreRanges(((0, 0), (2, 0)))
    shared = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    routed = _FakeExplicitCoreRanges(((1, 0), (1, 0)))
    inactive = _FakeExplicitCoreRanges(((2, 0), (2, 0)))
    geometry = SimpleNamespace(
        data_format="bf16",
        page_size=32,
        num_pages=3,
        total_size=96,
        tile_descriptor=None,
    )
    monkeypatch.setattr(
        kernel_runner, "cb_geometry", lambda _index, _cb: geometry
    )
    specs = [
        kernel_runner.KernelSpec(
            path="/tmp/shared_situ.cpp",
            thread_type="noc",
            tensor_indices=[],
            config=fake_ttnn.ReaderConfigDescriptor(),
            core_ranges=shared,
            used_cb_indices=[0],
        ),
        kernel_runner.KernelSpec(
            path="/tmp/routed_situ.cpp",
            thread_type="noc",
            tensor_indices=[],
            config=fake_ttnn.ReaderConfigDescriptor(),
            core_ranges=routed,
            used_cb_indices=[0],
        ),
        kernel_runner.KernelSpec(
            path="/tmp/inactive.cpp",
            thread_type="noc",
            tensor_indices=[],
            config=fake_ttnn.ReaderConfigDescriptor(),
            core_ranges=inactive,
            used_cb_indices=[],
        ),
    ]

    descriptors = kernel_runner.build_cb_descriptors_by_core(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[object()],
        core_ranges=full_grid,
        pages_by_core={0: [(shared, 2), (routed, 3), (inactive, 1)]},
        kernel_specs=specs,
    )

    claims = {
        next(
            iter(
                kernel_runner._core_range_coordinates(
                    descriptor.core_ranges, label="CB[0]"
                )
            )
        ): descriptor.total_size
        for descriptor in descriptors
    }
    assert claims == {(0, 0): 64, (1, 0): 96}


def test_serialized_layout_builder_matches_live_specialized_ranges(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    full_grid = _FakeExplicitCoreRanges(((0, 0), (1, 0)))
    active = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    inactive = _FakeExplicitCoreRanges(((1, 0), (1, 0)))
    geometry = SimpleNamespace(
        data_format="bf16",
        page_size=32,
        num_pages=2,
        total_size=64,
        tile_descriptor=None,
    )
    monkeypatch.setattr(
        kernel_runner, "cb_geometry", lambda _index, _cb: geometry
    )
    specs = [
        kernel_runner.KernelSpec(
            path="/tmp/active.cpp",
            thread_type="noc",
            tensor_indices=[],
            config=fake_ttnn.ReaderConfigDescriptor(),
            core_ranges=active,
            used_cb_indices=[0],
        ),
        kernel_runner.KernelSpec(
            path="/tmp/inactive.cpp",
            thread_type="noc",
            tensor_indices=[],
            config=fake_ttnn.ReaderConfigDescriptor(),
            core_ranges=inactive,
            used_cb_indices=[],
        ),
    ]

    live = kernel_runner.build_cb_descriptors(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[object()],
        core_ranges=full_grid,
        kernel_specs=specs,
    )
    emitted = kernel_runner.build_cb_descriptors_from_layouts(
        tensors=[_FakeTensorWithoutDevice()],
        cb_layouts=[((1, 2), 1, "bf16", 32, 64)],
        core_ranges=full_grid,
        kernel_specs=specs,
    )

    def descriptor_contract(descriptors):
        return [
            (
                descriptor.format_descriptors[0].buffer_index,
                descriptor.total_size,
                kernel_runner._core_range_coordinates(
                    descriptor.core_ranges, label="CB"
                ),
            )
            for descriptor in descriptors
        ]

    assert descriptor_contract(emitted) == descriptor_contract(live)


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


def test_run_kernel_applies_runtime_resource_factory(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    tensor = _FakeTensorWithoutDevice()
    owner = object()
    lifetime = []
    reader_args = [("core-0", [3, 2, 4, 5])]

    def factory(*, tensors, core_ranges, first_free_semaphore_id):
        assert tensors == [tensor]
        assert isinstance(core_ranges, _FakeCoreRanges)
        assert first_free_semaphore_id == 0
        return kernel_runner.ProgramRuntimeResources(
            semaphore_descriptors=["fabric-sem-0", "fabric-sem-1"],
            runtime_args_by_thread={"ncrisc": reader_args},
            defines_by_thread={"ncrisc": [("FABRIC_1D", "1")]},
            lifetimes=[owner],
        )

    spec = kernel_runner.KernelSpec(
        path="/tmp/kernel.cpp",
        thread_type="noc",
        tensor_indices=[],
        config=fake_ttnn.ReaderConfigDescriptor(),
    )
    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[spec],
        tensors=[tensor],
        cb_configs=[],
        core_ranges=_FakeCoreRanges(),
        runtime_resource_factory=factory,
        runtime_resource_lifetime=lifetime,
    )

    assert result["program"].kernels[0].runtime_args == reader_args
    assert result["program"].kernels[0].defines == [("FABRIC_1D", "1")]
    assert result["program"].semaphores == ["fabric-sem-0", "fabric-sem-1"]
    assert lifetime == [owner]


def test_cb_descriptor_override_accounts_for_disjoint_cores(monkeypatch):
    monkeypatch.setattr(kernel_runner, "DEFAULT_L1_CB_BUDGET_BYTES", 128)
    program_cores = _FakeExplicitCoreRanges(((0, 0), (1, 0)))
    descriptors = [
        _fake_cb_descriptor(
            0, _FakeExplicitCoreRanges(((0, 0), (0, 0))), total_size=96
        ),
        _fake_cb_descriptor(
            0, _FakeExplicitCoreRanges(((1, 0), (1, 0))), total_size=128
        ),
    ]

    result = kernel_runner.validate_cb_descriptors_override(
        descriptors=descriptors,
        program_core_ranges=program_cores,
        tensors=[_FakeTensorWithoutDevice()],
        num_cbs=1,
    )

    # The descriptor total is 224 bytes, but no core owns more than 128.
    assert result == descriptors


def test_cb_descriptor_override_rejects_overlapping_id_on_one_core(monkeypatch):
    monkeypatch.setattr(kernel_runner, "DEFAULT_L1_CB_BUDGET_BYTES", 128)
    core = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    descriptors = [
        _fake_cb_descriptor(0, core, total_size=32),
        _fake_cb_descriptor(0, core, total_size=64),
    ]

    with pytest.raises(ValueError, match="overlapping descriptors"):
        kernel_runner.validate_cb_descriptors_override(
            descriptors=descriptors,
            program_core_ranges=core,
            tensors=[_FakeTensorWithoutDevice()],
            num_cbs=1,
        )


def test_cb_descriptor_override_rejects_mismatched_page_format(monkeypatch):
    monkeypatch.setattr(kernel_runner, "DEFAULT_L1_CB_BUDGET_BYTES", 128)
    program_cores = _FakeExplicitCoreRanges(((0, 0), (1, 0)))
    descriptors = [
        _fake_cb_descriptor(
            0,
            _FakeExplicitCoreRanges(((0, 0), (0, 0))),
            total_size=64,
            page_size=32,
        ),
        _fake_cb_descriptor(
            0,
            _FakeExplicitCoreRanges(((1, 0), (1, 0))),
            total_size=64,
            page_size=64,
        ),
    ]

    with pytest.raises(ValueError, match="inconsistent page formats"):
        kernel_runner.validate_cb_descriptors_override(
            descriptors=descriptors,
            program_core_ranges=program_cores,
            tensors=[_FakeTensorWithoutDevice()],
            num_cbs=1,
        )


def test_cb_descriptor_override_rejects_per_core_budget_overflow(monkeypatch):
    monkeypatch.setattr(kernel_runner, "DEFAULT_L1_CB_BUDGET_BYTES", 128)
    core = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    descriptors = [
        _fake_cb_descriptor(0, core, total_size=96),
        _fake_cb_descriptor(1, core, total_size=64),
    ]

    with pytest.raises(ValueError, match="160 bytes on core \\(0, 0\\)"):
        kernel_runner.validate_cb_descriptors_override(
            descriptors=descriptors,
            program_core_ranges=core,
            tensors=[_FakeTensorWithoutDevice()],
            num_cbs=2,
        )


def test_cb_descriptor_override_uses_matching_per_core_l1_budget(monkeypatch):
    l1 = object()
    device = object()
    reports = SimpleNamespace(
        get_device_info=lambda _device: SimpleNamespace(
            address_at_first_l1_cb_buffer=0,
            cb_limit=256,
        ),
        get_buffer_pages=lambda _device: [
            SimpleNamespace(
                buffer_type=l1,
                page_address=128,
                core_x=0,
                core_y=0,
            ),
            SimpleNamespace(
                buffer_type=l1,
                page_address=192,
                core_x=1,
                core_y=0,
            ),
        ],
    )
    fake_ttnn = SimpleNamespace(
        BufferType=SimpleNamespace(L1=l1),
        _ttnn=SimpleNamespace(reports=reports),
    )
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    program = _FakeExplicitCoreRanges(((0, 0), (1, 0)))
    core_0 = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    core_1 = _FakeExplicitCoreRanges(((1, 0), (1, 0)))

    descriptors = kernel_runner.validate_cb_descriptors_override(
        descriptors=[
            _fake_cb_descriptor(0, core_0, total_size=96),
            _fake_cb_descriptor(0, core_1, total_size=160),
        ],
        program_core_ranges=program,
        tensors=[_FakeTensor(device)],
        num_cbs=1,
    )

    assert [descriptor.total_size for descriptor in descriptors] == [96, 160]


def test_cb_pages_by_core_specializes_selected_ids(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    monkeypatch.setattr(kernel_runner, "DEFAULT_L1_CB_BUDGET_BYTES", 160)
    program = _FakeExplicitCoreRanges(((0, 0), (1, 0)))
    core_0 = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    core_1 = _FakeExplicitCoreRanges(((1, 0), (1, 0)))
    geometries = {
        0: SimpleNamespace(
            data_format="bf16",
            page_size=32,
            num_pages=4,
            total_size=128,
            tile_descriptor=SimpleNamespace(height=1, width=32),
        ),
        1: SimpleNamespace(
            data_format="bf16",
            page_size=32,
            num_pages=1,
            total_size=32,
            tile_descriptor=SimpleNamespace(height=1, width=32),
        ),
    }
    monkeypatch.setattr(
        kernel_runner, "cb_geometry", lambda index, _cb: geometries[index]
    )

    descriptors = kernel_runner.build_cb_descriptors_by_core(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[object(), object()],
        core_ranges=program,
        pages_by_core={0: [(core_0, 1), (core_1, 4)]},
    )

    # Uniform descriptors stay first so their base addresses remain identical
    # across cores; specialized descriptors follow.
    assert [descriptor.total_size for descriptor in descriptors] == [
        32,
        32,
        32,
        128,
    ]
    assert [
        descriptor.format_descriptors[0].buffer_index
        for descriptor in descriptors
    ] == [1, 1, 0, 0]


def test_cb_pages_by_core_requires_exact_grid_partition(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    program = _FakeExplicitCoreRanges(((0, 0), (1, 0)))
    core_0 = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    geometry = SimpleNamespace(
        data_format="bf16",
        page_size=32,
        num_pages=1,
        total_size=32,
        tile_descriptor=SimpleNamespace(height=1, width=32),
    )
    monkeypatch.setattr(kernel_runner, "cb_geometry", lambda _index, _cb: geometry)

    with pytest.raises(ValueError, match="must cover the whole program grid"):
        kernel_runner.build_cb_descriptors_by_core(
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[object()],
            core_ranges=program,
            pages_by_core={0: [(core_0, 1)]},
        )


def test_cb_pages_by_core_preserves_compiler_maximum(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    program = _FakeExplicitCoreRanges(((0, 0), (1, 0)))
    geometry = SimpleNamespace(
        data_format="bf16",
        page_size=32,
        num_pages=4,
        total_size=128,
        tile_descriptor=SimpleNamespace(height=1, width=32),
    )
    monkeypatch.setattr(kernel_runner, "cb_geometry", lambda _index, _cb: geometry)

    with pytest.raises(ValueError, match="preserve the compiler-derived maximum"):
        kernel_runner.build_cb_descriptors_by_core(
            tensors=[_FakeTensorWithoutDevice()],
            cb_configs=[object()],
            core_ranges=program,
            pages_by_core={0: [(program, 2)]},
        )


def test_cb_pages_by_core_preserves_specialization_order(monkeypatch):
    monkeypatch.setattr(kernel_runner, "ttnn", _FakeTTNN())
    program = _FakeExplicitCoreRanges(((0, 0), (1, 0)))
    core_0 = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    core_1 = _FakeExplicitCoreRanges(((1, 0), (1, 0)))
    geometries = {
        0: SimpleNamespace(
            data_format="bf16",
            page_size=32,
            num_pages=4,
            total_size=128,
            tile_descriptor=SimpleNamespace(height=1, width=32),
        ),
        1: SimpleNamespace(
            data_format="bf16",
            page_size=32,
            num_pages=2,
            total_size=64,
            tile_descriptor=SimpleNamespace(height=1, width=32),
        ),
    }
    monkeypatch.setattr(
        kernel_runner, "cb_geometry", lambda index, _cb: geometries[index]
    )

    descriptors = kernel_runner.build_cb_descriptors_by_core(
        tensors=[_FakeTensorWithoutDevice()],
        cb_configs=[object(), object()],
        core_ranges=program,
        pages_by_core={
            1: [(core_0, 2), (core_1, 1)],
            0: [(core_0, 1), (core_1, 4)],
        },
    )

    assert [
        descriptor.format_descriptors[0].buffer_index
        for descriptor in descriptors
    ] == [1, 1, 0, 0]


def test_run_kernel_rejects_both_cb_override_forms(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    tensor = _FakeTensorWithoutDevice()
    core = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    descriptor = _fake_cb_descriptor(0, core, total_size=32)

    def factory(**_kwargs):
        return kernel_runner.ProgramRuntimeResources(
            cb_descriptors_override=[descriptor],
            cb_pages_by_core={0: [(core, 1)]},
        )

    with pytest.raises(ValueError, match="cannot set both"):
        kernel_runner.run_kernel_on_device(
            kernel_specs=[],
            tensors=[tensor],
            cb_configs=[object()],
            core_ranges=core,
            runtime_resource_factory=factory,
        )


def test_run_kernel_uses_validated_cb_descriptor_override(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    tensor = _FakeTensorWithoutDevice()
    core_ranges = _FakeExplicitCoreRanges(((0, 0), (0, 0)))
    descriptor = _fake_cb_descriptor(0, core_ranges, total_size=32)

    def fail_default_builder(**_kwargs):
        raise AssertionError("default whole-grid CB builder should not run")

    def factory(**_kwargs):
        return kernel_runner.ProgramRuntimeResources(
            cb_descriptors_override=[descriptor]
        )

    monkeypatch.setattr(kernel_runner, "build_cb_descriptors", fail_default_builder)

    result = kernel_runner.run_kernel_on_device(
        kernel_specs=[],
        tensors=[tensor],
        cb_configs=[object()],
        core_ranges=core_ranges,
        runtime_resource_factory=factory,
    )

    assert result["program"].cbs == [descriptor]


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


def test_emit_runner_source_omits_program_hash_by_default():
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "PROGRAM_HASH = None" in source


def test_emit_runner_source_preserves_specialized_cb_use(monkeypatch):
    fake_ttnn = _FakeTTNN()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    source = kernel_runner.emit_runner_source(
        kernel_specs=[
            kernel_runner.KernelSpec(
                path="/tmp/sparse.cpp",
                thread_type="noc",
                tensor_indices=[],
                config=fake_ttnn.ReaderConfigDescriptor(),
                core_ranges=_FakeExplicitCoreRanges(((1, 0), (1, 0))),
                used_cb_indices=[2],
            )
        ],
        cb_configs=[],
        grid_cols=2,
        grid_rows=1,
        num_tensors=1,
    )

    assert "KERNEL_USED_CB_INDICES = [" in source
    assert "    [2],  # noc" in source
    assert "used_cb_indices=KERNEL_USED_CB_INDICES[kernel_idx]" in source
    assert "build_cb_descriptors_from_layouts(" in source
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


def test_emit_runner_source_preserves_subtile_page_size(monkeypatch):
    class FakeTile:
        def __init__(self, shape):
            self.shape = shape

        def get_tile_size(self, _data_format):
            return self.shape[0] * self.shape[1] * 2

    fake_ttnn = SimpleNamespace(Tile=FakeTile)
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    cb = SimpleNamespace(
        dtype=SimpleNamespace(name="bfloat16"),
        tile=(8, 32),
        shape=(1, 9),
        block_count=4,
    )

    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[cb],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "((1, 9), 4, ttnn.bfloat16, 512, 18432),  # CB 0" in source


def test_emit_runner_source_handles_compiler_allocated_cb(monkeypatch):
    data_format = SimpleNamespace(name="bfloat16")
    monkeypatch.setattr(
        kernel_runner,
        "format_name_to_ttnn_dtype",
        lambda _name: data_format,
    )
    monkeypatch.setattr(
        kernel_runner,
        "tile_bytes_from_dtype",
        lambda _dtype: 2048,
    )
    cb = CompilerAllocatedDFBConfig(
        dfb_index=0,
        num_tiles=3,
        data_format="bfloat16",
        block_count=2,
    )

    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[cb],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
    )

    assert "((1, 3), 2, ttnn.bfloat16, 2048, 12288),  # CB 0" in source


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
