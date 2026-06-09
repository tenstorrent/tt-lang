# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only tests for ttl.kernel_runner resource allocation helpers."""

import pytest

from ttl import kernel_runner


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

    class KernelDescriptor:
        def __init__(
            self,
            kernel_source,
            core_ranges,
            compile_time_args,
            common_runtime_args,
            config,
        ):
            self.kernel_source = kernel_source
            self.core_ranges = core_ranges
            self.compile_time_args = compile_time_args
            self.common_runtime_args = common_runtime_args
            self.config = config

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


def test_emit_runner_source_uses_shared_pipe_resource_helpers():
    source = kernel_runner.emit_runner_source(
        kernel_specs=[],
        cb_configs=[],
        grid_cols=1,
        grid_rows=1,
        num_tensors=1,
        num_pipe_global_semaphores=3,
    )

    assert "NUM_PIPE_GLOBAL_SEMAPHORES = 3" in source
    assert "build_pipe_runtime_resources(" in source
    assert "build_kernel_descriptors(" in source
    assert "build_pipe_sync_semaphore_descriptors(" in source
    assert "build_generic_op_io_tensors(" in source
    assert "ttnn.create_global_semaphore(device, core_ranges, 0)" not in source
