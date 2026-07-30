# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for device options used by the TTL Python wrapper."""

from unittest import mock

import pytest

import ttl
import ttl.ttl_api as ttl_api


class _TensorWithDevice:
    def __init__(self, device):
        self._device = device

    def device(self):
        return self._device


class _GridSize:
    x = 1
    y = 1


class _BoundingBox:
    @staticmethod
    def grid_size():
        return _GridSize()


class _CoreRanges:
    @staticmethod
    def bounding_box():
        return _BoundingBox()


class _DeviceGrid:
    x = 8
    y = 8


class _DeviceWithMeshShape:
    def __init__(self, shape):
        self.shape = shape

    @staticmethod
    def compute_with_storage_grid_size():
        return _DeviceGrid()


class _DeviceWithArchMethod:
    def __init__(self, arch):
        self._arch = arch

    def arch(self):
        return self._arch


class _DeviceWithArchAttribute:
    def __init__(self, arch):
        self.arch = arch


class _DeviceWithRaisingArch:
    @property
    def arch(self):
        raise RuntimeError("device handle closed")


class TestDeviceTargetArch:
    @pytest.fixture(autouse=True)
    def _patch_tensor_detection(self):
        with mock.patch.object(
            ttl_api, "is_ttnn_tensor", lambda arg: isinstance(arg, _TensorWithDevice)
        ):
            yield

    def test_arch_method(self):
        device = _DeviceWithArchMethod("Arch.BLACKHOLE")
        assert ttl_api._device_target_arch((_TensorWithDevice(device),)) == "blackhole"

    def test_arch_attribute(self):
        device = _DeviceWithArchAttribute("Arch.WORMHOLE_B0")
        assert (
            ttl_api._device_target_arch((_TensorWithDevice(device),)) == "wormhole_b0"
        )

    def test_arch_without_dot_prefix(self):
        device = _DeviceWithArchAttribute("BLACKHOLE")
        assert ttl_api._device_target_arch((_TensorWithDevice(device),)) == "blackhole"

    def test_unknown_arch_returns_normalized_string(self):
        device = _DeviceWithArchAttribute("future_arch")
        assert (
            ttl_api._device_target_arch((_TensorWithDevice(device),)) == "future_arch"
        )

    def test_no_recognized_arch_attribute_returns_none(self):
        assert ttl_api._device_target_arch((_TensorWithDevice(object()),)) is None

    def test_no_tensor_args_returns_none(self):
        assert ttl_api._device_target_arch(()) is None

    def test_raising_arch_attribute_returns_none(self):
        # hasattr() swallows the AttributeError-or-otherwise; detection
        # falls through to the next attribute and ultimately returns None
        # when none resolve.
        assert (
            ttl_api._device_target_arch((_TensorWithDevice(_DeviceWithRaisingArch()),))
            is None
        )


class TestMeshProgramPlacement:
    @pytest.fixture(autouse=True)
    def _patch_tensor_detection(self):
        with mock.patch.object(
            ttl_api, "is_ttnn_tensor", lambda arg: isinstance(arg, _TensorWithDevice)
        ):
            yield

    def test_default_mesh_program_placement_covers_mesh(self):
        tensor = _TensorWithDevice(_DeviceWithMeshShape((2, 4)))

        placements = ttl_api._default_mesh_program_placements((tensor,))

        assert len(placements) == 1
        assert placements[0].start == (0, 0)
        assert placements[0].end == (1, 3)

    def test_default_mesh_program_placement_skips_single_device(self):
        tensor = _TensorWithDevice(_DeviceWithMeshShape((1, 1)))

        assert ttl_api._default_mesh_program_placements((tensor,)) is None

    def test_device_domain_mesh_program_placement_covers_domain(self):
        domain = ttl.DeviceDomain((1, 4))

        placements = ttl_api._default_mesh_program_placements_with_domain((), domain)

        assert len(placements) == 1
        assert placements[0].start == (0, 0)
        assert placements[0].end == (0, 3)

    def test_device_domain_mesh_program_placement_matches_mesh_tensor(self):
        domain = ttl.DeviceDomain((1, 4))
        tensor = _TensorWithDevice(_DeviceWithMeshShape((1, 4)))

        placements = ttl_api._default_mesh_program_placements_with_domain(
            (tensor,), domain
        )

        assert placements == [ttl_api.MeshProgramPlacement((0, 0), (0, 3))]

    def test_device_domain_mesh_program_placement_rejects_mismatch(self):
        domain = ttl.DeviceDomain((1, 2))
        tensor = _TensorWithDevice(_DeviceWithMeshShape((1, 4)))

        with pytest.raises(ValueError, match="does not match"):
            ttl_api._default_mesh_program_placements_with_domain((tensor,), domain)

    def test_device_domain_mesh_program_placement_supports_product(self):
        domain = ttl.DeviceDomain.product(board=(1,), device=(4,))

        placements = ttl_api._default_mesh_program_placements_with_domain((), domain)

        assert placements == [ttl_api.MeshProgramPlacement((0, 0), (0, 3))]

    def test_compile_kernel_forwards_device_domain_to_lowering(self, monkeypatch):
        domain = ttl.DeviceDomain((1, 2))
        calls = []

        monkeypatch.setattr(ttl_api, "_get_registered_threads", lambda: [object()])
        monkeypatch.setattr(ttl_api, "_build_operation_pipenets", lambda *_: object())
        monkeypatch.setattr(ttl_api, "_collect_cb_configs", lambda *_: [])

        def fake_lower_program_to_kernel(**kwargs):
            calls.append(kwargs)
            return "compiled"

        monkeypatch.setattr(
            ttl_api, "_lower_program_to_kernel", fake_lower_program_to_kernel
        )

        def kernel():
            pass

        result = ttl_api._compile_kernel(
            kernel,
            (),
            {},
            (1, 1),
            [],
            [],
            0,
            "L1",
            True,
            0,
            device_domain=domain,
        )

        assert result == "compiled"
        assert calls[0]["device_domain"] is domain
        assert calls[0]["mesh_program_placements"] == [
            ttl_api.MeshProgramPlacement((0, 0), (0, 1))
        ]

    def test_compiled_kernel_forwards_mesh_program_placements(self, monkeypatch):
        placement = ttl_api.MeshProgramPlacement((0, 0), (0, 3))
        calls = []

        def fake_run_kernel_on_device(**kwargs):
            calls.append(kwargs)
            return "result"

        monkeypatch.setattr(ttl_api, "run_kernel_on_device", fake_run_kernel_on_device)
        compiled_kernel = ttl_api.CompiledTTNNKernel(
            kernel_paths=[],
            kernel_configs=[],
            kernel_arg_specs=[],
            num_tensors=1,
            core_ranges=_CoreRanges(),
            kernel_tensor_indices=[],
            mesh_program_placements=[placement],
        )

        result = compiled_kernel(_TensorWithDevice(_DeviceWithMeshShape((1, 4))))

        assert result == "result"
        assert calls[0]["mesh_program_placements"] == [placement]
        assert (
            calls[0]["fabric_direction_cache"]
            is compiled_kernel._fabric_direction_cache
        )
