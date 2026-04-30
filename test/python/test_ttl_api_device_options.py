# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for device target-arch detection used by the TTL Python wrapper."""

from unittest import mock

import pytest

import ttl.ttl_api as ttl_api


class _TensorWithDevice:
    def __init__(self, device):
        self._device = device

    def device(self):
        return self._device


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
