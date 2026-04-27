# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for device-derived TTL compiler options."""

from unittest import mock

import pytest

import ttl.ttl_api as ttl_api
from ttl.compiler_options import CompilerOptions


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


class TestDeviceCompilerDefaults:
    @pytest.fixture(autouse=True)
    def _patch_tensor_detection(self):
        with mock.patch.object(
            ttl_api, "is_ttnn_tensor", lambda arg: isinstance(arg, _TensorWithDevice)
        ):
            yield

    def test_default_reduce_full_fp32_stays_enabled_on_blackhole(self):
        device = _DeviceWithArchMethod("Arch.BLACKHOLE")
        opts = ttl_api._effective_compiler_options_for_device(
            CompilerOptions(), (_TensorWithDevice(device),)
        )
        assert opts.reduce_full_fp32 is True

    def test_default_reduce_full_fp32_disabled_on_non_blackhole(self):
        device = _DeviceWithArchAttribute("Arch.WORMHOLE_B0")
        opts = ttl_api._effective_compiler_options_for_device(
            CompilerOptions(), (_TensorWithDevice(device),)
        )
        assert opts.reduce_full_fp32 is False

    def test_explicit_reduce_full_fp32_preserved_on_non_blackhole(self):
        device = _DeviceWithArchAttribute("Arch.WORMHOLE_B0")
        opts = ttl_api._effective_compiler_options_for_device(
            CompilerOptions.from_string("--ttl-reduce-full-fp32"),
            (_TensorWithDevice(device),),
        )
        assert opts.reduce_full_fp32 is True

    def test_device_target_arch_uses_arch_method(self):
        device = _DeviceWithArchMethod("Arch.BLACKHOLE")
        assert ttl_api._device_target_arch((_TensorWithDevice(device),)) == "blackhole"

    def test_device_target_arch_uses_arch_attribute(self):
        device = _DeviceWithArchAttribute("Arch.WORMHOLE_B0")
        assert (
            ttl_api._device_target_arch((_TensorWithDevice(device),)) == "wormhole_b0"
        )

    def test_unknown_arch_preserves_default_options(self):
        opts = ttl_api._effective_compiler_options_for_device(
            CompilerOptions(), (_TensorWithDevice(object()),)
        )
        assert opts == CompilerOptions()
