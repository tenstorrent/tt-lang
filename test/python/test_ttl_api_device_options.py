# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TT device options used by the TTL Python wrapper."""

from unittest import mock

import pytest

import ttl.dialects.ttl as ttl
import ttl.ttl_api as ttl_api
from ttl import ProgramRuntimeResources
from ttl.constants import SUPPORTED_MATH_FIDELITIES
from ttl.ir import Context, Module


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

    def test_quasar_arch_is_rejected(self):
        device = _DeviceWithArchMethod("Arch.QUASAR")
        with pytest.raises(ValueError, match="Unsupported TT device architecture"):
            ttl_api._device_target_arch((_TensorWithDevice(device),))

    def test_unknown_arch_is_rejected(self):
        device = _DeviceWithArchAttribute("future_arch")
        with pytest.raises(ValueError, match="Unsupported TT device architecture"):
            ttl_api._device_target_arch((_TensorWithDevice(device),))

    def test_no_recognized_arch_attribute_is_rejected(self):
        with pytest.raises(
            ValueError, match="Unsupported or undetectable TT device architecture"
        ):
            ttl_api._device_target_arch((_TensorWithDevice(object()),))

    def test_no_tensor_args_returns_none(self):
        assert ttl_api._device_target_arch(()) is None

    def test_raising_arch_attribute_is_rejected(self):
        with pytest.raises(
            ValueError, match="Unsupported or undetectable TT device architecture"
        ):
            ttl_api._device_target_arch((_TensorWithDevice(_DeviceWithRaisingArch()),))

    def test_different_device_architectures_are_rejected(self):
        args = (
            _TensorWithDevice(_DeviceWithArchAttribute("Arch.WORMHOLE_B0")),
            _TensorWithDevice(_DeviceWithArchAttribute("Arch.BLACKHOLE")),
        )
        with pytest.raises(ValueError, match="different TT device architectures"):
            ttl_api._device_target_arch(args)


@pytest.mark.parametrize("logical_selectors", [None, [], [None]])
def test_resource_factory_requires_complete_logical_selectors(logical_selectors):
    with pytest.raises(
        ValueError,
        match="runtime_resource_factory requires .*logical-kernel selector",
    ):
        ttl_api.CompiledTTNNKernel(
            kernel_paths=[("kernel.cpp", "compute")],
            kernel_configs=[object()],
            kernel_arg_specs=[[]],
            num_tensors=1,
            core_ranges=object(),
            kernel_tensor_indices=[[]],
            kernel_logical_selectors=logical_selectors,
            operation_name="resource_operation",
            runtime_resource_factory=lambda **_kwargs: ProgramRuntimeResources(),
        )


class TestKernelI32ArrayAttr:
    def test_optional_attribute_may_be_absent(self):
        context = Context()
        ttl.ensure_dialects_registered(context)

        with context:
            module = Module.parse("module { func.func @reader() { return } }")
            assert (
                ttl_api._get_optional_kernel_i32_array_attr(
                    module, "reader", "ttl.pipe_computed_address_dfb_indices"
                )
                == []
            )

    def test_optional_attribute_is_read_when_present(self):
        context = Context()
        ttl.ensure_dialects_registered(context)

        with context:
            module = Module.parse(
                """
                module {
                  func.func @reader() attributes {
                    ttl.pipe_computed_address_dfb_indices = array<i32: 2, 5>
                  } {
                    return
                  }
                }
                """
            )
            assert ttl_api._get_optional_kernel_i32_array_attr(
                module, "reader", "ttl.pipe_computed_address_dfb_indices"
            ) == [2, 5]

    def test_optional_attribute_is_validated_when_present(self):
        context = Context()
        ttl.ensure_dialects_registered(context)

        with context:
            module = Module.parse(
                """
                module {
                  func.func @reader() attributes {
                    ttl.pipe_computed_address_dfb_indices = 2 : i32
                  } {
                    return
                  }
                }
                """
            )
            with pytest.raises(ValueError, match="Expected DenseI32ArrayAttr"):
                ttl_api._get_optional_kernel_i32_array_attr(
                    module, "reader", "ttl.pipe_computed_address_dfb_indices"
                )

    def test_required_attribute_must_be_present(self):
        context = Context()
        ttl.ensure_dialects_registered(context)

        with context:
            module = Module.parse(
                """
                module {
                  func.func @compute_kernel() attributes {
                    dst_full_sync_en = false,
                    fp32_dest_acc_en = false
                  } {
                    return
                  }
                }
                """
            )
            with pytest.raises(
                ValueError,
                match="Required compiler-generated attribute "
                "'ttl.unpack_to_dest_fp32' is missing",
            ):
                ttl_api._get_kernel_i32_array_attr(
                    module, "compute_kernel", "ttl.unpack_to_dest_fp32"
                )


class TestMathFidelity:
    @pytest.mark.parametrize("math_fidelity", SUPPORTED_MATH_FIDELITIES)
    def test_maps_supported_value(self, math_fidelity):
        fidelity_value = object()
        ttnn_module = mock.Mock()
        setattr(ttnn_module.MathFidelity, math_fidelity, fidelity_value)
        config = mock.Mock()

        ttl_api._set_math_fidelity(config, ttnn_module, math_fidelity)

        assert config.math_fidelity is fidelity_value

    def test_missing_ttnn_value_is_rejected(self):
        ttnn_module = mock.Mock()
        del ttnn_module.MathFidelity.HiFi4

        with pytest.raises(RuntimeError, match="does not provide MathFidelity.HiFi4"):
            ttl_api._set_math_fidelity(mock.Mock(), ttnn_module, "HiFi4")
