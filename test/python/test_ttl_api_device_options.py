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


def _descriptor_candidate(
    name,
    coordinates,
    *,
    cpp_source="kernel body",
    function_attributes=(("ttkernel.thread", "noc"),),
):
    metadata = ttl_api._KernelDescriptorMetadata(
        thread_type="noc",
        noc_role=0,
        math_fidelity=None,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
        unpack_to_dest_fp32=(),
        tensor_indices=(0,),
        runtime_arg_signature=("tensor_address",),
        function_attributes=function_attributes,
        dynamic_noc=False,
    )
    return ttl_api._KernelDescriptorCandidate(
        name=name,
        core_coordinates=coordinates,
        cpp_source=cpp_source,
        metadata=metadata,
        runtime_arg_spec=(),
    )


def test_equivalent_specialized_kernels_share_descriptor():
    candidates = [
        _descriptor_candidate("reader_c0_0", ((0, 0),)),
        _descriptor_candidate("reader_c0_1", ((0, 1),)),
    ]

    groups = ttl_api._group_equivalent_specialized_kernels(candidates)

    assert [[candidate.name for candidate in group] for group in groups] == [
        ["reader_c0_0", "reader_c0_1"]
    ]


def test_descriptor_metadata_difference_prevents_sharing():
    candidates = [
        _descriptor_candidate("reader_c0_0", ((0, 0),)),
        _descriptor_candidate(
            "reader_c0_1",
            ((0, 1),),
            function_attributes=(("ttkernel.thread", "noc"), ("mode", "1")),
        ),
    ]

    groups = ttl_api._group_equivalent_specialized_kernels(candidates)

    assert [[candidate.name for candidate in group] for group in groups] == [
        ["reader_c0_0"],
        ["reader_c0_1"],
    ]


def test_conflicting_specialized_processor_assignment_is_rejected():
    candidates = [
        _descriptor_candidate("reader_a", ((0, 0),), cpp_source="body a"),
        _descriptor_candidate("reader_b", ((0, 0),), cpp_source="body b"),
    ]

    with pytest.raises(ValueError, match="both assign processor"):
        ttl_api._group_equivalent_specialized_kernels(candidates)
