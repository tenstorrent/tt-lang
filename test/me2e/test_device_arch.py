# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ME2E target architecture detection and propagation."""

import pytest

import ttl.dialects.ttl as ttl
from ttl.ir import Context, Module

from .builder.device_arch import get_mock_arch_from_device
from .builder.pipeline import compile_ttl_to_ttkernel


class FakeDevice:
    def __init__(self, architecture: str):
        self.architecture = architecture


class DeviceWithRaisingArch:
    @property
    def arch(self):
        raise RuntimeError("device handle closed")


@pytest.mark.parametrize(
    "architecture, expected",
    [
        ("Arch.WORMHOLE_B0", "wormhole_b0"),
        ("Arch.BLACKHOLE", "blackhole"),
    ],
)
def test_detect_supported_device_architecture(architecture, expected):
    assert get_mock_arch_from_device(FakeDevice(architecture)) == expected


def test_compiler_only_architecture_defaults_to_wormhole():
    assert get_mock_arch_from_device(None) == "wormhole_b0"


def test_unknown_device_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unsupported TT device architecture"):
        get_mock_arch_from_device(FakeDevice("Arch.UNKNOWN"))


def test_undetectable_device_architecture_is_rejected():
    with pytest.raises(
        ValueError, match="Unsupported or undetectable TT device architecture"
    ):
        get_mock_arch_from_device(DeviceWithRaisingArch())


def test_quasar_device_architecture_is_rejected():
    with pytest.raises(ValueError, match="unsupported Gen2 runtime APIs"):
        get_mock_arch_from_device(FakeDevice("Arch.QUASAR"))


def test_pipeline_attaches_typed_target_architecture():
    context = Context()
    ttl.ensure_dialects_registered(context)

    with context:
        module = Module.parse("module {}", context)
        compile_ttl_to_ttkernel(
            module, FakeDevice("Arch.BLACKHOLE"), maximize_dst=False
        )

        assert (
            str(module.operation.attributes["ttl.target_arch"])
            == "#ttcore.arch<blackhole>"
        )
