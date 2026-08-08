# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Device architecture detection for ME2E tests.

Provides logic for detecting device architecture from ttnn device objects.
"""


def get_mock_arch_from_device(device) -> str:
    """
    Detect device architecture from ttnn device to use as mock arch.

    Args:
        device: TTNN device object, or None.

    Returns:
        Architecture string (e.g., "wormhole_b0", "blackhole") for mock system desc.
        Defaults to "wormhole_b0" for compiler-only tests without a device.

    Raises:
        ValueError: If a device is present but its architecture is unsupported
            or cannot be detected.
    """
    if device is None:
        return "wormhole_b0"

    arch_attrs = [
        "arch",
        "architecture",
        "chip_type",
        "device_type",
        "_arch",
        "_architecture",
    ]

    for attr in arch_attrs:
        try:
            arch_value = getattr(device, attr)
        except Exception:
            continue
        if callable(arch_value):
            try:
                arch_value = arch_value()
            except Exception:
                continue
        arch = str(arch_value).lower().rsplit(".", maxsplit=1)[-1]
        if arch == "wormhole_b0":
            return arch
        if arch == "blackhole":
            return arch
        if arch == "quasar":
            raise ValueError(
                "Quasar compute kernels require unsupported Gen2 runtime APIs"
            )
        raise ValueError(f"Unsupported TT device architecture: {arch}")

    raise ValueError("Unsupported or undetectable TT device architecture")
