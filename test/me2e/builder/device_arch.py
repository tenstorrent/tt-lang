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

    # Try to detect architecture from device.
    # Common attributes to check (may vary by ttnn version):
    arch_attrs = [
        "arch",
        "architecture",
        "chip_type",
        "device_type",
        "_arch",
        "_architecture",
    ]

    for attr in arch_attrs:
        if hasattr(device, attr):
            arch_value = getattr(device, attr)
            # Try calling if it's a method/function.
            if callable(arch_value):
                try:
                    arch_value = arch_value()
                except Exception:
                    continue
            # Convert to string for comparison (handles enums, strings, etc.)
            arch_str = str(arch_value)
            arch_lower = arch_str.lower()
            if "wormhole" in arch_lower:
                return "wormhole_b0"
            if "blackhole" in arch_lower:
                return "blackhole"
            if "quasar" in arch_lower:
                return "quasar"

    raise ValueError("Unsupported or undetectable TT device architecture")
