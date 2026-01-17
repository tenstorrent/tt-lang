# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for tt-lang Python tests."""

import glob
import importlib.util
import os

import pytest

# Lit tests that should not be collected by pytest (they have # RUN: directives)
collect_ignore = [
    "conftest.py",
    "test_ttnn_interop_add.py",
    "test_dram_interleaved_add.py",
    "test_large_dram_streaming.py",
    "utils.py",
]

# =============================================================================
# Feature detection
# =============================================================================

_ttnn_available = False
if importlib.util.find_spec("ttnn") is not None:
    _ttnn_available = True

# Check for hardware: simulator, env var, or physical device
if os.environ.get("TT_METAL_SIMULATOR"):
    _hardware_available = True
elif os.environ.get("TTLANG_HAS_DEVICE") == "1":
    _hardware_available = True
elif glob.glob("/dev/tenstorrent*"):
    _hardware_available = True
else:
    _hardware_available = False

# Set compile-only mode if no hardware
if not _hardware_available:
    os.environ["TTLANG_COMPILE_ONLY"] = "1"


# =============================================================================
# Pytest markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_ttnn: skip test if ttnn is not available"
    )
    config.addinivalue_line(
        "markers", "requires_device: skip test if no TT device is available"
    )
    config.addinivalue_line(
        "markers", "skip_if_wormhole: skip test if running on Wormhole B0"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available features."""
    skip_ttnn = pytest.mark.skip(reason="TTNN not available")
    skip_device = pytest.mark.skip(reason="No Tenstorrent device available")

    # Import here to avoid circular imports and only when needed
    try:
        from test_helpers import is_wormhole_b0

        skip_wormhole = pytest.mark.skip(reason="Test skipped on Wormhole B0")
    except ImportError:
        is_wormhole_b0 = None
        skip_wormhole = None

    for item in items:
        if "requires_ttnn" in item.keywords and not _ttnn_available:
            item.add_marker(skip_ttnn)
        if "requires_device" in item.keywords and not _hardware_available:
            item.add_marker(skip_device)
        if (
            "skip_if_wormhole" in item.keywords
            and is_wormhole_b0 is not None
            and is_wormhole_b0()
        ):
            item.add_marker(skip_wormhole)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ttnn_device():
    """Fixture that provides a TTNN device, skipping if unavailable."""
    if not _ttnn_available:
        pytest.skip("TTNN not available")
    if not _hardware_available:
        pytest.skip("No Tenstorrent device available")

    import ttnn

    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


# Alias for convenience - most tests use 'device' as the fixture name
@pytest.fixture
def device(ttnn_device):
    """Alias for ttnn_device fixture."""
    return ttnn_device
