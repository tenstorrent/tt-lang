# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for tt-lang Python tests."""

import os
import sys

import pytest

# Add test root to path for shared utilities.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ttlang_test_utils import is_hardware_available, is_ttnn_available

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

_ttnn_available = is_ttnn_available()
_hardware_available = is_hardware_available()

# Set compile-only mode if no hardware.
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


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available features."""
    skip_ttnn = pytest.mark.skip(reason="TTNN not available")
    skip_device = pytest.mark.skip(reason="No Tenstorrent device available")

    for item in items:
        if "requires_ttnn" in item.keywords and not _ttnn_available:
            item.add_marker(skip_ttnn)
        if "requires_device" in item.keywords and not _hardware_available:
            item.add_marker(skip_device)


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
