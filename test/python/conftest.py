# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for tt-lang Python tests."""

import atexit
import os
import sys

import pytest

# =============================================================================
# Temp file cleanup for dynamically generated kernels
# =============================================================================
# Source files can't be deleted immediately after loading - inspect.findsource()
# needs them during kernel compilation. Track and cleanup at exit instead.
#
# Set TTLANG_KEEP_GENERATED_KERNELS=1 to preserve temp files for debugging.

temp_kernel_files = []


def _cleanup_temp_kernel_files():
    if os.environ.get("TTLANG_KEEP_GENERATED_KERNELS"):
        if temp_kernel_files:
            print(f"\nPreserving {len(temp_kernel_files)} temp kernel file(s):")
            for path in temp_kernel_files:
                print(f"  {path}")
        return
    for path in temp_kernel_files:
        try:
            os.unlink(path)
        except OSError:
            pass


atexit.register(_cleanup_temp_kernel_files)

# Add test root to path for shared utilities.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ttlang_test_utils import is_hardware_available, is_ttnn_available

# =============================================================================
# Feature detection
# =============================================================================

_ttnn_available = is_ttnn_available()
_hardware_available = is_hardware_available()

# Lit tests that should not be collected by pytest (they have # RUN: directives)
collect_ignore = [
    "conftest.py",
    "test_ttnn_interop_add.py",
    "test_dram_interleaved_add.py",
    "test_large_dram_streaming.py",
    "utils.py",
]


# Set compile-only mode if no hardware.
if not _hardware_available:
    os.environ["TTLANG_COMPILE_ONLY"] = "1"


# =============================================================================
# Pytest markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_device: skip test if no TT device is available"
    )


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
