# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest fixtures for E2E tests.

Provides device management, skip conditions, and test metadata extraction.
"""

import pytest

# Check for ttnn availability.
try:
    import importlib.util

    TTNN_AVAILABLE = importlib.util.find_spec("ttnn") is not None
    ttnn = None  # Don't import at module level
except Exception:
    TTNN_AVAILABLE = False
    ttnn = None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_device: mark test as requiring Tenstorrent hardware"
    )
    config.addinivalue_line(
        "markers",
        "skip_target(*targets): skip test for specific hardware targets",
    )
    config.addinivalue_line(
        "markers",
        "only_target(*targets): run test only on specific hardware targets",
    )
    config.addinivalue_line(
        "markers",
        "order(after=...): specify test execution order (requires pytest-order)",
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--dump-mlir",
        action="store_true",
        default=False,
        help="Save generated MLIR to build directory",
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection based on platform.

    Handles:
    - Skipping tests if ttnn unavailable
    - skip_target marker processing
    - only_target marker processing
    """
    if not TTNN_AVAILABLE:
        skip_ttnn = pytest.mark.skip(reason="ttnn not available")
        for item in items:
            item.add_marker(skip_ttnn)
        return

    # Check for hardware availability (similar to test/python/conftest.py)
    import glob
    import os

    hardware_available = False
    if os.environ.get("TT_METAL_SIMULATOR"):
        hardware_available = True
    elif os.environ.get("TTLANG_HAS_DEVICE") == "1":
        hardware_available = True
    elif glob.glob("/dev/tenstorrent*"):
        hardware_available = True

    skip_device = pytest.mark.skip(reason="No Tenstorrent device available")

    for item in items:
        # Skip tests requiring device if hardware not available.
        if "requires_device" in item.keywords and not hardware_available:
            item.add_marker(skip_device)

        # Process skip_target markers.
        for marker in item.iter_markers(name="skip_target"):
            # Would need device to detect system type - skip for now
            pass

        # Process only_target markers.
        for marker in item.iter_markers(name="only_target"):
            # Would need device to detect system type - skip for now
            pass


@pytest.fixture
def device():
    """
    Fixture that provides a TTNN device, skipping if unavailable.

    Matches the pattern from test/python/conftest.py for consistency.
    """
    if not TTNN_AVAILABLE:
        pytest.skip("ttnn not available")

    # Check for hardware availability
    import glob
    import os

    hardware_available = False
    if os.environ.get("TT_METAL_SIMULATOR"):
        hardware_available = True
    elif os.environ.get("TTLANG_HAS_DEVICE") == "1":
        hardware_available = True
    elif glob.glob("/dev/tenstorrent*"):
        hardware_available = True

    if not hardware_available:
        pytest.skip("No Tenstorrent device available")

    # Import ttnn here (not at module level)
    import ttnn

    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture
def dump_mlir(request):
    """Fixture to get --dump-mlir option value."""
    return request.config.getoption("--dump-mlir")
