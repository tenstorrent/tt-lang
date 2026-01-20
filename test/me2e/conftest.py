# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest fixtures for ME2E tests.

Provides device management, skip conditions, and test metadata extraction.
"""

import os
import sys

import pytest

# Add test root to path for shared utilities.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ttlang_test_utils import is_hardware_available, is_ttnn_available

TTNN_AVAILABLE = is_ttnn_available()


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


def _normalize_nodeid_to_test_me2e_format(file_path: str) -> str | None:
    """
    Normalize a pytest nodeid file path to test.me2e.* format.

    Args:
        file_path: File path from pytest nodeid (e.g., "ops/test_binary.py",
                   "test_compute_ops.py", "test/me2e/test_compute_ops.py").

    Returns:
        Normalized module path in test.me2e.* format, or None if normalization fails.
    """
    if file_path.startswith("ops/"):
        # Relative: ops/test_binary.py -> test.me2e.ops.test_binary
        return "test.me2e." + file_path.replace("/", ".").replace(".py", "")
    elif file_path.startswith("test/me2e/"):
        # Absolute: test/me2e/test_compute_ops.py -> test.me2e.test_compute_ops
        return file_path.replace("/", ".").replace(".py", "")
    elif file_path.startswith("me2e/"):
        # Partial: me2e/test_compute_ops.py -> test.me2e.test_compute_ops
        return "test." + file_path.replace("/", ".").replace(".py", "")
    elif "/" not in file_path and file_path.endswith(".py"):
        # No directory prefix: test_compute_ops.py -> test.me2e.test_compute_ops
        # Assume it's in test/me2e directory when run from there.
        return "test.me2e." + file_path.replace(".py", "")
    else:
        # Cannot normalize to test.me2e.* format.
        return None


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection based on platform.

    Handles:
    - Skipping tests if ttnn unavailable
    - skip_target marker processing
    - only_target marker processing
    - Marking tests from XFAILS list as xfail
    """
    if not TTNN_AVAILABLE:
        skip_ttnn = pytest.mark.skip(reason="ttnn not available")
        for item in items:
            item.add_marker(skip_ttnn)
        return

    # Import XFAILS dictionary.
    try:
        from .ops.XFAILS import XFAIL_TESTS as xfail_dict
    except ImportError:
        xfail_dict = {}

    # Check for hardware availability.
    hardware_available = is_hardware_available()
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

        # Mark tests from XFAILS list as xfail.
        # Convert item.nodeid to test.me2e.* format and match against XFAILS dictionary.
        # XFAILS format: "test.me2e.ops.test_binary::TestAddFloat32::test_validate_golden"
        nodeid_parts = item.nodeid.split("::")
        if len(nodeid_parts) >= 2 and xfail_dict:
            file_path = nodeid_parts[0]
            test_suffix = "::".join(nodeid_parts[1:])

            # Normalize file path to test.me2e.* format.
            module_path = _normalize_nodeid_to_test_me2e_format(file_path)
            if module_path is None:
                # Skip if we can't normalize to test.me2e.* format.
                continue

            # Build identifier in test.me2e.* format.
            test_identifier = "::".join([module_path, test_suffix])

            # Match against XFAILS dictionary.
            xfail_reason = xfail_dict.get(test_identifier)
            if xfail_reason:
                item.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=False))


@pytest.fixture
def device():
    """
    Fixture that provides a TTNN device, skipping if unavailable.

    Matches the pattern from test/python/conftest.py for consistency.
    """
    if not TTNN_AVAILABLE:
        pytest.skip("ttnn not available")

    if not is_hardware_available():
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
