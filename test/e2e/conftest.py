# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest fixtures for E2E tests.

Provides device management, skip conditions, and test metadata extraction.
"""

import os
from typing import Optional

import pytest

from .utils import TTLCompileException, TTLRuntimeException, TTLGoldenException

# Check for ttnn availability.
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None

# Device caching for efficiency.
_current_device = None
_current_system_type: Optional[str] = None


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
        "--sys-desc",
        action="store",
        default=None,
        help="Path to system descriptor file",
    )
    parser.addoption(
        "--dump-mlir",
        action="store_true",
        default=False,
        help="Save generated MLIR to build directory",
    )


def _get_system_type() -> Optional[str]:
    """
    Detect the system type from the current device.

    Returns:
        System type string ('n150', 'n300', 'p150', 'p300', 'llmbox', 'tg') or None.
    """
    global _current_system_type

    if _current_system_type is not None:
        return _current_system_type

    if not TTNN_AVAILABLE:
        return None

    try:
        device = ttnn.open_device(device_id=0)
        # Default to n150 for single device.
        # Real detection would inspect chip type and count.
        _current_system_type = "n150"
        ttnn.close_device(device)
        return _current_system_type
    except Exception:
        return None


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

    system_type = _get_system_type()

    for item in items:
        # Process skip_target markers.
        for marker in item.iter_markers(name="skip_target"):
            targets = set(marker.args)
            if system_type and system_type in targets:
                reason = marker.kwargs.get("reason", "")
                item.add_marker(
                    pytest.mark.skip(reason=f"Skipped for {system_type}. {reason}")
                )

        # Process only_target markers.
        for marker in item.iter_markers(name="only_target"):
            targets = set(marker.args)
            if system_type and system_type not in targets:
                reason = marker.kwargs.get("reason", "")
                item.add_marker(
                    pytest.mark.skip(reason=f"Only runs on {targets}. {reason}")
                )


@pytest.fixture(scope="session")
def device():
    """
    Session-scoped fixture for Tenstorrent device.

    Opens the device once for all tests and closes it when done.
    """
    global _current_device

    if not TTNN_AVAILABLE:
        pytest.skip("ttnn not available")

    if _current_device is not None:
        yield _current_device
        return

    dev = ttnn.open_device(device_id=0)
    _current_device = dev
    yield dev
    ttnn.close_device(dev)
    _current_device = None


@pytest.fixture(scope="session")
def system_desc_path(request):
    """
    Path to the system descriptor file.

    Uses --sys-desc option if provided, SYSTEM_DESC_PATH env var,
    or generates one from the current device.
    """
    # Check command line option first.
    cli_path = request.config.getoption("--sys-desc")
    if cli_path and os.path.exists(cli_path):
        return cli_path

    # Check environment variable.
    env_path = os.environ.get("SYSTEM_DESC_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Generate system descriptor if not provided.
    try:
        from _ttmlir_runtime import runtime

        system_desc = runtime.get_current_system_desc()
        generated_path = "/tmp/ttlang_e2e_system.ttsys"
        system_desc.store(generated_path)
        return generated_path
    except (ImportError, Exception) as e:
        pytest.skip(f"Cannot get system descriptor: {e}")


@pytest.fixture
def dump_mlir(request):
    """Fixture to get --dump-mlir option value."""
    return request.config.getoption("--dump-mlir")
