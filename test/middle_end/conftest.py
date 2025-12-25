# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest fixtures for middle-end tests.

Provides device management, skip conditions, and common test utilities.
"""

import os

import pytest

# Check for ttnn availability.
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_device: mark test as requiring Tenstorrent hardware"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require unavailable dependencies."""
    if not TTNN_AVAILABLE:
        skip_ttnn = pytest.mark.skip(reason="ttnn not available")
        for item in items:
            item.add_marker(skip_ttnn)


@pytest.fixture(scope="session")
def device():
    """
    Session-scoped fixture for Tenstorrent device.

    Opens the device once for all tests and closes it when done.
    """
    if not TTNN_AVAILABLE:
        pytest.skip("ttnn not available")

    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="session")
def system_desc_path():
    """
    Path to the system descriptor file.

    Uses SYSTEM_DESC_PATH environment variable if set, otherwise
    generates one from the current device.
    """
    path = os.environ.get("SYSTEM_DESC_PATH")
    if path and os.path.exists(path):
        return path

    # Generate system descriptor if not provided.
    try:
        from _ttmlir_runtime import runtime

        system_desc = runtime.get_current_system_desc()
        generated_path = "/tmp/ttlang_test_system.ttsys"
        system_desc.store(generated_path)
        return generated_path
    except (ImportError, Exception) as e:
        pytest.skip(f"Cannot get system descriptor: {e}")


@pytest.fixture
def assert_close():
    """
    Fixture providing tensor comparison with appropriate tolerances.

    Returns a function that compares tensors with bfloat16-appropriate tolerances.
    """
    import torch

    def _assert_close(
        actual: torch.Tensor,
        expected: torch.Tensor,
        rtol: float = 5e-2,
        atol: float = 1e-1,
    ):
        """Compare tensors with tolerance appropriate for bfloat16."""
        # Convert to float32 for comparison to avoid bfloat16 precision issues.
        actual_f32 = actual.float()
        expected_f32 = expected.float()

        if not torch.allclose(actual_f32, expected_f32, rtol=rtol, atol=atol):
            max_abs_diff = (actual_f32 - expected_f32).abs().max().item()
            max_rel_diff = (
                ((actual_f32 - expected_f32).abs() / (expected_f32.abs() + 1e-8))
                .max()
                .item()
            )
            pytest.fail(
                f"Tensor mismatch:\n"
                f"  Max absolute diff: {max_abs_diff:.6f} (tol: {atol})\n"
                f"  Max relative diff: {max_rel_diff:.6f} (tol: {rtol})\n"
                f"  Shape: {actual.shape}"
            )

    return _assert_close

