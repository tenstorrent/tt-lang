# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Common test utilities for tt-lang Python tests.

Provides consistent TTNN import handling and device availability checking.
When no hardware is available, TTLANG_COMPILE_ONLY is set automatically.

Device availability is determined at CMake configure time by checking for
/dev/tenstorrent* files, avoiding the slow ttnn.GetNumAvailableDevices() call.
"""

import os
import sys

# Check device availability from CMake-generated config (fast path)
# Falls back to checking environment if config not available
_hardware_available = False

try:
    # Try to import CMake-generated config first (fast - no ttnn import needed)
    from test_config import HAS_TT_DEVICE

    _hardware_available = HAS_TT_DEVICE
except ImportError:
    # Config not available (running outside build dir) - check env or device files
    import glob

    if os.environ.get("TTLANG_HAS_DEVICE") == "1":
        _hardware_available = True
    elif glob.glob("/dev/tenstorrent*"):
        _hardware_available = True

# Set compile-only mode if no hardware
if not _hardware_available:
    os.environ["TTLANG_COMPILE_ONLY"] = "1"

# Try to import TTNN
ttnn = None
_ttnn_available = False

try:
    import ttnn as _ttnn

    ttnn = _ttnn
    _ttnn_available = True
except ImportError:
    pass


def require_ttnn():
    """Exit test if TTNN is not available."""
    if not _ttnn_available:
        print("TTNN not available - exiting")
        sys.exit(0)


def is_hardware_available():
    """Check if Tenstorrent hardware is available for running kernels."""
    return _hardware_available


def is_ttnn_available():
    """Check if TTNN library is available."""
    return _ttnn_available


def skip_without_hardware(message: str = "Skipping - no hardware available"):
    """Print message and exit if no hardware is available.

    Use this at the start of `if __name__ == "__main__":` blocks in tests
    that need to run kernels on actual hardware.
    """
    if not _hardware_available:
        print(message)
        sys.exit(0)


# =============================================================================
# Tensor comparison utilities
# =============================================================================


def assert_pcc(golden, actual, threshold=0.99):
    """Assert Pearson correlation coefficient between tensors exceeds threshold.

    Args:
        golden: Expected tensor values
        actual: Actual tensor values from computation
        threshold: Minimum PCC required (default 0.99)

    Raises:
        AssertionError: If PCC is below threshold
    """
    import torch

    golden_flat = golden.flatten().float()
    actual_flat = actual.flatten().float()

    # Handle constant tensors (no variance)
    if golden_flat.std() == 0 and actual_flat.std() == 0:
        # Both constant - check if same constant
        if torch.allclose(golden_flat, actual_flat):
            return 1.0
        else:
            raise AssertionError(
                f"Both tensors are constant but differ: "
                f"golden={golden_flat[0].item()}, actual={actual_flat[0].item()}"
            )

    if golden_flat.std() == 0 or actual_flat.std() == 0:
        raise AssertionError(
            f"Cannot compute PCC: one tensor is constant "
            f"(golden std={golden_flat.std()}, actual std={actual_flat.std()})"
        )

    # Compute Pearson correlation
    golden_centered = golden_flat - golden_flat.mean()
    actual_centered = actual_flat - actual_flat.mean()

    numerator = (golden_centered * actual_centered).sum()
    denominator = torch.sqrt((golden_centered**2).sum() * (actual_centered**2).sum())

    pcc = numerator / denominator

    if pcc < threshold:
        raise AssertionError(
            f"PCC {pcc:.6f} is below threshold {threshold}. "
            f"Golden: mean={golden_flat.mean():.4f}, std={golden_flat.std():.4f}. "
            f"Actual: mean={actual_flat.mean():.4f}, std={actual_flat.std():.4f}."
        )

    return pcc.item()


def assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, verbose=True):
    """Assert tensors are element-wise close within tolerance.

    Args:
        actual: Actual tensor from computation
        expected: Expected tensor values
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: If True, print diff stats on failure

    Raises:
        AssertionError: If tensors differ beyond tolerance
    """
    import torch

    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = (actual - expected).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        msg = (
            f"Tensors not close: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
            f"rtol={rtol}, atol={atol}"
        )

        if verbose:
            # Find location of max difference
            max_idx = diff.argmax().item()
            actual_val = actual.flatten()[max_idx].item()
            expected_val = expected.flatten()[max_idx].item()
            msg += (
                f"\nMax diff at flat index {max_idx}: "
                f"actual={actual_val:.6e}, expected={expected_val:.6e}"
            )

        raise AssertionError(msg)
