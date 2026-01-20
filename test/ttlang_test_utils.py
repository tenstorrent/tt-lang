# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared test utilities for tt-lang test suite.

Provides unified feature detection (ttnn availability, hardware detection),
tensor creation helpers, and comparison utilities. Used across pytest conftest
files, lit configuration, and test scripts.

Device availability is determined by checking environment variables and
/dev/tenstorrent* files, avoiding the slow ttnn.GetNumAvailableDevices() call.
"""

import glob
import importlib.util
import os
import sys

# =============================================================================
# Feature detection
# =============================================================================

# Check device availability from CMake-generated config (fast path).
# Falls back to checking environment if config not available.
_hardware_available = False

try:
    # Try to import CMake-generated config first (fast - no ttnn import needed).
    from test_config import HAS_TT_DEVICE

    _hardware_available = HAS_TT_DEVICE
except ImportError:
    # Config not available (running outside build dir) - check env or device files.
    if os.environ.get("TT_METAL_SIMULATOR"):
        _hardware_available = True
    elif os.environ.get("TTLANG_HAS_DEVICE") == "1":
        _hardware_available = True
    elif glob.glob("/dev/tenstorrent*"):
        _hardware_available = True

# Set compile-only mode if no hardware.
if not _hardware_available:
    os.environ["TTLANG_COMPILE_ONLY"] = "1"

# Check if TTNN is available (lightweight check without importing).
_ttnn_available = False
try:
    _ttnn_available = importlib.util.find_spec("ttnn") is not None
except Exception:
    pass

ttnn = None  # Lazy import - loaded when first needed


def _get_ttnn():
    """Lazy import of ttnn module."""
    global ttnn, _ttnn_available
    if ttnn is None and _ttnn_available:
        try:
            import ttnn as _ttnn

            ttnn = _ttnn
        except (ImportError, ModuleNotFoundError):
            _ttnn_available = False
            ttnn = None
    return ttnn


def is_ttnn_available() -> bool:
    """
    Check if ttnn module is available without importing it.

    Uses importlib.util.find_spec for lightweight detection that avoids
    the overhead of actually importing ttnn (which can be slow).

    Returns:
        True if ttnn can be imported, False otherwise.
    """
    return _ttnn_available


def is_hardware_available() -> bool:
    """
    Check if Tenstorrent hardware is available.

    Checks in order:
    1. TT_METAL_SIMULATOR environment variable (simulation mode)
    2. TTLANG_HAS_DEVICE environment variable (set by CMake)
    3. Physical device files (/dev/tenstorrent*)

    Returns:
        True if hardware or simulator is available, False otherwise.
    """
    return _hardware_available


def require_ttnn():
    """Exit test if TTNN is not available."""
    if not _ttnn_available:
        print("TTNN not available - exiting")
        sys.exit(0)


def require_hardware(message: str = "Skipping test - no hardware available"):
    """Exit early if no hardware available.

    Use this at the start of `if __name__ == "__main__":` blocks in tests
    that need access to Tenstorrent hardware (even just for compilation).

    Note: This does NOT check the TTLANG_COMPILE_ONLY env var - tests can still compile
    kernels in compile-only mode, they just won't execute on device.
    """
    if not _hardware_available:
        print(message)
        sys.exit(0)


# =============================================================================
# Tensor creation utilities
# =============================================================================


def to_dram(torch_tensor, device):
    """Create a TTNN tensor in DRAM from a torch tensor.

    Args:
        torch_tensor: Source torch tensor (typically bfloat16)
        device: TTNN device handle

    Returns:
        TTNN tensor in DRAM with TILE_LAYOUT
    """
    ttnn = _get_ttnn()
    if ttnn is None:
        raise RuntimeError("TTNN not available")
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def to_l1(torch_tensor, device):
    """Create a TTNN tensor in L1 from a torch tensor.

    Creates in DRAM first then moves to L1 (required by TTNN).

    Args:
        torch_tensor: Source torch tensor (typically bfloat16)
        device: TTNN device handle

    Returns:
        TTNN tensor in L1 with TILE_LAYOUT
    """
    ttnn = _get_ttnn()
    if ttnn is None:
        raise RuntimeError("TTNN not available")
    dram_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.to_memory_config(dram_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)


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

    # Handle constant tensors (no variance).
    if golden_flat.std() == 0 and actual_flat.std() == 0:
        # Both constant - check if same constant.
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

    # Compute Pearson correlation.
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
            # Find location of max difference.
            max_idx = diff.argmax().item()
            actual_val = actual.flatten()[max_idx].item()
            expected_val = expected.flatten()[max_idx].item()
            msg += (
                f"\nMax diff at flat index {max_idx}: "
                f"actual={actual_val:.6e}, expected={expected_val:.6e}"
            )

        raise AssertionError(msg)


__all__ = [
    "is_ttnn_available",
    "is_hardware_available",
    "require_ttnn",
    "require_hardware",
    "to_dram",
    "to_l1",
    "assert_pcc",
    "assert_allclose",
]
