# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test utilities for ME2E tests.

Provides exception types, comparison functions, and helper utilities.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# =============================================================================
# Custom Exceptions for Failure Classification
# =============================================================================


class TTLCompileException(Exception):
    """Raised when TTL MLIR compilation fails."""

    pass


class TTLRuntimeException(Exception):
    """Raised when runtime execution fails."""

    pass


class TTLGoldenException(Exception):
    """Raised when golden comparison fails."""

    pass


# =============================================================================
# ULP (Units of Least Precision) Utilities
# =============================================================================


def ulp(x: torch.Tensor) -> torch.Tensor:
    """
    Return Unit of Least Precision for each element of a given tensor.

    Based on Goldberg's definition:
    "What every computer scientist should know about floating-point arithmetic"
    https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html

    Args:
        x: Input tensor.

    Returns:
        Tensor of ULP values for each element.
    """
    import math

    # Use torch.abs(x) to ensure symmetry ULP(-x) == ULP(x)
    abs_x = torch.abs(x)
    next_val = torch.nextafter(abs_x, torch.tensor(math.inf, dtype=x.dtype))
    ulp_value = next_val - abs_x

    # Special case: if abs_x == dtype.max, then next == inf, leading to ULP(x) == inf
    # Fix by manually calculating ULP at max value
    dtype_max = torch.finfo(x.dtype).max
    max_epsilon = dtype_max - torch.nextafter(
        torch.tensor(dtype_max, dtype=x.dtype),
        torch.tensor(-math.inf, dtype=x.dtype),
    )
    ulp_value = torch.where(abs_x == dtype_max, max_epsilon, ulp_value)

    return ulp_value


def get_default_ulp_threshold(dtype: torch.dtype) -> float:
    """
    Get default ULP threshold for a given dtype.

    Args:
        dtype: PyTorch dtype.

    Returns:
        Maximum acceptable ULP difference.
    """
    if dtype in (torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
        return 0.0  # Exact comparison for integer types

    # ULP thresholds based on dtype precision
    if dtype == torch.bfloat16:
        return 2.0  # BF16: 7-8 bits mantissa, allow 2 ULP
    elif dtype == torch.float16:
        return 2.0  # FP16: 10-11 bits mantissa, allow 2 ULP
    elif dtype == torch.float32:
        return 1.0  # FP32: 23-24 bits mantissa, allow 1 ULP
    elif dtype == torch.float64:
        return 1.0  # FP64: 52-53 bits mantissa, allow 1 ULP
    else:
        return 2.0  # Default: allow 2 ULP


# =============================================================================
# Comparison Result
# =============================================================================


@dataclass
class ComparisonResult:
    """Result of ULP-based tensor comparison."""

    passed: bool
    max_ulp: float
    mean_ulp: float
    message: str = ""


def compare_tensors(
    golden: torch.Tensor,
    calculated: torch.Tensor,
    ulp_threshold: Optional[float] = None,
) -> ComparisonResult:
    """
    Compare two tensors using ULP (Units of Least Precision).

    Measures error in representable floating-point values. Hardware-accurate
    and scale-independent.

    Args:
        golden: Expected tensor.
        calculated: Actual tensor.
        ulp_threshold: Max ULP difference (auto-computed from dtype if None).

    Returns:
        ComparisonResult with pass/fail status and ULP metrics.
    """
    ulp_threshold = ulp_threshold or get_default_ulp_threshold(golden.dtype)

    # Handle empty tensors
    if golden.numel() == 0 and calculated.numel() == 0:
        return ComparisonResult(True, 0.0, 0.0, "Both tensors empty")

    # Compute ULP error
    # ULP is measured according to the golden tensor
    ulp_value = ulp(golden.type(calculated.dtype))

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)
        ulp_value = ulp_value.type(golden.dtype)

    # Handle non-finite values
    mask_finite = torch.isfinite(golden) & torch.isfinite(calculated)
    if not torch.all(mask_finite):
        # Check if non-finite values match
        both_nan = torch.isnan(golden) & torch.isnan(calculated)
        both_inf = (
            torch.isinf(golden)
            & torch.isinf(calculated)
            & (torch.sign(golden) == torch.sign(calculated))
        )
        nonfinite_match = both_nan | both_inf
        if not torch.all(mask_finite | nonfinite_match):
            return ComparisonResult(
                False, float("inf"), float("inf"), "Non-finite values mismatch"
            )

    # Compute ULP differences (only for finite values)
    ulp_tensor = torch.abs(calculated - golden) / ulp_value
    ulp_tensor = torch.where(
        mask_finite, ulp_tensor, torch.tensor(0.0, dtype=ulp_tensor.dtype)
    )

    max_ulp = torch.max(ulp_tensor).item()
    mean_ulp = (
        torch.mean(ulp_tensor[mask_finite]).item() if torch.any(mask_finite) else 0.0
    )

    # Check threshold
    if max_ulp <= ulp_threshold:
        return ComparisonResult(True, max_ulp, mean_ulp, "")

    # Failed: build detailed error message
    idx = torch.argmax(ulp_tensor)
    pos = tuple(int(i) for i in torch.unravel_index(idx, golden.shape))
    msg = (
        f"ULP {max_ulp:.2f} > {ulp_threshold:.2f} @ {list(pos)}: "
        f"|{calculated[pos]:.6f} - {golden[pos]:.6f}| / {ulp_value[pos]:.6e}"
    )
    return ComparisonResult(False, max_ulp, mean_ulp, msg)


def compare_tensors_ulp(
    result: torch.Tensor,
    golden: torch.Tensor,
) -> tuple[float, float]:
    """
    Compute ULP statistics for tensor comparison.

    Simple utility that returns (max_ulp, mean_ulp) for use in assertions.

    Args:
        result: Actual tensor.
        golden: Expected tensor.

    Returns:
        Tuple of (max_ulp, mean_ulp).
    """
    comparison = compare_tensors(golden, result)
    return comparison.max_ulp, comparison.mean_ulp
