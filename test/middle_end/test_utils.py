# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test utilities for middle-end tests.

Provides pytest helpers, exception types, and comparison functions for
the declarative test framework.
"""

from typing import Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass

import numpy as np
import pytest
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
# Marks Class for Declarative Skip/XFail
# =============================================================================


class Marks:
    """
    Fluent interface for pytest marks via | operator.

    Allows declarative annotation of test parameters with skip, xfail, etc.

    Example:
        UNARY_OPS = [
            ComputeOpSpec("exp", ...),
            ComputeOpSpec("sqrt", ...) | Marks(pytest.mark.xfail(reason="domain")),
        ]
    """

    def __init__(self, *marks):
        """
        Initialize with pytest marks.

        Args:
            *marks: Variable number of pytest.mark objects.
        """
        self.marks = marks

    def __ror__(self, lhs):
        """
        Apply marks to a test parameter.

        Args:
            lhs: Test parameter to mark.

        Returns:
            pytest.param: Marked test parameter.
        """
        return pytest.param(lhs, marks=self.marks)


# =============================================================================
# ID Generation Helpers
# =============================================================================


def shape_str(shape) -> str:
    """
    Convert shape tuple to 'HxWxD' string format.

    Args:
        shape: Tuple or list of dimensions.

    Returns:
        String representation (e.g., '32x32' for shape (32, 32)).
    """
    return "x".join(map(str, shape))


def shapes_list_str(shapes) -> str:
    """
    Convert list of shapes to string joined by '-'.

    Args:
        shapes: Sequence of shape tuples.

    Returns:
        String representation (e.g., '1x2-3x4' for [(1, 2), (3, 4)]).
    """
    return "-".join(shape_str(s) for s in shapes)


def torch_dtype_to_abbrev(dtype: torch.dtype) -> str:
    """
    Convert torch dtype to abbreviated string.

    Args:
        dtype: PyTorch dtype.

    Returns:
        Abbreviated string (e.g., 'bf16' for torch.bfloat16).
    """
    mapping = {
        torch.float32: "f32",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.int32: "i32",
        torch.int16: "i16",
        torch.int8: "i8",
        torch.uint8: "u8",
        torch.bool: "bool",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


# =============================================================================
# PCC-based Golden Comparison
# =============================================================================


def mask_inf_nan(tensor: torch.Tensor) -> torch.Tensor:
    """
    Mask inf and nan values with zeros.

    Args:
        tensor: Input tensor.

    Returns:
        Tensor with inf/nan values replaced by 0.
    """
    tensor = tensor.clone()
    mask = torch.logical_or(
        torch.isnan(tensor),
        torch.logical_or(torch.isinf(tensor), torch.isneginf(tensor)),
    )
    tensor[mask] = 0
    return tensor


def compute_pcc(
    golden: torch.Tensor,
    calculated: torch.Tensor,
    atol: float = 1e-08,
    rtol: float = 1e-05,
) -> float:
    """
    Compute Pearson correlation coefficient between two tensors.

    Args:
        golden: Expected tensor.
        calculated: Actual tensor.
        atol: Absolute tolerance for edge cases.
        rtol: Relative tolerance for edge cases.

    Returns:
        PCC value in range [0, 1].
    """
    if golden.numel() == 0 and calculated.numel() == 0:
        return 1.0 if golden.shape == calculated.shape else 0.0
    elif golden.numel() == 0 or calculated.numel() == 0:
        return 0.0

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        return 1.0
    elif torch.any(golden.bool()) != torch.any(calculated.bool()):
        return 0.0
    elif torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        return 0.0

    golden = mask_inf_nan(golden)
    calculated = mask_inf_nan(calculated)

    if torch.equal(golden, calculated):
        return 1.0

    if golden.dtype == torch.bfloat16:
        golden = golden.float()
    if calculated.dtype == torch.bfloat16:
        calculated = calculated.float()

    if golden.numel() == 1:
        return float(torch.isclose(golden, calculated, atol=atol, rtol=rtol))

    if torch.max(golden) == torch.min(golden) and torch.max(calculated) == torch.min(
        calculated
    ):
        return float(
            torch.isclose(
                torch.max(golden), torch.max(calculated), atol=atol, rtol=rtol
            ).item()
        )

    # Compute correlation coefficient.
    cal_pcc = np.ma.corrcoef(
        np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
        np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
    )
    mask = np.ones(cal_pcc.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    cal_pcc = np.min(cal_pcc[mask])

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return 1.0

    return float(cal_pcc)


def get_atol_rtol_pcc(
    golden: torch.Tensor,
    calculated: torch.Tensor,
    atol: float = 1e-08,
    rtol: float = 1e-05,
) -> Tuple[float, float, float]:
    """
    Compute absolute tolerance, relative tolerance, and PCC.

    Args:
        golden: Expected tensor.
        calculated: Actual tensor.
        atol: Absolute tolerance threshold.
        rtol: Relative tolerance threshold.

    Returns:
        Tuple of (actual_atol, actual_rtol, pcc).
    """
    if not torch.is_floating_point(golden):
        golden = golden.to(torch.float64)
    if not torch.is_floating_point(calculated):
        calculated = calculated.to(torch.float64)

    if golden.numel() == 0 or calculated.numel() == 0:
        cal_atol = 0.0
        cal_rtol = 0.0
    else:
        cal_atol = torch.max(torch.abs(golden - calculated)).item()
        denom = torch.abs(calculated)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        cal_rtol = torch.max(torch.abs((golden - calculated) / denom)).item()

    if golden.numel() == 1 and golden.item() != 0:
        cal_pcc = (
            1.0
            if torch.nn.functional.cosine_similarity(
                golden.float().unsqueeze(0),
                calculated.float().unsqueeze(0),
                dim=0,
            ).item()
            else 0.0
        )
    else:
        cal_pcc = compute_pcc(golden, calculated, atol, rtol)

    return (cal_atol, cal_rtol, cal_pcc)


# =============================================================================
# Platform/System Constants
# =============================================================================

# Supported backends.
ALL_BACKENDS: Set[str] = {"ttnn", "ttmetal"}

# Supported system types.
ALL_SYSTEMS: Set[str] = {"n150", "n300", "llmbox", "tg", "p150", "p300"}


@dataclass
class ComparisonResult:
    """Result of a golden comparison."""

    passed: bool
    pcc: float
    atol: float
    rtol: float
    max_abs_diff: float
    mean_abs_error: float
    cosine_sim: float
    message: str = ""


def compare_tensors(
    golden: torch.Tensor,
    calculated: torch.Tensor,
    pcc_threshold: float = 0.99,
    atol: float = 1e-1,
    rtol: float = 5e-2,
    check_pcc: bool = True,
    check_atol: bool = False,
    check_rtol: bool = False,
) -> ComparisonResult:
    """
    Compare two tensors with multiple metrics.

    Args:
        golden: Expected tensor.
        calculated: Actual tensor.
        pcc_threshold: Minimum acceptable PCC.
        atol: Absolute tolerance threshold.
        rtol: Relative tolerance threshold.
        check_pcc: Whether to check PCC.
        check_atol: Whether to check atol.
        check_rtol: Whether to check rtol.

    Returns:
        ComparisonResult with pass/fail status and metrics.
    """
    golden_f32 = golden.float()
    calculated_f32 = calculated.float()

    cal_atol, cal_rtol, cal_pcc = get_atol_rtol_pcc(golden_f32, calculated_f32)

    max_abs_diff = torch.max(torch.abs(golden_f32 - calculated_f32)).item()
    mean_abs_error = torch.mean(torch.abs(golden_f32 - calculated_f32)).item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        golden_f32.flatten().unsqueeze(0),
        calculated_f32.flatten().unsqueeze(0),
    ).item()

    passed = True
    message = ""

    if check_pcc and cal_pcc < pcc_threshold:
        passed = False
        message = f"PCC {cal_pcc:.4f} < threshold {pcc_threshold}"
    if check_atol and cal_atol > atol:
        passed = False
        message = f"ATOL {cal_atol:.4f} > threshold {atol}"
    if check_rtol and cal_rtol > rtol:
        passed = False
        message = f"RTOL {cal_rtol:.4f} > threshold {rtol}"

    # Default to allclose check if no explicit checks.
    if not (check_pcc or check_atol or check_rtol):
        passed = torch.allclose(calculated_f32, golden_f32, rtol=rtol, atol=atol)
        if not passed:
            message = f"allclose failed: max_diff={max_abs_diff:.4f}"

    return ComparisonResult(
        passed=passed,
        pcc=cal_pcc,
        atol=cal_atol,
        rtol=cal_rtol,
        max_abs_diff=max_abs_diff,
        mean_abs_error=mean_abs_error,
        cosine_sim=cosine_sim,
        message=message,
    )
