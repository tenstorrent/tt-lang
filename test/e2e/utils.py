# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test utilities for E2E tests.

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


@dataclass
class ComparisonResult:
    """Result of a golden comparison."""

    passed: bool
    pcc: float
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
    error_tol: Optional[float] = None,
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
        error_tol: If provided, overrides atol/rtol checks with single tolerance.

    Returns:
        ComparisonResult with pass/fail status and metrics.
    """
    golden_f32 = golden.float()
    calculated_f32 = calculated.float()

    cal_pcc = compute_pcc(golden_f32, calculated_f32)

    max_abs_diff = torch.max(torch.abs(golden_f32 - calculated_f32)).item()
    mean_abs_error = torch.mean(torch.abs(golden_f32 - calculated_f32)).item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        golden_f32.flatten().unsqueeze(0),
        calculated_f32.flatten().unsqueeze(0),
    ).item()

    passed = True
    message = ""

    if error_tol is not None:
        # Use single error tolerance
        if max_abs_diff > error_tol:
            passed = False
            message = f"Max abs diff {max_abs_diff:.4f} > tolerance {error_tol}"
    else:
        # Use PCC/atol/rtol checks
        if check_pcc and cal_pcc < pcc_threshold:
            passed = False
            message = f"PCC {cal_pcc:.4f} < threshold {pcc_threshold}"

        # Default to allclose check
        if not torch.allclose(calculated_f32, golden_f32, rtol=rtol, atol=atol):
            passed = False
            if not message:
                message = f"allclose failed: max_diff={max_abs_diff:.4f}"

    return ComparisonResult(
        passed=passed,
        pcc=cal_pcc,
        max_abs_diff=max_abs_diff,
        mean_abs_error=mean_abs_error,
        cosine_sim=cosine_sim,
        message=message,
    )
