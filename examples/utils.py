# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for comparing tensor outputs in tests."""

import torch


def assert_pcc(golden, actual, threshold=0.99):
    """
    Assert Pearson correlation coefficient is above threshold.

    Args:
        golden: Expected tensor
        actual: Actual output tensor
        threshold: Minimum acceptable PCC (default 0.99)

    Raises:
        AssertionError: If PCC < threshold
    """
    combined = torch.stack([golden.flatten(), actual.flatten()])
    pcc = torch.corrcoef(combined)[0, 1].item()
    assert (
        pcc >= threshold
    ), f"Expected pcc {pcc} >= {threshold}\ngolden:\n{golden}\nactual:\n{actual}"


def assert_allclose(
    actual,
    expected,
    rtol=1e-5,
    atol=1e-8,
    verbose=True,
):
    """
    Assert tensors are close with detailed error reporting.

    Computes both absolute and relative errors with informative failure messages
    showing error statistics and worst-case locations.

    Args:
        actual: Actual output tensor
        expected: Expected tensor
        rtol: Relative tolerance (default 1e-5)
        atol: Absolute tolerance (default 1e-8)
        verbose: Show detailed error statistics on failure (default True)

    Raises:
        AssertionError: If tensors don't match within tolerance

    Examples:
        >>> out = model(input)
        >>> assert_allclose(out, expected, rtol=1e-4, atol=1e-6)
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}"
        )

    # Compute element-wise absolute error
    abs_diff = torch.abs(actual - expected)
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()

    # Compute element-wise relative error with epsilon for numerical stability
    eps = 1e-10
    rel_error = abs_diff / (torch.abs(expected) + eps)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    # Check if within tolerance
    is_close = torch.allclose(actual, expected, rtol=rtol, atol=atol)

    if not is_close and verbose:
        # Find locations of worst errors
        abs_error_flat = abs_diff.flatten()
        rel_error_flat = rel_error.flatten()

        worst_abs_idx = abs_error_flat.argmax()
        worst_rel_idx = rel_error_flat.argmax()

        # Convert flat indices to coordinates
        worst_abs_coord = torch.unravel_index(worst_abs_idx, actual.shape)
        worst_rel_coord = torch.unravel_index(worst_rel_idx, actual.shape)

        error_msg = f"""
Tensor comparison failed!

Error Statistics:
  Absolute Error:
    Mean: {mean_abs_error:.6e}
    Max:  {max_abs_error:.6e} at {tuple(c.item() for c in worst_abs_coord)}
          actual={actual[worst_abs_coord].item():.6f}, expected={expected[worst_abs_coord].item():.6f}

  Relative Error:
    Mean: {mean_rel_error:.6e}
    Max:  {max_rel_error:.6e} at {tuple(c.item() for c in worst_rel_coord)}
          actual={actual[worst_rel_coord].item():.6f}, expected={expected[worst_rel_coord].item():.6f}

Thresholds:
  rtol: {rtol}
  atol: {atol}

Shape: {actual.shape}
Mismatched elements: {(abs_diff > atol + rtol * torch.abs(expected)).sum().item()} / {actual.numel()}
"""
        raise AssertionError(error_msg)

    if not is_close:
        raise AssertionError(
            f"Tensors not close: max_abs_error={max_abs_error:.6e}, "
            f"max_rel_error={max_rel_error:.6e}, rtol={rtol}, atol={atol}"
        )
