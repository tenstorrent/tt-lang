# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for tensor comparison and diagnostics."""

import torch
from loguru import logger


def check_tensors_match(
    expected: torch.Tensor, actual: torch.Tensor, rtol: float = 5e-2, atol: float = 1e-1
) -> bool:
    """
    Check if two tensors match within tolerance and log results.

    Returns True if tensors match, False otherwise.
    """
    eps = torch.finfo(expected.dtype).eps
    matching = torch.allclose(expected, actual, rtol=rtol, atol=atol)

    abs_diff = torch.abs(expected - actual)
    rel_diff = abs_diff / (torch.abs(expected) + eps)
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()

    status = "PASS" if matching else "FAIL"
    logger.info(
        f"{status} | dtype eps={eps:.2e} | max abs diff={max_abs_diff:.2e} (tol={atol:.2e}) | max rel diff={max_rel_diff:.2e} (tol={rtol:.2e})"
    )

    return matching


def log_mismatch_diagnostics(
    expected: torch.Tensor,
    actual: torch.Tensor,
    inputs: dict[str, torch.Tensor] | None = None,
):
    """
    Log detailed diagnostics when tensors don't match.

    Args:
        expected: Expected tensor values
        actual: Actual tensor values
        inputs: Optional dict of input tensors for debugging (e.g., {"A": tensor_a, "B": tensor_b})
    """
    # Filter out NaN/inf for better diagnostics
    valid_mask = torch.isfinite(expected) & torch.isfinite(actual)
    if valid_mask.any():
        diff = torch.abs(expected[valid_mask] - actual[valid_mask])
        logger.info(f"Max difference (finite values): {diff.max()}")
        logger.info(f"Mean difference (finite values): {diff.mean()}")

    logger.info(f"NaN in expected: {torch.isnan(expected).sum()}")
    logger.info(f"NaN in actual: {torch.isnan(actual).sum()}")
    logger.info(f"Inf in expected: {torch.isinf(expected).sum()}")
    logger.info(f"Inf in actual: {torch.isinf(actual).sum()}")

    # Find where the inf values are
    inf_mask = torch.isinf(actual)
    if inf_mask.any():
        inf_indices = torch.nonzero(inf_mask)
        logger.info(f"Inf locations (first 5): {inf_indices[:5]}")

        # Log corresponding input values if provided
        if inputs:
            for idx in inf_indices[:2]:
                idx_tuple = tuple(idx.tolist())
                input_vals = ", ".join(
                    f"{name}={tensor[idx_tuple]}" for name, tensor in inputs.items()
                )
                logger.info(f"  At {list(idx_tuple)}: {input_vals}")
