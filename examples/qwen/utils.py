# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for Qwen tt-lang implementation."""

import os
from pathlib import Path

import torch

TILE_SIZE = 32
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "qwen2.5-0.5b.pt")


def load_checkpoint(path: str = WEIGHTS_PATH) -> dict:
    """Load the pre-extracted Qwen checkpoint."""
    print(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, weights_only=True)
    print(f"  {len(ckpt['layers'])} layers loaded")
    return ckpt


def to_device(tensor: torch.Tensor, device) -> "ttnn.Tensor":
    """Convert a torch tensor to a TTNN device tensor in tile layout."""
    import ttnn

    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def from_device(tensor) -> torch.Tensor:
    """Convert a TTNN device tensor back to torch."""
    import ttnn

    return ttnn.to_torch(tensor)


def alloc_output(shape: tuple, device) -> "ttnn.Tensor":
    """Allocate a zeroed output tensor on device."""
    import ttnn

    return to_device(torch.zeros(shape, dtype=torch.bfloat16), device)


def pcc(result: torch.Tensor, expected: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors."""
    r = result.float().flatten()
    e = expected.float().flatten()
    if r.numel() == 0:
        return 1.0
    corr = torch.corrcoef(torch.stack([r, e]))[0, 1].item()
    return corr


def assert_pcc(result: torch.Tensor, expected: torch.Tensor, threshold: float = 0.99, label: str = ""):
    """Assert PCC meets threshold, print diagnostics."""
    score = pcc(result, expected)
    max_diff = (result.float() - expected.float()).abs().max().item()
    status = "PASS" if score >= threshold else "FAIL"
    tag = f" [{label}]" if label else ""
    print(f"  {status}{tag}: PCC={score:.6f} (threshold={threshold}), max_diff={max_diff:.6f}")
    assert score >= threshold, f"PCC {score:.6f} < {threshold}{tag}"
    return score
