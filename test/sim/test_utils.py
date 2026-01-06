# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Common test utilities for simulator tests."""

import torch
from python.sim import ttnn
from python.sim.typedefs import Size, Shape

# Constants
TILE_SHAPE: Shape = (32, 32)


# ============================================================================
# ttnn.Tensor helpers
# ============================================================================


def make_ones_tile() -> ttnn.Tensor:
    """Create a 32x32 tile filled with ones."""
    t = ttnn.rand(TILE_SHAPE)
    t.to_torch().fill_(1.0)
    return t


def make_zeros_tile() -> ttnn.Tensor:
    """Create a 32x32 tile filled with zeros."""
    return ttnn.empty(TILE_SHAPE)


def make_full_tile(value: float) -> ttnn.Tensor:
    """Create a 32x32 tile filled with a specific value."""
    t = ttnn.rand(TILE_SHAPE)
    t.to_torch().fill_(value)
    return t


def make_rand_tensor(rows: Size, cols: Size) -> ttnn.Tensor:
    """Create a random ttnn.Tensor of given size."""
    return ttnn.rand((rows, cols))


def make_zeros_tensor(rows: Size, cols: Size) -> ttnn.Tensor:
    """Create a zero ttnn.Tensor of given size."""
    return ttnn.empty((rows, cols))


def make_ones_tensor(rows: Size, cols: Size) -> ttnn.Tensor:
    """Create a ttnn.Tensor filled with ones."""
    t = ttnn.rand((rows, cols))
    t.to_torch().fill_(1.0)
    return t


def make_full_tensor(rows: Size, cols: Size, value: float) -> ttnn.Tensor:
    """Create a ttnn.Tensor filled with a specific value."""
    t = ttnn.rand((rows, cols))
    t.to_torch().fill_(value)
    return t


def make_arange_tensor(rows: Size, cols: Size) -> ttnn.Tensor:
    """Create a ttnn.Tensor with sequential values (0, 1, 2, ...)."""
    t = ttnn.rand((rows, cols))
    t.to_torch().copy_(
        torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    )
    return t


def tensors_equal(
    t1: ttnn.Tensor, t2: ttnn.Tensor, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Check if two ttnn.Tensors are approximately equal."""
    return bool(torch.allclose(t1.to_torch(), t2.to_torch(), rtol=rtol, atol=atol))


def tensors_exact_equal(t1: ttnn.Tensor, t2: ttnn.Tensor) -> bool:
    """Check if two ttnn.Tensors are exactly equal (element-wise)."""
    return bool((t1.to_torch() == t2.to_torch()).all().item())


# ============================================================================
# torch.Tensor helpers (for tests that still use torch directly)
# ============================================================================


def make_randn(h: Size, w: Size) -> torch.Tensor:
    """Create a random torch.Tensor with given dimensions."""
    return torch.randn(h, w)


def make_zeros(h: Size, w: Size) -> torch.Tensor:
    """Create a zero-filled torch.Tensor with given dimensions."""
    return torch.zeros(h, w)


def make_ones(h: Size, w: Size) -> torch.Tensor:
    """Create a ones-filled torch.Tensor with given dimensions."""
    return torch.ones(h, w)


def make_full(shape: Shape, value: float) -> torch.Tensor:
    """Create a torch.Tensor filled with a specific value."""
    return torch.full(shape, value)


def all_equal(tensor: torch.Tensor, value: float) -> bool:
    """Check if all elements in torch.Tensor equal the given value."""
    return bool((tensor == value).all().item())
