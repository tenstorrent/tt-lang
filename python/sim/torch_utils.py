# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch Type Utilities

This module provides wrapper functions around PyTorch operations to avoid
'# type: ignore' comments throughout src files.

WHY THIS IS NEEDED:
==================

PyTorch has extremely complex function overloads that confuse static type checkers:

1. torch.randn() has 20+ overloads for different parameter combinations
2. torch.zeros() has similar complexity with device, dtype, requires_grad options
3. torch.all() returns Union[Tensor, bool] depending on context
4. Type checkers can't always infer which overload to use

This creates noise in src files with many '# type: ignore' comments, making
code harder to read and potentially masking real type issues.

SOLUTION:
=========

These wrapper functions provide simplified, single-purpose interfaces that:
- Hide PyTorch's complex overloads from the type checker
- Provide clear APIs
- Maintain full runtime functionality
- Keep type safety for the rest of the codebase

This approach is better than:
- Globally disabling type checking (weakens safety everywhere)
- Per-file suppressions (still clutters code)
- Complex type stubs (maintenance overhead)

USAGE:
======

Instead of:
    tensor = torch.randn(64, 64)  # type: ignore
    assert torch.all(tensor > 0)  # type: ignore

Use:
    from . import torch_utils as tu
    tensor = tu.randn(64, 64)
    assert tu.all_true(tensor > 0)
"""

import torch
from typing import Tuple, Union, List


# Tensor creation utilities
def randn(*shape: int) -> torch.Tensor:
    """Create a tensor with random normal values. Simplifies torch.randn overloads."""
    return torch.randn(*shape)  # type: ignore


def zeros(*shape: int) -> torch.Tensor:
    """Create a tensor filled with zeros. Simplifies torch.zeros overloads."""
    return torch.zeros(*shape)  # type: ignore


def ones(*shape: int) -> torch.Tensor:
    """Create a tensor filled with ones. Simplifies torch.ones overloads."""
    return torch.ones(*shape)  # type: ignore


def full(shape: Tuple[int, ...], fill_value: Union[int, float]) -> torch.Tensor:
    """Create a tensor filled with a specific value. Simplifies torch.full overloads."""
    return torch.full(shape, fill_value)  # type: ignore


# Assertion utilities for tests
def all_true(condition: torch.Tensor) -> bool:
    """Check if all elements are True. Simplifies torch.all return type complexity."""
    result = torch.all(condition)  # type: ignore
    # torch.all can return Tensor or bool depending on context - ensure we get bool
    return bool(result)


def allclose(
    a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08
) -> bool:
    """Check if tensors are element-wise close. Simplifies torch.allclose overloads."""
    return bool(torch.allclose(a, b, rtol=rtol, atol=atol))


def equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Check if tensors are exactly equal. Simplifies torch.equal overloads."""
    return bool(torch.equal(a, b))


# Common tensor operations
def cat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Concatenate tensors. Simplifies torch.cat overloads."""
    return torch.cat(tensors, dim=dim)  # type: ignore


def stack(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Stack tensors. Simplifies torch.stack overloads."""
    return torch.stack(tensors, dim=dim)  # type: ignore


# Tile calculation utilities
def tile_count(tensor_shape: Tuple[int, ...], tile_shape: Tuple[int, ...]) -> int:
    """
    Calculate the total number of tiles in a tensor.

    Args:
        tensor_shape: Shape of the tensor (height, width, ...)
        tile_shape: Shape of each tile (height, width, ...)

    Returns:
        Total number of tiles needed to represent the tensor

    Example:
        For a (64, 128) tensor with tile_shape=(32, 32):
        tile_count((64, 128), (32, 32)) = (64//32) * (128//32) = 2 * 4 = 8 tiles
    """
    from numpy import prod

    if len(tensor_shape) != len(tile_shape):
        raise ValueError(
            f"tensor_shape and tile_shape must have same dimensions: {len(tensor_shape)} vs {len(tile_shape)}"
        )
    return int(
        prod(
            [
                tensor_dim // tile_dim
                for tensor_dim, tile_dim in zip(tensor_shape, tile_shape)
            ]
        )
    )


def is_tiled(tensor: torch.Tensor, tile_shape: Tuple[int, ...]) -> bool:
    """
    Check if a tensor's dimensions are compatible with the given tile shape.

    A tensor is considered "tiled" if all its dimensions are evenly
    divisible by the corresponding tile shape dimensions.

    Args:
        tensor: Tensor to check for tile compatibility
        tile_shape: Shape of tiles to check compatibility against

    Returns:
        True if the tensor dimensions are tile-aligned, False otherwise

    Example:
        tensor = torch.randn(64, 64)
        assert is_tiled(tensor, (32, 32)) == True  # 64 % 32 == 0

        tensor = torch.randn(65, 64)
        assert is_tiled(tensor, (32, 32)) == False  # 65 % 32 != 0

        # Works with different tile shapes
        tensor = torch.randn(96, 64)
        assert is_tiled(tensor, (32, 16)) == True  # 96 % 32 == 0, 64 % 16 == 0
    """
    if len(tensor.shape) != len(tile_shape):
        return False
    return all(dim % tile_dim == 0 for dim, tile_dim in zip(tensor.shape, tile_shape))
