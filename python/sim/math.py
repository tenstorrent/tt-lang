# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT-Lang math functions for block operations.

This module provides math functions that operate on blocks, matching the
ttl.math API from the TT-Lang specification.
"""

from typing import List

from .block import Block
from .typedefs import Shape


def broadcast(block: Block, dims: List[int]) -> Block:
    """Broadcast a block along specified dimensions.

    Marks a block for explicit broadcasting along the specified dimensions.
    The block must have size 1 in the dimensions being broadcast.
    When used in a binary operation, the block will be expanded to match
    the shape of the other operand.

    Args:
        block: Input block to broadcast (can be Block or WaitContext)
        dims: List of dimension indices to broadcast along (0-indexed)

    Returns:
        Block marked for broadcasting

    Raises:
        ValueError: If any of the specified dimensions don't have size 1

    Example:
        # Broadcast a (1, 1) block along dimension 1 (columns)
        b_cb = ttl.make_circular_buffer_like(B, shape=(1, 1))
        with b_cb.wait() as b_blk:
            b_broadcast = ttl.math.broadcast(b_blk, dims=[1])
            # b_broadcast can now be added to blocks with shape (1, N)
            y = a_blk + b_broadcast

    From the specification:
        The broadcast function produces a block with shape expanded to be
        compatible with the outer part of the expression.

        Example: y.store(b * ttl.math.broadcast(a, dims=[1]))
        Here the `*` is the outer expression, and if `b` has shape (N, M),
        then `a` must have shape (N, 1).
    """
    # Unwrap WaitContext/ReserveContext if needed
    if hasattr(block, "block"):
        block = block.block()

    # Validate that the dimensions being broadcast have size 1
    block_shape = block._shape
    for dim in dims:
        if dim >= len(block_shape):
            raise ValueError(
                f"Cannot broadcast along dimension {dim}: block has shape {block_shape} "
                f"with only {len(block_shape)} dimensions"
            )
        if block_shape[dim] != 1:
            raise ValueError(
                f"Cannot broadcast along dimension {dim}: dimension must have size 1, "
                f"but has size {block_shape[dim]}"
            )

    # Create a new block with the same data but marked for broadcasting
    tensors = block.to_list()
    return Block.from_list(tensors, shape=block_shape, broadcast_dims=dims)
