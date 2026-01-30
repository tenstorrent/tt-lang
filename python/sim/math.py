# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT-Lang math functions for block operations.

This module provides math functions that operate on blocks, matching the
ttl.math API from the TT-Lang specification.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

from .block import Block

if TYPE_CHECKING:
    from .cb import ReserveContext, WaitContext


def broadcast(
    block: Union[Block, "ReserveContext", "WaitContext"],
    _unused_arg: Optional[Any] = None,
    dims: Optional[List[int]] = None,
) -> Block:
    """Broadcast a block along specified dimensions.

    Validates that the block has size 1 in the dimensions being broadcast.
    The actual broadcasting happens automatically in binary operations.

    Note: With implicit broadcasting enabled, this function is optional.
    Binary operations like `a * b` will automatically broadcast if shapes
    are compatible (one dimension is 1). This function can still be used
    for explicit validation and documentation of broadcasting intent.

    Args:
        block: Input block to broadcast (can be Block or WaitContext)
        _unused_arg: Unused argument for compatibility (typically output block shape hint)
        dims: List of dimension indices to broadcast along (0-indexed)

    Returns:
        The same block (unchanged, as broadcasting is now implicit)

    Raises:
        ValueError: If any of the specified dimensions don't have size 1

    Example:
        # Explicit broadcasting (validated but automatic)
        b_cb = ttl.make_circular_buffer_like(B, shape=(1, 1))
        with b_cb.wait() as b_blk:
            b_broadcast = ttl.math.broadcast(b_blk, dims=[1])
            y = a_blk + b_broadcast  # Broadcasts automatically

        # Implicit broadcasting (also works without ttl.math.broadcast)
        with b_cb.wait() as b_blk:
            y = a_blk + b_blk  # Broadcasts automatically if compatible

    From the specification:
        The broadcast function produces a block with shape expanded to be
        compatible with the outer part of the expression.

        Example: y.store(b * ttl.math.broadcast(a, dims=[1]))
        Here the `*` is the outer expression, and if `b` has shape (N, M),
        then `a` must have shape (N, 1).
    """
    if dims is None:
        raise ValueError("dims parameter is required for broadcast()")

    # Unwrap WaitContext/ReserveContext if needed
    actual_block: Block = block.block() if hasattr(block, "block") else block  # type: ignore[union-attr]

    # Validate that the dimensions being broadcast have size 1
    block_shape = actual_block._shape  # type: ignore[attr-defined]
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

    # Broadcasting is now implicit, so just return the block unchanged
    return actual_block
