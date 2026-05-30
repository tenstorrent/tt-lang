# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT-Lang block-manipulation functions for the ttl.block namespace.

This module provides shape-manipulation and utility functions that operate on
blocks, matching the ttl.block API from the TT-Lang specification (v0.17+).

Functions: broadcast, fill, mask, mask_posinf, where, squeeze, unsqueeze,
transpose.
"""

from typing import Callable, List, Tuple

import torch

from .constants import TILE_SHAPE
from .dfb import Block, check_same_layout, track_source_blocks
from .blockstate import BlockAcquisition, KernelType
from .ttnnsim import ROW_MAJOR_LAYOUT, Tensor


def _apply_binary_op(
    a: Block, b: Block, op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Block:
    # Layout is checked before shape so a layout error wins when both
    # mismatch: shape comparison only makes sense between like-laid-out
    # operands.
    check_same_layout(a, b)
    a_shape = a._shape  # type: ignore[attr-defined]
    b_shape = b._shape  # type: ignore[attr-defined]
    if a_shape != b_shape:
        raise ValueError(
            f"Shape mismatch in binary op: a has shape {a_shape}, b has shape {b_shape}"
        )
    layout = a.layout
    a_tensors = [t.to_torch() for t in a.to_list()]
    b_tensors = [t.to_torch() for t in b.to_list()]
    result_torch: List[torch.Tensor] = [
        op(a_t, b_t) for a_t, b_t in zip(a_tensors, b_tensors)
    ]
    result_list: List[Tensor] = [Tensor(t, layout) for t in result_torch]
    result_block = Block.from_list(result_list, shape=a_shape)  # type: ignore[attr-defined]
    track_source_blocks(result_block, a, b)
    return result_block


def _apply_ternary_op(
    a: Block,
    b: Block,
    c: Block,
    op: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> Block:
    # Layout is checked before shape so a layout error wins when both
    # mismatch (see ``_apply_binary_op`` for the rationale).
    check_same_layout(a, b, c)
    a_shape = a._shape  # type: ignore[attr-defined]
    b_shape = b._shape  # type: ignore[attr-defined]
    c_shape = c._shape  # type: ignore[attr-defined]
    if not (a_shape == b_shape == c_shape):
        raise ValueError(
            f"Shape mismatch in ternary op: a has shape {a_shape}, "
            f"b has shape {b_shape}, c has shape {c_shape}"
        )
    layout = a.layout
    a_tensors = [t.to_torch() for t in a.to_list()]
    b_tensors = [t.to_torch() for t in b.to_list()]
    c_tensors = [t.to_torch() for t in c.to_list()]
    result_torch: List[torch.Tensor] = [
        op(a_t, b_t, c_t) for a_t, b_t, c_t in zip(a_tensors, b_tensors, c_tensors)
    ]
    result_list: List[Tensor] = [Tensor(t, layout) for t in result_torch]
    result_block = Block.from_list(result_list, shape=a_shape)  # type: ignore[attr-defined]
    track_source_blocks(result_block, a, b, c)
    return result_block


def broadcast(
    block: Block,
    dims: List[int],
    shape: Tuple[int, ...],
) -> Block:
    """Broadcast a block over specified dimensions to a target shape.

    Dimension indexing uses standard Python convention: positive dim 0 is the
    outermost dimension, dim 1 is the next, and so on. Negative indices count
    from the innermost: dim -1 is the innermost (last) dimension, dim -2 is
    the next-to-innermost, and so on.

    The block must have grid size 1 in each dimension listed in dims. The target
    shape must match the block shape in all non-broadcast dimensions.

    Not supported for Row-Major layout blocks.  Requires a block with at least
    two grid dimensions; 1-D blocks should be ``ttl.block.unsqueeze``'d by the
    caller before broadcasting.  ``dims`` must be non-empty - matching the
    compiler's behaviour, an empty ``dims`` is a misuse rather than a no-op.

    Args:
        block: Input block to broadcast. Must have grid size 1 in each dim in dims.
        dims: List of grid dimension indices to broadcast along (non-empty).
        shape: Target grid shape of the result block.

    Returns:
        A new temporary Block with the specified shape.

    Raises:
        ValueError: If the block is row-major, has fewer than 2 grid dims,
            ``dims`` is empty, ``shape`` rank does not match the block, any
            broadcast dim is out of range, the block's grid size is not 1 in
            a broadcast dim, or a non-broadcast dim does not match the target.

    Examples:
        # a_blk shape (N, 1): broadcast along innermost (cols) to (N, M)
        a_bcast = ttl.block.broadcast(a_blk, dims=[-1], shape=(N, M))
        # b_blk shape (1, M): broadcast along outermost (rows) to (N, M)
        b_bcast = ttl.block.broadcast(b_blk, dims=[0], shape=(N, M))
        y_blk.store(a_bcast + b_bcast)
    """
    block_shape = block._shape  # type: ignore[attr-defined]
    ndim = len(block_shape)
    shape = tuple(shape)

    if block.layout == ROW_MAJOR_LAYOUT:
        raise ValueError("broadcast is not supported for Row-Major layout blocks")

    if ndim < 2:
        raise ValueError(
            f"broadcast requires a block with at least 2 grid dimensions; got "
            f"shape {block_shape}. Use ttl.block.unsqueeze first."
        )

    if not dims:
        raise ValueError(
            "broadcast requires at least one dim to broadcast along; got dims=[]"
        )

    if len(shape) != ndim:
        raise ValueError(
            f"broadcast shape {shape} has {len(shape)} dimensions but block has {ndim}"
        )

    # Normalize dims and validate that block has grid size 1 in each broadcast dim.
    internal_dims = {d % ndim for d in dims}
    for d in dims:
        if d >= ndim or d < -ndim:
            raise ValueError(
                f"Cannot broadcast along dimension {d}: block has shape {block_shape} "
                f"with only {ndim} dimensions"
            )
        nd = d % ndim
        if block_shape[nd] != 1:
            raise ValueError(
                f"Cannot broadcast along dimension {d}: block grid size must be 1, "
                f"but is {block_shape[nd]}"
            )

    # Validate non-broadcast dims match
    for i, (s, t) in enumerate(zip(block_shape, shape)):
        if i not in internal_dims and s != t:
            raise ValueError(
                f"broadcast shape mismatch at dimension {i}: block has {s}, target has {t}"
            )

    # Expand the element tensor to match the target shape.  The reshape /
    # indexing below treats the last two dims of ``shape`` as the within-tile
    # axes, so this function only handles blocks with at least two grid
    # dimensions; 1-D blocks are rejected by the ``ndim < 2`` guard above.
    elem = block._buf.to_torch()  # type: ignore[attr-defined]

    batch = block_shape[:-2]
    TM_s, TK_s = block_shape[-2], block_shape[-1]
    tile_h = elem.shape[-2] // TM_s if TM_s > 0 else 1
    tile_w = elem.shape[-1] // TK_s if TK_s > 0 else 1

    TM_t, TK_t = shape[-2], shape[-1]
    target_batch = shape[:-2]

    # Reshape: (*batch, TM_s, tile_h, TK_s, tile_w)
    exposed = elem.reshape(*batch, TM_s, tile_h, TK_s, tile_w)

    # Spec step (1): within-tile broadcast.  Fires when the broadcast hits one
    # or both of the last two (tile) dimensions of `shape`.  Per the spec the
    # source places vector data in row 0 / column 0 / position (0, 0) of each
    # tile; step (1) replicates that seed value across the rest of the tile so
    # step (2) can then replicate the now-uniform tile across the grid.
    bcast_row = (ndim - 2) in internal_dims  # outer tile dim -> seed row 0
    bcast_col = (ndim - 1) in internal_dims  # inner tile dim -> seed col 0
    if bcast_row:
        exposed = (
            exposed[..., :, 0:1, :, :]
            .expand(*batch, TM_s, tile_h, TK_s, tile_w)
            .contiguous()
        )
    if bcast_col:
        exposed = (
            exposed[..., :, :, :, 0:1]
            .expand(*batch, TM_s, tile_h, TK_s, tile_w)
            .contiguous()
        )

    # Spec step (2): across-tile broadcast at grid level.
    expanded = exposed.expand(*target_batch, TM_t, tile_h, TK_t, tile_w)
    # Fuse back: (*target_batch, TM_t*tile_h, TK_t*tile_w)
    result_elem = expanded.reshape(
        *target_batch, TM_t * tile_h, TK_t * tile_w
    ).contiguous()

    result_block = Block(
        tensor=Tensor(result_elem),
        shape=shape,
        acquisition=BlockAcquisition.RESERVE,
        kernel_type=KernelType.COMPUTE,
        is_temporary=True,
    )
    track_source_blocks(result_block, block)
    return result_block


def fill(value: float, shape: Tuple[int, ...]) -> Block:
    """Return a temporary tiled block of the specified shape filled with value.

    Args:
        value: The scalar value to fill every element with.
        shape: Grid shape of the resulting block (at least 2-dimensional).

    Returns:
        A temporary Block of the specified shape with every element set to value.
    """
    shape = tuple(shape)
    if len(shape) < 2:
        raise ValueError(
            "fill requires a shape with at least 2 dimensions for tiled layout"
        )

    tile_h, tile_w = TILE_SHAPE
    batch = shape[:-2]
    TM, TK = shape[-2], shape[-1]

    elem = torch.full(
        (*batch, TM * tile_h, TK * tile_w),
        value,
        dtype=torch.bfloat16,
    )
    return Block(
        tensor=Tensor(elem),
        shape=shape,
        acquisition=BlockAcquisition.RESERVE,
        kernel_type=KernelType.COMPUTE,
        is_temporary=True,
    )


def mask(expr: Block, mask_blk: Block) -> Block:
    """Mask a block by replacing masked elements with 0.

    Args:
        expr: Input block
        mask_blk: Mask block (elements equal to 1 are masked)

    Returns:
        Block with masked elements replaced by 0
    """

    def _op(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return torch.where(t2 == 1, torch.tensor(0.0, dtype=t1.dtype), t1)

    return _apply_binary_op(expr, mask_blk, _op)


def mask_posinf(expr: Block, mask_blk: Block) -> Block:
    """Mask a block by replacing masked elements with positive infinity.

    Args:
        expr: Input block
        mask_blk: Mask block (elements equal to 1 are masked)

    Returns:
        Block with masked elements replaced by positive infinity
    """

    def _op(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return torch.where(t2 == 1, torch.tensor(float("inf"), dtype=t1.dtype), t1)

    return _apply_binary_op(expr, mask_blk, _op)


def where(condition: Block, true_value: Block, false_value: Block) -> Block:
    """Conditional element selection.

    Args:
        condition: Condition block (elements equal to 1 are true, 0 are false)
        true_value: Block to select from when condition is true
        false_value: Block to select from when condition is false

    Returns:
        Block with elements selected based on condition
    """

    def _op(cond: torch.Tensor, tv: torch.Tensor, fv: torch.Tensor) -> torch.Tensor:
        return torch.where(cond == 1, tv, fv)

    return _apply_ternary_op(condition, true_value, false_value, _op)


def squeeze(block: Block, dims: List[int]) -> Block:
    """Remove size-1 dimensions from the block grid shape.

    All positions in ``dims`` are interpreted relative to the original shape.
    Removed dimensions must each have grid size 1. Dimension indexing uses
    standard Python convention: positive 0 is outermost, negative -1 is
    innermost.

    Args:
        block: Input block.
        dims: Grid dimensions to remove. Each must have grid size 1.

    Returns:
        A new temporary Block with the specified dimensions removed.

    Examples:
        # shape (1, N, M) -> squeeze(dims=[0]) -> shape (N, M)
        result = ttl.block.squeeze(a, dims=[0])
        # shape (N, M, 1) -> squeeze(dims=[-1]) -> shape (N, M)
        result = ttl.block.squeeze(a, dims=[-1])
        # shape (1, N, 1, M) -> squeeze(dims=[0, 2]) -> shape (N, M)
        result = ttl.block.squeeze(a, dims=[0, 2])
        # shape (N, 1, M, 1) -> squeeze(dims=[-1, -3]) -> shape (N, M)
        result = ttl.block.squeeze(a, dims=[-1, -3])
    """
    block_shape = block._shape  # type: ignore[attr-defined]
    ndim = len(block_shape)

    norm_dims: set[int] = set()
    for d in dims:
        if d >= ndim or d < -ndim:
            raise ValueError(
                f"Cannot squeeze dimension {d}: block has shape {block_shape} "
                f"with only {ndim} dimensions"
            )
        nd = d % ndim
        if block_shape[nd] != 1:
            raise ValueError(
                f"Cannot squeeze dimension {d}: grid size is {block_shape[nd]}, expected 1"
            )
        norm_dims.add(nd)

    new_shape = tuple(s for i, s in enumerate(block_shape) if i not in norm_dims)
    tiles = block.to_list()
    result = Block.from_list(tiles, new_shape)
    track_source_blocks(result, block)
    return result


def unsqueeze(block: Block, dims: List[int]) -> Block:
    """Add size-1 dimensions to the block grid shape.

    Position values in ``dims`` refer to positions in the resulting shape.
    Dimension indexing uses standard Python convention: positive 0 is
    outermost, negative -1 is innermost.

    Args:
        block: Input block.
        dims: Positions in the resulting shape at which to insert size-1
            dimensions.

    Returns:
        A new temporary Block with the specified size-1 dimensions inserted.

    Examples:
        # shape (N, M) -> unsqueeze(dims=[0]) -> shape (1, N, M)
        result = ttl.block.unsqueeze(a, dims=[0])
        # shape (N, M) -> unsqueeze(dims=[-1]) -> shape (N, M, 1)
        result = ttl.block.unsqueeze(a, dims=[-1])
        # shape (N, M) -> unsqueeze(dims=[0, 2]) -> shape (1, N, 1, M)
        result = ttl.block.unsqueeze(a, dims=[0, 2])
        # shape (N, M) -> unsqueeze(dims=[-1, -3]) -> shape (N, 1, M, 1)
        result = ttl.block.unsqueeze(a, dims=[-1, -3])
    """
    block_shape = block._shape  # type: ignore[attr-defined]
    ndim = len(block_shape)
    new_ndim = ndim + len(dims)

    norm_positions: List[int] = []
    for d in dims:
        if d >= new_ndim or d < -new_ndim:
            raise ValueError(
                f"Cannot unsqueeze at dimension {d}: resulting shape would have "
                f"{new_ndim} dimensions"
            )
        norm_positions.append(d % new_ndim)

    result_list: List[int] = list(block_shape)
    for pos in sorted(set(norm_positions)):
        result_list.insert(pos, 1)
    new_shape = tuple(result_list)

    tiles = block.to_list()
    result = Block.from_list(tiles, new_shape)
    track_source_blocks(result, block)
    return result


def transpose(block: Block) -> Block:
    """Transpose a 2D block (swap tile-grid rows and columns).

    Performs a grid-level transpose: block shape (M, N) becomes (N, M).
    Each 32x32 tile also has its rows and columns swapped.
    Supported only for two-dimensional blocks.

    Args:
        block: Input block with shape (M, N).

    Returns:
        Block with shape (N, M), where each tile is transposed.
    """
    if len(block._shape) != 2:  # type: ignore[attr-defined]
        raise ValueError(
            f"transpose requires a 2-D block grid, got shape {block._shape}"  # type: ignore[attr-defined]
        )

    layout = block.layout
    transposed_tiles = [Tensor(t.to_torch().T, layout) for t in block.to_list()]

    M, N = block._shape  # type: ignore[attr-defined]
    reordered_tiles: List[Tensor] = []
    for j in range(N):
        for i in range(M):
            reordered_tiles.append(transposed_tiles[i * N + j])

    result_block = Block.from_list(reordered_tiles, shape=(N, M))
    track_source_blocks(result_block, block)
    return result_block
