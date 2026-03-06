# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT-Lang math functions for block operations.

This module provides math functions that operate on blocks, matching the
ttl.math API from the TT-Lang specification.

Most functions are auto-generated from PyTorch equivalents using a mapping
system similar to ttnnsim.py. Special functions like broadcast and reductions
are implemented manually.
"""

import math as _math
from itertools import product as _iter_product
from typing import Any, Callable, List, Optional, Set

import torch

from .blockstate import BlockAcquisition, ThreadType
from .diagnostics import (
    lazy_import_diagnostics,
    find_user_code_location,
    format_core_ranges,
)
from .dfb import Block, track_source_blocks, matmul
from .ttnnsim import Tensor

_ = matmul

# Track 1D broadcast warnings by (filename, line) -> set of core_ids
# This allows us to deduplicate warnings and show which cores hit each location
_broadcast_1d_warnings: dict[tuple[str, int], set[str]] = {}


def _warn_1d_broadcast_unsupported() -> None:
    """Issue a warning that 1D broadcast is not supported on current hardware.

    Tracks which cores hit each source location and only prints once per location,
    showing the list of cores that encountered the issue.
    """
    # Find user code location (skip this function and caller)
    source_file, source_line = find_user_code_location(skip_frames=2)

    # Should always find user code in the call stack
    assert (
        source_file is not None and source_line is not None
    ), "Could not determine source location for 1D broadcast warning"

    # Get the current core ID
    from .greenlet_scheduler import get_current_core_id

    core_id = get_current_core_id()

    # Track this core hitting this location
    location_key = (source_file, source_line)
    if location_key not in _broadcast_1d_warnings:
        _broadcast_1d_warnings[location_key] = set()
        first_occurrence = True
    else:
        first_occurrence = False

    _broadcast_1d_warnings[location_key].add(core_id)

    # Only print on first occurrence for this location
    if first_occurrence:
        # Format the core list
        cores = _broadcast_1d_warnings[location_key]
        unique_cores = sorted(cores, key=lambda x: (len(x), x))

        if len(unique_cores) == 1 and unique_cores[0] != "unknown":
            cores_label = unique_cores[0]
        else:
            # Extract numeric core IDs (from "core0", "core1", etc.)
            try:
                core_numbers = [
                    int(core[4:])
                    for core in unique_cores
                    if core.startswith("core") and core[4:].isdigit()
                ]
                if core_numbers:
                    cores_label = f"cores: {format_core_ranges(core_numbers)}"
                else:
                    cores_label = f"cores: {', '.join(unique_cores)}"
            except (ValueError, IndexError):
                cores_label = f"cores: {', '.join(unique_cores)}"

        # Try to use diagnostics module for nice formatting
        try:
            diagnostics = lazy_import_diagnostics()
            SourceDiagnostic = diagnostics.SourceDiagnostic

            # Read source lines
            with open(source_file, "r") as f:
                source_lines = f.read().splitlines()

            # Format warning using diagnostics module
            diag = SourceDiagnostic(source_lines, source_file)
            warning_msg = diag.format_error(
                line=source_line,
                col=1,
                message=f"1D broadcast is not supported on current hardware ({cores_label})",
                label="warning",
            )
            print(warning_msg, flush=True)
        except (ImportError, IOError, OSError):
            # Fall back to simple warning if diagnostics unavailable or file unreadable
            print(
                f"warning: 1D broadcast is not supported on current hardware ({cores_label})",
                flush=True,
            )


def broadcast(
    block: Block,
    _unused_arg: Optional[Any] = None,
    dims: Optional[List[int]] = None,
) -> Block:
    """Broadcast a block along specified dimensions.

    This function replicates values within each tile along the specified dimensions.
    After reduce operations store values at specific positions (e.g., reduce_max with
    dims=[0] stores max at column 0), broadcast replicates those values.

    Dimension indexing uses the innermost-first convention: dims=[0] refers to the
    innermost (last) dimension of the block shape, dims=[1] to the next-to-innermost,
    and so on. This matches the convention used throughout the ttl.math API.

    For a 2-D grid block of shape (N, M):
    - dims=[0] (innermost/columns): takes values from column 0 and replicates across
      all columns. Block must have size 1 in M.
    - dims=[1] (next-to-innermost/rows): takes values from row 0 and replicates across
      all rows. Block must have size 1 in N.

    For ND grids, batch dimensions (all dims before the last tile.ndim grid dims)
    have no within-tile axis to broadcast; they are grid-level-only and the tile
    content is left unchanged.

    Args:
        block: Input block to broadcast
        _unused_arg: Unused argument for compatibility (typically output block shape hint)
        dims: List of dimension indices to broadcast along (0=innermost)

    Returns:
        Block with values replicated along the specified dimensions
    """
    if dims is None:
        raise ValueError("dims parameter is required for broadcast()")

    # Validate that the dimensions being broadcast have size 1 at grid level.
    # User dims use innermost-first convention: translate to internal (outermost-first).
    block_shape = block._shape  # type: ignore[attr-defined]
    ndim = len(block_shape)

    # Check if this is a 1D broadcast and issue a warning
    if ndim == 1:
        _warn_1d_broadcast_unsupported()

    for dim in dims:
        if dim >= ndim:
            raise ValueError(
                f"Cannot broadcast along dimension {dim}: block has shape {block_shape} "
                f"with only {ndim} dimensions"
            )
        internal_dim = ndim - 1 - dim
        if block_shape[internal_dim] != 1:
            raise ValueError(
                f"Cannot broadcast along dimension {dim}: dimension must have size 1, "
                f"but has size {block_shape[internal_dim]}"
            )

    # Map block-grid dimensions to within-tile dimensions.
    # The last tile.ndim grid dimensions correspond to tile-internal axes 0, 1, ...
    # Leading (batch) grid dimensions have no tile-internal counterpart.
    input_tensors = [t.to_torch() for t in block.to_list()]
    result_tensors: List[Tensor] = []

    for tile in input_tensors:
        tile_ndim = tile.ndim
        # Number of leading grid dims that are batch-only (no tile axis).
        num_batch_grid_dims = ndim - tile_ndim
        slices = [slice(None)] * tile_ndim
        for dim in dims:
            internal_dim = ndim - 1 - dim
            tile_dim = internal_dim - num_batch_grid_dims
            if tile_dim < 0:
                # Batch grid dimension: no within-tile axis, leave slice unchanged.
                continue
            slices[tile_dim] = slice(0, 1)

        # Extract the slice and expand back to original shape
        result_tile = tile[tuple(slices)].expand(tile.shape).clone()
        result_tensors.append(Tensor(result_tile))

    result_block = Block.from_list(result_tensors, block_shape)

    # Preserve source block tracking for wait() blocks
    if block._source_blocks:  # type: ignore[attr-defined]
        result_block._source_blocks = block._source_blocks.copy()  # type: ignore[attr-defined]

    # If block itself is a wait() block, add it to source_blocks
    if (
        not block._is_temporary  # type: ignore[attr-defined]
        and block.acquisition == BlockAcquisition.WAIT
        and block.thread_type == ThreadType.COMPUTE
    ):
        result_block._source_blocks.append(block)  # type: ignore[attr-defined]

    return result_block


# Helper function to create unary operation wrappers
def _create_unary_op_wrapper(
    name: str, torch_fn: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[[Block], Block]:
    """Create a wrapper function for a unary PyTorch operation.

    Args:
        name: Name of the operation
        torch_fn: PyTorch function to wrap

    Returns:
        Wrapper function that operates on Blocks
    """

    def wrapper(block: Block) -> Block:
        # Apply the operation to each tensor in the block
        result_torch: List[torch.Tensor] = [
            torch_fn(t.to_torch()) for t in block.to_list()
        ]

        result_list: List[Tensor] = [Tensor(t) for t in result_torch]
        result_block = Block.from_list(result_list, shape=block._shape)  # type: ignore[attr-defined]
        track_source_blocks(result_block, block)
        return result_block

    wrapper.__name__ = name
    wrapper.__doc__ = f"""{name.replace('_', ' ').title()} operation.

    Applies torch.{torch_fn.__name__} element-wise to each tensor in the block.

    Args:
        block: Input block

    Returns:
        Block with operation applied element-wise
    """
    return wrapper


# Mapping of ttl.math unary operations to PyTorch functions
# Only includes simple unary functions from TTLangSpecification.md
# Note: abs and neg are operators (__abs__, __neg__), not ttl.math functions
_TORCH_UNARY_OPS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # Basic unary math functions (from spec)
    "exp": torch.exp,
    "exp2": torch.exp2,
    "expm1": torch.expm1,
    "log": torch.log,
    "logp1": torch.log1p,  # spec calls it logp1, PyTorch calls it log1p
    "sqrt": torch.sqrt,
    "square": torch.square,
    "rsqrt": torch.rsqrt,
    "recip": torch.reciprocal,
    # Trigonometric unary math functions (from spec)
    "tan": torch.tan,
    "tanh": torch.tanh,
    "atan": torch.atan,
    "atanh": torch.atanh,
    "sin": torch.sin,
    "asin": torch.asin,
    "asinh": torch.asinh,
    "cos": torch.cos,
    "acos": torch.acos,
    "acosh": torch.acosh,
    # Simple activation functions (from spec) - no parameters
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "gelu": torch.nn.functional.gelu,
    "silu": torch.nn.functional.silu,
    "softsign": torch.nn.functional.softsign,  # type: ignore[dict-item]
    "hardsigmoid": torch.nn.functional.hardsigmoid,
    "selu": torch.nn.functional.selu,
}

# Auto-generate all simple unary operation functions
for _op_name, _torch_fn in _TORCH_UNARY_OPS.items():
    globals()[_op_name] = _create_unary_op_wrapper(
        _op_name, _torch_fn  # type: ignore[arg-type]
    )


# Helper function for binary operations
def _apply_binary_op(
    a: Block, b: Block, op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Block:
    """Apply a binary operation element-wise to two blocks.

    Both blocks must have the same shape; broadcasting between blocks of different
    shapes is not supported by this helper (use Block operator overloads instead).

    Args:
        a: First input block
        b: Second input block
        op: Binary operation to apply (takes two torch tensors)

    Returns:
        Block with operation applied element-wise

    Raises:
        ValueError: If a and b have different shapes.
    """
    a_shape = a._shape  # type: ignore[attr-defined]
    b_shape = b._shape  # type: ignore[attr-defined]
    if a_shape != b_shape:
        raise ValueError(
            f"Shape mismatch in binary op: a has shape {a_shape}, b has shape {b_shape}"
        )
    a_tensors = [t.to_torch() for t in a.to_list()]
    b_tensors = [t.to_torch() for t in b.to_list()]
    result_torch: List[torch.Tensor] = [
        op(a_t, b_t) for a_t, b_t in zip(a_tensors, b_tensors)
    ]
    result_list: List[Tensor] = [Tensor(t) for t in result_torch]

    result_block = Block.from_list(result_list, shape=a_shape)  # type: ignore[attr-defined]
    track_source_blocks(result_block, a, b)
    return result_block


# Helper function for unary operations with parameters
def _apply_unary_with_params(
    block: Block, op: Callable[[torch.Tensor], torch.Tensor]
) -> Block:
    """Apply a unary operation with parameters to each tensor in a block.

    Args:
        block: Input block
        op: Unary operation to apply (takes a torch tensor, returns a torch tensor)

    Returns:
        Block with operation applied element-wise
    """
    result_torch: List[torch.Tensor] = [op(t.to_torch()) for t in block.to_list()]
    result_list: List[Tensor] = [Tensor(t) for t in result_torch]

    result_block = Block.from_list(result_list, shape=block._shape)  # type: ignore[attr-defined]
    track_source_blocks(result_block, block)
    return result_block


# Binary operations
def max(a: Block, b: Block) -> Block:
    """Element-wise maximum of two blocks.

    Args:
        a: First input block
        b: Second input block

    Returns:
        Block with element-wise maximum
    """
    return _apply_binary_op(a, b, torch.maximum)


def min(a: Block, b: Block) -> Block:
    """Element-wise minimum of two blocks.

    Args:
        a: First input block
        b: Second input block

    Returns:
        Block with element-wise minimum
    """
    return _apply_binary_op(a, b, torch.minimum)


# Unary operations with scalar parameters
def rsub(a: Block, b: int) -> Block:
    """Subtract a from b where b is scalar unsigned integer (b - a).

    Args:
        a: Input block
        b: Scalar unsigned integer

    Returns:
        Block with b - a computed element-wise
    """
    return _apply_unary_with_params(a, lambda t: torch.tensor(b) - t)


# Activation functions with parameters
def relu_max(expr: Block, upper_limit: int) -> Block:
    """ReLU with upper limit.

    Equivalent to: ttl.math.relu(ttl.math.min(x, upper_limit))

    Args:
        expr: Input block
        upper_limit: Positive integer upper limit

    Returns:
        Block with ReLU applied with upper clipping
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.relu(t), max=upper_limit)

    return _apply_unary_with_params(expr, _op)


def relu_min(expr: Block, lower_limit: int) -> Block:
    """ReLU with lower limit.

    Equivalent to: ttl.math.relu(ttl.math.max(x, lower_limit))

    Args:
        expr: Input block
        lower_limit: Positive integer lower limit

    Returns:
        Block with ReLU applied with lower clipping
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.clamp(t, min=lower_limit))

    return _apply_unary_with_params(expr, _op)


def leaky_relu(expr: Block, slope: float) -> Block:
    """Leaky ReLU activation.

    Args:
        expr: Input block
        slope: Slope for negative values

    Returns:
        Block with Leaky ReLU applied
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(t, negative_slope=slope)

    return _apply_unary_with_params(expr, _op)


def elu(expr: Block, alpha: float) -> Block:
    """ELU activation.

    Args:
        expr: Input block
        alpha: Alpha parameter

    Returns:
        Block with ELU applied
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(t, alpha=alpha)

    return _apply_unary_with_params(expr, _op)


def celu(expr: Block, alpha: float, alpha_recip: float) -> Block:
    """CELU activation.

    Args:
        expr: Input block
        alpha: Alpha parameter
        alpha_recip: Reciprocal of alpha (for API compatibility)

    Returns:
        Block with CELU applied
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.celu(t, alpha=alpha)

    return _apply_unary_with_params(expr, _op)


def prelu(expr: Block, alpha: float) -> Block:
    """PReLU activation.

    Args:
        expr: Input block
        alpha: Slope for negative values

    Returns:
        Block with PReLU applied
    """
    # PyTorch's prelu expects weight parameter, use leaky_relu for scalar alpha

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(t, negative_slope=alpha)

    return _apply_unary_with_params(expr, _op)


def softplus(
    expr: Block, beta: float, beta_reciprocal: float, threshold: float
) -> Block:
    """Softplus activation.

    Args:
        expr: Input block
        beta: Beta parameter
        beta_reciprocal: Reciprocal of beta (for API compatibility)
        threshold: Threshold value

    Returns:
        Block with Softplus applied
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(t, beta=beta, threshold=threshold)

    return _apply_unary_with_params(expr, _op)


def hardtanh(expr: Block, min_val: float, max_val: float) -> Block:
    """Hardtanh activation.

    Args:
        expr: Input block
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Block with Hardtanh applied
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.hardtanh(t, min_val=min_val, max_val=max_val)

    return _apply_unary_with_params(expr, _op)


def _reduce_impl(
    block: Block,
    scaler: Block,
    dims: List[int],
    op: str,  # 'sum' or 'max'
) -> Block:
    """Shared implementation for reduce_sum and reduce_max over an ND block grid.

    Reduces the block along specified grid dimensions using torch operations.
    Each reduced dimension collapses to size 1 in the resulting grid.

    Dimension indexing uses the innermost-first convention: dims=[0] refers to
    the innermost (last) dimension of the block shape, dims=[1] to the next-to-
    innermost, and so on.

    Args:
        block: Input block.
        scaler: Scaler block; its first tile is multiplied into every result tile.
        dims: Grid dimensions to reduce over (0=innermost).
        op: 'sum' or 'max'.

    Returns:
        Reduced block with grid shape having each dimension in dims collapsed to 1.
    """
    block_shape = block._shape  # type: ignore[attr-defined]
    ndim = len(block_shape)
    dims_set: Set[int] = set(dims)

    for d in dims_set:
        if d >= ndim:
            raise ValueError(
                f"Cannot reduce along dimension {d}: block grid has only {ndim} dimensions"
            )

    # Translate user-facing dims (0=innermost) to internal grid dims (0=outermost).
    internal_dims_set = {ndim - 1 - d for d in dims_set}

    # Get the scaler
    scaler_tile = scaler.to_list()[0].to_torch()

    # Compute result grid shape
    result_shape = tuple(
        1 if i in internal_dims_set else block_shape[i] for i in range(ndim)
    )

    # Stack input tiles to reshape for reduction
    # Each output grid position gets contributions from multiple input positions
    input_tensors = [t.to_torch() for t in block.to_list()]
    result_tensors: List[Tensor] = []

    for out_idx in _iter_product(*[range(s) for s in result_shape]):
        # Collect all input tiles that contribute to this output position
        in_ranges = [
            (
                range(block_shape[i])
                if i in internal_dims_set
                else range(out_idx[i], out_idx[i] + 1)
            )
            for i in range(ndim)
        ]

        # Gather contributing tiles
        contributing_tiles: List[torch.Tensor] = []
        for in_idx in _iter_product(*in_ranges):
            flat = sum(
                in_idx[i] * _math.prod(block_shape[i + 1 :]) for i in range(ndim)
            )
            contributing_tiles.append(input_tensors[flat])

        # Reduce across contributing tiles using torch operations
        if len(contributing_tiles) == 1:
            result_tile = contributing_tiles[0]
        else:
            # Stack and reduce
            stacked = torch.stack(contributing_tiles, dim=0)
            if op == "sum":
                result_tile = stacked.sum(dim=0)
            else:  # max
                result_tile = stacked.max(dim=0).values

        # Apply scaler
        result_tensors.append(Tensor(result_tile * scaler_tile))

    result_block = Block.from_list(result_tensors, shape=result_shape)
    track_source_blocks(result_block, block, scaler)
    return result_block


def reduce_max(
    block: Block,
    scaler: Block,
    _output_hint: Optional[Block] = None,
    dims: Optional[List[int]] = None,
) -> Block:
    """Scaled maximum reduction over an ND block grid.

    See _reduce_impl for full semantics. dims must be non-empty and every
    element must be a valid grid dimension index.

    Args:
        block: Input block.
        scaler: Scaler block; its first tile is multiplied into every result tile.
        _output_hint: Unused output block hint (kept for API compatibility).
        dims: Grid dimensions to reduce over (0-indexed).

    Returns:
        Block with reduced dimensions.
    """
    if dims is None or not dims:
        raise ValueError("dims parameter must contain at least one dimension")
    return _reduce_impl(block, scaler, dims, "max")


def reduce_sum(
    block: Block,
    scaler: Block,
    _output_hint: Optional[Block] = None,
    dims: Optional[List[int]] = None,
) -> Block:
    """Scaled sum reduction over an ND block grid.

    See _reduce_impl for full semantics. dims must be non-empty and every
    element must be a valid grid dimension index.

    Args:
        block: Input block.
        scaler: Scaler block; its first tile is multiplied into every result tile.
        _output_hint: Unused output block hint (kept for API compatibility).
        dims: Grid dimensions to reduce over (0-indexed).

    Returns:
        Block with reduced dimensions.
    """
    if dims is None or not dims:
        raise ValueError("dims parameter must contain at least one dimension")
    return _reduce_impl(block, scaler, dims, "sum")


# Clean up temporary variables
for _name in ["_op_name", "_torch_fn"]:
    globals().pop(_name, None)


def transpose(block: Block, _output_hint: Optional[Block] = None) -> Block:
    """Transpose a 2D tile tensor (swap width and height).

    Performs width-height transpose on input tiles. Each 32x32 tile has its
    rows and columns swapped.

    The input tensor shape [M, N] becomes output shape [N, M] in tiles.

    Args:
        block: Input block with shape (M, N)
        _output_hint: Optional output block hint (unused in simulator)

    Returns:
        Block with shape (N, M), where each tile is transposed
    """
    if len(block._shape) != 2:  # type: ignore[attr-defined]
        raise ValueError(
            f"transpose requires a 2-D block grid, got shape {block._shape}"  # type: ignore[attr-defined]
        )

    # Transpose each tile (swap rows/columns within tiles)
    transposed_tiles = [Tensor(t.to_torch().T) for t in block.to_list()]

    # Also swap the tile grid dimensions: (M, N) -> (N, M)
    M, N = block._shape  # type: ignore[attr-defined]

    # Reorder tiles to match transposed grid: tile[i,j] -> tile[j,i]
    reordered_tiles: List[Tensor] = []
    for j in range(N):
        for i in range(M):
            reordered_tiles.append(transposed_tiles[i * N + j])

    result_block = Block.from_list(reordered_tiles, shape=(N, M))
    track_source_blocks(result_block, block)
    return result_block
