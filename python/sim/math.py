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
from typing import Callable, List, Optional, Set, Tuple

import torch

from .dfb import Block, check_same_layout, track_source_blocks, matmul
from .ttnnsim import ROW_MAJOR_LAYOUT, Tensor
from .typedefs import PositiveInt

_ = matmul


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
        layout = block.layout
        result_torch: List[torch.Tensor] = [
            torch_fn(t.to_torch()) for t in block.to_list()
        ]

        result_list: List[Tensor] = [Tensor(t, layout) for t in result_torch]
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
_TORCH_UNARY_OPS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # Basic unary math functions (from spec)
    "abs": torch.abs,
    "neg": torch.neg,
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
    # Rounding functions (from spec) - simple unary
    "floor": torch.floor,
    "ceil": torch.ceil,
    "frac": torch.frac,
    "trunc": torch.trunc,
    "sign": torch.sign,
    "signbit": torch.signbit,
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
    # Layout is checked before shape so a layout error wins when both
    # mismatch (see ``block._apply_binary_op`` for the rationale).
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
    layout = block.layout
    result_torch: List[torch.Tensor] = [op(t.to_torch()) for t in block.to_list()]
    result_list: List[Tensor] = [Tensor(t, layout) for t in result_torch]

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
def rsub(a: Block, b: PositiveInt) -> Block:
    """Subtract a from b where b is scalar unsigned integer (b - a).

    Args:
        a: Input block
        b: Scalar unsigned integer

    Returns:
        Block with b - a computed element-wise
    """
    return _apply_unary_with_params(a, lambda t: torch.tensor(b) - t)


# Activation functions with parameters
def relu_max(expr: Block, upper_limit: PositiveInt) -> Block:
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


def relu_min(expr: Block, lower_limit: PositiveInt) -> Block:
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


def leaky_relu(expr: Block, slope: PositiveInt) -> Block:
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


def elu(expr: Block, alpha: PositiveInt) -> Block:
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


def celu(expr: Block, alpha: PositiveInt, alpha_recip: PositiveInt) -> Block:
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


def prelu(expr: Block, alpha: PositiveInt) -> Block:
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
    expr: Block, beta: PositiveInt, beta_reciprocal: PositiveInt, threshold: PositiveInt
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


def hardtanh(expr: Block, min_val: PositiveInt, max_val: PositiveInt) -> Block:
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


# Rounding functions with parameters
def round(expr: Block, decimals: PositiveInt = 0) -> Block:
    """Round to specified number of decimal places.

    Args:
        expr: Input block
        decimals: Number of decimal places to round to

    Returns:
        Block with values rounded to specified decimal places
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.round(t, decimals=decimals)

    return _apply_unary_with_params(expr, _op)


def clamp(expr: Block, min: PositiveInt, max: PositiveInt) -> Block:
    """Clamp values to specified min and max.

    Args:
        expr: Input block
        min: Minimum value
        max: Maximum value

    Returns:
        Block with values clamped to [min, max]
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t, min=min, max=max)

    return _apply_unary_with_params(expr, _op)


def threshold(expr: Block, threshold: PositiveInt, value: PositiveInt) -> Block:
    """Replace values greater than threshold with specified value.

    Args:
        expr: Input block
        threshold: Threshold value
        value: Replacement value for elements > threshold

    Returns:
        Block with thresholding applied
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        # Spec: replace values GREATER THAN threshold (not <= like torch.threshold)
        return torch.where(t > threshold, torch.tensor(value, dtype=t.dtype), t)

    return _apply_unary_with_params(expr, _op)


def _reduce_impl(
    block: Block,
    dims: List[int],
    shape: Tuple[int, ...],
    op: str,  # 'sum' or 'max'
) -> Block:
    """Shared implementation for reduce_sum and reduce_max over an ND block grid.

    Reduces the block along specified grid dimensions using torch operations.
    Each reduced dimension collapses to size 1 in the resulting grid.

    Dimension indexing uses standard Python convention: positive dim 0 is the
    outermost dimension, dim 1 is the next, and so on. Negative dims count from
    the innermost: dim -1 is the innermost (last) dimension, dim -2 is the
    next-to-innermost, and so on.

    Args:
        block: Input block.
        dims: Grid dimensions to reduce over (standard Python indexing).
        shape: Expected result grid shape; must contain 1 in each reduced dimension.
        op: 'sum' or 'max'.

    Returns:
        Reduced block with the specified result shape.
    """
    block_shape = block._shape  # type: ignore[attr-defined]
    ndim = len(block_shape)
    dims_set: Set[int] = set(dims)
    shape = tuple(shape)

    if block.layout == ROW_MAJOR_LAYOUT:
        raise ValueError("reduce is not supported for Row-Major layout blocks")

    if len(shape) != ndim:
        raise ValueError(
            f"reduce shape {shape} has {len(shape)} dimensions but block has {ndim}"
        )

    for d in dims_set:
        if d >= ndim or d < -ndim:
            raise ValueError(
                f"Cannot reduce along dimension {d}: block grid has only {ndim} dimensions"
            )

    # Translate user-facing dims to internal grid indices using standard Python
    # indexing: d % ndim maps both positive and negative dims correctly.
    internal_dims_set = {d % ndim for d in dims_set}

    # Compute and validate result grid shape
    result_shape = tuple(
        1 if i in internal_dims_set else block_shape[i] for i in range(ndim)
    )
    if shape != result_shape:
        raise ValueError(
            f"reduce shape {shape} does not match expected result shape {result_shape} "
            f"(block shape {block_shape}, reducing dims {dims})"
        )

    # Stack input tiles to reshape for reduction
    # Each output grid position gets contributions from multiple input positions
    input_tensors = [t.to_torch() for t in block.to_list()]

    # Spec step (2) fires when the user reduces along one or both of the last
    # two (tile) dimensions of `block_shape`.  In that case the within-tile
    # collapse stores the scalar/row/column result in row 0 / col 0 / (0, 0)
    # of each output tile, matching the spec's data-placement convention.
    reduce_row = (ndim - 2) in internal_dims_set
    reduce_col = (ndim - 1) in internal_dims_set

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

        # Spec step (1): elementwise reduce across contributing tiles.
        if len(contributing_tiles) == 1:
            result_tile = contributing_tiles[0]
        else:
            stacked = torch.stack(contributing_tiles, dim=0)
            if op == "sum":
                result_tile = stacked.sum(dim=0)
            else:  # max
                result_tile = stacked.max(dim=0).values

        # Spec step (2): within-tile reduce along the requested tile dims.
        # The result lives at position 0 of each collapsed dim (per the spec
        # convention - col 0 / row 0 / (0, 0)); the rest of the tile is
        # filled with zeros.  Index expressions use Ellipsis so the same
        # branches work for 1-D, 2-D and 3-D tile shapes.
        if reduce_row or reduce_col:
            new_tile = torch.zeros_like(result_tile)
            if reduce_row and reduce_col:
                if op == "sum":
                    seed = result_tile.sum(dim=(-1, -2))
                else:
                    seed = result_tile.amax(dim=(-1, -2))
                new_tile[(..., 0, 0)] = seed
            elif reduce_col:
                if op == "sum":
                    seed = result_tile.sum(dim=-1)
                else:
                    seed = result_tile.amax(dim=-1)
                new_tile[(..., 0)] = seed
            else:  # reduce_row only
                if op == "sum":
                    seed = result_tile.sum(dim=-2)
                else:
                    seed = result_tile.amax(dim=-2)
                new_tile[(..., 0, slice(None))] = seed
            result_tile = new_tile

        result_tensors.append(Tensor(result_tile, block.layout))

    result_block = Block.from_list(result_tensors, shape=result_shape)
    track_source_blocks(result_block, block)
    return result_block


def reduce_max(
    block: Block,
    dims: List[int],
    shape: Tuple[int, ...],
) -> Block:
    """Maximum reduction over an ND block grid.

    Reduces the block along specified grid dimensions by taking the element-wise
    maximum across contributing tiles. Not supported for Row-Major layout blocks.

    Dimension indexing uses standard Python convention: positive dim 0 is the
    outermost dimension, negative dim -1 is the innermost.

    Args:
        block: Input block.
        dims: Grid dimensions to reduce over (standard Python indexing).
        shape: Result grid shape; must contain 1 in each dimension in dims
            and match the block shape in all other dimensions.

    Returns:
        Block with reduced dimensions matching shape.
    """
    if not dims:
        raise ValueError("dims parameter must contain at least one dimension")
    return _reduce_impl(block, dims, shape, "max")


def reduce_sum(
    block: Block,
    dims: List[int],
    shape: Tuple[int, ...],
) -> Block:
    """Sum reduction over an ND block grid.

    Reduces the block along specified grid dimensions by summing contributing
    tiles element-wise. Not supported for Row-Major layout blocks.

    Dimension indexing uses standard Python convention: positive dim 0 is the
    outermost dimension, negative dim -1 is the innermost.

    Args:
        block: Input block.
        dims: Grid dimensions to reduce over (standard Python indexing).
        shape: Result grid shape; must contain 1 in each dimension in dims
            and match the block shape in all other dimensions.

    Returns:
        Block with reduced dimensions matching shape.
    """
    if not dims:
        raise ValueError("dims parameter must contain at least one dimension")
    return _reduce_impl(block, dims, shape, "sum")


# Clean up temporary variables
_cleanup_name: Optional[str] = None
for _cleanup_name in ("_op_name", "_torch_fn"):
    globals().pop(_cleanup_name, None)
if _cleanup_name is not None:  # Always true after loop executes
    del _cleanup_name
