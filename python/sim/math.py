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
from typing import Callable, List, Optional, Set, Union

import torch

from .constants import TILE_SHAPE
from .context import get_context
from .diagnostics import warn_once_per_location
from .greenlet_scheduler import get_current_core_id
from .dfb import Block, track_source_blocks, matmul
from .blockstate import BlockAcquisition, ThreadType
from .ttnnsim import Tensor, TILE_LAYOUT
from .typedefs import PositiveInt

_ = matmul


def _warn_1d_broadcast_unsupported() -> None:
    """Issue a warning that 1D broadcast is not supported on current hardware.

    Tracks which cores hit each source location and only prints once per location,
    showing the list of cores that encountered the issue.
    """
    warn_once_per_location(
        get_context().warnings.broadcast_1d_warnings,
        "1D broadcast is not supported on current hardware",
        get_current_core_id(),
    )


def broadcast(
    block: Block,
    output_hint: Optional[Block] = None,
    *,
    dims: Optional[List[int]] = None,
    shape: Optional[List[int]] = None,
) -> Block:
    """Broadcast a block along specified dimensions.

    Supports three modes:

    1. **Spec form** (``shape`` provided): expand each grid dim listed in
       ``dims`` from 1 to the corresponding ``shape`` entry by replicating
       tiles. Matches the ``ttl.block.broadcast(expr, dims, shape)``
       compiler API at the grid level.
    2. **Eager expansion** (``output_hint`` provided): expand the per-tile
       buffer to match the output hint's element shape via torch.expand.
       Used by legacy sim call sites that allocate 1x1 element-size tiles.
    3. **Lazy expansion** (neither provided): mark the block with broadcast
       metadata; expansion happens later when the block is stored or used.

    Dimension indexing uses standard Python convention: positive dim 0 is
    the outermost; ``-1`` is the innermost.

    Args:
        block: Input block to broadcast
        output_hint: Optional output block providing target shape (legacy form)
        dims: List of dimension indices to broadcast along
        shape: Target shape (spec form)

    Returns:
        Materialized Block in modes 1 and 2; the same block with broadcast
        metadata in mode 3.
    """
    if dims is None:
        raise ValueError("dims parameter is required for broadcast()")

    block_shape = block._shape  # type: ignore[attr-defined]
    element_shape = block._element_shape  # type: ignore[attr-defined]
    ndim = len(block_shape)

    if ndim == 1:
        _warn_1d_broadcast_unsupported()

    for dim in dims:
        if dim >= ndim or dim < -ndim:
            raise ValueError(
                f"Cannot broadcast along dimension {dim}: block has shape {block_shape} "
                f"with only {ndim} dimensions"
            )

    if shape is not None:
        # Spec-form: shape is a Shape (tuple/list of grid dims).
        target_shape = tuple(shape)
        target_element_shape = None
        if len(target_shape) != ndim:
            raise ValueError(
                f"shape size {len(target_shape)} does not match input rank {ndim}"
            )
        norm_dims = {d + ndim if d < 0 else d for d in dims}
        for i in range(ndim):
            if i in norm_dims:
                if block_shape[i] != 1:
                    raise ValueError(
                        f"broadcast dim {i} requires input grid shape 1, got "
                        f"{block_shape[i]}"
                    )
            elif block_shape[i] != target_shape[i]:
                raise ValueError(
                    f"Non-broadcast dim {i}: input has {block_shape[i]} but "
                    f"shape has {target_shape[i]}"
                )

        # For TILE_LAYOUT inputs the spec requires within-tile expansion on
        # broadcast dims that touch the innermost two axes (every "tile" is
        # logically TILE_SHAPE even when the dfb element shape stores a
        # degenerate size-1 dim, e.g. a (N, 1) source tensor). Derive the
        # effective per-tile size accordingly.
        is_tile_layout = block.layout == TILE_LAYOUT

        def _per_tile_size(i: int) -> int:
            base = element_shape[i] // block_shape[i] if block_shape[i] else 1
            if is_tile_layout and i >= ndim - 2 and base == 1:
                return TILE_SHAPE[i - (ndim - 2)]
            return base if base > 0 else 1

        target_element_shape = tuple(
            target_shape[i] * _per_tile_size(i) for i in range(ndim)
        )

        src_tensor = block._buf.to_torch()  # type: ignore[attr-defined]
        # Pad degenerate-tile broadcast dims to the full per-tile size so the
        # repeat below produces TILE_SHAPE-aligned output tiles.
        pad_factors = tuple(
            _per_tile_size(i) if (i in norm_dims and element_shape[i] == 1) else 1
            for i in range(ndim)
        )
        if any(f != 1 for f in pad_factors):
            src_tensor = src_tensor.repeat(*pad_factors)
        # Block-level broadcast replicates whole tiles along broadcast dims;
        # torch.repeat (not expand) handles non-singleton source dims, which
        # arise when the per-tile element width is > 1.
        src_shape = src_tensor.shape
        repeat_factors = tuple(
            target_element_shape[i] // src_shape[i] if src_shape[i] > 0 else 1
            for i in range(ndim)
        )
        expanded_tensor = src_tensor.repeat(*repeat_factors)
        result_block = Block(
            tensor=Tensor(expanded_tensor.contiguous()),
            shape=target_shape,
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.COMPUTE,
            is_temporary=True,
        )
        track_source_blocks(result_block, block)
        return result_block

    for dim in dims:
        if element_shape[dim] != 1:
            raise ValueError(
                f"Cannot broadcast along dimension {dim}: dimension must have element size 1, "
                f"but has element size {element_shape[dim]}"
            )

    if output_hint is not None:
        target_shape = output_hint._shape  # type: ignore[attr-defined]
        target_element_shape = output_hint._element_shape  # type: ignore[attr-defined]

        if len(target_shape) != len(block_shape):
            raise ValueError(
                f"Broadcast output hint has {len(target_shape)} dimensions, "
                f"but source block has {len(block_shape)} dimensions"
            )

        src_tensor = block._buf.to_torch()  # type: ignore[attr-defined]
        expanded_tensor = src_tensor.expand(*target_element_shape)

        result_block = Block(
            tensor=Tensor(expanded_tensor.contiguous()),
            shape=target_shape,
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.COMPUTE,
            is_temporary=True,
        )
        track_source_blocks(result_block, block)
        return result_block

    block._broadcast_dims = tuple(dims)  # type: ignore[attr-defined]
    return block


def block_fill(value, *, shape, dtype=None) -> Block:
    """Spec-form ttl.block.fill(value, shape).

    Returns a temporary tiled Block of the given grid ``shape`` filled with
    ``value``. ``dtype`` defaults to bf16 to match the spec/sim convention;
    pass a ttnn/torch dtype to override.
    """
    shape_tuple = tuple(int(s) for s in shape)
    if len(shape_tuple) < 2:
        raise ValueError(
            "fill requires a shape with at least 2 dimensions for tiled layout"
        )
    if any(s <= 0 for s in shape_tuple):
        raise ValueError(f"fill shape must be all-positive, got {shape_tuple}")

    # In the sim, ttnn dtypes (ttnn.bfloat16, ttnn.float32, ...) are aliased
    # directly to the matching torch.dtype, so the same value works either way.
    torch_dtype = torch.bfloat16 if dtype is None else dtype

    tile_h, tile_w = TILE_SHAPE
    batch = shape_tuple[:-2]
    rows_tiles, cols_tiles = shape_tuple[-2], shape_tuple[-1]
    elem = torch.full(
        (*batch, rows_tiles * tile_h, cols_tiles * tile_w),
        float(value),
        dtype=torch_dtype,
    )
    return Block(
        tensor=Tensor(elem),
        shape=shape_tuple,
        acquisition=BlockAcquisition.RESERVE,
        thread_type=ThreadType.COMPUTE,
        is_temporary=True,
    )


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
    """Apply a ternary operation element-wise to three blocks.

    All blocks must have the same shape.

    Args:
        a: First input block
        b: Second input block
        c: Third input block
        op: Ternary operation to apply (takes three torch tensors)

    Returns:
        Block with operation applied element-wise

    Raises:
        ValueError: If blocks have different shapes.
    """
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


# Fill, mask and where functions
def fill(out_blk: Block, value: float) -> Block:
    """Return a temporary block with the same shape as out_blk filled with value.

    Args:
        out_blk: Block whose shape determines the result shape.
        value: The scalar value to fill every element with.

    Returns:
        A temporary Block with the same shape as out_blk, every element set to value.
    """

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, value)

    return _apply_unary_with_params(out_blk, _op)


def mask(expr: Block, mask: Block) -> Block:
    """Mask a block by replacing masked elements with 0.

    Args:
        expr: Input block
        mask: Mask block (elements equal to 1 are masked)

    Returns:
        Block with masked elements replaced by 0
    """

    def _op(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Mask: where mask == 1, replace with 0, else keep original
        return torch.where(t2 == 1, torch.tensor(0.0, dtype=t1.dtype), t1)

    return _apply_binary_op(expr, mask, _op)


def mask_posinf(expr: Block, mask: Block) -> Block:
    """Mask a block by replacing masked elements with positive infinity.

    Args:
        expr: Input block
        mask: Mask block (elements equal to 1 are masked)

    Returns:
        Block with masked elements replaced by positive infinity
    """

    def _op(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Mask: where mask == 1, replace with +inf, else keep original
        return torch.where(t2 == 1, torch.tensor(float("inf"), dtype=t1.dtype), t1)

    return _apply_binary_op(expr, mask, _op)


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


def _reduce_impl(
    block: Block,
    scaler: Union[Block, int, float],
    dims: List[int],
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
        scaler: Scaler block or numeric constant multiplied into every result
            tile.
        dims: Grid dimensions to reduce over (standard Python indexing).
        op: 'sum' or 'max'.

    Returns:
        Reduced block with grid shape having each dimension in dims collapsed to 1.
    """
    block_shape = block._shape  # type: ignore[attr-defined]
    ndim = len(block_shape)
    dims_set: Set[int] = set(dims)

    for d in dims_set:
        if d >= ndim or d < -ndim:
            raise ValueError(
                f"Cannot reduce along dimension {d}: block grid has only {ndim} dimensions"
            )

    # Translate user-facing dims to internal grid indices using standard Python
    # indexing: d % ndim maps both positive and negative dims correctly.
    internal_dims_set = {d % ndim for d in dims_set}

    # Compute result grid shape
    result_shape = tuple(
        1 if i in internal_dims_set else block_shape[i] for i in range(ndim)
    )

    # Stack input tiles to reshape for reduction
    # Each output grid position gets contributions from multiple input positions
    input_tensors = [t.to_torch() for t in block.to_list()]

    # Get the scaler. Numeric constants match compiler lowering, which
    # materializes them as a 1x1 fill tile before reduce.
    if isinstance(scaler, Block):
        scaler_tile = scaler.to_list()[0].to_torch()
    else:
        scaler_tile = torch.full_like(input_tensors[0], float(scaler))

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
        result_tensors.append(Tensor(result_tile * scaler_tile, block.layout))

    result_block = Block.from_list(result_tensors, shape=result_shape)
    if isinstance(scaler, Block):
        track_source_blocks(result_block, block, scaler)
    else:
        track_source_blocks(result_block, block)
    return result_block


def reduce_max(
    block: Block,
    scaler: Union[Block, int, float],
    _output_hint: Optional[Block] = None,
    dims: Optional[List[int]] = None,
) -> Block:
    """Scaled maximum reduction over an ND block grid.

    See _reduce_impl for full semantics. dims must be non-empty and every
    element must be a valid grid dimension index.

    Args:
        block: Input block.
        scaler: Scaler block or numeric constant multiplied into every result
            tile.
        _output_hint: Unused output block hint (kept for API compatibility).
        dims: Grid dimensions to reduce over (standard Python indexing).

    Returns:
        Block with reduced dimensions.
    """
    if dims is None or not dims:
        raise ValueError("dims parameter must contain at least one dimension")
    return _reduce_impl(block, scaler, dims, "max")


def reduce_sum(
    block: Block,
    scaler: Union[Block, int, float],
    _output_hint: Optional[Block] = None,
    dims: Optional[List[int]] = None,
) -> Block:
    """Scaled sum reduction over an ND block grid.

    See _reduce_impl for full semantics. dims must be non-empty and every
    element must be a valid grid dimension index.

    Args:
        block: Input block.
        scaler: Scaler block or numeric constant multiplied into every result
            tile.
        _output_hint: Unused output block hint (kept for API compatibility).
        dims: Grid dimensions to reduce over (standard Python indexing).

    Returns:
        Block with reduced dimensions.
    """
    if dims is None or not dims:
        raise ValueError("dims parameter must contain at least one dimension")
    return _reduce_impl(block, scaler, dims, "sum")


# Clean up temporary variables
_cleanup_name: Optional[str] = None
for _cleanup_name in ("_op_name", "_torch_fn"):
    globals().pop(_cleanup_name, None)
if _cleanup_name is not None:  # Always true after loop executes
    del _cleanup_name


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
    layout = block.layout
    transposed_tiles = [Tensor(t.to_torch().T, layout) for t in block.to_list()]

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
