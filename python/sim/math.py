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

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import torch

from .block import Block, BlockAcquisition, ThreadType

if TYPE_CHECKING:
    from .cb import ReserveContext, WaitContext


def _track_source_blocks(result_block: Block, *input_blocks: Block) -> None:
    """Track source wait() blocks for proper state management.

    Adds input wait() blocks to the result block's _source_blocks list so that
    when the result is stored, the sources can be marked as consumed.

    Args:
        result_block: The result block to track sources for
        *input_blocks: Input blocks that contributed to the result
    """
    for block in input_blocks:
        # Unwrap context managers if needed
        actual_block = block
        if hasattr(block, "_block"):
            actual_block = block._block  # type: ignore[attr-defined]

        if isinstance(actual_block, Block):
            if (
                not actual_block._is_temporary  # type: ignore[attr-defined]
                and actual_block._acquisition == BlockAcquisition.WAIT  # type: ignore[attr-defined]
                and actual_block._thread_type == ThreadType.COMPUTE  # type: ignore[attr-defined]
            ):
                result_block._source_blocks.append(actual_block)  # type: ignore[attr-defined]
            elif actual_block._is_temporary:  # type: ignore[attr-defined]
                result_block._source_blocks.extend(actual_block._source_blocks)  # type: ignore[attr-defined]


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

    # Broadcasting is now implicit - return a new temporary block that preserves source tracking
    # Create a new block to avoid modifying the original, and preserve source blocks
    result_block = Block.from_list(actual_block.to_list(), actual_block._shape)  # type: ignore[attr-defined]

    # Preserve source block tracking for wait() blocks
    if hasattr(actual_block, "_source_blocks"):
        result_block._source_blocks = actual_block._source_blocks.copy()  # type: ignore[attr-defined]

    # If actual_block itself is a wait() block, add it to source_blocks
    if (
        hasattr(actual_block, "_is_temporary")
        and not actual_block._is_temporary  # type: ignore[attr-defined]
        and hasattr(actual_block, "_acquisition")
        and actual_block._acquisition
        == BlockAcquisition.WAIT  # type: ignore[attr-defined]
        and hasattr(actual_block, "_thread_type")
        and actual_block._thread_type
        == ThreadType.COMPUTE  # type: ignore[attr-defined]
    ):
        result_block._source_blocks.append(actual_block)  # type: ignore[attr-defined]

    return result_block


# Helper function to create unary operation wrappers
def _create_unary_op_wrapper(name: str, torch_fn: Callable) -> Callable:
    """Create a wrapper function for a unary PyTorch operation.

    Args:
        name: Name of the operation
        torch_fn: PyTorch function to wrap

    Returns:
        Wrapper function that operates on Blocks
    """

    def wrapper(block: Block) -> Block:
        # Apply the operation to each tensor in the block
        result_tensors = [torch_fn(t.to_torch()) for t in block.to_list()]

        # Import Tensor here to avoid circular dependency
        from .ttnnsim import Tensor

        result_list = [Tensor(t) for t in result_tensors]
        result_block = Block.from_list(result_list, shape=block._shape)  # type: ignore[attr-defined]
        _track_source_blocks(result_block, block)
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
_TORCH_UNARY_OPS = {
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
    "softsign": torch.nn.functional.softsign,
    "hardsigmoid": torch.nn.functional.hardsigmoid,
    "selu": torch.nn.functional.selu,
}

# Auto-generate all simple unary operation functions
for _op_name, _torch_fn in _TORCH_UNARY_OPS.items():
    globals()[_op_name] = _create_unary_op_wrapper(_op_name, _torch_fn)


# Helper function for binary operations
def _apply_binary_op(a: Block, b: Block, op: Callable) -> Block:
    """Apply a binary operation element-wise to two blocks.

    Args:
        a: First input block
        b: Second input block
        op: Binary operation to apply (takes two torch tensors)

    Returns:
        Block with operation applied element-wise
    """
    from .ttnnsim import Tensor

    a_tensors = [t.to_torch() for t in a.to_list()]
    b_tensors = [t.to_torch() for t in b.to_list()]
    result_tensors = [op(a_t, b_t) for a_t, b_t in zip(a_tensors, b_tensors)]
    result_list = [Tensor(t) for t in result_tensors]

    result_block = Block.from_list(result_list, shape=a._shape)  # type: ignore[attr-defined]
    _track_source_blocks(result_block, a, b)
    return result_block


# Helper function for unary operations with parameters
def _apply_unary_with_params(block: Block, op: Callable) -> Block:
    """Apply a unary operation with parameters to each tensor in a block.

    Args:
        block: Input block
        op: Unary operation to apply (takes a torch tensor, returns a torch tensor)

    Returns:
        Block with operation applied element-wise
    """
    from .ttnnsim import Tensor

    result_tensors = [op(t.to_torch()) for t in block.to_list()]
    result_list = [Tensor(t) for t in result_tensors]

    result_block = Block.from_list(result_list, shape=block._shape)  # type: ignore[attr-defined]
    _track_source_blocks(result_block, block)
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


def matmul(a: Block, b: Block, _output_hint: Optional[Block] = None) -> Block:
    """Matrix multiplication of two blocks.

    Performs matrix multiplication across the tile grid. If block a has shape (M, K)
    and block b has shape (K, N), the result will have shape (M, N).

    Each output tile [i, j] is computed as the sum of torch.matmul(a[i, k], b[k, j])
    for all k from 0 to K-1.

    Args:
        a: First input block with shape (M, K)
        b: Second input block with shape (K, N)
        _output_hint: Optional output block hint (unused in simulator)

    Returns:
        Block with shape (M, N) containing the matrix multiplication result

    Note:
        This is equivalent to the @ operator. In the spec, matmul is BlockExpr.__matmul__,
        but this function is provided for convenience in the simulator.
    """
    from .ttnnsim import Tensor

    # Get block shapes
    a_shape = a._shape  # type: ignore[attr-defined]
    b_shape = b._shape  # type: ignore[attr-defined]

    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(
            f"matmul requires 2D blocks, got shapes {a_shape} and {b_shape}"
        )

    M, K = a_shape
    K_b, N = b_shape

    if K != K_b:
        raise ValueError(
            f"Inner dimensions must match for matmul: {a_shape} @ {b_shape}"
        )

    # Get all tiles as torch tensors
    a_tensors = a.to_list()
    b_tensors = b.to_list()

    # Compute result tile-by-tile
    # Output tile [i, j] = sum over k of (a[i, k] @ b[k, j])
    result_tensors = []
    for i in range(M):
        for j in range(N):
            # Accumulate contributions from all k
            acc = None
            for k in range(K):
                a_tile = a_tensors[i * K + k].to_torch()
                b_tile = b_tensors[k * N + j].to_torch()
                partial = torch.matmul(a_tile, b_tile)

                if acc is None:
                    acc = partial
                else:
                    acc = acc + partial

            result_tensors.append(Tensor(acc))

    result_block = Block.from_list(result_tensors, shape=(M, N))
    _track_source_blocks(result_block, a, b)
    return result_block


# Unary operations with scalar parameters
def rsub(a: Block, b: int) -> Block:
    """Subtract a from b where b is scalar unsigned integer (b - a).

    Args:
        a: Input block
        b: Scalar unsigned integer

    Returns:
        Block with b - a computed element-wise
    """
    return _apply_unary_with_params(a, lambda t: b - t)


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
    return _apply_unary_with_params(
        expr, lambda t: torch.clamp(torch.relu(t), max=upper_limit)
    )


def relu_min(expr: Block, lower_limit: int) -> Block:
    """ReLU with lower limit.

    Equivalent to: ttl.math.relu(ttl.math.max(x, lower_limit))

    Args:
        expr: Input block
        lower_limit: Positive integer lower limit

    Returns:
        Block with ReLU applied with lower clipping
    """
    return _apply_unary_with_params(
        expr, lambda t: torch.relu(torch.clamp(t, min=lower_limit))
    )


def leaky_relu(expr: Block, slope: float) -> Block:
    """Leaky ReLU activation.

    Args:
        expr: Input block
        slope: Slope for negative values

    Returns:
        Block with Leaky ReLU applied
    """
    return _apply_unary_with_params(
        expr, lambda t: torch.nn.functional.leaky_relu(t, negative_slope=slope)
    )


def elu(expr: Block, alpha: float) -> Block:
    """ELU activation.

    Args:
        expr: Input block
        alpha: Alpha parameter

    Returns:
        Block with ELU applied
    """
    return _apply_unary_with_params(
        expr, lambda t: torch.nn.functional.elu(t, alpha=alpha)
    )


def celu(expr: Block, alpha: float, alpha_recip: float) -> Block:
    """CELU activation.

    Args:
        expr: Input block
        alpha: Alpha parameter
        alpha_recip: Reciprocal of alpha (for API compatibility)

    Returns:
        Block with CELU applied
    """
    return _apply_unary_with_params(
        expr, lambda t: torch.nn.functional.celu(t, alpha=alpha)
    )


def prelu(expr: Block, alpha: float) -> Block:
    """PReLU activation.

    Args:
        expr: Input block
        alpha: Slope for negative values

    Returns:
        Block with PReLU applied
    """
    # PyTorch's prelu expects weight parameter, use leaky_relu for scalar alpha
    return _apply_unary_with_params(
        expr, lambda t: torch.nn.functional.leaky_relu(t, negative_slope=alpha)
    )


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
    return _apply_unary_with_params(
        expr, lambda t: torch.nn.functional.softplus(t, beta=beta, threshold=threshold)
    )


def hardtanh(expr: Block, min_val: float, max_val: float) -> Block:
    """Hardtanh activation.

    Args:
        expr: Input block
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Block with Hardtanh applied
    """
    return _apply_unary_with_params(
        expr,
        lambda t: torch.nn.functional.hardtanh(t, min_val=min_val, max_val=max_val),
    )


def reduce_max(
    block: Block,
    scaler: Block,
    _output_hint: Optional[Block] = None,
    dims: Optional[List[int]] = None,
) -> Block:
    """Scaled maximum reduction.

    Computes the scaled maximum reduction over specified dimensions.
    The result is the maximum value along the specified dimensions, scaled by the scaler.

    Args:
        block: Input block to reduce
        scaler: Scaler block
        _output_hint: Optional output block hint (unused in simulator)
        dims: List of dimension indices to reduce over (0-indexed)
              Example: [0] for rows, [1] for columns, [0, 1] for all

    Returns:
        Block with reduced dimensions

    Example:
        # Reduce over rows (dimension 0)
        with a_cb.wait() as a_blk, s_cb.wait() as s_blk:
            result = ttl.math.reduce_max(a_blk, s_blk, dims=[0])

    From the specification:
        Scaled maximum reduction over specified dimensions.
        Example for reduction over rows: ttl.math.reduce_max(a, s, dims=[0])
        Example for reduction over rows and columns: ttl.math.reduce_max(a, s, dims=[0, 1])
    """

    if dims is None or not dims:
        raise ValueError("dims parameter must contain at least one dimension")

    # Import Tensor here to avoid circular dependency
    from .ttnnsim import Tensor

    # Reduction operates on TILE GRID: combines tiles along specified grid dimensions
    # dims=[0] means combine tiles along grid rows (element-wise max)
    # dims=[1] means combine tiles along grid columns (element-wise max)
    # The result tile grid shrinks in the reduced dimensions
    block_shape = block._shape  # type: ignore[attr-defined]
    M, N = block_shape
    input_tensors = [t.to_torch() for t in block.to_list()]

    # Validate dims
    for dim in dims:
        if dim >= 2:
            raise ValueError(
                f"Cannot reduce along dimension {dim}: block grid has only 2 dimensions"
            )

    # Calculate result grid shape
    result_M = 1 if 0 in dims else M
    result_N = 1 if 1 in dims else N

    # For each result tile position, gather and combine input tiles element-wise
    result_tensors = []
    for res_i in range(result_M):
        for res_j in range(result_N):
            # Collect all tiles that contribute
            tiles_to_max = []
            for i in range(M):
                for j in range(N):
                    # Include if dims match or dimension is being reduced
                    if (0 in dims or i == res_i) and (1 in dims or j == res_j):
                        tile_idx = i * N + j
                        tiles_to_max.append(input_tensors[tile_idx])

            # Take element-wise max across all contributing tiles
            result_tile = tiles_to_max[0]
            for tile in tiles_to_max[1:]:
                result_tile = torch.maximum(result_tile, tile)

            # Apply scaling
            scaler_tile = scaler.to_list()[0].to_torch()
            result_tile = result_tile * scaler_tile
            result_tensors.append(Tensor(result_tile))

    result_block = Block.from_list(result_tensors, shape=(result_M, result_N))
    _track_source_blocks(result_block, block, scaler)
    return result_block


def reduce_sum(
    block: Block,
    scaler: Block,
    _output_hint: Optional[Block] = None,
    dims: Optional[List[int]] = None,
) -> Block:
    """Scaled sum reduction.

    Computes the scaled sum reduction over specified dimensions.
    The result is the sum of values along the specified dimensions, scaled by the scaler.

    Args:
        block: Input block to reduce
        scaler: Scaler block
        _output_hint: Optional output block hint (unused in simulator)
        dims: List of dimension indices to reduce over (0-indexed)
              Example: [0] for rows, [1] for columns, [0, 1] for all

    Returns:
        Block with reduced dimensions

    Example:
        # Reduce over rows (dimension 0)
        with a_cb.wait() as a_blk, s_cb.wait() as s_blk:
            result = ttl.math.reduce_sum(a_blk, s_blk, dims=[0])

    From the specification:
        Scaled sum reduction over specified dimensions.
        Example for reduction over rows: ttl.math.reduce_sum(a, s, dims=[0])
        Example for reduction over rows and columns: ttl.math.reduce_sum(a, s, dims=[0, 1])
    """

    if dims is None or not dims:
        raise ValueError("dims parameter must contain at least one dimension")

    # Import Tensor here to avoid circular dependency
    from .ttnnsim import Tensor

    # Reduction operates on TILE GRID: combines tiles along specified grid dimensions
    # dims=[0] means combine tiles along grid rows (element-wise sum)
    # dims=[1] means combine tiles along grid columns (element-wise sum)
    # The result tile grid shrinks in the reduced dimensions
    block_shape = block._shape  # type: ignore[attr-defined]
    M, N = block_shape
    input_tensors = [t.to_torch() for t in block.to_list()]

    # Validate dims
    for dim in dims:
        if dim >= 2:
            raise ValueError(
                f"Cannot reduce along dimension {dim}: block grid has only 2 dimensions"
            )

    # Calculate result grid shape
    result_M = 1 if 0 in dims else M
    result_N = 1 if 1 in dims else N

    # For each result tile position, gather and combine input tiles element-wise
    result_tensors = []
    for res_i in range(result_M):
        for res_j in range(result_N):
            # Collect all tiles that contribute
            tiles_to_sum = []
            for i in range(M):
                for j in range(N):
                    # Include if dims match or dimension is being reduced
                    if (0 in dims or i == res_i) and (1 in dims or j == res_j):
                        tile_idx = i * N + j
                        tiles_to_sum.append(input_tensors[tile_idx])

            # Take element-wise sum across all contributing tiles
            result_tile = tiles_to_sum[0].clone()
            for tile in tiles_to_sum[1:]:
                result_tile = result_tile + tile

            # Apply scaling
            scaler_tile = scaler.to_list()[0].to_torch()
            result_tile = result_tile * scaler_tile
            result_tensors.append(Tensor(result_tile))

    result_block = Block.from_list(result_tensors, shape=(result_M, result_N))
    _track_source_blocks(result_block, block, scaler)
    return result_block


# Clean up temporary variables
del _op_name, _torch_fn


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
    from .ttnnsim import Tensor

    # Transpose each tile (swap rows/columns within tiles)
    transposed_tiles = [Tensor(t.to_torch().T) for t in block.to_list()]

    # Also swap the tile grid dimensions: (M, N) -> (N, M)
    M, N = block._shape  # type: ignore[attr-defined]

    # Reorder tiles to match transposed grid: tile[i,j] -> tile[j,i]
    reordered_tiles = []
    for j in range(N):
        for i in range(M):
            reordered_tiles.append(transposed_tiles[i * N + j])

    result_block = Block.from_list(reordered_tiles, shape=(N, M))
    _track_source_blocks(result_block, block)
    return result_block
