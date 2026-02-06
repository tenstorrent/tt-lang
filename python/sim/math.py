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

    This function replicates values within each tile along the specified dimensions.
    After reduce operations store values at specific positions (e.g., reduce_max with
    dims=[0] stores max per row at column 0), broadcast replicates those values.

    For dims=[1] (broadcast along columns):
    - Takes values from column 0 of each row
    - Replicates them across all columns in that row

    For dims=[0] (broadcast along rows):
    - Takes values from row 0 of each column
    - Replicates them across all rows in that column

    Args:
        block: Input block to broadcast (can be Block or WaitContext)
        _unused_arg: Unused argument for compatibility (typically output block shape hint)
        dims: List of dimension indices to broadcast along (0-indexed)

    Returns:
        Block with values replicated along the specified dimensions
    """
    if dims is None:
        raise ValueError("dims parameter is required for broadcast()")

    from .ttnnsim import Tensor

    # Unwrap WaitContext/ReserveContext if needed
    actual_block: Block = block.block() if hasattr(block, "block") else block  # type: ignore[union-attr]

    # Validate that the dimensions being broadcast have size 1 at grid level
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

    # Perform within-tile broadcasting
    input_tensors = [t.to_torch() for t in actual_block.to_list()]
    result_tensors = []

    for tile in input_tensors:
        result_tile = tile.clone()
        if 0 in dims and 1 in dims:
            # Broadcast both dimensions: replicate (0,0) to entire tile
            result_tile = torch.full_like(tile, tile[0, 0].item())
        elif 1 in dims:
            # Broadcast along columns: replicate column 0 across all columns
            # Each row gets its value from column 0 replicated
            col0 = tile[:, 0:1]  # Shape: (32, 1)
            result_tile = col0.expand(-1, tile.shape[1]).clone()
        elif 0 in dims:
            # Broadcast along rows: replicate row 0 across all rows
            # Each column gets its value from row 0 replicated
            row0 = tile[0:1, :]  # Shape: (1, 32)
            result_tile = row0.expand(tile.shape[0], -1).clone()
        result_tensors.append(Tensor(result_tile))

    result_block = Block.from_list(result_tensors, block_shape)

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

    Reduction operates at two levels:
    1. Within each tile: reduces along the specified tensor dimensions
    2. Across tiles: combines tiles in the grid along the specified dimensions

    For dims=[0] (reduce rows):
    - Within each tile: compute max per row (across columns), store at column 0
    - Across tiles: combine tiles along grid row dimension

    For dims=[1] (reduce columns):
    - Within each tile: compute max per column (across rows), store at row 0
    - Across tiles: combine tiles along grid column dimension

    Args:
        block: Input block to reduce
        scaler: Scaler block
        _output_hint: Optional output block hint (unused in simulator)
        dims: List of dimension indices to reduce over (0-indexed)
              Example: [0] for rows, [1] for columns, [0, 1] for all

    Returns:
        Block with reduced dimensions
    """

    if dims is None or not dims:
        raise ValueError("dims parameter must contain at least one dimension")

    from .ttnnsim import Tensor

    block_shape = block._shape  # type: ignore[attr-defined]
    M, N = block_shape
    input_tensors = [t.to_torch() for t in block.to_list()]
    scaler_tile = scaler.to_list()[0].to_torch()

    for dim in dims:
        if dim >= 2:
            raise ValueError(
                f"Cannot reduce along dimension {dim}: block grid has only 2 dimensions"
            )

    # Step 1: Within-tile reduction
    # dims=[0] means reduce across columns within each row -> one value per row at col 0
    # dims=[1] means reduce across rows within each column -> one value per col at row 0
    reduced_tiles = []
    for tile in input_tensors:
        result_tile = torch.zeros_like(tile)
        if 0 in dims and 1 in dims:
            # Full reduction: single max value at position (0, 0)
            max_val = tile.max()
            result_tile[0, 0] = max_val
        elif 0 in dims:
            # Reduce across columns (dim 1) for each row -> store at column 0
            row_maxes = tile.max(dim=1).values  # shape: (32,)
            result_tile[:, 0] = row_maxes
        elif 1 in dims:
            # Reduce across rows (dim 0) for each column -> store at row 0
            col_maxes = tile.max(dim=0).values  # shape: (32,)
            result_tile[0, :] = col_maxes
        reduced_tiles.append(result_tile)

    # Step 2: Grid-level reduction (combine tiles across specified grid dimensions)
    result_M = 1 if 0 in dims else M
    result_N = 1 if 1 in dims else N

    result_tensors = []
    for res_i in range(result_M):
        for res_j in range(result_N):
            tiles_to_max = []
            for i in range(M):
                for j in range(N):
                    if (0 in dims or i == res_i) and (1 in dims or j == res_j):
                        tile_idx = i * N + j
                        tiles_to_max.append(reduced_tiles[tile_idx])

            result_tile = tiles_to_max[0]
            for tile in tiles_to_max[1:]:
                result_tile = torch.maximum(result_tile, tile)

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

    Reduction operates at two levels:
    1. Within each tile: reduces along the specified tensor dimensions
    2. Across tiles: combines tiles in the grid along the specified dimensions

    For dims=[0] (reduce rows):
    - Within each tile: compute sum per row (across columns), store at column 0
    - Across tiles: combine tiles along grid row dimension

    For dims=[1] (reduce columns):
    - Within each tile: compute sum per column (across rows), store at row 0
    - Across tiles: combine tiles along grid column dimension

    Args:
        block: Input block to reduce
        scaler: Scaler block
        _output_hint: Optional output block hint (unused in simulator)
        dims: List of dimension indices to reduce over (0-indexed)
              Example: [0] for rows, [1] for columns, [0, 1] for all

    Returns:
        Block with reduced dimensions
    """

    if dims is None or not dims:
        raise ValueError("dims parameter must contain at least one dimension")

    from .ttnnsim import Tensor

    block_shape = block._shape  # type: ignore[attr-defined]
    M, N = block_shape
    input_tensors = [t.to_torch() for t in block.to_list()]
    scaler_tile = scaler.to_list()[0].to_torch()

    for dim in dims:
        if dim >= 2:
            raise ValueError(
                f"Cannot reduce along dimension {dim}: block grid has only 2 dimensions"
            )

    # Step 1: Within-tile reduction
    # dims=[0] means reduce across columns within each row -> one value per row at col 0
    # dims=[1] means reduce across rows within each column -> one value per col at row 0
    reduced_tiles = []
    for tile in input_tensors:
        result_tile = torch.zeros_like(tile)
        if 0 in dims and 1 in dims:
            # Full reduction: single sum value at position (0, 0)
            sum_val = tile.sum()
            result_tile[0, 0] = sum_val
        elif 0 in dims:
            # Reduce across columns (dim 1) for each row -> store at column 0
            row_sums = tile.sum(dim=1)  # shape: (32,)
            result_tile[:, 0] = row_sums
        elif 1 in dims:
            # Reduce across rows (dim 0) for each column -> store at row 0
            col_sums = tile.sum(dim=0)  # shape: (32,)
            result_tile[0, :] = col_sums
        reduced_tiles.append(result_tile)

    # Step 2: Grid-level reduction (combine tiles across specified grid dimensions)
    result_M = 1 if 0 in dims else M
    result_N = 1 if 1 in dims else N

    result_tensors = []
    for res_i in range(result_M):
        for res_j in range(result_N):
            tiles_to_sum = []
            for i in range(M):
                for j in range(N):
                    if (0 in dims or i == res_i) and (1 in dims or j == res_j):
                        tile_idx = i * N + j
                        tiles_to_sum.append(reduced_tiles[tile_idx])

            result_tile = tiles_to_sum[0].clone()
            for tile in tiles_to_sum[1:]:
                result_tile = result_tile + tile

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
