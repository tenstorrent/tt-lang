# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DSL operators for tensor operations and data movement."""

from __future__ import annotations

from typing import Tuple, Union

from ttmlir.dialects import arith
from ttmlir.ir import IntegerAttr, IntegerType, RankedTensorType, Type

# Re-export generated elementwise operations
from ._generated_elementwise import *  # noqa: F401,F403
from ._generated_elementwise import __all__ as _generated_all
from ._src.ttl_ast import syntax
from .dialects import ttl


def _get_constant_int(val):
    """Extract Python int from MLIR arith.ConstantOp or return as-is if already int."""
    if isinstance(val, int):
        return val
    if isinstance(val, arith.ConstantOp):
        return val.literal_value
    raise ValueError(f"Expected int or arith.ConstantOp, got {type(val)}")


# Type aliases for common patterns
CoreCoordinate = Tuple[int, int]
IndexedTensor = Union["TensorBlock", Tuple["TensorBlock", Tuple[int, ...]]]

# Module-level grid storage for grid_size() function
# Sentinel value (-1, -1) makes uninitialized reads obvious
_current_grid: Tuple[int, int] = (-1, -1)


def _set_current_grid(grid: Tuple[int, int]) -> None:
    """Set the current grid dimensions. Called before compiling threads."""
    global _current_grid
    _current_grid = grid


def _get_current_grid() -> Tuple[int, int]:
    """Get the current grid dimensions."""
    return _current_grid


@syntax("!tensor")
class TensorBlock:
    """
    Represents a block of tensor data in the TTL dialect.

    TensorBlock supports arithmetic operations through operator
    overloading. Operations generate TTL high-level ops that get lowered
    to ttl.compute blocks.
    """

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __add__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """
        Element-wise addition using ttl.add.

        Args:
            rhs: Right operand tensor. Must have the same shape as self.

        Returns:
            Result tensor with the same shape as inputs.
        """
        return ttl.add(ast_self.type, ast_self, rhs)

    def __sub__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Element-wise subtraction using ttl.sub."""
        return ttl.sub(ast_self.type, ast_self, rhs)

    def __mul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Element-wise multiplication using ttl.mul."""
        return ttl.mul(ast_self.type, ast_self, rhs)

    def __matmul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Matrix multiplication via @ operator is not supported. Use matmul(a, b, c) instead."""
        raise NotImplementedError(
            "Matrix multiplication via @ operator is not supported. "
            "Use matmul(a, b, c) function instead for accumulating matmul: c += a @ b"
        )

    def store(ast_self: TensorBlock, rhs: TensorBlock) -> None:
        """Store result tensor to CB by propagating CB association from output view."""
        # ast_self is the result of attach_cb(tensor, cb) from reserve()
        # Extract the CB operand and attach it to the result tensor
        cb = ast_self.owner.operands[1]
        return ttl.attach_cb(rhs.type, rhs, cb)


@syntax("!ttl.transfer_handle")
class CopyTransferHandler:
    """
    Transfer handle for asynchronous copy operations.

    CopyTransferHandler objects are returned by copy() calls and must be
    explicitly waited on to ensure transfer completion.
    """

    def wait(ast_self: CopyTransferHandler):
        """Block until the copy operation completes."""
        return ttl.wait(ast_self)


def _make_tensor_slice(tensor, indices, slice_shape):
    """Create a ttl.tensor_slice from a tensor, tile indices, and shape.

    Args:
        tensor: The source tensor to slice from
        indices: (row, col) tile indices for the slice start position
        slice_shape: (rows, cols) shape for the slice in tiles
    """
    tensor_type = tensor.type
    if not isinstance(tensor_type, RankedTensorType):
        raise ValueError(f"Expected RankedTensorType, got {tensor_type}")

    # TTL tensors are 4D internally: [grid_row, grid_col, shard_row, shard_col]
    # User provides 2D tile coordinates
    if tensor_type.rank != 4:
        raise ValueError(f"Expected rank-4 TTL tensor, got rank {tensor_type.rank}")

    if len(indices) != 2:
        raise ValueError(f"Expected 2 tile indices (row, col), got {len(indices)}")

    row_idx, col_idx = indices

    # Build result type: [grid_row, grid_col, slice_rows, slice_cols]
    orig_shape = list(tensor_type.shape)
    reduced_shape = orig_shape[:2] + list(slice_shape)
    result_type = RankedTensorType.get(
        reduced_shape, tensor_type.element_type, tensor_type.encoding
    )
    return ttl.tensor_slice(result_type, tensor, row_idx, col_idx)


def _is_block(value) -> bool:
    """Check if a value is a block (result of cb.reserve() or cb.wait()).

    A block is a tensor with an attached CB, produced by ttl.attach_cb.
    """
    if not hasattr(value, "owner") or value.owner is None:
        return False
    owner_name = value.owner.name
    return owner_name == "ttl.attach_cb"


def _get_cb_from_block(block):
    """Extract the CB from a block (result of ttl.attach_cb).

    The attach_cb op has signature: (tensor, cb) -> tensor
    So the CB is operand[1].
    """
    return block.owner.operands[1]


def _get_cb_shape(cb_val):
    """Extract the block shape from a CB value."""
    cb_type = ttl.CircularBufferType.maybe_downcast(cb_val.type)
    if cb_type is None:
        raise ValueError(f"Expected CircularBufferType, got {cb_val.type}")
    return list(cb_type.shape)


def _process_tensor_subscript(subscript_tuple, cb_shape):
    """Process tensor subscript and create tensor slice.

    Args:
        subscript_tuple: (tensor, indices) where indices are [(value, is_range), ...]
        cb_shape: [rows, cols] shape from the CB

    Returns:
        Tensor slice with shape matching cb_shape
    """
    tensor, indices = subscript_tuple

    if len(indices) != 2:
        raise ValueError(f"Expected 2 indices (row, col), got {len(indices)}")

    cb_is_multi_tile = cb_shape[0] > 1 or cb_shape[1] > 1
    uses_ranges = any(is_range for _, is_range in indices)

    if cb_is_multi_tile and not uses_ranges:
        raise ValueError(
            f"CB shape {cb_shape} requires range syntax (e.g., tensor[0:2, 0:2]), "
            f"but got index syntax (e.g., tensor[0, 0])"
        )

    # TODO: Validate that range size matches CB shape (requires runtime or
    # constant folding to compare end - start with cb_shape dimensions).

    start_indices = [value for value, _ in indices]
    return _make_tensor_slice(tensor, start_indices, cb_shape)


@syntax("copy")
def copy(src, dst) -> CopyTransferHandler:
    """
    Initiate an asynchronous data transfer using ttl.copy.

    Args:
        src: Source tensor/slice (for reads) or block (for writes)
        dst: Destination block (for reads) or tensor/slice (for writes)

    Returns:
        CopyTransferHandler handle that must be waited on for completion

    For multi-tile CBs (shape > 1x1), use range syntax: tensor[0:2, 0:2]
    For single-tile CBs (shape 1x1), use index syntax: tensor[0, 0]
    """
    src_is_subscript = isinstance(src, tuple)
    dst_is_subscript = isinstance(dst, tuple)

    # Identify the block argument to get CB shape
    if dst_is_subscript:
        if not _is_block(src):
            raise ValueError("copy() with tensor subscript dst requires block src")
        cb_shape = _get_cb_shape(_get_cb_from_block(src))
    elif src_is_subscript:
        if not _is_block(dst):
            raise ValueError("copy() with tensor subscript src requires block dst")
        cb_shape = _get_cb_shape(_get_cb_from_block(dst))
    else:
        raise ValueError(
            "copy() requires at least one tensor subscript argument "
            "(e.g., tensor[row, col] or tensor[r0:r1, c0:c1])"
        )

    # Process subscripted tensors into tensor slices
    if src_is_subscript:
        src = _process_tensor_subscript(src, cb_shape)
    if dst_is_subscript:
        dst = _process_tensor_subscript(dst, cb_shape)

    ctx = src.type.context

    # Check if src/dst is a block (result of cb.reserve()/cb.wait())
    src_is_block = _is_block(src)
    dst_is_block = _is_block(dst)

    # Extract CB from block if needed
    src_cb = _get_cb_from_block(src) if src_is_block else None
    dst_cb = _get_cb_from_block(dst) if dst_is_block else None

    if dst_is_block and not src_is_block:
        # Read: device tensor/slice -> block (CB)
        xf_type = Type.parse("!ttl.transfer_handle<read>", ctx)
        return ttl.copy(xf_type, src, dst_cb)
    elif src_is_block and not dst_is_block:
        # Write: block (CB) -> device tensor/slice
        xf_type = Type.parse("!ttl.transfer_handle<write>", ctx)
        return ttl.copy(xf_type, src_cb, dst)
    else:
        raise ValueError(
            f"copy() requires exactly one block argument (result of cb.reserve() or cb.wait()). "
            f"Got src_is_block={src_is_block}, dst_is_block={dst_is_block}"
        )


@syntax("core")
def core(*, dims):
    """
    Get the coordinates of the current core.

    Currently only dims=2 is supported (temporary restriction).

    Args:
        dims: Number of dimensions to return (must be 2)

    Returns:
        For dims=2: Tuple (x, y) where x is column coordinate and y is row coordinate

    Raises:
        ValueError: If dims is not 2

    Example:
        x, y = ttl.core(dims=2)
    """
    dims_val = _get_constant_int(dims)
    if dims_val != 2:
        raise ValueError(
            f"core() currently only supports dims=2, got dims={dims_val}. "
            "Multi-dimensional grids are not yet supported."
        )
    return (ttl.core_x(), ttl.core_y())


def grid_size(*, dims):
    """
    Get the size of the grid.

    Currently only dims=2 is supported (temporary restriction).

    Args:
        dims: Number of dimensions to return (must be 2)

    Returns:
        For dims=2: Tuple (x_size, y_size) where x_size is columns and y_size is rows

    Raises:
        ValueError: If dims is not 2

    Example:
        x_size, y_size = ttl.grid_size(dims=2)
    """
    dims_val = _get_constant_int(dims)
    if dims_val != 2:
        raise ValueError(
            f"grid_size() currently only supports dims=2, got dims={dims_val}. "
            "Multi-dimensional grids are not yet supported."
        )
    # grid is stored as (rows, cols), spec returns (x, y) = (cols, rows)
    rows, cols = _get_current_grid()
    return (cols, rows)

@syntax("matmul")
def matmul(a: TensorBlock, b: TensorBlock, c: TensorBlock) -> TensorBlock:
    """
    Matrix multiplication with accumulation: result = a @ b + c.

    This is a fused multiply-add operation where 'c' is the accumulator.
    The result is written back, modeling the hardware behavior where
    matmul accumulates into the DST register.

    Args:
        a: Left operand matrix tile
        b: Right operand matrix tile
        c: Accumulator tile (read and accumulated into)

    Returns:
        Result tile containing a @ b + c
    """
    # Use high-level matmul op (like binary ops use ttl.add, not ttl.tile_add).
    # This gets lowered to ttl.compute with ttl.tile_matmul in the body.
    return ttl.matmul(c.type, a, b, c)


@syntax("power")
def power(input: TensorBlock, exponent: int) -> TensorBlock:
    """
    Element-wise power with scalar exponent: result = input ^ exponent.

    Raises each element of the input tensor to the given integer power.

    Args:
        input: Input tensor tile
        exponent: Integer exponent to raise each element to

    Returns:
        Result tile containing input ^ exponent (element-wise)
    """
    exp_val = _get_constant_int(exponent)
    return ttl.power(input.type, input, exp_val)


@syntax("where")
def where(
    condition: TensorBlock, true_val: TensorBlock, false_val: TensorBlock
) -> TensorBlock:
    """
    Element-wise conditional selection: result = condition ? true_val : false_val.

    For each element, if the condition is non-zero, selects from true_val;
    otherwise selects from false_val.

    Args:
        condition: Condition tensor (non-zero = true)
        true_val: Values to select when condition is true
        false_val: Values to select when condition is false

    Returns:
        Result tile with selected values
    """
    return ttl.where(condition, true_val, false_val, results=[true_val.type])


@syntax("reduce_sum")
def reduce_sum(input: TensorBlock, scaler: TensorBlock, output: TensorBlock, dim: str = "scalar") -> TensorBlock:
    """
    Reduce tensor by summing along a dimension.

    Args:
        input: Input tensor tile
        scaler: Scaler tensor tile (typically initialized with 1.0 for simple sum)
        output: Output tensor tile (used to track output CB for init)
        dim: Reduction dimension - "row", "col", or "scalar" (default)

    Returns:
        Result tile with reduced values
    """
    # Map string to integer value (Row=0, Col=1, Scalar=2)
    dim_map = {"row": 0, "col": 1, "scalar": 2}
    reduce_dim_val = dim_map.get(dim, 2)  # Default to scalar
    # Create an IntegerAttr for the reduce_dim
    ctx = input.type.context
    i32_type = IntegerType.get_signless(32, ctx)
    reduce_dim_attr = IntegerAttr.get(i32_type, reduce_dim_val)
    return ttl.reduce_sum(output.type, input, scaler, output, reduce_dim_attr)


@syntax("reduce_max")
def reduce_max(input: TensorBlock, scaler: TensorBlock, output: TensorBlock, dim: str = "scalar") -> TensorBlock:
    """
    Reduce tensor by taking maximum along a dimension.

    Args:
        input: Input tensor tile
        scaler: Scaler tensor tile (typically initialized with 1.0)
        output: Output tensor tile (used to track output CB for init)
        dim: Reduction dimension - "row", "col", or "scalar" (default)

    Returns:
        Result tile with reduced values (max of each row/col/all)
    """
    # Map string to integer value (Row=0, Col=1, Scalar=2)
    dim_map = {"row": 0, "col": 1, "scalar": 2}
    reduce_dim_val = dim_map.get(dim, 2)  # Default to scalar
    # Create an IntegerAttr for the reduce_dim
    ctx = input.type.context
    i32_type = IntegerType.get_signless(32, ctx)
    reduce_dim_attr = IntegerAttr.get(i32_type, reduce_dim_val)
    return ttl.reduce_max(output.type, input, scaler, output, reduce_dim_attr)


@syntax("transpose")
def transpose(input: TensorBlock, output: TensorBlock) -> TensorBlock:
    """
    Transpose a 32x32 tile (width-height swap).

    Swaps the last two dimensions of the tile, effectively rotating the data.
    This is a tile-level transpose operation (32x32 only).

    Args:
        input: Input tensor tile
        output: Output tensor tile (used to track output CB for init)

    Returns:
        Result tile with transposed values
    """
    return ttl.transpose(output.type, input, output)


__all__ = [
    "TensorBlock",
    "CopyTransferHandler",
    "copy",
    "core",
    "grid_size",
    "matmul",
    "power",
    "where",
    "reduce_sum",
    "reduce_max",
    "transpose",
    *_generated_all,
]
