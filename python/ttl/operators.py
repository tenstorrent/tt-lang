# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DSL operators for tensor operations and data movement."""

from __future__ import annotations

from typing import List, Tuple, Union

from ttmlir.dialects import arith
from ttmlir.ir import RankedTensorType, Type

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


def _get_constant_float(val):
    """Extract Python float from MLIR arith.ConstantOp or return as-is if already float."""
    if isinstance(val, (float, int)):
        return float(val)
    if isinstance(val, arith.ConstantOp):
        return float(val.literal_value)
    raise ValueError(f"Expected float or arith.ConstantOp, got {type(val)}")


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

    def __truediv__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Element-wise division using ttl.div."""
        return ttl.div(ast_self.type, ast_self, rhs)

    def __matmul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Matrix multiplication via @ operator. Equivalent to ttl.math.matmul."""
        return matmul(ast_self, rhs)

    def store(ast_self: TensorBlock, rhs: TensorBlock) -> None:
        """Store result tensor to the output CB reserve view.

        Emits ttl.store with the result tensor and reserve view.
        """
        if not _is_block(ast_self):
            raise ValueError(
                "store() must be called on a block acquired from reserve(), not a regular tensor"
            )
        reserve = _get_reserve_from_block(ast_self)
        ttl.store(rhs, reserve)

    def push(ast_self: TensorBlock) -> None:
        """
        Signal that data is ready in the circular buffer (producer release).

        Finalizes a reserve() operation by signaling that the block has been
        written and is ready for consumers. This operation is non-blocking.

        Must be called on a block acquired via reserve().

        Example:
            block = cb.reserve()
            ttl.copy(data, block).wait()
            block.push()  # Signal data ready
        """
        if not _is_block(ast_self):
            raise ValueError(
                "push() must be called on a block acquired from reserve(), not a regular tensor"
            )
        cb = _get_cb_from_block(ast_self)
        ttl.cb_push(cb)

    def pop(ast_self: TensorBlock) -> None:
        """
        Signal that data has been consumed (consumer release).

        Finalizes a wait() operation by signaling that the block has been
        consumed and space is available for producers. This operation is non-blocking.

        Must be called on a block acquired via wait().

        Example:
            block = cb.wait()
            result = compute(block)
            block.pop()  # Signal consumption complete
        """
        if not _is_block(ast_self):
            raise ValueError(
                "pop() must be called on a block acquired from wait(), not a regular tensor"
            )
        cb = _get_cb_from_block(ast_self)
        ttl.cb_pop(cb)


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
        indices: Tile indices for the slice start position (one per tensor dim)
        slice_shape: CB shape in tiles (same rank as tensor)
    """
    tensor_type = tensor.type
    if not isinstance(tensor_type, RankedTensorType):
        raise ValueError(f"Expected RankedTensorType, got {tensor_type}")

    if tensor_type.rank < 2:
        raise ValueError(
            f"TTL tensors must have at least 2 dimensions, got rank {tensor_type.rank}"
        )

    if len(indices) != tensor_type.rank:
        raise ValueError(
            f"Expected {tensor_type.rank} tile indices for rank-{tensor_type.rank} "
            f"tensor, got {len(indices)}"
        )

    if len(slice_shape) != tensor_type.rank:
        raise ValueError(
            f"CB shape rank ({len(slice_shape)}) must match tensor rank "
            f"({tensor_type.rank})"
        )

    result_type = RankedTensorType.get(
        list(slice_shape), tensor_type.element_type, tensor_type.encoding
    )
    return ttl.tensor_slice(result_type, tensor, indices)


def _is_block(value) -> bool:
    """Check if a value is a block (result of cb.reserve() or cb.wait()).

    A block is a tensor with an attached CB, produced by ttl.attach_cb.
    """
    if not hasattr(value, "owner") or value.owner is None:
        return False
    owner_name = value.owner.name
    return owner_name == "ttl.attach_cb"


def _get_reserve_from_block(block):
    """Extract the reserve view from a block (result of ttl.attach_cb).

    The attach_cb op has signature: (tensor, cb) -> tensor
    So the reserve/wait tensor is operand[0].
    """
    if block.owner.name != "ttl.attach_cb":
        raise ValueError(f"expected block from ttl.attach_cb, got {block.owner.name}")
    return block.owner.operands[0]


def _get_cb_from_block(block):
    """Extract the CB from a block (result of ttl.attach_cb).

    The attach_cb op has signature: (tensor, cb) -> tensor
    So the CB is operand[1].
    """
    if block.owner.name != "ttl.attach_cb":
        raise ValueError(f"expected block from ttl.attach_cb, got {block.owner.name}")
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
        cb_shape: Shape from the CB (matches tensor rank)

    Returns:
        Tensor slice with shape matching cb_shape
    """
    tensor, indices = subscript_tuple

    tensor_type = tensor.type
    if not isinstance(tensor_type, RankedTensorType):
        raise ValueError(f"Expected RankedTensorType, got {tensor_type}")

    expected_indices = tensor_type.rank
    if len(indices) != expected_indices:
        raise ValueError(
            f"Expected {expected_indices} indices for rank-{tensor_type.rank} "
            f"tensor, got {len(indices)}"
        )

    cb_is_multi_tile = any(d > 1 for d in cb_shape)
    uses_ranges = any(is_range for _, is_range in indices)

    if cb_is_multi_tile and not uses_ranges:
        raise ValueError(
            f"CB shape {cb_shape} requires range syntax "
            f"(e.g., tensor[0:2, 0:2]), but got index syntax"
        )

    # TODO: Validate that range size matches CB shape (requires runtime or
    # constant folding to compare end - start with cb_shape dimensions).

    start_indices = [value for value, _ in indices]
    return _make_tensor_slice(tensor, start_indices, cb_shape)


def _is_pipe(val):
    """Check if a value is a pipe (either MLIR PipeType or Python Pipe with MLIR value)."""
    # Check for MLIR PipeType value
    if hasattr(val, "type"):
        type_str = str(val.type)
        if "ttl.pipe" in type_str:
            return True
    # Check for Python Pipe object with MLIR value
    from .pipe import Pipe

    return isinstance(val, Pipe) and hasattr(val, "_mlir_value")


def _get_pipe_mlir_value(pipe):
    """Get the MLIR value for a pipe (either MLIR value or Python Pipe object)."""
    # If it's already an MLIR value, return it
    if hasattr(pipe, "type"):
        type_str = str(pipe.type)
        if "ttl.pipe" in type_str:
            return pipe
    # If it's a Python Pipe object, get its MLIR value
    return pipe._mlir_value


@syntax("copy")
def copy(src, dst) -> CopyTransferHandler:
    """
    Initiate an asynchronous data transfer using ttl.copy.

    Args:
        src: Source tensor/slice (for reads), block (for writes), or Pipe (for pipe receive)
        dst: Destination block (for reads), tensor/slice (for writes), or Pipe (for pipe send)

    Returns:
        CopyTransferHandler handle that must be waited on for completion

    For multi-tile CBs (shape > 1x1), use range syntax: tensor[0:2, 0:2]
    For single-tile CBs (shape 1x1), use index syntax: tensor[0, 0]

    For pipe transfers:
        ttl.copy(block, pipe) - send from CB to pipe (multicast write)
        ttl.copy(pipe, block) - receive from pipe to CB (no-op, data arrives via multicast)
    """
    # Check for pipe operands first
    src_is_pipe = _is_pipe(src)
    dst_is_pipe = _is_pipe(dst)

    if src_is_pipe or dst_is_pipe:
        # Pipe transfer: CB <-> Pipe
        if src_is_pipe and dst_is_pipe:
            raise ValueError("copy() cannot transfer directly between two pipes")

        if dst_is_pipe:
            # CB -> Pipe (send via multicast)
            if not _is_block(src):
                raise ValueError(
                    "copy() to pipe requires block src (from cb.reserve() or cb.wait())"
                )
            src_cb = _get_cb_from_block(src)
            pipe_val = _get_pipe_mlir_value(dst)
            ctx = src_cb.type.context
            xf_type = Type.parse("!ttl.transfer_handle<write>", ctx)
            return ttl.copy(xf_type, src_cb, pipe_val)
        else:
            # Pipe -> CB (receive, data arrives via multicast from source)
            # No transfer kind - data is already in CB after source's write barrier
            if not _is_block(dst):
                raise ValueError(
                    "copy() from pipe requires block dst (from cb.reserve() or cb.wait())"
                )
            dst_cb = _get_cb_from_block(dst)
            pipe_val = _get_pipe_mlir_value(src)
            ctx = dst_cb.type.context
            xf_type = Type.parse("!ttl.transfer_handle", ctx)
            return ttl.copy(xf_type, pipe_val, dst_cb)

    # Non-pipe transfers: tensor subscript <-> block
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
    # grid is stored as (cols, rows) = (x, y), matching tt-metal convention
    return _get_current_grid()


@syntax("signpost")
def signpost(name: str):
    """
    Mark a profiling scope visible in Tracy.

    Use as a context manager to wrap a region of interest:

        with ttl.signpost("my_region"):
            ...

    Generates a DeviceZoneScopedN in the emitted C++ code. Enable
    TTLANG_SIGNPOST_PROFILE=1 to collect per-region cycle counts.

    Args:
        name: Name for the profiling region (must be a string literal)
    """
    return ttl.signpost(name)


@syntax("broadcast")
def broadcast(input: TensorBlock, *, dims: List[int]) -> TensorBlock:
    """
    Broadcast over specified dimensions.

    Only 2D tensors are supported for broadcast (hardware constraint).
    The output CB is determined by the ttl.store on this op's result.
    For shape expansion (output DFB larger than input DFB), the compiler
    derives the output shape from the store target.

    Args:
        input: Input tensor (CB-attached)
        dims: Dimensions to broadcast over

    Returns:
        Result tensor with broadcast values
    """
    from ttmlir.ir import IntegerAttr, IntegerType

    if isinstance(input.type, RankedTensorType) and input.type.rank != 2:
        raise ValueError(
            f"broadcast only supports 2D tensors, got rank {input.type.rank}. "
            "Use 2D tensors for broadcast operations."
        )

    dims_set = set(dims)
    if dims_set == {0}:
        bcast_val = 2  # Row
    elif dims_set == {1}:
        bcast_val = 1  # Col
    elif dims_set == {0, 1}:
        bcast_val = 3  # Scalar
    else:
        raise ValueError(f"Invalid dims: {dims}. Must be [0], [1], or [0, 1]")

    ctx = input.type.context
    i32_type = IntegerType.get_signless(32, ctx)
    bcast_attr = IntegerAttr.get(i32_type, bcast_val)
    return ttl.bcast(input.type, input, bcast_attr)


@syntax("matmul")
def matmul(a: TensorBlock, b: TensorBlock) -> TensorBlock:
    """Matrix multiplication A * B. Both operands must be CB-attached."""
    a_type = a.type
    b_type = b.type
    if isinstance(a_type, RankedTensorType) and isinstance(
        b_type, RankedTensorType
    ):
        out_shape = list(a_type.shape)
        out_shape[-1] = b_type.shape[-1]
        result_type = RankedTensorType.get(
            out_shape, a_type.element_type, a_type.encoding
        )
    else:
        result_type = a_type
    return ttl.matmul(result_type, a, b)


@syntax("transpose")
def transpose(input: TensorBlock) -> TensorBlock:
    """Transpose a 2D tensor (swap rows and columns).

    Reads from the input CB, writes result to DST.
    The output CB is determined by the ttl.store on this op's result.
    """
    from ttmlir.ir import RankedTensorType

    input_type = input.type
    shape = list(input_type.shape)
    shape[0], shape[1] = shape[1], shape[0]
    result_type = RankedTensorType.get(
        shape, input_type.element_type, input_type.encoding
    )
    return ttl.transpose(result_type, input)


@syntax("reduce_sum")
def reduce_sum(
    input: TensorBlock, scaler: TensorBlock, dims: List[int]
) -> TensorBlock:
    """Reduce tensor by summing along specified dimensions (PyTorch semantics).

    Args:
        input: Input tensor (CB-attached)
        scaler: Scaler tensor for reduction (CB-attached)
        dims: Dimensions to collapse (PyTorch convention) -
              [0] collapses rows, [1] collapses columns, [0, 1] for scalar
    """
    from ttmlir.ir import IntegerAttr, IntegerType, RankedTensorType

    dims_set = set(dims)
    if dims_set == {0}:
        reduce_dim_val = 1  # Col reduce: collapse rows -> (1, M)
    elif dims_set == {1}:
        reduce_dim_val = 0  # Row reduce: collapse columns -> (N, 1)
    elif dims_set == {0, 1}:
        reduce_dim_val = 2  # Scalar
    else:
        raise ValueError(f"Invalid dims: {dims}. Must be [0], [1], or [0, 1]")

    input_type = input.type
    in_shape = list(input_type.shape)
    if dims_set == {0}:
        out_shape = [1, in_shape[1]]
    elif dims_set == {1}:
        out_shape = [in_shape[0], 1]
    else:  # Scalar
        out_shape = [1, 1]
    result_type = RankedTensorType.get(
        out_shape, input_type.element_type, input_type.encoding
    )

    ctx = input.type.context
    i32_type = IntegerType.get_signless(32, ctx)
    reduce_type_attr = IntegerAttr.get(i32_type, 0)  # Sum = 0
    reduce_dim_attr = IntegerAttr.get(i32_type, reduce_dim_val)
    return ttl.reduce(
        result_type, input, scaler, reduce_type_attr, reduce_dim_attr
    )


@syntax("reduce_max")
def reduce_max(
    input: TensorBlock, scaler: TensorBlock, dims: List[int]
) -> TensorBlock:
    """Reduce tensor by taking max along specified dimensions (PyTorch semantics).

    Args:
        input: Input tensor (CB-attached)
        scaler: Scaler tensor for reduction (CB-attached)
        dims: Dimensions to collapse (PyTorch convention) -
              [0] collapses rows, [1] collapses columns, [0, 1] for scalar
    """
    from ttmlir.ir import IntegerAttr, IntegerType, RankedTensorType

    dims_set = set(dims)
    if dims_set == {0}:
        reduce_dim_val = 1  # Col reduce: collapse rows -> (1, M)
    elif dims_set == {1}:
        reduce_dim_val = 0  # Row reduce: collapse columns -> (N, 1)
    elif dims_set == {0, 1}:
        reduce_dim_val = 2  # Scalar
    else:
        raise ValueError(f"Invalid dims: {dims}. Must be [0], [1], or [0, 1]")

    input_type = input.type
    in_shape = list(input_type.shape)
    if dims_set == {0}:
        out_shape = [1, in_shape[1]]
    elif dims_set == {1}:
        out_shape = [in_shape[0], 1]
    else:  # Scalar
        out_shape = [1, 1]
    result_type = RankedTensorType.get(
        out_shape, input_type.element_type, input_type.encoding
    )

    ctx = input.type.context
    i32_type = IntegerType.get_signless(32, ctx)
    reduce_type_attr = IntegerAttr.get(i32_type, 1)  # Max = 1
    reduce_dim_attr = IntegerAttr.get(i32_type, reduce_dim_val)
    return ttl.reduce(
        result_type, input, scaler, reduce_type_attr, reduce_dim_attr
    )


@syntax("power")
def power(input: TensorBlock, exponent) -> TensorBlock:
    """Raise tensor elements to an integer power."""
    from ttmlir.ir import IntegerAttr, IntegerType

    exp_val = _get_constant_int(exponent)
    ctx = input.type.context
    i32_type = IntegerType.get_signless(32, ctx)
    exp_attr = IntegerAttr.get(i32_type, exp_val)
    return ttl.power(input.type, input, exp_attr)


@syntax("fill")
def fill(output: TensorBlock, value) -> TensorBlock:
    """Fill a tensor with a constant f32 value."""
    from ttmlir.ir import FloatAttr, F32Type

    fill_val = _get_constant_float(value)
    ctx = output.type.context
    value_attr = FloatAttr.get(F32Type.get(ctx), fill_val)
    return ttl.fill(output.type, value_attr)


@syntax("where")
def where(
    condition: TensorBlock, true_value: TensorBlock, false_value: TensorBlock
) -> TensorBlock:
    """Element-wise conditional selection: cond ? true_val : false_val."""
    return ttl.where(true_value.type, condition, true_value, false_value)


__all__ = [
    "TensorBlock",
    "CopyTransferHandler",
    "copy",
    "core",
    "grid_size",
    "signpost",
    "matmul",
    "transpose",
    "reduce_sum",
    "reduce_max",
    "power",
    "fill",
    "where",
    *_generated_all,
]
