# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DSL operators for tensor operations and data movement."""

from __future__ import annotations

from typing import List, Tuple, Union

from ttl.dialects import arith
from ttl.ir import RankedTensorType, Type, FloatAttr, F32Type, IndexType

# Re-export generated elementwise operations
from ._generated_elementwise import *  # noqa: F401,F403
from ._generated_elementwise import __all__ as _generated_all
from ._src.ttl_ast import syntax
from ttl.dialects import ttl
from .pipe import Pipe


def _get_constant_int(val):
    """Extract Python int from MLIR arith.ConstantOp or return as-is if already int."""
    if isinstance(val, int):
        return val
    if isinstance(val, arith.ConstantOp):
        return val.literal_value
    raise ValueError(f"Expected int or arith.ConstantOp, got {type(val)}")


def _as_host_scalar(val):
    """Return `val` as a Python float if it is a Python int/float, an
    arith.ConstantOp, or an MLIR Value defined by arith.ConstantOp.
    Otherwise return None. Torch 0-dim tensors are not recognized."""
    if isinstance(val, (int, float)):
        return float(val)
    const_op = val if isinstance(val, arith.ConstantOp) else None
    if const_op is None and isinstance(getattr(val, "owner", None), arith.ConstantOp):
        const_op = val.owner
    if const_op is None:
        return None
    try:
        return float(const_op.literal_value)
    except (TypeError, ValueError):
        return None


def _get_constant_float(val):
    """Extract Python float from `val` (Python int/float or arith.ConstantOp).
    Raises ValueError if `val` is not a recognized host scalar."""
    result = _as_host_scalar(val)
    if result is None:
        raise ValueError(f"Expected float or arith.ConstantOp, got {type(val)}")
    return result


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

    def __mul__(ast_self: TensorBlock, rhs) -> TensorBlock:
        """Multiplication.

        If `rhs` is a host-side scalar (Python int/float or torch 0-dim
        float tensor), emit `ttl.mul_unary_const(self, rhs)`. Otherwise
        treat `rhs` as a TensorBlock and emit `ttl.mul`.
        """
        c = _as_host_scalar(rhs)
        if c is not None:
            ctx = ast_self.type.context
            value_attr = FloatAttr.get(F32Type.get(ctx), c)
            return ttl.mul_unary_const(ast_self, value_attr)
        return ttl.mul(ast_self.type, ast_self, rhs)

    def __rmul__(ast_self: TensorBlock, lhs) -> TensorBlock:
        """Reflected multiplication for `scalar * self`."""
        c = _as_host_scalar(lhs)
        if c is not None:
            ctx = ast_self.type.context
            value_attr = FloatAttr.get(F32Type.get(ctx), c)
            return ttl.mul_unary_const(ast_self, value_attr)
        return NotImplemented

    def __truediv__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Element-wise division using ttl.div."""
        return ttl.div(ast_self.type, ast_self, rhs)

    def __matmul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Matrix multiplication using ttl.matmul.

        Computes C[M,N] = A[M,K] * B[K,N]. Both operands must be
        CB-attached tensors of tiles.
        """
        lhs_type = ast_self.type
        rhs_type = rhs.type
        lhs_shape = list(lhs_type.shape)
        rhs_shape = list(rhs_type.shape)
        result_shape = [lhs_shape[0], rhs_shape[1]]
        result_type = RankedTensorType.get(
            result_shape, lhs_type.element_type, lhs_type.encoding
        )
        return ttl.matmul(result_type, ast_self, rhs)

    def store(ast_self: TensorBlock, rhs: TensorBlock) -> None:
        """Store result tensor to the output CB reserve view (overwrite).

        Emits ttl.store with the result tensor and reserve view.
        Always overwrites the CB slot. For accumulation, use ``+=``.
        """
        if not _is_block(ast_self):
            raise ValueError(
                "store() must be called on a block acquired from reserve(), not a regular tensor"
            )
        reserve = _get_reserve_from_block(ast_self)
        ttl.store(rhs, reserve)

    def __iadd__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Accumulate into a reserved block via L1 packer accumulation.

        Emits ttl.store with the ``accumulate`` attribute. When used
        inside a loop, the compiler inserts ``pack_reconfig_l1_acc``
        guards so that each iteration adds to the existing L1 value
        instead of overwriting.

        This is an interim mechanism; the spec's full pattern
        (``fill`` + lazy ``BlockExpr`` ``+=`` + ``store``) is deferred
        to the BlockExpr PR (#446).
        """
        if not _is_block(ast_self):
            raise ValueError(
                "+= must be called on a block acquired from reserve(), not a regular tensor"
            )
        reserve = _get_reserve_from_block(ast_self)
        ttl.store(rhs, reserve, accumulate=True)
        return ast_self

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
    if hasattr(val, "type") and ttl.PipeType.maybe_downcast(val.type):
        return True
    return isinstance(val, Pipe) and hasattr(val, "_mlir_value")


def _get_pipe_mlir_value(pipe):
    """Get the MLIR value for a pipe (either MLIR value or Python Pipe object)."""
    if hasattr(pipe, "type") and ttl.PipeType.maybe_downcast(pipe.type):
        return pipe
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


@syntax("node")
def node(*, dims):
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
        x, y = ttl.node(dims=2)
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
def broadcast(
    input: TensorBlock, output: TensorBlock, *, dims: List[int]
) -> TensorBlock:
    """
    Broadcast over specified dimensions.

    Only 2D tensors are supported for broadcast (hardware constraint).

    ``dims`` uses the same indexing as PyTorch ``dim`` arguments: each index must
    lie in ``[-ndim, ndim - 1]`` for ``ndim == 2`` (outermost is ``0`` or ``-2``,
    innermost is ``1`` or ``-1``). Duplicate indices after normalization are
    allowed (e.g. ``[0, -2]`` is row broadcast).

    Args:
        input: Input tensor (CB-attached)
        output: Output tensor (CB-attached, used for output CB tracking)
        dims: Dimensions to broadcast over

    Returns:
        Result tensor with broadcast values
    """
    from ttl.ir import IntegerAttr, IntegerType

    if isinstance(input.type, RankedTensorType) and input.type.rank != 2:
        raise ValueError(
            f"broadcast only supports 2D tensors, got rank {input.type.rank}. "
            "Use 2D tensors for broadcast operations."
        )

    rank = 2
    if not dims:
        raise ValueError("dims must be a non-empty list of dimension indices")

    for d in dims:
        if d < -rank or d >= rank:
            raise ValueError(
                f"Invalid broadcast dimension {d}: for rank-{rank} tensors, "
                f"each index must satisfy {-rank} <= dim <= {rank - 1} "
                "(PyTorch-style dim indexing)"
            )

    dims_set = {d % rank for d in dims}
    if dims_set == {0}:
        bcast_val = 2  # Row
    elif dims_set == {1}:
        bcast_val = 1  # Col
    elif dims_set == {0, 1}:
        bcast_val = 3  # Scalar
    else:
        raise ValueError(
            f"Invalid dims: {dims}. After normalization, expect row [0]/[-2], "
            f"col [1]/[-1], or both for scalar broadcast (e.g. [0,1] or [-2,-1])"
        )

    ctx = input.type.context
    i32_type = IntegerType.get_signless(32, ctx)
    bcast_attr = IntegerAttr.get(i32_type, bcast_val)
    return ttl.bcast(output.type, input, output, bcast_attr)


def _reduce_impl(
    input: TensorBlock,
    dims: List[int],
    reduce_type: int,
) -> TensorBlock:
    """Shared implementation for reduce_sum and reduce_max."""
    from ttl.ir import IntegerAttr, IntegerType, DenseI64ArrayAttr

    input_type = input.type
    input_shape = list(input_type.shape)
    rank = len(input_shape)
    if rank != 2:
        raise ValueError(f"reduce only supports 2D tensors, got rank {rank}")
    if not dims:
        raise ValueError("dims must be non-empty")

    for d in dims:
        if d < -rank or d >= rank:
            raise ValueError(
                f"dim {d} out of range for rank {rank}: "
                f"must be in [{-rank}, {rank - 1}]"
            )
    norm_dims = sorted({d % rank for d in dims})

    result_shape = [1 if i in norm_dims else s for i, s in enumerate(input_shape)]
    result_type = RankedTensorType.get(
        result_shape, input_type.element_type, input_type.encoding
    )

    ctx = input_type.context
    i32_type = IntegerType.get_signless(32, ctx)
    reduce_type_attr = IntegerAttr.get(i32_type, reduce_type)
    dims_attr = DenseI64ArrayAttr.get(dims, ctx)
    scaler_type = RankedTensorType.get(
        [1, 1], input_type.element_type, input_type.encoding
    )
    scaler = ttl.fill(scaler_type, FloatAttr.get(F32Type.get(ctx), 1.0))
    return ttl.reduce(result_type, input, scaler, reduce_type_attr, dims_attr)


@syntax("reduce_sum")
def reduce_sum(input: TensorBlock, *, dims: List[int]) -> TensorBlock:
    """Sum reduction over specified dimensions.

    To scale the result by a constant, multiply: `c * reduce_sum(x, dims=...)`.
    """
    return _reduce_impl(input, dims, reduce_type=0)


@syntax("reduce_max")
def reduce_max(input: TensorBlock, *, dims: List[int]) -> TensorBlock:
    """Max reduction over specified dimensions.

    To scale the result by a constant, multiply: `c * reduce_max(x, dims=...)`.
    """
    return _reduce_impl(input, dims, reduce_type=1)


@syntax("transpose")
def transpose(input: TensorBlock) -> TensorBlock:
    """Transpose a 2D block: (M, N) -> (N, M)."""
    input_type = input.type
    input_shape = list(input_type.shape)
    if len(input_shape) != 2:
        raise ValueError(
            f"transpose only supports 2D tensors, got rank {len(input_shape)}"
        )
    result_shape = [input_shape[1], input_shape[0]]
    result_type = RankedTensorType.get(
        result_shape, input_type.element_type, input_type.encoding
    )
    return ttl.transpose(result_type, input)


@syntax("fill")
def fill(output: TensorBlock, value) -> TensorBlock:
    """Fill a tensor with a constant f32 value."""
    fill_val = _get_constant_float(value)
    ctx = output.type.context
    value_attr = FloatAttr.get(F32Type.get(ctx), fill_val)
    return ttl.fill(output.type, value_attr)


def _is_supported_typecast_dtype(ttcore_dtype) -> bool:
    from ttl.dialects import ttcore

    return ttcore_dtype in {
        ttcore.DataType.Float32,
        ttcore.DataType.BFloat16,
        ttcore.DataType.BFP_BFloat8,
        ttcore.DataType.BFP_BFloat4,
    }


def _is_supported_typecast_tile_type(tile_type) -> bool:
    from ttl.dialects import ttcore

    return _is_supported_typecast_dtype(ttcore.DataType(tile_type.data_type_as_int))


@syntax("typecast")
def typecast(input: TensorBlock, dtype) -> TensorBlock:
    """
    Elementwise typecast: convert each element of ``input`` to ``dtype``.

    Args:
        input: Input tensor (CB-attached). Each element is a tile.
        dtype: Target data type. Accepts a ``ttcore.DataType`` enum value
            or a torch/ttnn dtype convertible via ``dtype_utils``.

    Returns:
        Result tensor with the same shape as ``input`` but with the element
        type derived from ``dtype``.
    """
    from ttl.dialects import ttcore
    from .dtype_utils import tensor_dtype_to_ttcore_datatype

    if isinstance(dtype, ttcore.DataType):
        ttcore_dtype = dtype
    else:
        ttcore_dtype = tensor_dtype_to_ttcore_datatype(dtype)
    if not _is_supported_typecast_dtype(ttcore_dtype):
        raise ValueError(
            f"typecast only supports floating-point destination dtypes, got {dtype}"
        )

    input_type = input.type
    if not isinstance(input_type, RankedTensorType):
        raise ValueError(f"typecast expects a RankedTensorType input, got {input_type}")

    ctx = input_type.context
    input_tile = ttcore.ir.TileType.maybe_downcast(input_type.element_type)
    if input_tile is None:
        raise ValueError(
            f"typecast expects tile-typed elements, got {input_type.element_type}"
        )
    if not _is_supported_typecast_tile_type(input_tile):
        raise ValueError(
            "typecast only supports floating-point input tile dtypes, got "
            f"{input_tile}"
        )

    out_tile_type = ttcore.ir.TileType.get(
        ctx, input_tile.shape[0], input_tile.shape[1], ttcore_dtype
    )
    result_type = RankedTensorType.get(
        input_type.shape, out_tile_type, input_type.encoding
    )
    return ttl.typecast(result_type, input)


def _get_block_scalar_type(block):
    """Extract the scalar MLIR type from a block's tensor element type.

    For tiled blocks (!ttcore.tile<H, W, dtype>), returns the corresponding
    scalar type (f32 for Float32, bf16 for BFloat16).
    For row-major blocks, returns the element type directly.
    """
    from ttl.dialects import ttcore
    from ttl.ir import BF16Type

    block_type = block.type
    if not isinstance(block_type, RankedTensorType):
        raise ValueError(f"Expected RankedTensorType block, got {block_type}")

    elem_type = block_type.element_type
    tile_type = ttcore.ir.TileType.maybe_downcast(elem_type)
    if tile_type is not None:
        dtype = ttcore.DataType(tile_type.data_type_as_int)
        ctx = block_type.context
        if dtype == ttcore.DataType.Float32:
            return F32Type.get(ctx)
        if dtype == ttcore.DataType.BFloat16:
            return BF16Type.get(ctx)
        raise ValueError(
            f"raw element access only supports f32 and bf16, got tile dtype {dtype}"
        )
    if elem_type == F32Type.get(block_type.context):
        return elem_type
    if elem_type == BF16Type.get(block_type.context):
        return elem_type
    raise ValueError(
        f"raw element access only supports f32 and bf16, got element type {elem_type}"
    )


@syntax("raw_element_read")
def raw_element_read(block, *coords):
    """Read a scalar element from a block at flat coordinates.

    Coordinates are scalar-element positions within the block. The number
    of coordinates must match the block tensor rank.

    For tiled blocks, lowering decomposes them into tile + intra-tile offsets.

    Only supported in data movement (noc) threads.

    Args:
        block: Block tensor (from cb.reserve() or cb.wait())
        *coords: Index values matching the block tensor rank

    Returns:
        Scalar value matching the block's element dtype
    """
    if len(coords) < 1:
        raise ValueError("raw_element_read requires at least one coordinate")
    scalar_type = _get_block_scalar_type(block)
    ctx = block.type.context
    index_vals = []
    for c in coords:
        if isinstance(c, int):
            index_vals.append(arith.ConstantOp(IndexType.get(ctx), c))
        elif hasattr(c, "type") and isinstance(c.type, IndexType):
            index_vals.append(c)
        else:
            index_vals.append(arith.IndexCastOp(IndexType.get(ctx), c))
    return ttl.raw_element_read(scalar_type, block, index_vals)


@syntax("raw_element_write")
def raw_element_write(block, *args):
    """Write a scalar value to a block at flat coordinates.

    Coordinates are scalar-element positions within the block. The number
    of coordinates must match the block tensor rank. The last argument
    is the value to write; all preceding arguments are coordinates.

    For tiled blocks, lowering decomposes them into tile + intra-tile offsets.

    Only supported in data movement (noc) threads.

    Args:
        block: Block tensor (from cb.reserve() or cb.wait())
        *args: N index values followed by the scalar value to write.

    Example:
        ttl.raw_element_write(block, row, col, val)
    """

    if len(args) < 2:
        raise ValueError(
            "raw_element_write requires at least one coordinate and a value"
        )
    coord_args = args[:-1]
    val = args[-1]
    ctx = block.type.context
    index_vals = []
    for c in coord_args:
        if isinstance(c, int):
            index_vals.append(arith.ConstantOp(IndexType.get(ctx), c))
        elif hasattr(c, "type") and isinstance(c.type, IndexType):
            index_vals.append(c)
        else:
            index_vals.append(arith.IndexCastOp(IndexType.get(ctx), c))
    ttl.raw_element_write(block, index_vals, val)


__all__ = [
    "TensorBlock",
    "CopyTransferHandler",
    "copy",
    "core",
    "grid_size",
    "signpost",
    "fill",
    "typecast",
    "raw_element_read",
    "raw_element_write",
    *_generated_all,
]
