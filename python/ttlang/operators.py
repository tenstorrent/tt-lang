# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DSL operators for tensor operations, data movement, and memory transactions."""

from __future__ import annotations

from typing import List, Callable, Optional, Tuple, Union

from ttmlir.ir import *
from ttmlir.dialects import arith, linalg, d2m

from .dialects import ttl
from ._src.ttl_ast import syntax
from pykernel._src.utils import _asindex

# Type aliases for common patterns
CoreCoordinate = Tuple[int, int]
IndexedTensor = Union["TensorBlock", Tuple["TensorBlock", Tuple[int, ...]]]


def _create_linalg_generic(
    lhs: TensorBlock,
    rhs: TensorBlock,
    output_shape: List[int],
    affine_maps: List[AffineMap],
    iterator_types: List[str],
    tile_op_builder: Callable,
) -> TensorBlock:
    """
    Create a linalg.generic operation with a D2M tile operation in the body.

    This helper encapsulates the common pattern for creating linalg.generic
    operations with D2M tile operations for elementwise and reduction operations.

    Args:
        lhs: Left-hand side operand
        rhs: Right-hand side operand
        output_shape: Shape of the output tensor
        affine_maps: List of AffineMap objects for indexing
        iterator_types: List of iterator type strings ("parallel" or "reduction")
        tile_op_builder: Function that takes (result_type, *block_args) and creates tile op

    Returns:
        Result of the linalg.generic operation
    """
    ctx = lhs.type.context

    out_type = RankedTensorType.get(
        output_shape, lhs.type.element_type, lhs.type.encoding
    )
    empty = d2m.empty(out_type)

    affine_maps_attr = ArrayAttr.get([AffineMapAttr.get(m) for m in affine_maps])

    iter_types_attr = ArrayAttr.get(
        [Attribute.parse(f"#linalg.iterator_type<{it}>", ctx) for it in iterator_types]
    )

    # Linalg.generic: affine_maps = [input1_map, input2_map, output_map]
    num_inputs = len(affine_maps) - 1
    if num_inputs != 2:
        raise ValueError(f"Function only supports 2 inputs, got {num_inputs}")
    inputs = [lhs, rhs]

    generic_op = linalg.GenericOp(
        result_tensors=[out_type],
        inputs=inputs,
        outputs=[empty],
        indexing_maps=affine_maps_attr,
        iterator_types=iter_types_attr,
    )

    block_arg_types = [inp.type.element_type for inp in inputs] + [
        empty.type.element_type
    ]
    block = generic_op.regions[0].blocks.append(*block_arg_types)

    with InsertionPoint(block):
        tile_result = tile_op_builder(lhs.type.element_type, *block.arguments)
        linalg.YieldOp([tile_result])

    return generic_op.result


def _create_linalg_generic_unary(
    input_tensor: TensorBlock,
    output_shape: List[int],
    affine_maps: List[AffineMap],
    iterator_types: List[str],
    tile_op_builder: Callable,
) -> TensorBlock:
    """
    Create a linalg.generic operation for unary operations.

    Args:
        input_tensor: Input operand
        output_shape: Shape of the output tensor
        affine_maps: List of AffineMap objects for indexing
        iterator_types: List of iterator type strings ("parallel" or "reduction")
        tile_op_builder: Function that takes (result_type, *block_args) and creates tile op

    Returns:
        Result of the linalg.generic operation
    """
    ctx = input_tensor.type.context

    out_type = RankedTensorType.get(
        output_shape, input_tensor.type.element_type, input_tensor.type.encoding
    )
    empty = d2m.empty(out_type)

    affine_maps_attr = ArrayAttr.get([AffineMapAttr.get(m) for m in affine_maps])

    iter_types_attr = ArrayAttr.get(
        [Attribute.parse(f"#linalg.iterator_type<{it}>", ctx) for it in iterator_types]
    )

    # Linalg.generic: affine_maps = [input_map, output_map]
    num_inputs = len(affine_maps) - 1
    if num_inputs != 1:
        raise ValueError(f"Function only supports 1 input, got {num_inputs}")
    inputs = [input_tensor]

    generic_op = linalg.GenericOp(
        result_tensors=[out_type],
        inputs=inputs,
        outputs=[empty],
        indexing_maps=affine_maps_attr,
        iterator_types=iter_types_attr,
    )

    block_arg_types = [inp.type.element_type for inp in inputs] + [
        empty.type.element_type
    ]
    block = generic_op.regions[0].blocks.append(*block_arg_types)

    with InsertionPoint(block):
        tile_result = tile_op_builder(input_tensor.type.element_type, *block.arguments)
        linalg.YieldOp([tile_result])

    return generic_op.result


def _unary_elementwise(input_tensor: TensorBlock, tile_op: Callable) -> TensorBlock:
    """
    Create a unary elementwise operation with identity indexing.

    Args:
        input_tensor: Input tensor operand
        tile_op: D2M tile operation (e.g., d2m.tile_exp, d2m.tile_sqrt)

    Returns:
        Result tensor with same shape as input
    """
    if not isinstance(input_tensor.type, RankedTensorType):
        raise TypeError(
            f"Expected RankedTensorType, got {type(input_tensor.type).__name__}"
        )

    ctx = input_tensor.type.context
    rank = len(input_tensor.type.shape)
    identity_map = AffineMap.get_identity(rank, ctx)

    return _create_linalg_generic_unary(
        input_tensor,
        output_shape=list(input_tensor.type.shape),
        affine_maps=[identity_map, identity_map],
        iterator_types=["parallel"] * rank,
        tile_op_builder=lambda result_type, in_arg, out_arg: tile_op(
            result_type, in_arg
        ),
    )


def _binary_elementwise(
    lhs: TensorBlock, rhs: TensorBlock, tile_op: Callable
) -> TensorBlock:
    """
    Create a binary elementwise operation with identity indexing.

    Args:
        lhs: Left operand tensor
        rhs: Right operand tensor
        tile_op: D2M tile operation (e.g., d2m.tile_add, d2m.tile_mul)

    Returns:
        Result tensor with same shape as inputs
    """
    if not isinstance(lhs.type, RankedTensorType):
        raise TypeError(f"Expected RankedTensorType, got {type(lhs.type).__name__}")

    ctx = lhs.type.context
    rank = len(lhs.type.shape)
    identity_map = AffineMap.get_identity(rank, ctx)

    return _create_linalg_generic(
        lhs,
        rhs,
        output_shape=list(lhs.type.shape),
        affine_maps=[identity_map] * 3,
        iterator_types=["parallel"] * rank,
        tile_op_builder=lambda result_type, lhs_arg, rhs_arg, out_arg: tile_op(
            result_type, lhs_arg, rhs_arg
        ),
    )


@syntax("!tensor")
class TensorBlock:
    """
    Represents a block of tensor data in the TTL dialect.

    TensorBlock supports arithmetic and matrix operations through operator
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

    def __sub__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise subtraction using ttl.sub."""
        return ttl.sub(ast_self.type, ast_self, rhs)

    def __mul__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise multiplication using ttl.mul."""
        return ttl.mul(ast_self.type, ast_self, rhs)

    def __matmul__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """
        Matrix multiplication is not yet supported in TTL mode.
        """
        raise NotImplementedError("Matrix multiplication not yet supported in TTL mode")

    def store(ast_self: "TensorBlock", rhs: "TensorBlock") -> None:
        """Store result tensor to CB by propagating CB association from output view."""
        # ast_self is the result of attach_cb(tensor, cb) from reserve()
        # Extract the CB operand and attach it to the result tensor
        cb = ast_self.owner.operands[1]
        return ttl.attach_cb(rhs.type, rhs, cb)


@syntax("!ttl.transfer_handle")
class MemTx:
    """
    Memory transaction handle for asynchronous copy operations.

    MemTx objects are returned by dma()/copy() calls and must be explicitly
    waited on to ensure transfer completion.
    """

    def wait(ast_self: MemTx):
        """Block until the copy operation completes."""
        return ttl.wait(ast_self)


def _determine_transfer_direction(src, dst) -> str:
    """Determine transfer direction based on src/dst tensor types."""
    # If source has MetalLayoutAttr (device tensor), it's a read from device
    # If dest has MetalLayoutAttr (device tensor), it's a write to device
    src_type = src.type if hasattr(src, "type") else None
    dst_type = dst.type if hasattr(dst, "type") else None

    # Check if source has device layout (MetalLayoutAttr)
    src_is_device = False
    dst_is_device = False

    if src_type and isinstance(src_type, RankedTensorType):
        if src_type.encoding is not None:
            src_is_device = True

    if dst_type and isinstance(dst_type, RankedTensorType):
        if dst_type.encoding is not None:
            dst_is_device = True

    # read = device -> CB, write = CB -> device
    if src_is_device and not dst_is_device:
        return "read"
    elif not src_is_device and dst_is_device:
        return "write"
    else:
        # Default to read for now
        return "read"


@syntax("dma")
def dma(
    src: IndexedTensor,
    dst: IndexedTensor,
    core: Optional[CoreCoordinate] = None,
    mcast: Optional[CoreCoordinate] = None,
) -> MemTx:
    """
    DEPRECATED: Use copy() instead.

    Initiate an asynchronous data transfer using ttl.copy.
    """
    import warnings
    warnings.warn(
        "dma() is deprecated, use copy() instead",
        DeprecationWarning,
        stacklevel=2
    )
    if core is not None or mcast is not None:
        raise NotImplementedError("core and mcast parameters not supported, use copy() instead")

    src_indices = None
    dst_indices = None
    if isinstance(src, tuple):
        src, src_indices = src
    if isinstance(dst, tuple):
        dst, dst_indices = dst

    if src_indices is not None or dst_indices is not None:
        raise NotImplementedError("Indexed tensor access not supported, use copy() instead")

    return copy(src, dst)


@syntax("copy")
def copy(src, dst) -> MemTx:
    """
    Initiate an asynchronous data transfer using ttl.copy.

    Args:
        src: Source tensor (for reads) or CB (for writes)
        dst: Destination CB (for reads) or tensor (for writes)

    Returns:
        MemTx handle that must be waited on for completion
    """
    # TODO: Support non-zero indices for tensor accessors
    if isinstance(src, tuple):
        src, indices = src
        # indices are MLIR ConstantOp objects, extract literal values
        idx_vals = [getattr(i, 'literal_value', i) for i in indices]
        assert idx_vals == [0, 0], f"Only [0, 0] index supported, got {idx_vals}"
    if isinstance(dst, tuple):
        dst, indices = dst
        idx_vals = [getattr(i, 'literal_value', i) for i in indices]
        assert idx_vals == [0, 0], f"Only [0, 0] index supported, got {idx_vals}"

    ctx = src.type.context

    src_type_str = str(src.type)
    dst_type_str = str(dst.type)
    src_is_cb = src_type_str.startswith("!ttl.cb")
    dst_is_cb = dst_type_str.startswith("!ttl.cb")

    if dst_is_cb and not src_is_cb:
        # Read: device tensor -> CB
        xf_type = Type.parse("!ttl.transfer_handle<read>", ctx)
        return ttl.copy(xf_type, src, dst)
    elif src_is_cb and not dst_is_cb:
        # Write: CB -> device tensor
        xf_type = Type.parse("!ttl.transfer_handle<write>", ctx)
        return ttl.copy(xf_type, src, dst)
    else:
        raise ValueError(
            f"copy() requires exactly one CB argument. "
            f"Got src={src_type_str}, dst={dst_type_str}"
        )


# Unary element-wise operations
@syntax("exp")
def exp(input_tensor: TensorBlock) -> TensorBlock:
    """Element-wise exponential function."""
    return _unary_elementwise(input_tensor, d2m.tile_exp)


@syntax("sqrt")
def sqrt(input_tensor: TensorBlock) -> TensorBlock:
    """Element-wise square root function."""
    return _unary_elementwise(input_tensor, d2m.tile_sqrt)


@syntax("rsqrt")
def rsqrt(input_tensor: TensorBlock) -> TensorBlock:
    """Element-wise reciprocal square root function."""
    return _unary_elementwise(input_tensor, d2m.tile_rsqrt)


@syntax("recip")
def recip(input_tensor: TensorBlock) -> TensorBlock:
    """Element-wise reciprocal function."""
    return _unary_elementwise(input_tensor, d2m.tile_recip)


@syntax("maximum")
def maximum(lhs: TensorBlock, rhs: TensorBlock) -> TensorBlock:
    """Element-wise maximum function."""
    return _binary_elementwise(lhs, rhs, d2m.tile_maximum)


def _extract_int_from_mlir_value(value):
    """Extract integer from MLIR constant or return the value as-is."""
    if hasattr(value, "literal_value"):
        return value.literal_value
    elif hasattr(value, "value"):
        return int(value.value)
    else:
        return int(value)


def _create_reduction_linalg(
    a: TensorBlock,
    b: TensorBlock,
    dim: int,
    tile_op_builder: Callable,
) -> TensorBlock:
    """
    Create linalg.generic for reduction operations (sum, max).

    Generates a reduction over the specified dimension with broadcasting scaler.
    The scaler B value at b[0,0] is broadcast to all elements of A before reduction.

    Hardware behavior: tile_reduce_sum only fills certain positions in output tile:
    - dim=1 (column reduction): only column 0 has valid data
    - dim=0 (row reduction): only row 0 has valid data
    Use broadcast_reduce_result() to fill all positions.

    Args:
        a: Input tensor (full dimensions accessed)
        b: Scaler tensor (only b[0,0] is used, broadcast to all elements)
        dim: Reduction dimension (0 for rows, 1 for columns)
        tile_op_builder: Function that creates the tile operation

    Returns:
        Result of the linalg.generic operation (only partial positions filled)
    """
    if not isinstance(a.type, RankedTensorType):
        raise TypeError(f"Expected RankedTensorType, got {type(a.type).__name__}")

    from ttmlir.dialects.d2m import ReduceDim

    dim_value = _extract_int_from_mlir_value(dim)

    # Hardware ReduceDim mapping: dim=0 (reduce rows) -> C, dim=1 (reduce cols) -> R
    reduce_dim_attr = ReduceDim.R if dim_value == 1 else ReduceDim.C

    ctx = a.type.context
    rank = len(a.type.shape)
    out_shape = list(a.type.shape)

    # Indexing maps: input uses all dims, scaler broadcasts from (0,0), output collapses reduced dim
    identity_map = AffineMap.get_identity(rank, ctx)
    zero_map = AffineMap.get(
        2, 0, [AffineConstantExpr.get(0, ctx), AffineConstantExpr.get(0, ctx)], ctx
    )

    if dim_value == 1:
        # Column reduction: iterate rows (parallel), reduce cols (reduction)
        # Output only to column 0 (hardware behavior)
        output_map = AffineMap.get(
            2, 0, [AffineDimExpr.get(0, ctx), AffineConstantExpr.get(0, ctx)], ctx
        )
        iter_types = ["parallel", "reduction"]
    else:
        # Row reduction: reduce rows (reduction), iterate cols (parallel)
        # Output only to row 0 (hardware behavior)
        output_map = AffineMap.get(
            2, 0, [AffineConstantExpr.get(0, ctx), AffineDimExpr.get(1, ctx)], ctx
        )
        iter_types = ["reduction", "parallel"]

    out_type = RankedTensorType.get(out_shape, a.type.element_type, a.type.encoding)
    empty = d2m.empty(out_type)

    affine_maps_attr = ArrayAttr.get(
        [
            AffineMapAttr.get(identity_map),
            AffineMapAttr.get(zero_map),
            AffineMapAttr.get(output_map),
        ]
    )
    iter_types_attr = ArrayAttr.get(
        [Attribute.parse(f"#linalg.iterator_type<{it}>", ctx) for it in iter_types]
    )

    generic_op = linalg.GenericOp(
        result_tensors=[out_type],
        inputs=[a, b],
        outputs=[empty],
        indexing_maps=affine_maps_attr,
        iterator_types=iter_types_attr,
    )

    block_arg_types = [
        a.type.element_type,
        b.type.element_type,
        empty.type.element_type,
    ]
    block = generic_op.regions[0].blocks.append(*block_arg_types)

    with InsertionPoint(block):
        reduce_dim_mlir_attr = Attribute.parse(
            f"#d2m<reduce_dim {reduce_dim_attr.name}>", ctx
        )
        tile_result = tile_op_builder(
            a.type.element_type,
            block.arguments[0],
            block.arguments[1],
            block.arguments[2],
            reduce_dim_mlir_attr,
        )
        linalg.YieldOp([tile_result])

    return generic_op.result


@syntax("bcast")
def bcast(input_tensor: TensorBlock, dim: int = 1) -> TensorBlock:
    """
    Broadcast a tile along the specified dimension.

    Hardware operation that broadcasts values within a tile:
    - dim=1: Broadcast column 0 to all columns (after column reduction)
    - dim=0: Broadcast row 0 to all rows (after row reduction)

    Args:
        input_tensor: Input tensor (e.g., result from reduce_sum)
        dim: Dimension to broadcast (0=rows, 1=columns)

    Returns:
        Tensor with broadcasted values
    """
    if not isinstance(input_tensor.type, RankedTensorType):
        raise TypeError(
            f"Expected RankedTensorType, got {type(input_tensor.type).__name__}"
        )

    ctx = input_tensor.type.context
    rank = len(input_tensor.type.shape)
    dim_value = _extract_int_from_mlir_value(dim)

    identity_map = AffineMap.get_identity(rank, ctx)

    out_type = RankedTensorType.get(
        list(input_tensor.type.shape),
        input_tensor.type.element_type,
        input_tensor.type.encoding,
    )
    empty = d2m.empty(out_type)

    affine_maps_attr = ArrayAttr.get(
        [AffineMapAttr.get(identity_map), AffineMapAttr.get(identity_map)]
    )
    iter_types_attr = ArrayAttr.get(
        [
            Attribute.parse("#linalg.iterator_type<parallel>", ctx),
            Attribute.parse("#linalg.iterator_type<parallel>", ctx),
        ]
    )

    generic_op = linalg.GenericOp(
        result_tensors=[out_type],
        inputs=[input_tensor],
        outputs=[empty],
        indexing_maps=affine_maps_attr,
        iterator_types=iter_types_attr,
    )

    block_arg_types = [input_tensor.type.element_type, empty.type.element_type]
    block = generic_op.regions[0].blocks.append(*block_arg_types)

    with InsertionPoint(block):
        from ttmlir.dialects.d2m import TileBcastType

        bcast_type = TileBcastType.Col if dim_value == 1 else TileBcastType.Row
        bcast_type_attr = Attribute.parse(
            f"#d2m<tile_bcast_type {bcast_type.name.lower()}>", ctx
        )

        tile_result = d2m.tile_bcast(
            input_tensor.type.element_type, block.arguments[0], bcast_type_attr
        )
        linalg.YieldOp([tile_result])

    return generic_op.result


# Reduction operations
@syntax("reduce_sum")
def reduce_sum(a: TensorBlock, b: TensorBlock, dim: int = 1) -> TensorBlock:
    """
    Sum reduction: result <- sum<dim>(A * B).

    The scaler B value at b[0,0] is broadcast to all elements of A before reduction.

    Hardware behavior: Only certain positions in output tile contain valid data:
    - dim=1 (column reduction): only column 0 is valid
    - dim=0 (row reduction): only row 0 is valid

    Use bcast() operator to broadcast the result to all positions.

    Args:
        a: Input tensor
        b: Scaler tensor (only b[0,0] is used, broadcast to all elements)
        dim: Reduction dimension (0=rows, 1=columns)

    Returns:
        Reduced tensor with only first column (dim=1) or first row (dim=0) valid.
    """
    print(
        "WARNING: reduce_sum uses DST register as accumulator. "
        "Ensure output tensor is pre-initialized with zeros to avoid garbage accumulation. "
        "See GitHub issue #31 for details."
    )
    return _create_reduction_linalg(a, b, dim, d2m.tile_reduce_sum)


@syntax("reduce_max")
def reduce_max(a: TensorBlock, b: TensorBlock, dim: int = 1) -> TensorBlock:
    """
    Max reduction: result <- max<dim>(A * B).

    The scaler B value at b[0,0] is broadcast to all elements of A before reduction.

    Hardware behavior: Only certain positions in output tile contain valid data:
    - dim=1 (column reduction): only column 0 is valid
    - dim=0 (row reduction): only row 0 is valid

    Use bcast() operator to broadcast the result to all positions.

    Args:
        a: Input tensor
        b: Scaler tensor (only b[0,0] is used, broadcast to all elements)
        dim: Reduction dimension (0=rows, 1=columns)

    Returns:
        Reduced tensor with only first column (dim=1) or first row (dim=0) valid.
    """
    print(
        "WARNING: reduce_max uses DST register as accumulator. "
        "Ensure output tensor is pre-initialized with zeros to avoid garbage accumulation. "
        "See GitHub issue #31 for details."
    )
    return _create_reduction_linalg(a, b, dim, d2m.tile_reduce_max)


@syntax("transpose")
def transpose(input_tensor: TensorBlock) -> TensorBlock:
    """Transpose the input tensor (swap last two dimensions)."""
    if not isinstance(input_tensor.type, RankedTensorType):
        raise TypeError(
            f"Expected RankedTensorType, got {type(input_tensor.type).__name__}"
        )

    ctx = input_tensor.type.context
    rank = len(input_tensor.type.shape)

    # Output shape has last two dims swapped
    out_shape = list(input_tensor.type.shape)
    if rank >= 2:
        out_shape[-2], out_shape[-1] = out_shape[-1], out_shape[-2]

    identity_map = AffineMap.get_identity(rank, ctx)

    return _create_linalg_generic_unary(
        input_tensor,
        output_shape=out_shape,
        affine_maps=[identity_map, identity_map],
        iterator_types=["parallel"] * rank,
        tile_op_builder=lambda result_type, in_arg, out_arg: d2m.tile_transpose(
            result_type, in_arg
        ),
    )
