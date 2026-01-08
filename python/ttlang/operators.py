# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DSL operators for tensor operations and data movement."""

from __future__ import annotations

from typing import Tuple, Union

from ttmlir.dialects import tensor
from ttmlir.ir import Type

# Re-export generated elementwise operations
from ._generated_elementwise import *  # noqa: F401,F403
from ._generated_elementwise import __all__ as _generated_all
from ._src.ttl_ast import syntax
from .dialects import ttl

# Type aliases for common patterns
CoreCoordinate = Tuple[int, int]
IndexedTensor = Union["TensorBlock", Tuple["TensorBlock", Tuple[int, ...]]]


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
        """Matrix multiplication is not yet supported in TTL mode."""
        raise NotImplementedError("Matrix multiplication not yet supported in TTL mode")

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


# Sentinel value for dynamic dimensions in tensor.extract_slice
# This is ShapedType::kDynamic in MLIR
DYNAMIC_SENTINEL = -9223372036854775808  # INT64_MIN


def _try_get_constant_int(value):
    """Try to extract constant integer from an MLIR arith.constant op.

    Returns (int_value, None) if the value is from arith.constant,
    or (None, value) if it's a dynamic value (e.g., loop induction variable).
    """
    # Handle case where we get the Op object instead of its result Value
    if hasattr(value, "result"):
        op = value
        result_val = value.result
    elif hasattr(value, "owner"):
        op = value.owner
        result_val = value
    else:
        # Not a recognized MLIR type, treat as dynamic
        return (None, value)

    # Check if owner is actually an Op (not a Block, which happens for loop vars)
    if not hasattr(op, "name"):
        return (None, result_val)

    if op.name != "arith.constant":
        return (None, result_val)

    # Get the value attribute from arith.constant
    attr = op.attributes["value"]
    return (int(attr), None)


def _make_extract_slice(src_tensor, indices):
    """Create a tensor.extract_slice from a tensor and tile indices.

    The indices are tile coordinates. For DMA operations, we extract a
    single tile at the specified position.

    Handles both 2D tensors (MLIR lit tests) and 4D device-shaped tensors
    (Python-generated). For 4D tensors, the indices map to shard dimensions
    (last two), with zeros for grid dimensions.

    Supports both constant indices (compile-time known) and dynamic indices
    (e.g., loop induction variables).
    """
    if len(indices) != 2:
        raise ValueError(f"Tensor slice requires exactly 2 indices, got {len(indices)}")

    row_const, row_dyn = _try_get_constant_int(indices[0])
    col_const, col_dyn = _try_get_constant_int(indices[1])

    tensor_type = src_tensor.type
    shape = list(tensor_type.shape)
    rank = len(shape)

    # Build offset arrays - use DYNAMIC_SENTINEL for dynamic offsets
    # and collect dynamic values separately
    dynamic_offsets = []

    if rank == 2:
        # 2D logical shape (lit tests): offsets are tile indices directly
        static_offsets = []
        if row_const is not None:
            static_offsets.append(row_const)
        else:
            static_offsets.append(DYNAMIC_SENTINEL)
            dynamic_offsets.append(row_dyn)
        if col_const is not None:
            static_offsets.append(col_const)
        else:
            static_offsets.append(DYNAMIC_SENTINEL)
            dynamic_offsets.append(col_dyn)
        static_sizes = shape
        static_strides = [1, 1]
    elif rank == 4:
        # 4D device shape [grid_y, grid_x, shard_tiles_y, shard_tiles_x]
        # Grid dims are always 0 (static), tile indices go to shard dims
        static_offsets = [0, 0]  # Grid offsets are always 0
        # Shard row offset
        if row_const is not None:
            static_offsets.append(row_const)
        else:
            static_offsets.append(DYNAMIC_SENTINEL)
            dynamic_offsets.append(row_dyn)
        # Shard col offset
        if col_const is not None:
            static_offsets.append(col_const)
        else:
            static_offsets.append(DYNAMIC_SENTINEL)
            dynamic_offsets.append(col_dyn)
        static_sizes = [1, 1, 1, 1]
        static_strides = [1, 1, 1, 1]
    else:
        raise ValueError(
            f"Expected 2D or 4D tensor, got {rank}D tensor with shape {shape}"
        )

    # Build result type with the extracted shape
    from ttmlir.ir import RankedTensorType
    result_type = RankedTensorType.get(
        static_sizes, tensor_type.element_type, tensor_type.encoding
    )

    return tensor.extract_slice(
        result_type,
        src_tensor,
        dynamic_offsets,
        [],  # dynamic sizes (always static)
        [],  # dynamic strides (always static)
        static_offsets,
        static_sizes,
        static_strides,
    )


@syntax("copy")
def copy(src, dst) -> CopyTransferHandler:
    """
    Initiate an asynchronous data transfer using ttl.copy.

    Args:
        src: Source tensor/slice (for reads) or CB (for writes)
        dst: Destination CB (for reads) or tensor/slice (for writes)

    Returns:
        CopyTransferHandler handle that must be waited on for completion
    """
    # Handle subscripted tensors by creating tensor.extract_slice
    if isinstance(src, tuple):
        src_tensor, indices = src
        src = _make_extract_slice(src_tensor, indices)
    if isinstance(dst, tuple):
        dst_tensor, indices = dst
        dst = _make_extract_slice(dst_tensor, indices)

    ctx = src.type.context

    src_type_str = str(src.type)
    dst_type_str = str(dst.type)
    src_is_cb = src_type_str.startswith("!ttl.cb")
    dst_is_cb = dst_type_str.startswith("!ttl.cb")

    if dst_is_cb and not src_is_cb:
        # Read: device tensor/slice -> CB
        xf_type = Type.parse("!ttl.transfer_handle<read>", ctx)
        return ttl.copy(xf_type, src, dst)
    elif src_is_cb and not dst_is_cb:
        # Write: CB -> device tensor/slice
        xf_type = Type.parse("!ttl.transfer_handle<write>", ctx)
        return ttl.copy(xf_type, src, dst)
    else:
        raise ValueError(
            f"copy() requires exactly one CB argument. "
            f"Got src={src_type_str}, dst={dst_type_str}"
        )


__all__ = [
    "TensorBlock",
    "CopyTransferHandler",
    "copy",
    *_generated_all,
]
