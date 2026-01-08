# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DSL operators for tensor operations and data movement."""

from __future__ import annotations

from typing import Tuple, Union

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


def _make_tensor_slice(tensor, indices):
    """Create a ttl.tensor_slice from a tensor and tile indices."""
    if len(indices) != 2:
        raise ValueError(f"Tensor slice requires exactly 2 indices, got {len(indices)}")

    row_idx, col_idx = indices
    # Result type is inferred from tensor type via TypesMatchWith trait
    return ttl.tensor_slice(tensor, row_idx, col_idx)


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
    # Handle subscripted tensors by creating tensor slices
    if isinstance(src, tuple):
        tensor, indices = src
        src = _make_tensor_slice(tensor, indices)
    if isinstance(dst, tuple):
        tensor, indices = dst
        dst = _make_tensor_slice(tensor, indices)

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
