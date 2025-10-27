# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DSL operators for tensor operations, circular buffers, DMA, and semaphores."""

from ttmlir.ir import *
from ttmlir.dialects import d2m, arith, linalg
from ttmlir.dialects._linalg_ops_gen import GenericOp

from ._src.d2m_ast import syntax
from ._src.utils import _asindex


@syntax("!tensor")
class TensorBlock:
    """
    Represents a block of tensor data in the D2M dialect.

    TensorBlock supports arithmetic and matrix operations through operator
    overloading. Operations generate linalg.generic ops with D2M tile operations.
    """

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __add__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise addition generating linalg.generic with d2m.tile_add."""
        lhs = ast_self
        assert isinstance(lhs.type, RankedTensorType)

        out_type = lhs.type
        empty = d2m.empty(out_type)

        ctx = lhs.type.context
        rank = len(lhs.type.shape)
        identity_map = AffineMap.get_identity(rank, ctx)
        affine_maps_attr = ArrayAttr.get([AffineMapAttr.get(identity_map)] * 3)

        iter_types_attr = ArrayAttr.get([
            Attribute.parse('#linalg.iterator_type<parallel>', ctx) for _ in range(rank)
        ])

        generic_op = GenericOp(
            result_tensors=[out_type],
            inputs=[lhs, rhs],
            outputs=[empty],
            indexing_maps=affine_maps_attr,
            iterator_types=iter_types_attr
        )

        block = generic_op.regions[0].blocks.append(
            lhs.type.element_type,
            rhs.type.element_type,
            empty.type.element_type
        )

        with InsertionPoint(block):
            tile_result = d2m.tile_add(
                lhs.type.element_type,
                block.arguments[0],
                block.arguments[1]
            )
            linalg.YieldOp([tile_result])

        return generic_op.result

    def __sub__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise subtraction."""
        return arith.subf(ast_self, rhs)

    def __mul__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise multiplication."""
        return arith.mulf(ast_self, rhs)

    def __truediv__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise division."""
        return arith.divf(ast_self, rhs)

    def __matmul__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Matrix multiplication generating linalg.generic with d2m.tile_matmul."""
        lhs = ast_self
        assert isinstance(lhs.type, RankedTensorType)

        out_shape = list(lhs.type.shape)
        out_shape[-1] = rhs.type.shape[-1]

        out_type = RankedTensorType.get(
            out_shape, lhs.type.element_type, lhs.type.encoding
        )
        empty = d2m.empty(out_type)

        ctx = lhs.type.context
        matmul_maps = [
            AffineMap.get(3, 0, [AffineDimExpr.get(0, ctx), AffineDimExpr.get(2, ctx)], ctx),
            AffineMap.get(3, 0, [AffineDimExpr.get(2, ctx), AffineDimExpr.get(1, ctx)], ctx),
            AffineMap.get(3, 0, [AffineDimExpr.get(0, ctx), AffineDimExpr.get(1, ctx)], ctx),
        ]
        affine_maps_attr = ArrayAttr.get([AffineMapAttr.get(m) for m in matmul_maps])

        iter_types_attr = ArrayAttr.get([
            Attribute.parse('#linalg.iterator_type<parallel>', ctx),
            Attribute.parse('#linalg.iterator_type<parallel>', ctx),
            Attribute.parse('#linalg.iterator_type<reduction>', ctx)
        ])

        generic_op = GenericOp(
            result_tensors=[out_type],
            inputs=[lhs, rhs],
            outputs=[empty],
            indexing_maps=affine_maps_attr,
            iterator_types=iter_types_attr
        )

        block = generic_op.regions[0].blocks.append(
            lhs.type.element_type,
            rhs.type.element_type,
            empty.type.element_type
        )

        with InsertionPoint(block):
            tile_result = d2m.tile_matmul(
                lhs.type.element_type,
                block.arguments[0],
                block.arguments[1],
                block.arguments[2]
            )
            linalg.YieldOp([tile_result])

        return generic_op.result

    def store(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Store operation for writing tensor data."""
        return d2m.store(ast_self, rhs)


@syntax("!d2m.cb")
class CircularBuffer:
    """
    Circular buffer for inter-thread communication.

    Circular buffers provide producer-consumer synchronization between
    compute and data movement threads.
    """

    def pop(ast_self) -> TensorBlock:
        """Wait for and consume data from the circular buffer."""
        return d2m.wait(d2m.ir.CBType.cast(ast_self.type).getUnderlying(), ast_self)

    def reserve(ast_self) -> TensorBlock:
        """Reserve space in the circular buffer for writing."""
        return d2m.reserve(d2m.ir.CBType.cast(ast_self.type).getUnderlying(), ast_self)


@syntax("!d2m.mem_tx")
class MemTx:
    """
    Memory transaction handle for asynchronous DMA operations.

    MemTx objects are returned by dma() calls and must be explicitly
    waited on to ensure DMA completion.
    """

    def wait(ast_self):
        """Block until the DMA operation completes."""
        return d2m.dma_wait(ast_self)


@syntax("dma")
def dma(src, dst, core=None, mcast=None) -> MemTx:
    """
    Initiate an asynchronous DMA transfer.

    Args:
        src: Source tensor or tuple of (tensor, indices)
        dst: Destination tensor or tuple of (tensor, indices)
        core: Optional tuple specifying core coordinates for multicast
        mcast: Optional tuple specifying multicast dimensions

    Returns:
        MemTx handle that must be waited on for completion
    """
    src_indices = None
    dst_indices = None
    if isinstance(src, tuple):
        src, src_indices = src
    if isinstance(dst, tuple):
        dst, dst_indices = dst
    return d2m.dma(
        src,
        _asindex(src_indices),
        dst,
        _asindex(dst_indices),
        _asindex(core),
        _asindex(mcast),
    )


@syntax("!d2m.semaphore")
class Semaphore:
    """
    Semaphore for multi-core synchronization.

    Semaphores enable coordination between cores through set, increment,
    and wait operations with optional multicast.
    """

    def set(ast_self, value, core=None, mcast=None):
        """
        Set semaphore value, optionally multicasting to other cores.

        Args:
            value: Value to set
            core: Target core coordinates for multicast
            mcast: Multicast dimensions
        """
        return d2m.semaphore_set(
            ast_self, _asindex(value), _asindex(core), _asindex(mcast)
        )

    def inc(ast_self, value, core=None, mcast=None):
        """
        Increment semaphore value, optionally multicasting to other cores.

        Args:
            value: Increment amount
            core: Target core coordinates for multicast
            mcast: Multicast dimensions
        """
        return d2m.semaphore_inc(
            ast_self, _asindex(value), _asindex(core), _asindex(mcast)
        )

    def wait(ast_self, value, reset=None):
        """
        Wait for semaphore to reach a value, optionally resetting after.

        Args:
            value: Value to wait for
            reset: Optional value to reset semaphore to after waiting
        """
        return d2m.semaphore_wait(
            ast_self, _asindex(value), reset_value=_asindex(reset)
        )
