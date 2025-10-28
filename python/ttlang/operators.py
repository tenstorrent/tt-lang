# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DSL operators for tensor operations, DMA, and memory transactions."""

from typing import List, Callable

from ttmlir.ir import *
from ttmlir.dialects import d2m, arith, linalg
from ttmlir.dialects._linalg_ops_gen import GenericOp

from ._src.d2m_ast import syntax
from pykernel._src.utils import _asindex


def _create_linalg_generic(
    lhs,
    rhs,
    output_shape: List[int],
    affine_maps: List[AffineMap],
    iterator_types: List[str],
    tile_op_builder: Callable,
) -> "TensorBlock":
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

    num_inputs = len(affine_maps) - 1
    inputs = [lhs, rhs] if num_inputs == 2 else [lhs] + [rhs] * (num_inputs - 1)

    generic_op = GenericOp(
        result_tensors=[out_type],
        inputs=inputs[:num_inputs],
        outputs=[empty],
        indexing_maps=affine_maps_attr,
        iterator_types=iter_types_attr,
    )

    block_arg_types = [inp.type.element_type for inp in inputs[:num_inputs]] + [
        empty.type.element_type
    ]
    block = generic_op.regions[0].blocks.append(*block_arg_types)

    with InsertionPoint(block):
        tile_result = tile_op_builder(lhs.type.element_type, *block.arguments)
        linalg.YieldOp([tile_result])

    return generic_op.result


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

        ctx = lhs.type.context
        rank = len(lhs.type.shape)
        identity_map = AffineMap.get_identity(rank, ctx)

        return _create_linalg_generic(
            lhs,
            rhs,
            output_shape=list(lhs.type.shape),
            affine_maps=[identity_map] * 3,
            iterator_types=["parallel"] * rank,
            tile_op_builder=lambda result_type, lhs_arg, rhs_arg, out_arg: d2m.tile_add(
                result_type, lhs_arg, rhs_arg
            ),
        )

    def __sub__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise subtraction."""
        # TODO: Generate linalg.generic with d2m.tile_sub instead of arith.subf
        return arith.subf(ast_self, rhs)

    def __mul__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise multiplication."""
        # TODO: Generate linalg.generic with d2m.tile_mul instead of arith.mulf
        return arith.mulf(ast_self, rhs)

    def __truediv__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise division."""
        # TODO: Generate linalg.generic with d2m.tile_div instead of arith.divf
        return arith.divf(ast_self, rhs)

    def __matmul__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Matrix multiplication generating linalg.generic with d2m.tile_matmul."""
        lhs = ast_self
        assert isinstance(lhs.type, RankedTensorType)

        out_shape = list(lhs.type.shape)
        out_shape[-1] = rhs.type.shape[-1]

        ctx = lhs.type.context
        matmul_maps = [
            AffineMap.get(
                3, 0, [AffineDimExpr.get(0, ctx), AffineDimExpr.get(2, ctx)], ctx
            ),
            AffineMap.get(
                3, 0, [AffineDimExpr.get(2, ctx), AffineDimExpr.get(1, ctx)], ctx
            ),
            AffineMap.get(
                3, 0, [AffineDimExpr.get(0, ctx), AffineDimExpr.get(1, ctx)], ctx
            ),
        ]

        return _create_linalg_generic(
            lhs,
            rhs,
            output_shape=out_shape,
            affine_maps=matmul_maps,
            iterator_types=["parallel", "parallel", "reduction"],
            tile_op_builder=lambda result_type, lhs_arg, rhs_arg, acc_arg: d2m.tile_matmul(
                result_type, lhs_arg, rhs_arg, acc_arg
            ),
        )

    def store(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Store operation for writing tensor data."""
        return d2m.store(ast_self, rhs)


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
