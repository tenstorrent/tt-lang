# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DSL operators for tensor operations, DMA, and memory transactions."""

from __future__ import annotations

from typing import List, Callable, Optional, Tuple, Union

from ttmlir.ir import *
from ttmlir.dialects import d2m, arith, linalg

from ._src.d2m_ast import syntax
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
    inp,
    output_shape: List[int],
    affine_maps: List[AffineMap],
    iterator_types: List[str],
    tile_op_builder: Callable,
) -> "TensorBlock":
    """
    Create a linalg.generic operation for unary D2M tile operations.

    Args:
        inp: Input operand
        output_shape: Shape of the output tensor
        affine_maps: List of AffineMap objects [input_map, output_map]
        iterator_types: List of iterator type strings ("parallel" or "reduction")
        tile_op_builder: Function that takes (result_type, input_arg, output_arg) and creates tile op

    Returns:
        Result of the linalg.generic operation
    """
    ctx = inp.type.context

    out_type = RankedTensorType.get(
        output_shape, inp.type.element_type, inp.type.encoding
    )
    empty = d2m.empty(out_type)

    affine_maps_attr = ArrayAttr.get([AffineMapAttr.get(m) for m in affine_maps])

    iter_types_attr = ArrayAttr.get(
        [Attribute.parse(f"#linalg.iterator_type<{it}>", ctx) for it in iterator_types]
    )

    generic_op = GenericOp(
        result_tensors=[out_type],
        inputs=[inp],
        outputs=[empty],
        indexing_maps=affine_maps_attr,
        iterator_types=iter_types_attr,
    )

    block_arg_types = [inp.type.element_type, empty.type.element_type]
    block = generic_op.regions[0].blocks.append(*block_arg_types)

    with InsertionPoint(block):
        tile_result = tile_op_builder(inp.type.element_type, *block.arguments)
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

    def __add__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """
        Element-wise addition generating linalg.generic with d2m.tile_add.

        Args:
            rhs: Right operand tensor. Must have the same shape as self.

        Returns:
            Result tensor with the same shape as inputs.
        """
        lhs = ast_self
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
            tile_op_builder=lambda result_type, lhs_arg, rhs_arg, out_arg: d2m.tile_add(
                result_type, lhs_arg, rhs_arg
            ),
        )

    def __sub__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise subtraction."""
        # TODO(#10): Generate linalg.generic with d2m.tile_sub instead of arith.subf
        return arith.subf(ast_self, rhs)

    def __mul__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise multiplication."""
        # TODO(#10): Generate linalg.generic with d2m.tile_mul instead of arith.mulf
        return arith.mulf(ast_self, rhs)

    def __truediv__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """Element-wise division."""
        # TODO(#10): Generate linalg.generic with d2m.tile_div instead of arith.divf
        return arith.divf(ast_self, rhs)

    def __matmul__(ast_self: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
        """
        Matrix multiplication generating linalg.generic with d2m.tile_matmul.

        Args:
            rhs: Right operand tensor. Shape must be compatible for matrix multiplication.
                 For 2D: (M, K) @ (K, N) -> (M, N)
                 Inner dimensions must match: lhs.shape[-1] == rhs.shape[-2]

        Returns:
            Result tensor with shape (..., M, N) where M = lhs.shape[-2], N = rhs.shape[-1]

        Raises:
            TypeError: If operands are not ranked tensors
            ValueError: If ranks are < 2 or inner dimensions don't match
        """
        lhs = ast_self

        if not isinstance(lhs.type, RankedTensorType):
            raise TypeError("lhs must be a ranked tensor")
        if not isinstance(rhs.type, RankedTensorType):
            raise TypeError("rhs must be a ranked tensor")

        # Validate ranks and shapes for matmul
        lhs_rank = len(lhs.type.shape)
        rhs_rank = len(rhs.type.shape)
        if lhs_rank < 2 or rhs_rank < 2:
            raise ValueError(
                f"matmul requires at least 2D tensors, got lhs rank {lhs_rank}, rhs rank {rhs_rank}"
            )
        if lhs.type.shape[-1] != rhs.type.shape[-2]:
            raise ValueError(
                f"matmul inner dimensions must match, got lhs[-1]={lhs.type.shape[-1]}, "
                f"rhs[-2]={rhs.type.shape[-2]}"
            )

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

    def transpose(ast_self: "TensorBlock") -> "TensorBlock":
        """Transpose operation generating linalg.generic with d2m.tile_transpose."""
        inp = ast_self
        assert isinstance(inp.type, RankedTensorType)

        ctx = inp.type.context
        rank = len(inp.type.shape)

        # Output shape is transposed
        out_shape = list(reversed(inp.type.shape))

        # Identity maps for input and output
        identity_map = AffineMap.get_identity(rank, ctx)

        return _create_linalg_generic_unary(
            inp,
            output_shape=out_shape,
            affine_maps=[identity_map, identity_map],
            iterator_types=["parallel"] * rank,
            tile_op_builder=lambda result_type, inp_arg, out_arg: d2m.tile_transpose(
                result_type, inp_arg
            ),
        )

    def exp(ast_self: "TensorBlock") -> "TensorBlock":
        """Exponential operation generating linalg.generic with d2m.tile_exp."""
        inp = ast_self
        assert isinstance(inp.type, RankedTensorType)

        ctx = inp.type.context
        rank = len(inp.type.shape)

        # Output shape is same as input
        out_shape = list(inp.type.shape)

        # Identity maps for input and output
        identity_map = AffineMap.get_identity(rank, ctx)

        return _create_linalg_generic_unary(
            inp,
            output_shape=out_shape,
            affine_maps=[identity_map, identity_map],
            iterator_types=["parallel"] * rank,
            tile_op_builder=lambda result_type, inp_arg, out_arg: d2m.tile_exp(
                result_type, inp_arg
            ),
        )


@syntax("!d2m.mem_tx")
class MemTx:
    """
    Memory transaction handle for asynchronous DMA operations.

    MemTx objects are returned by dma() calls and must be explicitly
    waited on to ensure DMA completion.
    """

    def wait(ast_self: MemTx):
        """Block until the DMA operation completes."""
        return d2m.dma_wait(ast_self)


@syntax("dma")
def dma(
    src: IndexedTensor,
    dst: IndexedTensor,
    core: Optional[CoreCoordinate] = None,
    mcast: Optional[CoreCoordinate] = None,
) -> MemTx:
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
