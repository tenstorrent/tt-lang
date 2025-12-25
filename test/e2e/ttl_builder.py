# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTL MLIR module builder for E2E tests.

Generates complete TTL modules with reader, compute, and writer kernels.
"""

from typing import List, Tuple

import torch
from ttmlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    FunctionType,
    RankedTensorType,
    AffineMap,
    AffineMapAttr,
    ArrayAttr,
    Attribute,
    IntegerAttr,
    IntegerType,
    IndexType,
    F32Type,
    BF16Type,
)
from ttmlir.dialects import func, tensor
from ttmlir.dialects import ttcore

import ttlang.dialects.ttl as ttl

from .config import TestConfig, BufferType


def _get_mlir_element_type(ctx: Context, dtype: torch.dtype):
    """Convert torch dtype to MLIR element type."""
    if dtype == torch.float32:
        return F32Type.get(ctx)
    elif dtype == torch.bfloat16:
        return BF16Type.get(ctx)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _get_tile_type(ctx: Context, dtype: torch.dtype):
    """Get the ttcore.tile type for the given dtype."""
    elem_type = _get_mlir_element_type(ctx, dtype)
    return ttcore.TileType.get(ctx, 32, 32, elem_type)


def _get_cb_type(
    ctx: Context, grid_shape: Tuple[int, int], dtype: torch.dtype, buffer_factor: int
):
    """Get the ttl.cb type for the given configuration."""
    tile_type = _get_tile_type(ctx, dtype)
    return ttl.ir.CBType.get(ctx, list(grid_shape), tile_type, buffer_factor)


def _get_layout_attr(ctx: Context, config: TestConfig):
    """Create the ttnn layout attribute for tensor types."""
    rows, cols = config.grid_shape
    buffer_type = "dram" if config.buffer_type == BufferType.DRAM else "l1"
    elem_type = "f32" if config.dtype == torch.float32 else "bf16"

    layout_str = (
        f"#ttnn.ttnn_layout<"
        f"(d0, d1) -> (d0, d1), <1x1>, "
        f"memref<{rows}x{cols}x!ttcore.tile<32x32, {elem_type}>, #ttnn.buffer_type<{buffer_type}>>, "
        f"<interleaved>>"
    )
    return Attribute.parse(layout_str, ctx)


def _get_tensor_type_with_layout(ctx: Context, config: TestConfig):
    """Get tensor type with TTNN layout encoding."""
    h, w = config.tensor_shape
    elem_type = _get_mlir_element_type(ctx, config.dtype)
    layout = _get_layout_attr(ctx, config)
    return RankedTensorType.get([h, w], elem_type, layout)


def _get_tile_tensor_type(ctx: Context, config: TestConfig):
    """Get tensor of tiles type (used in compute kernel)."""
    rows, cols = config.grid_shape
    tile_type = _get_tile_type(ctx, config.dtype)
    return RankedTensorType.get([rows, cols], tile_type)


def build_ttl_module(
    op_str: str,
    arity: int,
    config: TestConfig,
    torch_inputs: List[torch.Tensor],
) -> Module:
    """
    Build a complete TTL module for the given operation.

    Args:
        op_str: Operation name (e.g., "add", "exp").
        arity: Number of inputs (1 or 2).
        config: Test configuration.
        torch_inputs: Input tensors (for shape/dtype).

    Returns:
        MLIR Module containing reader, compute, and writer kernels.
    """
    ctx = Context()
    ttl.ensure_dialects_registered(ctx)

    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.create(loc)

        with InsertionPoint(module.body):
            # Build reader kernel
            _build_reader_kernel(module, arity, config, loc)

            # Build compute kernel
            _build_compute_kernel(module, op_str, arity, config, loc)

            # Build writer kernel
            _build_writer_kernel(module, config, loc)

        module.operation.verify()

    return module


def _build_reader_kernel(module, arity, config, loc):
    """Build the reader kernel (NOC thread)."""
    ctx = module.context
    tensor_type = _get_tensor_type_with_layout(ctx, config)
    input_types = [tensor_type] * arity

    func_type = FunctionType.get(input_types, [])
    reader_name = f"reader_{'binary' if arity == 2 else 'unary'}"
    reader_func = func.FuncOp(reader_name, func_type, loc=loc)

    thread_attr = Attribute.parse("#ttkernel.thread<noc>", ctx)
    reader_func.attributes["ttl.kernel_thread"] = thread_attr

    entry_block = reader_func.add_entry_block()

    with InsertionPoint(entry_block):
        cb_type = _get_cb_type(
            ctx, config.grid_shape, config.dtype, config.buffer_factor
        )

        for i in range(arity):
            cb_index = IntegerAttr.get(IndexType.get(ctx), i)
            buffer_factor = IntegerAttr.get(
                IntegerType.get_signless(64, ctx), config.buffer_factor
            )

            cb = ttl.bind_cb(
                cb_type, cb_index=cb_index, buffer_factor=buffer_factor, loc=loc
            )

            tensor_arg = entry_block.arguments[i]
            xf_type = ttl.ir.TransferHandleType.get(ctx, read=True)
            xf = ttl.copy(xf_type, tensor_arg, cb, loc=loc)
            ttl.wait(xf, loc=loc)

        func.ReturnOp([], loc=loc)


def _build_compute_kernel(module, op_str, arity, config, loc):
    """Build the compute kernel (compute thread) with tile operation."""
    ctx = module.context
    tile_tensor_type = _get_tile_tensor_type(ctx, config)
    input_types = [tile_tensor_type] * arity
    result_types = [tile_tensor_type]

    func_type = FunctionType.get(input_types, result_types)
    compute_name = f"compute_{op_str}"
    compute_func = func.FuncOp(compute_name, func_type, loc=loc)

    thread_attr = Attribute.parse("#ttkernel.thread<compute>", ctx)
    compute_func.attributes["ttl.kernel_thread"] = thread_attr

    entry_block = compute_func.add_entry_block()

    with InsertionPoint(entry_block):
        output = tensor.EmptyOp(tile_tensor_type, [], loc=loc).result

        cb_type_compute = _get_cb_type(ctx, config.grid_shape, config.dtype, 1)

        cbs = []
        for i in range(arity + 1):
            cb_index = IntegerAttr.get(IndexType.get(ctx), i)
            buffer_factor = IntegerAttr.get(IntegerType.get_signless(64, ctx), 1)
            cb = ttl.bind_cb(
                cb_type_compute,
                cb_index=cb_index,
                buffer_factor=buffer_factor,
                loc=loc,
            )
            cbs.append(cb)

        input_ready = []
        for i in range(arity):
            ready = ttl.cb_wait(tile_tensor_type, cbs[i], loc=loc)
            input_ready.append(ready)

        output_cb_idx = arity
        output_attached = ttl.attach_cb(
            tile_tensor_type, output, cbs[output_cb_idx], loc=loc
        )

        result = _build_compute_region(
            ctx,
            op_str,
            arity,
            input_ready,
            output_attached,
            cbs[output_cb_idx],
            config,
            loc,
        )

        func.ReturnOp([result], loc=loc)


def _build_compute_region(ctx, op_str, arity, inputs, output, output_cb, config, loc):
    """Build the ttl.compute region with the tile operation."""
    tile_tensor_type = _get_tile_tensor_type(ctx, config)
    tile_type = _get_tile_type(ctx, config.dtype)

    rows, cols = config.grid_shape
    identity_map = AffineMap.get_identity(2, context=ctx)

    num_operands = arity + 1
    indexing_maps = ArrayAttr.get(
        [AffineMapAttr.get(identity_map) for _ in range(num_operands)], ctx
    )

    iterator_types = ArrayAttr.get(
        [Attribute.parse("#linalg.iterator_type<parallel>", ctx) for _ in range(2)],
        ctx,
    )

    compute_op = ttl.ComputeOp(
        result=[tile_tensor_type],
        inputs=inputs,
        outputs=[output],
        indexing_maps=indexing_maps,
        iterator_types=iterator_types,
        loc=loc,
    )

    body_arg_types = [tile_type] * (arity + 1)
    body_block = compute_op.body.blocks.append(*body_arg_types)

    with InsertionPoint(body_block):
        input_tiles = [body_block.arguments[i] for i in range(arity)]

        # Apply tile operation
        tile_op_name = f"tile_{op_str}"
        if not hasattr(ttl, tile_op_name):
            raise ValueError(f"Unknown tile op: {tile_op_name}")

        tile_op_func = getattr(ttl, tile_op_name)

        if arity == 1:
            result_tile = tile_op_func(input_tiles[0], loc=loc)
        elif arity == 2:
            result_tile = tile_op_func(input_tiles[0], input_tiles[1], loc=loc)
        else:
            raise ValueError(f"Unsupported arity: {arity}")

        result_view = ttl.cb_reserve(tile_tensor_type, output_cb, loc=loc)
        ttl.store(result_tile, result_view, loc=loc)
        ttl.cb_push(output_cb, loc=loc)

        ttl.YieldOp([result_tile], loc=loc)

    return compute_op.results[0]


def _build_writer_kernel(module, config, loc):
    """Build the writer kernel (NOC thread)."""
    ctx = module.context
    tensor_type = _get_tensor_type_with_layout(ctx, config)
    func_type = FunctionType.get([tensor_type], [])

    writer_name = "writer_unary"
    writer_func = func.FuncOp(writer_name, func_type, loc=loc)

    thread_attr = Attribute.parse("#ttkernel.thread<noc>", ctx)
    writer_func.attributes["ttl.kernel_thread"] = thread_attr

    entry_block = writer_func.add_entry_block()

    with InsertionPoint(entry_block):
        # Output CB is at index = arity (but we don't know arity here, assume index from compute)
        # For simplicity, use a fixed index
        output_cb_index = 2  # Assuming binary op (0, 1 for inputs, 2 for output)

        cb_type = _get_cb_type(
            ctx, config.grid_shape, config.dtype, config.buffer_factor
        )

        cb_index_attr = IntegerAttr.get(IndexType.get(ctx), output_cb_index)
        buffer_factor_attr = IntegerAttr.get(
            IntegerType.get_signless(64, ctx), config.buffer_factor
        )

        cb = ttl.bind_cb(
            cb_type,
            cb_index=cb_index_attr,
            buffer_factor=buffer_factor_attr,
            loc=loc,
        )

        tile_tensor_type = _get_tile_tensor_type(ctx, config)
        view = ttl.cb_wait(tile_tensor_type, cb, loc=loc)

        out_tensor = entry_block.arguments[0]
        xf_type = ttl.ir.TransferHandleType.get(ctx, read=False)
        xf = ttl.copy(xf_type, cb, out_tensor, loc=loc)

        ttl.wait(xf, loc=loc)

        func.ReturnOp([], loc=loc)
