# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTL MLIR module builder for middle-end tests.

Generates complete TTL modules with reader, compute, and writer kernels
for testing the TTL dialect compilation pipeline.
"""

from typing import List, Tuple
from dataclasses import dataclass

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

from .op_specs import AnyOpSpec, get_ttl_tile_op_func
from .config_specs import TestConfig, BufferType

import torch


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
    # CB type: !ttl.cb<[rows, cols], element_type, buffer_factor>
    tile_type = _get_tile_type(ctx, dtype)
    return ttl.ir.CBType.get(ctx, list(grid_shape), tile_type, buffer_factor)


def _get_layout_attr(ctx: Context, config: TestConfig):
    """Create the ttnn layout attribute for tensor types."""
    # This creates the #ttnn.ttnn_layout attribute used in compute_with_data_movement.mlir.
    # For now, we use a simplified interleaved DRAM layout.
    # TODO: Support other memory layouts based on config.memory_layout.

    # Parse the layout attribute string.
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


@dataclass
class TTLModuleBuilder:
    """Builder for constructing TTL MLIR modules for testing."""

    op: AnyOpSpec
    config: TestConfig
    ctx: Context = None
    module: Module = None

    def build(self) -> Module:
        """Build the complete TTL module with reader, compute, and writer."""
        self.ctx = Context()

        # Ensure dialects are registered.
        ttl.ensure_dialects_registered(self.ctx)

        loc = Location.unknown(self.ctx)

        with self.ctx, loc:
            self.module = Module.create(loc)

            with InsertionPoint(self.module.body):
                # Build the three kernel functions.
                self._build_reader_kernel()
                self._build_compute_kernel()
                self._build_writer_kernel()

            # Verify the module.
            self.module.operation.verify()

        return self.module

    def _build_reader_kernel(self):
        """Build the reader kernel (NOC thread) that reads tensors to CBs."""
        loc = Location.unknown(self.ctx)

        # Function signature depends on arity.
        tensor_type = _get_tensor_type_with_layout(self.ctx, self.config)
        input_types = [tensor_type] * self.op.arity

        func_type = FunctionType.get(input_types, [])

        # Create function with ttl.kernel_thread = #ttkernel.thread<noc>.
        func_name = f"reader_{self.op.reader_type}"
        reader_func = func.FuncOp(func_name, func_type, loc=loc)

        # Add kernel thread attribute.
        thread_attr = Attribute.parse("#ttkernel.thread<noc>", self.ctx)
        reader_func.attributes["ttl.kernel_thread"] = thread_attr

        # Create entry block with arguments.
        entry_block = reader_func.add_entry_block()

        with InsertionPoint(entry_block):
            # Bind CBs for each input.
            cb_type = _get_cb_type(
                self.ctx,
                self.config.grid_shape,
                self.config.dtype,
                self.config.buffer_factor,
            )

            for i in range(self.op.arity):
                # Create bind_cb op.
                cb_index = IntegerAttr.get(IndexType.get(self.ctx), i)
                buffer_factor = IntegerAttr.get(
                    IntegerType.get_signless(64, self.ctx), self.config.buffer_factor
                )

                cb = ttl.bind_cb(
                    cb_type, cb_index=cb_index, buffer_factor=buffer_factor, loc=loc
                )

                # Copy tensor to CB.
                tensor_arg = entry_block.arguments[i]
                xf_type = ttl.ir.TransferHandleType.get(self.ctx, read=True)
                xf = ttl.copy(xf_type, tensor_arg, cb, loc=loc)

                # Wait for transfer.
                ttl.wait(xf, loc=loc)

            # Return.
            func.ReturnOp([], loc=loc)

    def _build_compute_kernel(self):
        """Build the compute kernel (compute thread) with tile operations."""
        loc = Location.unknown(self.ctx)

        # Function takes tile tensors and returns tile tensor.
        tile_tensor_type = _get_tile_tensor_type(self.ctx, self.config)
        input_types = [tile_tensor_type] * self.op.arity
        result_types = [tile_tensor_type]

        func_type = FunctionType.get(input_types, result_types)

        func_name = f"compute_{self.op.name}"
        compute_func = func.FuncOp(func_name, func_type, loc=loc)

        # Add kernel thread attribute.
        thread_attr = Attribute.parse("#ttkernel.thread<compute>", self.ctx)
        compute_func.attributes["ttl.kernel_thread"] = thread_attr

        entry_block = compute_func.add_entry_block()

        with InsertionPoint(entry_block):
            # Create output tensor.
            output = tensor.EmptyOp(tile_tensor_type, [], loc=loc).result

            # Bind CBs.
            cb_type_compute = _get_cb_type(
                self.ctx,
                self.config.grid_shape,
                self.config.dtype,
                1,  # Compute uses buffer_factor=1.
            )

            cbs = []
            for i in range(self.op.arity + 1):  # +1 for output CB.
                cb_index = IntegerAttr.get(IndexType.get(self.ctx), i)
                buffer_factor = IntegerAttr.get(
                    IntegerType.get_signless(64, self.ctx), 1
                )
                cb = ttl.bind_cb(
                    cb_type_compute,
                    cb_index=cb_index,
                    buffer_factor=buffer_factor,
                    loc=loc,
                )
                cbs.append(cb)

            # Wait for input CBs.
            input_ready = []
            for i in range(self.op.arity):
                ready = ttl.cb_wait(tile_tensor_type, cbs[i], loc=loc)
                input_ready.append(ready)

            # Attach output to its CB.
            output_cb_idx = self.op.arity
            output_attached = ttl.attach_cb(
                tile_tensor_type, output, cbs[output_cb_idx], loc=loc
            )

            # Build ttl.compute region with tile operation.
            result = self._build_compute_region(
                input_ready, output_attached, cbs[output_cb_idx]
            )

            # Return result.
            func.ReturnOp([result], loc=loc)

    def _build_compute_region(self, inputs: List, output: any, output_cb: any):
        """Build the ttl.compute region with the tile operation."""
        loc = Location.unknown(self.ctx)

        tile_tensor_type = _get_tile_tensor_type(self.ctx, self.config)
        tile_type = _get_tile_type(self.ctx, self.config.dtype)

        # Create identity affine maps for indexing.
        rows, cols = self.config.grid_shape
        identity_map = AffineMap.get_identity(2, context=self.ctx)

        # Build indexing_maps and iterator_types attributes.
        num_operands = self.op.arity + 1  # inputs + output
        indexing_maps = ArrayAttr.get(
            [AffineMapAttr.get(identity_map) for _ in range(num_operands)], self.ctx
        )

        iterator_types = ArrayAttr.get(
            [
                Attribute.parse("#linalg.iterator_type<parallel>", self.ctx)
                for _ in range(2)
            ],
            self.ctx,
        )

        # Create ttl.compute op.
        compute_op = ttl.ComputeOp(
            result=[tile_tensor_type],
            inputs=inputs,
            outputs=[output],
            indexing_maps=indexing_maps,
            iterator_types=iterator_types,
            loc=loc,
        )

        # Build the body block with tile arguments.
        body_arg_types = [tile_type] * (self.op.arity + 1)  # input tiles + output tile
        body_block = compute_op.body.blocks.append(*body_arg_types)

        with InsertionPoint(body_block):
            # Get input tile arguments.
            input_tiles = [body_block.arguments[i] for i in range(self.op.arity)]

            # Apply all tile operations in sequence.
            result_tile = self._apply_tile_ops(input_tiles)

            # Reserve output CB, store result, push.
            result_view = ttl.cb_reserve(tile_tensor_type, output_cb, loc=loc)
            ttl.store(result_tile, result_view, loc=loc)
            ttl.cb_push(output_cb, loc=loc)

            # Yield the result tile.
            ttl.YieldOp([result_tile], loc=loc)

        return compute_op.results[0]

    def _apply_tile_ops(self, input_tiles: list):
        """
        Apply all tile operations in sequence.

        For single ops: applies the single operation.
        For fused ops: chains operations, output of op[i] becomes input to op[i+1].
        """
        loc = Location.unknown(self.ctx)

        # Get the list of ops to apply (works for both ComputeOpSpec and FusedOpSpec).
        ttl_ops = self.op.ttl_ops

        # First operation consumes the inputs.
        first_op = ttl_ops[0]
        first_op_func = get_ttl_tile_op_func(ttl, first_op)

        if self.op.arity == 1:
            current_tile = first_op_func(input_tiles[0], loc=loc)
        elif self.op.arity == 2:
            current_tile = first_op_func(input_tiles[0], input_tiles[1], loc=loc)
        else:
            raise ValueError(f"Unsupported arity: {self.op.arity}")

        # Subsequent operations are unary, taking output of previous op.
        for ttl_op_name in ttl_ops[1:]:
            op_func = get_ttl_tile_op_func(ttl, ttl_op_name)
            current_tile = op_func(current_tile, loc=loc)

        return current_tile

    def _build_writer_kernel(self):
        """Build the writer kernel (NOC thread) that writes CB to output tensor."""
        loc = Location.unknown(self.ctx)

        # Function takes output tensor.
        tensor_type = _get_tensor_type_with_layout(self.ctx, self.config)
        func_type = FunctionType.get([tensor_type], [])

        func_name = "writer_unary"
        writer_func = func.FuncOp(func_name, func_type, loc=loc)

        # Add kernel thread attribute.
        thread_attr = Attribute.parse("#ttkernel.thread<noc>", self.ctx)
        writer_func.attributes["ttl.kernel_thread"] = thread_attr

        entry_block = writer_func.add_entry_block()

        with InsertionPoint(entry_block):
            # Output CB is at index = arity (after input CBs).
            output_cb_index = self.op.arity

            # Create a simpler CB type for the writer (just f32/bf16, not tile).
            elem_type = _get_mlir_element_type(self.ctx, self.config.dtype)
            rows, cols = self.config.grid_shape

            # For writer, we use the scalar element type in CB.
            cb_type = _get_cb_type(
                self.ctx,
                self.config.grid_shape,
                self.config.dtype,
                self.config.buffer_factor,
            )

            cb_index_attr = IntegerAttr.get(IndexType.get(self.ctx), output_cb_index)
            buffer_factor_attr = IntegerAttr.get(
                IntegerType.get_signless(64, self.ctx), self.config.buffer_factor
            )

            cb = ttl.bind_cb(
                cb_type,
                cb_index=cb_index_attr,
                buffer_factor=buffer_factor_attr,
                loc=loc,
            )

            # Wait for data in CB.
            tile_tensor_type = _get_tile_tensor_type(self.ctx, self.config)
            view = ttl.cb_wait(tile_tensor_type, cb, loc=loc)

            # Copy CB to output tensor.
            out_tensor = entry_block.arguments[0]
            xf_type = ttl.ir.TransferHandleType.get(self.ctx, read=False)
            xf = ttl.copy(xf_type, cb, out_tensor, loc=loc)

            # Wait for transfer.
            ttl.wait(xf, loc=loc)

            # Return.
            func.ReturnOp([], loc=loc)


def build_ttl_module(op: AnyOpSpec, config: TestConfig) -> Module:
    """
    Build a complete TTL module for the given operation and configuration.

    Args:
        op: The compute operation to test.
        config: The test configuration.

    Returns:
        MLIR Module containing reader, compute, and writer kernels.
    """
    builder = TTLModuleBuilder(op=op, config=config)
    return builder.build()
