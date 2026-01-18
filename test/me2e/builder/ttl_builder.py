# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTL MLIR module builder for E2E tests.

Generates TTL modules using high-level ops (ttl.add, ttl.exp, etc.) that
can be lowered through the compiler pipeline to executable kernels.

Two modes:
1. Compute-only: Just the compute function (for unit testing compiler passes).
2. Full E2E: Reader, compute, and writer functions (for device execution).
"""

from typing import List

import torch
from ttmlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    FunctionType,
    RankedTensorType,
    Type as MLIRType,
)
from ttmlir.dialects import func
from ttmlir.dialects import ttcore

import ttl.dialects.ttl as ttl

from ..config import E2EConfig
from .dm_threads import (
    generate_binary_reader_mlir,
    generate_unary_reader_mlir,
    generate_writer_mlir,
    generate_layout_attrs,
)
from .dtype_utils import torch_dtype_to_mlir_str, torch_dtype_to_ttcore_datatype


def _get_tile_type(ctx: Context, dtype: torch.dtype):
    """Get the ttcore.tile type for the given dtype."""
    dtype_int = torch_dtype_to_ttcore_datatype(dtype)
    return ttcore.ir.TileType.get(ctx, 32, 32, dtype_int)


def _get_tile_tensor_type(ctx: Context, config: E2EConfig):
    """Get tensor of tiles type."""
    rows, cols = config.grid_shape
    tile_type = _get_tile_type(ctx, config.dtype)
    return RankedTensorType.get([rows, cols], tile_type)


def _get_cb_type_str(config: E2EConfig) -> str:
    """Get the !ttl.cb type as a string for parsing."""
    rows, cols = config.grid_shape
    dtype_str = torch_dtype_to_mlir_str(config.dtype)
    return f"!ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {config.buffer_factor}>"


def build_ttl_module(
    op_str: str,
    arity: int,
    config: E2EConfig,
    torch_inputs: List[torch.Tensor],
) -> Module:
    """
    Build a compute-only TTL module for the given operation.

    This generates just the compute function without reader/writer threads.

    Args:
        op_str: Operation name (e.g., "add", "exp").
        arity: Number of inputs (1 or 2).
        config: Test configuration.
        torch_inputs: Input tensors (for shape/dtype).

    Returns:
        MLIR Module containing the TTL compute function.
    """
    ctx = Context()
    ttl.ensure_dialects_registered(ctx)

    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.create(loc)
        tile_tensor_type = _get_tile_tensor_type(ctx, config)

        with InsertionPoint(module.body):
            # Create function signature.
            input_types = [tile_tensor_type] * arity
            result_types = [tile_tensor_type]
            func_type = FunctionType.get(input_types, result_types)
            func_name = f"compute_{op_str}"
            compute_func = func.FuncOp(func_name, func_type, loc=loc)

            entry_block = compute_func.add_entry_block()

            with InsertionPoint(entry_block):
                # Parse CB type string for bind_cb result type.
                cb_type_str = _get_cb_type_str(config)
                cb_type = MLIRType.parse(cb_type_str, ctx)

                # Create CBs and attach to inputs.
                attached_inputs = []
                for i in range(arity):
                    cb = ttl.bind_cb(
                        cb_type,
                        cb_index=i,
                        buffer_factor=config.buffer_factor,
                        loc=loc,
                    )
                    attached = ttl.attach_cb(
                        tile_tensor_type,
                        entry_block.arguments[i],
                        cb,
                        loc=loc,
                    )
                    attached_inputs.append(attached)

                # Create output CB (required for convert-ttl-to-compute pass).
                output_cb = ttl.bind_cb(
                    cb_type,
                    cb_index=arity,  # Next index after inputs.
                    buffer_factor=config.buffer_factor,
                    loc=loc,
                )

                # Apply the operation.
                op_func = getattr(ttl, op_str, None)
                if op_func is None:
                    raise ValueError(f"Unknown TTL op: ttl.{op_str}")

                if arity == 1:
                    result = op_func(
                        tile_tensor_type,
                        attached_inputs[0],
                        loc=loc,
                    )
                elif arity == 2:
                    result = op_func(
                        tile_tensor_type,
                        attached_inputs[0],
                        attached_inputs[1],
                        loc=loc,
                    )
                else:
                    raise ValueError(f"Unsupported arity: {arity}")

                func.ReturnOp([result], loc=loc)

        module.operation.verify()

    return module


def build_e2e_module_mlir(
    op_str: str,
    arity: int,
    config: E2EConfig,
) -> str:
    """
    Build complete E2E MLIR module string with reader, compute, and writer threads.

    This generates a full module suitable for device execution.

    Args:
        op_str: Operation name (e.g., "add", "exp").
        arity: Number of inputs (1 or 2).
        config: Test configuration.

    Returns:
        MLIR source string with all three thread functions.
    """
    grid_shape = config.grid_shape
    dtype = config.dtype
    buffer_factor = config.buffer_factor

    # Generate layout attributes.
    layout_attrs = generate_layout_attrs(grid_shape, dtype)

    # Generate reader.
    if arity == 2:
        reader_mlir = generate_binary_reader_mlir(grid_shape, dtype, buffer_factor)
    else:
        reader_mlir = generate_unary_reader_mlir(grid_shape, dtype, buffer_factor)

    # Generate compute (using string template for now since we need kernel_thread attr).
    rows, cols = grid_shape
    dtype_str = torch_dtype_to_mlir_str(dtype)

    # Get the tile op name - map from high-level op to tile op.
    tile_op_name = f"tile_{op_str}"

    if arity == 2:
        compute_mlir = f"""
// Compute thread for {op_str} binary operation.
func.func @compute_{op_str}(%arg0: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>,
                            %arg1: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>)
    -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
    attributes {{ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %output = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>

  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {buffer_factor}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {buffer_factor}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
  %cb_out = ttl.bind_cb {{cb_index = 2, buffer_factor = {buffer_factor}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>

  // Wait for data from reader.
  %a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
  %b = ttl.cb_wait %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
  %output_cb = ttl.attach_cb %output, %cb_out : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>

  // Compute using ttl.compute with tile op.
  %result = ttl.compute
      ins(%a, %b : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>,
                   tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>)
      outs(%output_cb : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>)
      {{indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]}} {{
  ^bb0(%a_tile: !ttcore.tile<32x32, {dtype_str}>,
       %b_tile: !ttcore.tile<32x32, {dtype_str}>,
       %out_tile: !ttcore.tile<32x32, {dtype_str}>):
    %sum = ttl.{tile_op_name} %a_tile, %b_tile : !ttcore.tile<32x32, {dtype_str}>
    %result_view = ttl.cb_reserve %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
    ttl.store %sum, %result_view : !ttcore.tile<32x32, {dtype_str}>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
    ttl.cb_push %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
    ttl.yield %sum : !ttcore.tile<32x32, {dtype_str}>
  }} -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>

  func.return %result : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
}}
"""
        output_cb_index = 2
    else:
        compute_mlir = f"""
// Compute thread for {op_str} unary operation.
func.func @compute_{op_str}(%arg0: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>)
    -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
    attributes {{ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %output = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>

  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {buffer_factor}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
  %cb_out = ttl.bind_cb {{cb_index = 1, buffer_factor = {buffer_factor}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>

  // Wait for data from reader.
  %a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
  %output_cb = ttl.attach_cb %output, %cb_out : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>

  // Compute using ttl.compute with tile op.
  %result = ttl.compute
      ins(%a : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>)
      outs(%output_cb : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>)
      {{indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]}} {{
  ^bb0(%a_tile: !ttcore.tile<32x32, {dtype_str}>,
       %out_tile: !ttcore.tile<32x32, {dtype_str}>):
    %res = ttl.{tile_op_name} %a_tile : !ttcore.tile<32x32, {dtype_str}>
    %result_view = ttl.cb_reserve %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
    ttl.store %res, %result_view : !ttcore.tile<32x32, {dtype_str}>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
    ttl.cb_push %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
    ttl.yield %res : !ttcore.tile<32x32, {dtype_str}>
  }} -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>

  func.return %result : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>
}}
"""
        output_cb_index = 1

    # Generate writer.
    writer_mlir = generate_writer_mlir(
        grid_shape, dtype, buffer_factor, output_cb_index
    )

    # Combine into full module.
    return f"""// Auto-generated E2E MLIR module for {op_str} operation.
// Arity: {arity}, Grid: {grid_shape}, Dtype: {dtype}

{layout_attrs}

module {{
{reader_mlir}
{compute_mlir}
{writer_mlir}
}}
"""


def build_e2e_module(
    op_str: str,
    arity: int,
    config: E2EConfig,
) -> Module:
    """
    Build complete E2E TTL module with reader, compute, and writer threads.

    This generates a full module suitable for device execution.

    Args:
        op_str: Operation name (e.g., "add", "exp").
        arity: Number of inputs (1 or 2).
        config: Test configuration.

    Returns:
        MLIR Module with all three thread functions.
    """
    mlir_str = build_e2e_module_mlir(op_str, arity, config)
    return build_ttl_module_from_mlir(mlir_str)


def build_ttl_module_from_mlir(mlir_str: str) -> Module:
    """
    Build a TTL module from an MLIR string.

    This allows tests to provide custom MLIR for edge cases that are
    difficult to generate programmatically.

    Args:
        mlir_str: The MLIR source code as a string.

    Returns:
        MLIR Module parsed from the string.
    """
    ctx = Context()
    ttl.ensure_dialects_registered(ctx)

    with ctx:
        module = Module.parse(mlir_str, ctx)
        module.operation.verify()

    return module
