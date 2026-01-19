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


def _generate_binary_compute_mlir(
    op_str: str,
    rows: int,
    cols: int,
    dtype_str: str,
    buffer_factor: int,
    num_iterations: int,
) -> str:
    """
    Generate compute thread MLIR for binary operations.

    Uses high-level tensor ops (ttl.add, ttl.mul) with proper CB lifecycle:
    cb_wait -> compute -> cb_reserve -> attach_cb -> cb_push -> cb_pop (inputs).
    """
    cb_type = (
        f"!ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>"
    )
    tensor_type = f"tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>"

    # Build loop body or single iteration.
    if num_iterations > 1:
        loop_start = f"""
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c{num_iterations} = arith.constant {num_iterations} : index
  scf.for %iter = %c0 to %c{num_iterations} step %c1 {{"""
        loop_end = """
  }"""
        indent = "    "
    else:
        loop_start = ""
        loop_end = ""
        indent = "  "

    return f"""
// Compute thread for {op_str} binary operation.
// Uses high-level tensor ops with proper CB lifecycle (wait/push/pop).
func.func @compute_{op_str}() attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {buffer_factor}}} : {cb_type}
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {buffer_factor}}} : {cb_type}
  %cb_out = ttl.bind_cb {{cb_index = 2, buffer_factor = {buffer_factor}}} : {cb_type}
{loop_start}
{indent}// Wait for input data from reader.
{indent}%a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> {tensor_type}
{indent}%a_attached = ttl.attach_cb %a, %cb0 : ({tensor_type}, {cb_type}) -> {tensor_type}
{indent}%b = ttl.cb_wait %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> {tensor_type}
{indent}%b_attached = ttl.attach_cb %b, %cb1 : ({tensor_type}, {cb_type}) -> {tensor_type}

{indent}// Reserve output CB.
{indent}%out_reserved = ttl.cb_reserve %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> {tensor_type}
{indent}%out_attached = ttl.attach_cb %out_reserved, %cb_out : ({tensor_type}, {cb_type}) -> {tensor_type}

{indent}// Compute: apply {op_str} operation.
{indent}%result = ttl.{op_str} %a_attached, %b_attached : {tensor_type}, {tensor_type} -> {tensor_type}
{indent}%result_attached = ttl.attach_cb %result, %cb_out : ({tensor_type}, {cb_type}) -> {tensor_type}

{indent}// Push output, pop inputs.
{indent}ttl.cb_push %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
{indent}ttl.cb_pop %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
{indent}ttl.cb_pop %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
{loop_end}
  return
}}
"""


def _generate_unary_compute_mlir(
    op_str: str,
    rows: int,
    cols: int,
    dtype_str: str,
    buffer_factor: int,
    num_iterations: int,
) -> str:
    """
    Generate compute thread MLIR for unary operations.

    Uses high-level tensor ops (ttl.exp, ttl.sqrt) with proper CB lifecycle.
    """
    cb_type = (
        f"!ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>"
    )
    tensor_type = f"tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>>"

    # Build loop body or single iteration.
    if num_iterations > 1:
        loop_start = f"""
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c{num_iterations} = arith.constant {num_iterations} : index
  scf.for %iter = %c0 to %c{num_iterations} step %c1 {{"""
        loop_end = """
  }"""
        indent = "    "
    else:
        loop_start = ""
        loop_end = ""
        indent = "  "

    return f"""
// Compute thread for {op_str} unary operation.
// Uses high-level tensor ops with proper CB lifecycle (wait/push/pop).
func.func @compute_{op_str}() attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {buffer_factor}}} : {cb_type}
  %cb_out = ttl.bind_cb {{cb_index = 1, buffer_factor = {buffer_factor}}} : {cb_type}
{loop_start}
{indent}// Wait for input data from reader.
{indent}%a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> {tensor_type}
{indent}%a_attached = ttl.attach_cb %a, %cb0 : ({tensor_type}, {cb_type}) -> {tensor_type}

{indent}// Reserve output CB.
{indent}%out_reserved = ttl.cb_reserve %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> {tensor_type}
{indent}%out_attached = ttl.attach_cb %out_reserved, %cb_out : ({tensor_type}, {cb_type}) -> {tensor_type}

{indent}// Compute: apply {op_str} operation.
{indent}%result = ttl.{op_str} %a_attached : {tensor_type} -> {tensor_type}
{indent}%result_attached = ttl.attach_cb %result, %cb_out : ({tensor_type}, {cb_type}) -> {tensor_type}

{indent}// Push output, pop input.
{indent}ttl.cb_push %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
{indent}ttl.cb_pop %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
{loop_end}
  return
}}
"""


def build_e2e_module_mlir(
    op_str: str,
    arity: int,
    config: E2EConfig,
) -> str:
    """
    Build complete E2E MLIR module string with reader, compute, and writer threads.

    This generates a full module suitable for device execution. The generated MLIR
    uses high-level tensor operations (ttl.add, ttl.exp, etc.) that get lowered
    through the compiler pipeline.

    For multi-tile grids, generates scf.for loops to iterate over tiles.

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

    # Total tiles and iterations needed.
    total_tiles = config.num_tiles
    # For now, process 1 tile per iteration (CB shape 1x1).
    # TODO: Support blocking (process multiple tiles per iteration).
    num_iterations = total_tiles

    # Generate reader with loop support.
    if arity == 2:
        reader_mlir = generate_binary_reader_mlir(
            grid_shape, dtype, buffer_factor, num_iterations
        )
    else:
        reader_mlir = generate_unary_reader_mlir(
            grid_shape, dtype, buffer_factor, num_iterations
        )

    # Generate compute with high-level ops (not ttl.compute regions).
    # Use 1x1 CB shape for simplicity - one tile per iteration.
    rows, cols = 1, 1
    dtype_str = torch_dtype_to_mlir_str(dtype)

    if arity == 2:
        compute_mlir = _generate_binary_compute_mlir(
            op_str, rows, cols, dtype_str, buffer_factor, num_iterations
        )
        output_cb_index = 2
    else:
        compute_mlir = _generate_unary_compute_mlir(
            op_str, rows, cols, dtype_str, buffer_factor, num_iterations
        )
        output_cb_index = 1

    # Generate writer with loop support.
    writer_mlir = generate_writer_mlir(
        grid_shape, dtype, buffer_factor, output_cb_index, num_iterations
    )

    # Combine into full module.
    return f"""// Auto-generated E2E MLIR module for {op_str} operation.
// Arity: {arity}, Grid: {grid_shape}, Dtype: {dtype}, Iterations: {num_iterations}

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
