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
    IndexType,
    IntegerType,
    IntegerAttr,
    ArrayAttr,
    StringAttr,
    Attribute,
)
from ttmlir.dialects import func, scf, arith, ttkernel
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


def _build_compute_func(
    module: Module,
    ctx: Context,
    loc: Location,
    op_str: str,
    arity: int,
    rows: int,
    cols: int,
    dtype: torch.dtype,
    buffer_factor: int,
    num_iterations: int,
) -> None:
    """
    Build compute thread function using the Python builder API.

    Creates a func.func with CB operations and optional scf.for loop for
    multi-tile iteration. Uses high-level tensor ops (ttl.add, ttl.exp, etc.)
    with proper CB lifecycle: cb_wait -> compute -> cb_reserve -> cb_push -> cb_pop.

    Args:
        module: The MLIR module to add the function to.
        ctx: The MLIR context.
        loc: The location for operations.
        op_str: Operation name (e.g., "add", "exp").
        arity: Number of inputs (1 or 2).
        rows: Number of tile rows per iteration.
        cols: Number of tile columns per iteration.
        dtype: Data type for tiles.
        buffer_factor: Circular buffer factor.
        num_iterations: Number of loop iterations (1 = no loop).
    """
    # Get types.
    dtype_int = torch_dtype_to_ttcore_datatype(dtype)
    tile_type = ttcore.ir.TileType.get(ctx, 32, 32, dtype_int)
    tensor_type = RankedTensorType.get([rows, cols], tile_type)
    cb_type = ttl.CircularBufferType.get(ctx, [rows, cols], tile_type, buffer_factor)

    with InsertionPoint(module.body):
        # Create function with no arguments and no results.
        func_type = FunctionType.get([], [])
        func_name = f"compute_{op_str}"
        compute_func = func.FuncOp(func_name, func_type, loc=loc)

        # Set kernel thread attributes.
        i32_type = IntegerType.get_signless(32, ctx)
        compute_func.attributes["ttl.base_cta_index"] = IntegerAttr.get(i32_type, 3)
        compute_func.attributes["ttl.crta_indices"] = ArrayAttr.get([], ctx)
        compute_func.attributes["ttl.kernel_thread"] = Attribute.parse(
            "#ttkernel.thread<compute>", ctx
        )

        entry_block = compute_func.add_entry_block()

        with InsertionPoint(entry_block):
            # Bind circular buffers.
            cb0 = ttl.bind_cb(cb_type, cb_index=0, buffer_factor=buffer_factor, loc=loc)
            if arity == 2:
                cb1 = ttl.bind_cb(
                    cb_type, cb_index=1, buffer_factor=buffer_factor, loc=loc
                )
                output_cb_index = 2
            else:
                output_cb_index = 1
            cb_out = ttl.bind_cb(
                cb_type, cb_index=output_cb_index, buffer_factor=buffer_factor, loc=loc
            )

            def build_loop_body():
                """Build the compute operations for one iteration."""
                # Wait for input data from reader.
                a = ttl.cb_wait(tensor_type, cb0, loc=loc)
                a_attached = ttl.attach_cb(tensor_type, a, cb0, loc=loc)

                if arity == 2:
                    b = ttl.cb_wait(tensor_type, cb1, loc=loc)
                    b_attached = ttl.attach_cb(tensor_type, b, cb1, loc=loc)

                # Reserve output CB.
                out_reserved = ttl.cb_reserve(tensor_type, cb_out, loc=loc)
                out_attached = ttl.attach_cb(tensor_type, out_reserved, cb_out, loc=loc)

                # Apply the operation.
                op_func = getattr(ttl, op_str, None)
                if op_func is None:
                    raise ValueError(f"Unknown TTL op: ttl.{op_str}")

                if arity == 1:
                    result = op_func(tensor_type, a_attached, loc=loc)
                else:
                    result = op_func(tensor_type, a_attached, b_attached, loc=loc)

                # Attach result to output CB (for data flow tracking).
                ttl.attach_cb(tensor_type, result, cb_out, loc=loc)

                # Push output, pop inputs.
                ttl.cb_push(cb_out, loc=loc)
                if arity == 2:
                    ttl.cb_pop(cb1, loc=loc)
                ttl.cb_pop(cb0, loc=loc)

            if num_iterations > 1:
                # Create loop bounds.
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                num_iters = arith.ConstantOp(IndexType.get(ctx), num_iterations).result

                # Create scf.for loop.
                for_op = scf.ForOp(c0, num_iters, c1)
                with InsertionPoint(for_op.body):
                    build_loop_body()
                    scf.YieldOp([])
            else:
                # Single iteration - no loop needed.
                build_loop_body()

            func.ReturnOp([], loc=loc)


def _generate_compute_mlir(
    op_str: str,
    arity: int,
    rows: int,
    cols: int,
    dtype: torch.dtype,
    buffer_factor: int,
    num_iterations: int,
) -> str:
    """
    Generate compute thread MLIR using the Python builder API.

    Builds the compute function programmatically and returns it as a string.
    This ensures type safety and consistency with the rest of the codebase.

    Args:
        op_str: Operation name (e.g., "add", "exp").
        arity: Number of inputs (1 or 2).
        rows: Number of tile rows per iteration.
        cols: Number of tile columns per iteration.
        dtype: Data type for tiles.
        buffer_factor: Circular buffer factor.
        num_iterations: Number of loop iterations (1 = no loop).

    Returns:
        MLIR string for the compute function.
    """
    ctx = Context()
    ttl.ensure_dialects_registered(ctx)
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.create(loc)
        _build_compute_func(
            module,
            ctx,
            loc,
            op_str,
            arity,
            rows,
            cols,
            dtype,
            buffer_factor,
            num_iterations,
        )
        module.operation.verify()

    # Extract just the function from the module string.
    # The module wrapper will be added by build_e2e_module_mlir.
    module_str = str(module)
    # Remove module wrapper to get just the function.
    lines = module_str.strip().split("\n")
    # Skip first line ('module {') and last line ('}')
    if lines[0].strip().startswith("module"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "}":
        lines = lines[:-1]
    return "\n".join(lines)


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

    # Use builder API to generate compute function.
    compute_mlir = _generate_compute_mlir(
        op_str, arity, rows, cols, dtype, buffer_factor, num_iterations
    )
    output_cb_index = arity  # Output CB follows input CBs.

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
