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

from typing import Callable, List

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
from .thread_builder import generate_layout_attrs
from .dm_builder import DMThreadBuilder
from .compute_builder import ComputeThreadBuilder
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


def _generate_compute_mlir(
    op_str: str,
    arity: int,
    config: E2EConfig,
) -> str:
    """
    Generate compute thread MLIR using the ComputeThreadBuilder.

    Builds the compute function programmatically and returns it as a string.

    Args:
        op_str: Operation name (e.g., "add", "exp").
        arity: Number of inputs (1 or 2).
        config: Test configuration.

    Returns:
        MLIR string for the compute function.
    """
    ctx = Context()
    ttl.ensure_dialects_registered(ctx)
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.create(loc)

        # Use ComputeThreadBuilder to build the function.
        builder = ComputeThreadBuilder(module, ctx, loc, config)
        builder.build_compute(op_str, arity)

        module.operation.verify()

    # Extract just the function from the module string.
    module_str = str(module)
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
    # Generate layout attributes.
    layout_attrs = generate_layout_attrs(config)

    # Use DMThreadBuilder for reader and writer.
    dm_builder = DMThreadBuilder(config)
    reader_mlir = dm_builder.build_reader(arity)
    output_cb_index = arity  # Output CB follows input CBs.
    writer_mlir = dm_builder.build_writer([output_cb_index])

    # Generate compute with ComputeThreadBuilder.
    compute_mlir = _generate_compute_mlir(op_str, arity, config)

    # Combine into full module.
    return f"""// Auto-generated E2E MLIR module for {op_str} operation.
// Arity: {arity}, Grid: {config.grid_shape}, Dtype: {config.dtype}, Iterations: {config.num_tiles}

{layout_attrs}

module {{
{reader_mlir}
{compute_mlir}
{writer_mlir}
}}
"""


def build_e2e_module_mlir_custom(
    name: str,
    arity: int,
    num_outputs: int,
    config: E2EConfig,
    compute_fn: Callable[[List, "ComputeThreadBuilder"], List],
) -> str:
    """
    Build E2E MLIR module with custom compute function.

    Use this for fused operations like exp(a + b) or sqrt(abs(a)).

    Args:
        name: Name for the compute function.
        arity: Number of inputs.
        num_outputs: Number of outputs.
        config: Test configuration.
        compute_fn: Callback that takes (inputs, builder) and returns output tensors.

    Returns:
        MLIR source string with all three thread functions.
    """
    # Generate layout attributes.
    layout_attrs = generate_layout_attrs(config)

    # Use DMThreadBuilder for reader and writer.
    dm_builder = DMThreadBuilder(config)
    reader_mlir = dm_builder.build_reader(arity)
    output_cbs = list(range(arity, arity + num_outputs))
    writer_mlir = dm_builder.build_writer(output_cbs)

    # Generate custom compute.
    ctx = Context()
    ttl.ensure_dialects_registered(ctx)
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.create(loc)
        builder = ComputeThreadBuilder(module, ctx, loc, config)

        # Wrap the compute_fn to pass the builder for type access.
        def wrapped_compute_fn(inputs):
            return compute_fn(inputs, builder)

        builder.build_compute_custom(
            name=name,
            input_cbs=list(range(arity)),
            output_cbs=output_cbs,
            compute_fn=wrapped_compute_fn,
        )
        module.operation.verify()

    # Extract compute function from module.
    module_str = str(module)
    lines = module_str.strip().split("\n")
    if lines[0].strip().startswith("module"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "}":
        lines = lines[:-1]
    compute_mlir = "\n".join(lines)

    # Combine into full module.
    return f"""// Auto-generated E2E MLIR module for {name} operation.
// Arity: {arity}, Outputs: {num_outputs}, Grid: {config.grid_shape}, Dtype: {config.dtype}

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
