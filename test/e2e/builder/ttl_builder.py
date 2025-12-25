# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTL MLIR module builder for E2E tests.

Generates TTL modules using high-level ops (ttl.add, ttl.exp, etc.) that
can be lowered through the compiler pipeline to executable kernels.
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
    Type as MLIRType,
)
from ttmlir.dialects import func
from ttmlir.dialects import ttcore

import ttlang.dialects.ttl as ttl

from ..config import E2EConfig


def _torch_dtype_to_ttcore_datatype(dtype: torch.dtype) -> int:
    """Convert torch dtype to ttcore DataType integer value."""
    if dtype == torch.float32:
        return int(ttcore.DataType.Float32)
    elif dtype == torch.bfloat16:
        return int(ttcore.DataType.BFloat16)
    elif dtype == torch.float16:
        return int(ttcore.DataType.Float16)
    else:
        raise ValueError(f"Unsupported dtype for tile: {dtype}")


def _torch_dtype_to_mlir_str(dtype: torch.dtype) -> str:
    """Convert torch dtype to MLIR type string."""
    if dtype == torch.float32:
        return "f32"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float16:
        return "f16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _get_tile_type(ctx: Context, dtype: torch.dtype):
    """Get the ttcore.tile type for the given dtype."""
    dtype_int = _torch_dtype_to_ttcore_datatype(dtype)
    return ttcore.ir.TileType.get(ctx, 32, 32, dtype_int)


def _get_tile_tensor_type(ctx: Context, config: E2EConfig):
    """Get tensor of tiles type."""
    rows, cols = config.grid_shape
    tile_type = _get_tile_type(ctx, config.dtype)
    return RankedTensorType.get([rows, cols], tile_type)


def _get_cb_type_str(config: E2EConfig) -> str:
    """Get the !ttl.cb type as a string for parsing."""
    rows, cols = config.grid_shape
    dtype_str = _torch_dtype_to_mlir_str(config.dtype)
    return f"!ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype_str}>, {config.buffer_factor}>"


def build_ttl_module(
    op_str: str,
    arity: int,
    config: E2EConfig,
    torch_inputs: List[torch.Tensor],
) -> Module:
    """
    Build a TTL module for the given operation using high-level TTL ops.

    Generates MLIR like:
    ```mlir
    func.func @compute_add(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>,
                           %arg1: tensor<4x4x!ttcore.tile<32x32, f32>>)
        -> tensor<4x4x!ttcore.tile<32x32, f32>> {
      %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<...>
      %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<...>
      %a = ttl.attach_cb %arg0, %cb0 : ...
      %b = ttl.attach_cb %arg1, %cb1 : ...
      %result = ttl.add %a, %b : ...
      func.return %result : ...
    }
    ```

    Args:
        op_str: Operation name (e.g., "add", "exp").
        arity: Number of inputs (1 or 2).
        config: Test configuration.
        torch_inputs: Input tensors (for shape/dtype).

    Returns:
        MLIR Module containing the TTL function.
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
