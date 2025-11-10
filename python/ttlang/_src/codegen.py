# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Code generation utilities for creating D2M generic functions and MLIR operations."""

import inspect
from typing import List, Callable, Any, Dict

from ttmlir.ir import *
from ttmlir.dialects import ttcore, d2m, func

from ..layouts import (
    create_metal_layout,
    create_stream_layout_for_input,
    compute_device_shape,
    StreamLayoutConfig,
    MetalLayoutConfig,
)
from ..constants import DEFAULT_TILE_SHAPE, DEFAULT_TILE_SIZE
from ..dtype_utils import torch_dtype_to_mlir_type, torch_dtype_to_ttcore_datatype


def create_tensor_type(
    ctx: Context,
    arg: Any,
    grid: List[int],
    tiled: bool,
    memory_space: str,
) -> Type:
    """
    Create a RankedTensorType for a kernel argument.

    Args:
        ctx: MLIR context
        arg: Tensor argument (must have .shape and .dtype attributes)
        grid: Grid dimensions (for computing shard shape)
        tiled: Unused (kept for compatibility)
        memory_space: Unused (function args always use System memory)

    Returns:
        RankedTensorType with per-core shard shape, scalar element type, and System memory space
    """
    logical_shape = list(arg.shape)
    shard_shape = [logical_shape[i] // grid[i] for i in range(len(logical_shape))]
    dtype = torch_dtype_to_mlir_type(arg.dtype, ctx)

    # HACK: Make function args 4D (device shape) to match what to_layout expects
    # This avoids rank mismatch in D2MSplitCompoundLayoutRewriter
    # TODO: Properly support 2D->4D rank change in compiler pass
    device_shape = list(grid) + shard_shape  # [grid_y, grid_x, shard_y, shard_x]

    # Function arguments are host buffers (System memory), not device buffers (L1/DRAM)
    layout = create_metal_layout(
        ctx,
        MetalLayoutConfig(
            logical_shape=logical_shape,  # Use full logical shape
            grid=grid,
            tiled=False,
            memory_space="System",
        ),
    )

    return RankedTensorType.get(device_shape, dtype, layout)


def affine_map_from_lambda(fn: Callable) -> AffineMap:
    """
    Convert a Python lambda function to an MLIR AffineMap.

    The lambda should take dimension parameters and return a tuple of
    dimension expressions or constants.

    Args:
        fn: Lambda function taking dimension parameters (e.g., lambda m, n: (m, n))

    Returns:
        AffineMap representing the indexing function

    Raises:
        TypeError: If lambda result contains unsupported types
    """

    class Dim:
        def __init__(self, position, name):
            self.position = position
            self.name = name

    dims = tuple(
        Dim(i, name) for i, name in enumerate(inspect.signature(fn).parameters)
    )
    num_dims = len(dims)
    results = fn(*dims)
    exprs = []
    for result in results:
        if isinstance(result, Dim):
            exprs.append(AffineDimExpr.get(result.position))
        elif isinstance(result, int):
            if result != 0:
                raise ValueError(
                    "The only integer constant allowed in an indexing_map is 0"
                )
            exprs.append(AffineConstantExpr.get(result))
        else:
            raise TypeError(
                f"Unsupported indexing_map result type `{type(result)}` for result `{result}`"
            )
    num_syms = 0
    return AffineMap.get(num_dims, num_syms, exprs)


def create_generic_func(
    ctx: Context,
    name: str,
    stream_func_arg_attrs: List[Attribute],
    grid: List[int],
    block_factors: List[int],
    indexing_maps: List[Callable],
    iterator_types: List[str],
    compiled_threads: List[Any],
    num_outs: int,
    user_args: List[Any],
    tiled: bool,
    memory_space: str,
):
    """
    Create a D2M generic function from compiled threads.

    This function orchestrates the creation of a d2m.generic operation that
    encapsulates compute and data movement threads with proper tensor types
    and stream layouts.

    Args:
        ctx: MLIR context
        name: Function name
        stream_func_arg_attrs: Stream attributes for function arguments
        grid: Grid dimensions
        block_factors: Block factors for each argument
        indexing_maps: List of lambda functions for indexing
        iterator_types: List of iterator type strings ("parallel", "reduction")
        compiled_threads: List of compiled thread objects
        num_outs: Number of output arguments
        user_args: Original user tensor arguments
        tiled: Whether to use tiled layout
        memory_space: "L1" or "DRAM"
    """
    if (
        isinstance(block_factors, list)
        and len(block_factors) > 0
        and isinstance(block_factors[0], tuple)
    ):
        block_factors = [b for bs in block_factors for b in bs]

    # TODO: Determine correct block_factors semantics for multi-region d2m.generic.
    # Currently forcing empty block_factors to enable "explicit datamovement form"
    # which allows multiple regions (datamovement + compute) without yield terminators.
    block_factors = []

    compiled_threads.sort(key=lambda ct: ct.kernel_type == "compute")

    ordered_tensor_args = []
    for arg in user_args:
        tensor_type = create_tensor_type(ctx, arg, grid, tiled, memory_space)
        ordered_tensor_args.append(tensor_type)

    arg_types = ordered_tensor_args
    # Currently only single output tensor is supported (num_outs=1 enforced in pykernel_gen)
    ret_type = ordered_tensor_args[-1]
    func_entry = func.FuncOp(name=name, type=(arg_types, [ret_type]))
    func_entry.arg_attrs = stream_func_arg_attrs
    func_bb = func_entry.add_entry_block()
    with InsertionPoint(func_bb):
        inputs = func_bb.arguments[:-num_outs]
        outputs = func_bb.arguments[-num_outs:]

        is_stream = []
        for attr in stream_func_arg_attrs[:-num_outs]:
            attr_dict = DictAttr(attr)
            stream_attr = attr_dict["d2m.stream"]
            is_stream.append(BoolAttr(stream_attr).value)

        # Insert to_layout operations to convert host tensors to device tensors
        # Function args are scalar host tensors, generic operates on tiled device tensors
        device_inputs = []
        for i, inp in enumerate(inputs):
            # Create device tensor type (tiled, L1 memory)
            device_layout = create_metal_layout(
                ctx,
                MetalLayoutConfig(
                    logical_shape=list(user_args[i].shape),
                    grid=grid,
                    tiled=tiled,
                    memory_space=memory_space,
                ),
            )

            # For to_layout, keep element type as scalar to avoid compound operation
            # The tiling happens inside d2m.generic, not in to_layout
            # Device tensor shape is just the shard shape (same as input for System→L1)
            dtype = torch_dtype_to_mlir_type(user_args[i].dtype, ctx)
            inp_tensor_type = RankedTensorType(inp.type)
            device_shape = list(inp_tensor_type.shape)  # Same shape as input

            device_tensor_type = RankedTensorType.get(device_shape, dtype, device_layout)

            # Insert to_layout: host → device
            device_buffer = d2m.EmptyOp(device_tensor_type)
            to_device = d2m.ToLayoutOp(
                [device_tensor_type], inp, device_buffer.result, layout=None
            )

            # Wrap with stream_layout if needed
            if is_stream[i]:
                device_with_stream = create_stream_layout_for_input(
                    ctx,
                    to_device.results[0],
                    StreamLayoutConfig(
                        logical_shape=list(user_args[i].shape),
                        grid=grid,
                        tiled=tiled,
                        memory_space=memory_space,
                    ),
                )
                device_inputs.append(device_with_stream)
            else:
                device_inputs.append(to_device.results[0])

        wrapped_inputs = device_inputs

        threads = ArrayAttr.get(
            [
                ct.func_entry.attributes[d2m.ir.ThreadAttr.name]
                for ct in compiled_threads
            ]
        )

        # Create device output buffer (scalar, L1 memory)
        # Keep scalar for to_layout to avoid compound operation
        output_idx = len(user_args) - num_outs
        device_output_layout = create_metal_layout(
            ctx,
            MetalLayoutConfig(
                logical_shape=list(user_args[output_idx].shape),
                grid=grid,
                tiled=tiled,
                memory_space=memory_space,
            ),
        )

        output_dtype = torch_dtype_to_mlir_type(user_args[output_idx].dtype, ctx)
        # Use same shape as the output argument (System memory)
        output_tensor_type = RankedTensorType(outputs[0].type)
        device_output_shape = list(output_tensor_type.shape)

        device_output_type = RankedTensorType.get(
            device_output_shape, output_dtype, device_output_layout
        )
        device_output_buffer = d2m.EmptyOp(device_output_type)

        # Generic operates on device tensors
        generic = d2m.GenericOp(
            [device_output_type],
            wrapped_inputs,
            [device_output_buffer.result],
            ttcore.ir.GridAttr.get(ctx, grid),
            block_factors,
            list(map(affine_map_from_lambda, indexing_maps)),
            ArrayAttr.get(
                list(
                    ttcore.ir.IteratorTypeAttr.get(
                        ctx, ttcore.IteratorType[i.title()].value
                    )
                    for i in iterator_types
                )
            ),
            threads,
            len(compiled_threads),
        )
        for compiled_thread, generic_region in zip(compiled_threads, generic.regions):
            compiled_thread.func_entry.entry_block.append_to(generic_region)
            last_op = generic_region.blocks[0].operations[-1]
            if isinstance(last_op, func.ReturnOp):
                last_op.erase()

        # Insert to_layout: device → host (write to existing output argument)
        to_host = d2m.ToLayoutOp(
            [ret_type], generic.results[0], outputs[0], layout=None
        )

        func.ReturnOp(to_host.results)


def copy_symbol_table_globals(
    module_symbol_table: SymbolTable,
    compiled_threads: List[Any],
    f_params: Dict[str, Any],
) -> None:
    """
    Copy global symbols from compiled threads to the module symbol table.

    Args:
        module_symbol_table: Module-level symbol table
        compiled_threads: List of compiled thread objects
        f_params: Function parameters dictionary
    """
    f_params_list = list(f_params.keys())
    for ct in compiled_threads:
        for op in ct.module.body:
            if "sym_name" not in op.attributes:
                continue
            sym_name = op.attributes["sym_name"]
            if sym_name.value in f_params and sym_name.value in ct.module_symbol_table:
                clone = op.clone()
                clone.index = IntegerAttr.get(
                    IntegerType.get_signed(32), f_params_list.index(sym_name.value)
                )
                module_symbol_table.insert(clone)
