# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
from typing import List, Set
from dataclasses import dataclass

from ttmlir.ir import *
from ttmlir.dialects import ttcore, d2m, func, arith

from pykernel._src.kernel_types import *
from pykernel._src.kernel_ast import TTCompilerBase
from .tensor_accessor import TensorAccessor

from ..layouts import create_metal_layout, compute_device_shape, MetalLayoutConfig
from ..dtype_utils import torch_dtype_to_mlir_type, torch_dtype_to_ttcore_datatype
from ..constants import DEFAULT_TILE_SHAPE, DEFAULT_TILE_SIZE


@dataclass(frozen=True)
class CompilerContext:
    """Immutable compilation context for D2M kernels."""

    grid: List[int]
    memory_space: str
    tiled: bool
    dram_streams: frozenset  # Names of streams backed by DRAM TensorAccessors


class D2MGenericCompiler(TTCompilerBase):
    _syntax = {}

    def __init__(self, name, kernel_type=None, captures={}, *args, **kwargs):
        super().__init__(name, kernel_type, *args, **kwargs)
        self.loc = Location.name(self.name)
        self.captures = captures
        self.streams: Set[str] = set()
        self.stream_memory_spaces: dict = {}
        self.supported_nodes.append(ast.AsyncFunctionDef)

        self.context = CompilerContext(
            grid=kwargs.get("grid", [1, 1]),
            memory_space=kwargs.get("memory_space", "L1"),
            tiled=kwargs.get("tiled", True),
            dram_streams=frozenset(kwargs.get("dram_streams", set())),
        )

        self._fn_map = {}
        self._fn_map["iter_index"] = (
            d2m.iter_index,
            [True],
        )  # True for arg as attribute
        self._fn_map["core_index"] = (
            d2m.core_index,
            [True],
        )  # True for arg as attribute
        for name, val in D2MGenericCompiler._syntax.items():
            self._fn_map[name] = val

    # Override to use i64 for all integer constants (attributes or not)
    # D2M ops require i64, and this reduces casts throughout the pipeline
    def visit_Constant(self, node):
        as_attr = getattr(node, "_ttkernel_as_attr", False)
        op_constructor = IntegerAttr.get if as_attr else arith.ConstantOp
        if callable(as_attr):
            return as_attr(node)
        elif isinstance(node.value, bool):
            return op_constructor(IntegerType.get_signless(1, self.ctx), node.value)
        elif isinstance(node.value, int):
            return op_constructor(IntegerType.get_signless(64, self.ctx), node.value)
        else:
            raise NotImplementedError(
                f"constant type {type(node.value).__name__} not implemented"
            )

    def _emit_entry(self, node):
        # TODO: add alloca args name into symbol table
        assert not self.func_entry, "Cannot declare function within a function"

        func_operand_types = []
        for i in range(len(node.args.args)):
            arg = node.args.args[i]

            if not arg.annotation:
                raise TypeError("All kernel arguments must have a type annotation")
            elif arg.annotation.id == "TensorBlock":
                shape = self.args[i].shape
                shard_shape = [
                    shape[j] // self.context.grid[j] for j in range(len(shape))
                ]
                dtype = torch_dtype_to_mlir_type(self.args[i].dtype, self.ctx)
                tensor_type = RankedTensorType.get(shard_shape, dtype)
                func_operand_types.append(tensor_type)
            elif arg.annotation.id == "CircularBuffer":
                shape = list(self.args[i].shape)
                dtype = torch_dtype_to_mlir_type(self.args[i].dtype, self.ctx)

                # Check if this CB receives data from a DRAM TensorAccessor
                # CBs receiving DRAM data should be scalar (tilization happens in compute)
                tensor_name = getattr(self.args[i], '_global_name', None)
                cb_receives_dram = tensor_name in self.context.dram_streams

                cb_tiled = self.context.tiled and not cb_receives_dram

                # Compute shard shape (tiles per core)
                layout = create_metal_layout(
                    self.ctx,
                    MetalLayoutConfig(
                        logical_shape=shape,
                        grid=self.context.grid,
                        tiled=cb_tiled,
                        memory_space=self.context.memory_space,
                    ),
                )
                tile_shape = DEFAULT_TILE_SHAPE if cb_tiled else [1, 1]
                device_shape = compute_device_shape(
                    layout, self.context.grid, shape, tile_shape
                )
                shard_shape = device_shape[len(device_shape) // 2 :]

                # CBs wrap the tensor type that enters the generic op
                # CBs receiving DRAM data use scalar type, others use tile type
                if cb_tiled:
                    ttcore_dtype = torch_dtype_to_ttcore_datatype(self.args[i].dtype)
                    element_type = ttcore.ir.TileType.get(
                        self.ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
                    )
                else:
                    element_type = dtype

                # CBs use local memory (no MetalLayoutAttr) - they represent per-core views
                cb_tensor_type = RankedTensorType.get(shard_shape, element_type, None)
                func_operand_types.append(d2m.ir.CBType.get(self.ctx, cb_tensor_type))
            elif arg.annotation.id == "Semaphore":
                func_operand_types.append(d2m.ir.SemaphoreType.get(self.ctx))
            else:
                raise TypeError(
                    f"Unknown kernel arguments type annotation {arg.annotation.id}"
                )

        self.func_entry = func.FuncOp(name=node.name, type=(func_operand_types, []))

        self.func_entry.attributes[d2m.ir.ThreadAttr.name] = d2m.ir.ThreadAttr.get(
            self.ctx, self.kernel_type
        )

        self.symbol_tables.append({})

        # prepopulate bb arguments into symbol table
        func_bb = self.func_entry.add_entry_block()
        for i, bb_arg in enumerate(func_bb.arguments):
            self.symbol_tables[-1][node.args.args[i].arg] = bb_arg

        # Add d2m module to symbol table
        self.symbol_tables[-1]["d2m"] = d2m

        self.module_symbol_table = SymbolTable(self.module.operation)

        # update basic block
        with InsertionPoint(func_bb):
            # prepopulate captures at the top of the scope
            for name, val in self.captures.items():
                assert isinstance(name, str)
                if isinstance(val, int):
                    self.symbol_tables[-1][name] = arith.ConstantOp(
                        IndexType.get(self.ctx), val
                    )
                elif isinstance(val, TensorAccessor):
                    with InsertionPoint.at_block_begin(self.module.body):
                        accessor_memory_space = (
                            val.memory_space
                            if val.memory_space is not None
                            else self.context.memory_space # Default to L1
                        )
                        # DRAM TensorAccessors use scalar format - tilization happens
                        # on-the-fly in compute when data is pulled from DRAM to CB
                        accessor_tiled = self.context.tiled and accessor_memory_space != "DRAM"
                        layout = create_metal_layout(
                            self.ctx,
                            MetalLayoutConfig(
                                logical_shape=val.shape,
                                grid=self.context.grid,
                                tiled=accessor_tiled,
                                memory_space=accessor_memory_space,
                            ),
                        )
                        tile_shape = (
                            DEFAULT_TILE_SHAPE if accessor_tiled else [1, 1]
                        )
                        device_shape = compute_device_shape(
                            layout, self.context.grid, val.shape, tile_shape
                        )

                        # Get dtype from TensorAccessor
                        stream_dtype = torch_dtype_to_mlir_type(val.dtype, self.ctx)
                        stream_ttcore_dtype = torch_dtype_to_ttcore_datatype(val.dtype)
                        element_type = (
                            ttcore.ir.TileType.get(
                                self.ctx,
                                DEFAULT_TILE_SIZE,
                                DEFAULT_TILE_SIZE,
                                stream_ttcore_dtype,
                            )
                            if accessor_tiled
                            else stream_dtype
                        )
                        tensor = RankedTensorType.get(
                            device_shape, element_type, layout
                        )
                        globalTensor = ttcore.GlobalOp(val.name, tensor)
                        self.module_symbol_table.insert(globalTensor.operation)
                    self.symbol_tables[-1][name] = ttcore.get_global(tensor, val.name)
                    self.streams.add(val.name)
                    self.stream_memory_spaces[val.name] = accessor_memory_space
                else:
                    raise TypeError(f"Invalid capture type for var {name}: {type(val)}")

            for target in node.body:
                self.visit(target)

            func.ReturnOp([])

        self.symbol_tables.pop()

    def visit_FunctionDef(self, node):
        with self.loc:
            return self._emit_entry(node)

    def visit_AsyncFunctionDef(self, node):
        with self.loc:
            return self._emit_entry(node)


def syntax(syntax_name):
    if syntax_name.startswith("!"):

        def _class_wrapper(cls):
            assert isinstance(cls, type)

            for name, method in cls.__dict__.items():
                if callable(method):
                    sig = inspect.signature(method)
                    first_arg_name = next(iter(sig.parameters.keys()))
                    if first_arg_name == "ast_self":
                        setattr(cls, name, staticmethod(method))
                        qualified = f"{syntax_name}.{name}"
                        D2MGenericCompiler._syntax[qualified] = method

            return cls

        return _class_wrapper
    else:

        def _fn_wrapper(fn):
            assert callable(fn)
            D2MGenericCompiler._syntax[fn.__name__] = fn
            return fn

        return _fn_wrapper
