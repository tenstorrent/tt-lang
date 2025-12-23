# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
from typing import List, Set
from dataclasses import dataclass

from ttmlir.ir import *
from ttmlir.dialects import ttcore, func, arith, ttkernel

from pykernel._src.kernel_types import *
from pykernel._src.kernel_ast import TTCompilerBase
from .tensor_accessor import TensorAccessor

from ..dialects import ttl
from ..layouts import create_ttnn_layout, TTNNLayoutConfig
from ..dtype_utils import tensor_dtype_to_mlir_type, tensor_dtype_to_ttcore_datatype
from ..constants import DEFAULT_TILE_SIZE


def _build_tensor_accessor_type(ctx, accessor, grid, tiled, memory_space):
    """Build MLIR tensor type for a TensorAccessor with TTNNLayoutAttr."""
    if not tiled:
        raise ValueError("Only tiled tensors supported for TTNN interop")
    if memory_space != "L1":
        raise ValueError(f"Only L1 memory space supported, got {memory_space}")
    if len(accessor.shape) != 2:
        raise ValueError(f"Only 2D tensors supported, got shape {accessor.shape}")

    layout = create_ttnn_layout(
        ctx,
        TTNNLayoutConfig(
            logical_shape=accessor.shape,
            grid=grid,
            dtype=accessor.dtype,
        ),
    )

    ttcore_dtype = tensor_dtype_to_ttcore_datatype(accessor.dtype)
    element_type = ttcore.ir.TileType.get(
        ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
    )

    # Device shape: grid dims + shard dims (1x1 tiles per core for single-core)
    shard_tiles = [
        accessor.shape[i] // grid[i] // DEFAULT_TILE_SIZE for i in range(2)
    ]
    device_shape = list(grid) + shard_tiles

    return RankedTensorType.get(device_shape, element_type, layout)


@dataclass(frozen=True)
class CompilerContext:
    """Immutable compilation context for TTL kernels."""

    grid: List[int]
    memory_space: str
    tiled: bool


class TTLGenericCompiler(TTCompilerBase):
    """Compiler that generates TTL dialect ops from Python AST."""

    _syntax = {}

    def __init__(self, name, kernel_type=None, captures={}, *args, **kwargs):
        super().__init__(name, kernel_type, *args, **kwargs)
        self.loc = Location.name(self.name)
        self.captures = captures
        self.streams: Set[str] = set()
        self.supported_nodes.append(ast.AsyncFunctionDef)
        self.supported_nodes.append(ast.With)

        self.context = CompilerContext(
            grid=kwargs.get("grid", [1, 1]),
            memory_space=kwargs.get("memory_space", "L1"),
            tiled=kwargs.get("tiled", True),
        )

        # Track CB info for binding inside function body
        self._cb_info: List[dict] = []  # [{name, shape, element_type, cb_index}, ...]
        self._next_cb_index = 0

        self._fn_map = {}
        for name, val in TTLGenericCompiler._syntax.items():
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

    def _get_ttkernel_thread_type(self) -> str:
        """Map kernel_type to ttkernel thread type string."""
        if self.kernel_type == "compute":
            return "compute"
        elif self.kernel_type == "datamovement":
            return "noc"
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _emit_entry(self, node):
        assert not self.func_entry, "Cannot declare function within a function"

        # Collect CB info for binding inside function body
        self._cb_info = []
        self._next_cb_index = 0

        for i in range(len(node.args.args)):
            arg = node.args.args[i]

            if not arg.annotation:
                raise TypeError("All kernel arguments must have a type annotation")
            elif arg.annotation.id == "TensorBlock":
                raise NotImplementedError("TensorBlock not yet supported in TTL mode")
            elif arg.annotation.id == "CircularBuffer":
                if not self.context.tiled:
                    raise ValueError("Only tiled CBs supported")
                shape = list(self.args[i].shape)
                if len(shape) != 2:
                    raise ValueError(f"Only 2D CBs supported, got shape {shape}")

                # Compute shard shape: tiles per core
                shard_shape = [
                    shape[i] // self.context.grid[i] // DEFAULT_TILE_SIZE
                    for i in range(2)
                ]

                ttcore_dtype = tensor_dtype_to_ttcore_datatype(self.args[i].dtype)
                element_type = ttcore.ir.TileType.get(
                    self.ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
                )

                self._cb_info.append({
                    "name": arg.arg,
                    "shard_shape": shard_shape,
                    "element_type": element_type,
                    "cb_index": self._next_cb_index,
                })
                self._next_cb_index += 1
            elif arg.annotation.id == "Semaphore":
                raise NotImplementedError("Semaphore not yet supported in TTL mode")
            else:
                raise TypeError(
                    f"Unknown kernel arguments type annotation {arg.annotation.id}"
                )

        # Collect TensorAccessor captures for function arguments
        self._tensor_accessor_names = []
        func_arg_types = []
        for name, val in self.captures.items():
            if isinstance(val, TensorAccessor):
                tensor_type = _build_tensor_accessor_type(
                    self.ctx, val, self.context.grid,
                    self.context.tiled, self.context.memory_space
                )
                self._tensor_accessor_names.append(name)
                func_arg_types.append(tensor_type)

        self.func_entry = func.FuncOp(name=node.name, type=(func_arg_types, []))

        # Set thread attribute: ttl.kernel_thread = #ttkernel.thread<compute/noc>
        thread_type = self._get_ttkernel_thread_type()
        thread_attr = ttkernel.ir.ThreadTypeAttr.get(self.ctx, thread_type)
        self.func_entry.attributes["ttl.kernel_thread"] = thread_attr

        self.symbol_tables.append({})
        func_bb = self.func_entry.add_entry_block()

        # Add ttl module to symbol table
        self.symbol_tables[-1]["ttl"] = ttl

        # Ensure TTL dialect is registered for type parsing
        ttl.ensure_dialects_registered(self.ctx)

        self.module_symbol_table = SymbolTable(self.module.operation)

        # Emit function body
        with InsertionPoint(func_bb):
            # Emit ttl.bind_cb ops at function entry for each CB argument
            for cb_info in self._cb_info:
                # Parse CB type: !ttl.cb<[shape], element_type, buffer_factor>
                shape_str = ", ".join(str(s) for s in cb_info["shard_shape"])
                elem_str = str(cb_info["element_type"])
                buffer_factor = 2  # Default buffer factor
                cb_type = Type.parse(
                    f"!ttl.cb<[{shape_str}], {elem_str}, {buffer_factor}>", self.ctx
                )

                # Emit: %cb = ttl.bind_cb {cb_index = N, buffer_factor = M} : !ttl.cb<...>
                cb_val = ttl.bind_cb(
                    cb_type,
                    cb_info["cb_index"],
                    buffer_factor=buffer_factor,
                )
                self.symbol_tables[-1][cb_info["name"]] = cb_val

            # Map TensorAccessor function arguments to symbol table
            for i, name in enumerate(self._tensor_accessor_names):
                self.symbol_tables[-1][name] = func_bb.arguments[i]
                self.streams.add(name)

            # Prepopulate other captures (non-TensorAccessor)
            for name, val in self.captures.items():
                if isinstance(val, TensorAccessor):
                    continue  # Already handled via function arguments
                assert isinstance(name, str)
                if isinstance(val, int):
                    self.symbol_tables[-1][name] = arith.ConstantOp(
                        IndexType.get(self.ctx), val
                    )
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

    def _get_cb_tensor_type(self, cb_val):
        """Extract the tensor type from a TTL CB type by parsing it."""
        # CB type is !ttl.cb<[shape], element_type, buffer_factor>
        cb_type_str = str(cb_val.type)
        # Parse: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
        # Element type may contain commas inside <>, so match up to last comma
        import re
        match = re.match(r"!ttl\.cb<\[([^\]]+)\], (.+), (\d+)>$", cb_type_str)
        if match:
            shape_str = match.group(1)
            elem_str = match.group(2)
            shape = [int(s.strip()) for s in shape_str.split(",")]
            elem_type = Type.parse(elem_str, self.ctx)
            return RankedTensorType.get(shape, elem_type)
        raise ValueError(f"Could not parse CB type: {cb_type_str}")

    def visit_With(self, node):
        """
        Handle 'with' for CircularBuffer acquire/release.

        Acquire ops (wait/reserve) are generated left-to-right.
        Release ops (pop/push) are generated in reverse order at scope end.

        Example:
            with lhs_cb.wait() as l, rhs_cb.wait() as r, out_cb.reserve() as o:
                ...
                # releases in reverse order: push(out), pop(rhs), pop(lhs)
        """
        with self.loc:
            # Process each with-item: acquire resources and track for release
            releases = []  # [(release_op, cb_val), ...] in acquisition order

            for item in node.items:
                context_expr = item.context_expr
                optional_vars = item.optional_vars

                if not isinstance(context_expr, ast.Call):
                    raise NotImplementedError(
                        "'with' requires a method call (e.g., cb.reserve())"
                    )

                if not isinstance(context_expr.func, ast.Attribute):
                    raise NotImplementedError(
                        "'with' requires a method call on an object"
                    )

                method_name = context_expr.func.attr
                cb_node = context_expr.func.value

                if method_name not in ("reserve", "wait"):
                    raise NotImplementedError(
                        f"'with' only supports 'reserve()' or 'wait()', got '{method_name}'"
                    )

                if not isinstance(cb_node, ast.Name):
                    raise NotImplementedError(
                        "'with' requires a simple variable (e.g., cb.reserve())"
                    )

                cb_table = self._var_exists(cb_node.id)
                if not cb_table:
                    raise NameError(f"'{cb_node.id}' not found in scope")
                cb_val = cb_table[cb_node.id]

                # Get tensor type from CB for reserve/wait result
                tensor_type = self._get_cb_tensor_type(cb_val)
                if method_name == "reserve":
                    acquire_result = ttl.cb_reserve(tensor_type, cb_val)
                    releases.append((ttl.cb_push, cb_val))
                else:  # wait
                    acquire_result = ttl.cb_wait(tensor_type, cb_val)
                    releases.append((ttl.cb_pop, cb_val))

                if optional_vars is not None:
                    if not isinstance(optional_vars, ast.Name):
                        raise NotImplementedError(
                            "'with ... as var' requires a simple variable name"
                        )
                    self.symbol_tables[-1][optional_vars.id] = acquire_result

            for stmt in node.body:
                self.visit(stmt)

            # Release in reverse order
            for release_op, cb_val in reversed(releases):
                release_op(cb_val)


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
                        TTLGenericCompiler._syntax[qualified] = method

            return cls

        return _class_wrapper
    else:

        def _fn_wrapper(fn):
            assert callable(fn)
            TTLGenericCompiler._syntax[fn.__name__] = fn
            return fn

        return _fn_wrapper
