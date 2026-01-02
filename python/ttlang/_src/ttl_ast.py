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
from .tensor_registry import get_tensor_global_index

from ..dialects import ttl
from ..dtype_utils import is_ttnn_tensor
from ..layouts import create_ttnn_layout, TTNNLayoutConfig
from ..dtype_utils import tensor_dtype_to_ttcore_datatype
from ..constants import DEFAULT_TILE_SIZE
from ..ttl_utils import get_thread_type_string


def _build_tensor_type(ctx, tensor, grid, tiled, memory_space):
    """Build MLIR tensor type for a ttnn tensor with TTNNLayoutAttr."""
    if not tiled:
        raise ValueError("Only tiled tensors supported for TTNN interop")
    if memory_space not in ("L1", "DRAM"):
        raise ValueError(f"Only L1 or DRAM memory space supported, got {memory_space}")
    if len(tensor.shape) != 2:
        raise ValueError(f"Only 2D tensors supported, got shape {tensor.shape}")

    layout = create_ttnn_layout(
        ctx,
        TTNNLayoutConfig(
            logical_shape=tensor.shape,
            grid=grid,
            dtype=tensor.dtype,
        ),
    )

    ttcore_dtype = tensor_dtype_to_ttcore_datatype(tensor.dtype)
    element_type = ttcore.ir.TileType.get(
        ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
    )

    # Device shape: grid dims + shard dims (1x1 tiles per core for single-core)
    shard_tiles = [tensor.shape[i] // grid[i] // DEFAULT_TILE_SIZE for i in range(2)]
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

    def _emit_cb_from_capture(self, cb):
        """Emit ttl.bind_cb for a captured CircularBuffer instance."""
        ttcore_dtype = tensor_dtype_to_ttcore_datatype(cb.dtype)
        element_type = ttcore.ir.TileType.get(
            self.ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
        )
        cb_type = ttl.CircularBufferType.get(
            self.ctx,
            list(cb.shape),
            element_type,
            cb.buffer_factor,
        )
        # Emit: %cb = ttl.bind_cb {cb_index = N, buffer_factor = M} : !ttl.cb<...>
        return ttl.bind_cb(cb_type, cb._cb_index, buffer_factor=cb.buffer_factor)

    def _emit_entry(self, node):
        assert not self.func_entry, "Cannot declare function within a function"

        if node.args.args:
            raise ValueError(
                "Thread functions must have no parameters. "
                "Use make_circular_buffer_like() in kernel body and capture CBs in closures."
            )

        # Collect tensor captures for function arguments
        self._tensor_accessor_names = []
        self._tensor_accessor_global_indices = []
        func_arg_types = []
        for name, val in self.captures.items():
            if is_ttnn_tensor(val):
                tensor_type = _build_tensor_type(
                    self.ctx,
                    val,
                    self.context.grid,
                    self.context.tiled,
                    self.context.memory_space,
                )
                self._tensor_accessor_names.append(name)
                self._tensor_accessor_global_indices.append(
                    get_tensor_global_index(val)
                )
                func_arg_types.append(tensor_type)

        self.func_entry = func.FuncOp(name=node.name, type=(func_arg_types, []))

        # Set thread attribute: ttl.kernel_thread = #ttkernel.thread<compute/noc>
        thread_type = get_thread_type_string(self.kernel_type)
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
            # Map TensorAccessor function arguments to symbol table
            for i, name in enumerate(self._tensor_accessor_names):
                self.symbol_tables[-1][name] = func_bb.arguments[i]
                self.streams.add(name)

            # Prepopulate other captures (non-tensor)
            from ..circular_buffer import CircularBuffer

            for name, val in self.captures.items():
                if is_ttnn_tensor(val):
                    continue  # Already handled via function arguments
                assert isinstance(name, str)
                if isinstance(val, int):
                    self.symbol_tables[-1][name] = arith.ConstantOp(
                        IndexType.get(self.ctx), val
                    )
                elif isinstance(val, CircularBuffer):
                    cb_val = self._emit_cb_from_capture(val)
                    self.symbol_tables[-1][name] = cb_val
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
        """Extract the tensor type from a TTL CB type."""
        cb_type = ttl.CircularBufferType.maybe_downcast(cb_val.type)
        if cb_type is None:
            raise ValueError(f"Expected CircularBufferType, got {cb_val.type}")
        return RankedTensorType.get(cb_type.shape, cb_type.element_type)

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
                    tensor = ttl.cb_reserve(tensor_type, cb_val)
                    releases.append((ttl.cb_push, cb_val))
                else:  # wait
                    tensor = ttl.cb_wait(tensor_type, cb_val)
                    releases.append((ttl.cb_pop, cb_val))

                # Attach CB to tensor so store() can find the CB association
                acquire_result = ttl.attach_cb(tensor.type, tensor, cb_val)

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
