# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
from dataclasses import dataclass
from typing import List, Optional, Set

from ttl.pykernel._src.kernel_ast import TTCompilerBase
from ttl.pykernel._src.utils import _get_type_str
from ttl.dialects import arith, func, ttcore, ttkernel
from ttl.ir import *

from ..constants import DEFAULT_TILE_SIZE
from ..diagnostics import TTLangCompileError
from ttl.dialects import ttl
from ..dtype_utils import is_ttnn_tensor, tensor_dtype_to_ttcore_datatype
from ..layouts import (
    LayoutConfig,
    create_layout,
    detect_memory_layout,
    TENSOR_MEMORY_LAYOUT_INTERLEAVED,
)
from ..ttl_utils import get_thread_type_string
from .auto_profile import (
    get_line_mapper,
    is_auto_profile_enabled,
)
from .tensor_registry import get_tensor_global_index, get_tensor_source


def _make_file_loc(ctx, source_file: str, node, line_offset: int = 0) -> Location:
    """Create an MLIR file location from an AST node."""
    if not hasattr(node, "lineno"):
        raise ValueError(f"AST node {type(node).__name__} has no line number")
    return Location.file(
        source_file, node.lineno + line_offset, node.col_offset + 1, ctx
    )


def _get_annotation_name(annotation):
    """Extract the type name from an annotation node.

    Handles both simple names (DataflowBuffer) and qualified names (ttl.DataflowBuffer).
    Returns the simple type name (e.g., 'DataflowBuffer') in both cases.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Attribute):
        return annotation.attr
    else:
        raise TypeError(f"Unsupported annotation type: {type(annotation)}")


def _raise_tensor_error(tensor, message: str):
    """Raise TTLangCompileError with tensor source location if available."""
    source_info = get_tensor_source(tensor)
    if source_info:
        source_file, line = source_info
        raise TTLangCompileError(message, source_file=source_file, line=line)
    raise ValueError(message)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _build_tensor_type(ctx, tensor, grid, tiled, memory_space):
    """Build MLIR tensor type with TTLLayoutAttr encoding."""
    if not tiled:
        raise ValueError("Only tiled tensors supported")
    if memory_space not in ("L1", "DRAM"):
        raise ValueError(f"Only L1 or DRAM memory space supported, got {memory_space}")
    if len(grid) != 2:
        raise ValueError(f"Only 2D grids supported, got grid {tuple(grid)}")

    shape = list(tensor.shape)
    if len(shape) < 2:
        _raise_tensor_error(
            tensor,
            f"Tensors must have at least 2 dimensions, got shape {tensor.shape}",
        )
    if any(d <= 0 for d in shape):
        _raise_tensor_error(
            tensor,
            f"All shape dimensions must be positive, got shape {tensor.shape}",
        )

    mem_layout = TENSOR_MEMORY_LAYOUT_INTERLEAVED
    if is_ttnn_tensor(tensor):
        mem_layout = detect_memory_layout(tensor)

    layout = create_layout(
        ctx,
        LayoutConfig(
            logical_shape=shape,
            grid=grid,
            dtype=tensor.dtype,
            memory_layout=mem_layout,
        ),
    )

    ttcore_dtype = tensor_dtype_to_ttcore_datatype(tensor.dtype)
    element_type = ttcore.ir.TileType.get(
        ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
    )

    # Device shape: batch dims preserved, last 2 dims converted to tile counts
    batch_dims = shape[:-2]
    tensor_rows, tensor_cols = shape[-2], shape[-1]
    total_row_tiles = _ceil_div(tensor_rows, DEFAULT_TILE_SIZE)
    total_col_tiles = _ceil_div(tensor_cols, DEFAULT_TILE_SIZE)
    device_shape = batch_dims + [total_row_tiles, total_col_tiles]

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

        # Debug location support
        self.debug_locations = kwargs.get("debug_locations", False)
        self.source_file = kwargs.get("_source_file", "<unknown>")
        self.source_lines = kwargs.get("_source_lines", [])
        self.line_offset = kwargs.get("_line_offset", 0)

        # Function globals for resolving module-level constants
        self.fn_globals = kwargs.get("_globals", {})

        # Track CB info for binding inside function body
        self._cb_info: List[dict] = []  # [{name, shape, element_type, cb_index}, ...]

        # Auto-profiling support
        self.auto_profile_enabled = is_auto_profile_enabled()
        self.line_mapper = get_line_mapper() if self.auto_profile_enabled else None
        if self.line_mapper:
            self.line_mapper.line_offset = self.line_offset
        self._current_signpost_line = None

        self._fn_map = {}
        for name, val in TTLGenericCompiler._syntax.items():
            self._fn_map[name] = val

        # Map id(PipeNet object) -> Python variable name the user assigned
        # it to. Populated from captures/globals at function entry and
        # from body-local PipeNet assignments. Read by `_emit_pipe_from_capture`
        # to stamp the user's variable name onto each `ttl.create_pipe`
        # so the verifier can name PipeNets by user-facing identifier.
        self._pipe_net_names: dict[int, str] = {}

    def _set_var(self, var_name, value):
        # Capture PipeNet variable names so the verifier can render
        # diagnostics in user-facing terms (e.g. `a_pipe_net.is_active()`
        # instead of `net_0.is_active()`). Body-local PipeNet assignments
        # are recorded here too — `a_pipe_net = ttl.PipeNet(a_pipes)`
        # evaluates the RHS at trace time and stores the resulting object.
        from ..pipe import PipeNet

        if isinstance(value, PipeNet):
            self._pipe_net_names.setdefault(id(value), var_name)
        super()._set_var(var_name, value)

    def _resolve_pipe_net_name(self, pipenet) -> str:
        """User's Python variable name for `pipenet`, or a synthetic
        `net_<id>` fallback so the IR attribute is always non-empty
        and diagnostics never need a name-vs-no-name special case."""
        name = self._pipe_net_names.get(id(pipenet))
        if name:
            return name
        return f"net_{pipenet.pipe_net_id}"

    def visit_Assign(self, node):
        """Handle tuple unpacking for TTL functions like core(dims=2)."""
        if not isinstance(node.targets[0], ast.Tuple):
            return super().visit_Assign(node)

        value = self.visit(node.value)
        if not isinstance(value, tuple):
            return super().visit_Assign(node)

        targets = node.targets[0].elts
        if len(value) != len(targets):
            raise ValueError(
                f"Cannot unpack {len(value)} values into {len(targets)} variables"
            )

        for elt, val in zip(targets, value):
            if not isinstance(elt, ast.Name):
                raise ValueError("Tuple unpacking requires simple variable names")
            self._set_var(elt.id, val)

    def _loc_for_node(self, node):
        """Return file location for node if debug_locations enabled, else name location."""
        if self.debug_locations and hasattr(node, "lineno"):
            return _make_file_loc(self.ctx, self.source_file, node, self.line_offset)
        return self.loc

    def _raise_error(self, node, message: str):
        """Raise a TTLangCompileError with source location from AST node."""
        line = node.lineno + self.line_offset if hasattr(node, "lineno") else None
        col = node.col_offset + 1 if hasattr(node, "col_offset") else None
        raise TTLangCompileError(
            message,
            source_file=self.source_file,
            line=line,
            col=col,
        )

    # Auto-profiling helpers for line-based signposting

    def _emit_signpost(self, name: str, is_end: bool = False):
        """Emit a signpost operation into the MLIR."""
        ttl.signpost(name, is_end=is_end)

    def _emit_line_signpost_if_needed(self, node):
        """Emit signposts at line boundaries for auto-profiling."""
        if not self.auto_profile_enabled or not hasattr(node, "lineno"):
            return

        file_lineno = node.lineno + self.line_offset
        if self._current_signpost_line == file_lineno:
            return

        if self._current_signpost_line is not None:
            self._emit_signpost(
                f"{self.name}_L{self._current_signpost_line}", is_end=True
            )

        if self.source_lines and 0 < node.lineno <= len(self.source_lines):
            source_line = self.source_lines[node.lineno - 1].strip()
        else:
            source_line = f"<line {file_lineno}>"

        base_name = f"{self.name}_L{file_lineno}"

        if self.line_mapper:
            self.line_mapper.register_signpost(base_name, file_lineno, source_line)

        self._emit_signpost(base_name)
        self._current_signpost_line = file_lineno

    def _close_final_signpost(self):
        """Close the final signpost at the end of function body."""
        if self.auto_profile_enabled and self._current_signpost_line is not None:
            self._emit_signpost(
                f"{self.name}_L{self._current_signpost_line}", is_end=True
            )
            self._current_signpost_line = None

    def _on_scope_exit(self):
        self._close_final_signpost()

    def _try_emit_auto_signposts(self, node, visit_fn):
        """Emit line-based signposts if auto-profiling is enabled."""
        self._emit_line_signpost_if_needed(node)
        return visit_fn()

    def _emit_op_signposts(self, op_name: str, node, op_fn, implicit=False):
        """Emit signposts for CB operations with op name included."""
        if not self.auto_profile_enabled:
            with self._loc_for_node(node):
                return op_fn()

        file_lineno = node.lineno + self.line_offset
        prefix = "implicit_" if implicit else ""
        base_name = f"{self.name}_L{file_lineno}_{prefix}{op_name}"

        if self.source_lines and 0 < node.lineno <= len(self.source_lines):
            source_line = self.source_lines[node.lineno - 1].strip()
        else:
            source_line = f"<line {file_lineno}>"

        if self.line_mapper:
            self.line_mapper.register_signpost(base_name, file_lineno, source_line)

        with self._loc_for_node(node):
            self._emit_signpost(base_name)
            result = op_fn()
            self._emit_signpost(base_name, is_end=True)
        return result

    def visit_Call(self, node):
        """Override to set location context, catch errors, and inject auto-profiling."""
        with self._loc_for_node(node):
            try:
                # Intercept print() to handle keyword arguments.
                if (
                    not isinstance(node.func, ast.Attribute)
                    and hasattr(node.func, "id")
                    and node.func.id == "print"
                ):
                    return self.visit_Print(node.args, node.keywords)

                # Check for PipeNet.if_src/if_dst calls
                if self._is_pipenet_callback_call(node):
                    return self._handle_pipenet_callback(node)

                # Check for PipeNet.is_src/is_dst/is_active predicate calls
                if self._is_pipenet_predicate_call(node):
                    return self._handle_pipenet_predicate(node)

                return self._try_emit_auto_signposts(
                    node, lambda: super(TTLGenericCompiler, self).visit_Call(node)
                )
            except (ValueError, TypeError, NotImplementedError) as e:
                if isinstance(e, TTLangCompileError):
                    raise
                self._raise_error(node, str(e))

    def visit_AugAssign(self, node):
        """Handle += on tensor blocks via the registered __iadd__ method."""
        with self._loc_for_node(node):
            target = self.visit(node.target)
            if (
                isinstance(node.op, ast.Add)
                and hasattr(target, "type")
                and isinstance(target.type, RankedTensorType)
            ):
                rhs = self.visit(node.value)
                mlir_type = _get_type_str(target.type)
                iadd_fn = self._fn_map.get(f"{mlir_type}.__iadd__")
                if iadd_fn:
                    result = iadd_fn(target, rhs)
                    self._set_var(node.target.id, result)
                    return
            return super().visit_AugAssign(node)

    def _is_pipenet_callback_call(self, node):
        """Check if this is a pipenet.if_src(fn) or pipenet.if_dst(fn) call."""
        if not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr not in ("if_src", "if_dst"):
            return False
        if not isinstance(node.func.value, ast.Name):
            self._raise_error(
                node,
                f"PipeNet.{node.func.attr}() requires a plain variable name "
                f"as receiver (e.g., `net.{node.func.attr}(...)`), "
                f"not an expression",
            )
        var_name = node.func.value.id
        tbl = self._var_exists(var_name)
        if not tbl:
            return False
        val = tbl[var_name]
        from ..pipe import PipeNet

        return isinstance(val, PipeNet)

    _PIPENET_PREDICATE_OPS = {
        "is_src": ttl.is_src,
        "is_dst": ttl.is_dst,
        "is_active": ttl.is_active,
    }

    def _is_pipenet_predicate_call(self, node):
        if not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr not in self._PIPENET_PREDICATE_OPS:
            return False
        if not isinstance(node.func.value, ast.Name):
            return False
        tbl = self._var_exists(node.func.value.id)
        if not tbl:
            return False
        from ..pipe import PipeNet

        return isinstance(tbl[node.func.value.id], PipeNet)

    def _handle_pipenet_predicate(self, node):
        from ..pipe import PipeNet

        method = node.func.attr
        var_name = node.func.value.id
        pipenet = self._var_exists(var_name)[var_name]
        assert isinstance(pipenet, PipeNet)
        if node.args or node.keywords:
            self._raise_error(node, f"PipeNet.{method}() takes no arguments")
        op = self._PIPENET_PREDICATE_OPS[method](
            pipe_net_id=IntegerAttr.get(
                IntegerType.get_signless(64, self.ctx), pipenet.pipe_net_id
            )
        )
        return op

    def _handle_pipenet_callback(self, node):
        """Handle pipenet.if_src(callback) or pipenet.if_dst(callback) calls."""
        from ..pipe import PipeNet, SrcPipeIdentity, DstPipeIdentity

        method_name = node.func.attr
        var_name = node.func.value.id
        tbl = self._var_exists(var_name)
        pipenet = tbl[var_name]

        # Get the callback argument
        if len(node.args) != 1:
            self._raise_error(
                node, f"PipeNet.{method_name}() requires exactly one callback argument"
            )
        callback_node = node.args[0]

        # Support lambda or named function reference
        if isinstance(callback_node, ast.Lambda):
            callback_body = callback_node.body
            if len(callback_node.args.args) != 1:
                self._raise_error(
                    callback_node,
                    f"PipeNet.{method_name}() callback must take exactly one argument (pipe)",
                )
            pipe_param_name = callback_node.args.args[0].arg
        elif isinstance(callback_node, ast.Name):
            fn_name = callback_node.id
            fn_table = self._var_exists(fn_name)
            if not fn_table:
                self._raise_error(callback_node, f"'{fn_name}' not found in scope")
            fn_def = fn_table[fn_name]
            if not isinstance(fn_def, ast.FunctionDef):
                self._raise_error(
                    callback_node,
                    f"PipeNet.{method_name}() requires a function, "
                    f"got {type(fn_def).__name__}",
                )
            if len(fn_def.args.args) != 1:
                self._raise_error(
                    callback_node,
                    f"PipeNet.{method_name}() callback must take exactly one argument (pipe)",
                )
            pipe_param_name = fn_def.args.args[0].arg
            callback_body = fn_def.body
        else:
            self._raise_error(
                callback_node,
                f"PipeNet.{method_name}() requires a lambda or function reference",
            )

        # Resolve the user's variable name for this PipeNet so the
        # verifier can render diagnostics in user-facing terms.
        # `_resolve_pipe_net_name` falls back to `net_<id>` if the
        # PipeNet wasn't bound to a named variable, so the attribute
        # is always non-empty.
        pipe_net_name = self._resolve_pipe_net_name(pipenet)

        # Iterate over all pipes and emit if_src/if_dst for each
        decl_file = getattr(pipenet, "_source_file", None)
        decl_line = getattr(pipenet, "_source_line", None)
        for pipe in pipenet.pipes:
            # Emit the pipe MLIR value
            pipe_val = self._emit_pipe_from_capture(
                pipe,
                pipe_net_name=pipe_net_name,
                source_file=decl_file,
                source_line=decl_line,
            )
            pipe._mlir_value = pipe_val

            # Create the appropriate PipeIdentity
            if method_name == "if_src":
                pipe_identity = SrcPipeIdentity(pipe)
                op = ttl.if_src(pipe_val)
            else:
                pipe_identity = DstPipeIdentity(pipe)
                op = ttl.if_dst(pipe_val)

            # Create body block and compile callback inside
            block = Block.create_at_start(op.body)
            with InsertionPoint(block):
                # Bind the pipe parameter to the MLIR pipe value.
                # TODO: bind to PipeIdentity instead so .src/.dst work
                # on the callback parameter per the spec.
                self.symbol_tables.append({})
                self.symbol_tables[-1][pipe_param_name] = pipe_val
                self.symbol_tables[-1][f"__{pipe_param_name}_identity"] = pipe_identity

                if isinstance(callback_body, list):
                    for stmt in callback_body:
                        self.visit(stmt)
                else:
                    self.visit(callback_body)

                self.symbol_tables.pop()
                ttl.yield_()

        return None  # Statement, no return value

    def visit_BinOp(self, node):
        """Override to inject auto-profiling and provide better error messages."""
        with self._loc_for_node(node):
            try:
                return self._try_emit_auto_signposts(
                    node, lambda: super(TTLGenericCompiler, self).visit_BinOp(node)
                )
            except (ValueError, TypeError, NotImplementedError) as e:
                if isinstance(e, TTLangCompileError):
                    raise
                self._raise_error(node, str(e))

    def visit_Compare(self, node):
        """Attach the comparison's AST source location to the emitted
        `arith.cmpi`, so verifier and runtime diagnostics that reference the
        predicate point at the comparison itself rather than the enclosing
        function or block."""
        with self._loc_for_node(node):
            try:
                return super(TTLGenericCompiler, self).visit_Compare(node)
            except (ValueError, TypeError, NotImplementedError) as e:
                if isinstance(e, TTLangCompileError):
                    raise
                self._raise_error(node, str(e))

    def visit_Name(self, node):
        """Override to check function globals for simple constants."""
        result = super().visit_Name(node)
        if result is not None:
            return result

        # Check if it's a module-level constant
        var_name = node.id
        if var_name in self.fn_globals:
            val = self.fn_globals[var_name]
            if isinstance(val, int):
                return arith.ConstantOp(
                    IntegerType.get_signless(64, self.ctx), val
                ).result
            if isinstance(val, float):
                return arith.ConstantOp(F32Type.get(self.ctx), val).result

        return None

    def _is_ttl_module_access(self, node):
        """Check if node is ttl.XXX access pattern."""
        return isinstance(node.value, ast.Name) and node.value.id == "ttl"

    def _is_ttl_math_access(self, node):
        """Check if node is ttl.math.XXX access pattern."""
        return (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "ttl"
            and node.value.attr == "math"
        )

    def _is_ttl_block_access(self, node):
        """Check if node is ttl.block.XXX access pattern."""
        return (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "ttl"
            and node.value.attr == "block"
        )

    # Spec change-log 0.17 (TTLangSpecification.md) moved these names from
    # ttl.math/ttl to the ttl.block namespace. Each entry restricts the names
    # to the listed namespace; calls under another namespace raise a clear
    # error pointing at the correct one.
    _NAMESPACE_OVERRIDES = {
        "broadcast": "ttl.block",
        "fill": "ttl.block",
    }

    def _resolve_ttl_function(self, node, func_args, kwargs):
        """Resolve and call a ttl.XXX, ttl.math.XXX, or ttl.block.XXX function."""
        if self._is_ttl_module_access(node):
            namespace = "ttl"
        elif self._is_ttl_math_access(node):
            namespace = "ttl.math"
        elif self._is_ttl_block_access(node):
            namespace = "ttl.block"
        else:
            return None

        required_namespace = self._NAMESPACE_OVERRIDES.get(node.attr)
        if required_namespace is not None and namespace != required_namespace:
            self._raise_error(
                node,
                f"{namespace}.{node.attr} is not available; use "
                f"{required_namespace}.{node.attr}",
            )

        fn = self._fn_map.get(node.attr)
        if fn is None:
            self._raise_error(node, f"Unknown function: {namespace}.{node.attr}")
        return fn(*func_args, **kwargs)

    def _resolve_chained_method_call(self, node, func_args, kwargs):
        """Handle chained calls like foo().bar() where node.value is a Call."""
        mlir_value = self.visit(node.value)
        if mlir_value is None:
            self._raise_error(node, "Chained call returned no value")
        mlir_type = _get_type_str(mlir_value.type)
        qualified_object_syntax = f"{mlir_type}.{node.attr}"
        fn = self._fn_map.get(qualified_object_syntax, None)
        if fn is None:
            self._raise_error(node, f"No method '{node.attr}' on type {mlir_type}")
        return fn(mlir_value, *func_args, **kwargs)

    def visit_Attribute(self, node, func_args=[], kwargs={}):
        """Override to set location context and catch errors for method calls."""
        with self._loc_for_node(node):
            try:
                # Handle ttl.XXX and ttl.math.XXX attribute access
                if (
                    self._is_ttl_module_access(node)
                    or self._is_ttl_math_access(node)
                    or self._is_ttl_block_access(node)
                ):
                    return self._resolve_ttl_function(node, func_args, kwargs)
                # Tensor-typed .shape: return the value's grid shape as a
                # Python tuple of ints. Lets users write `y_blk.shape` inside
                # @ttl.compute / @ttl.datamovement to derive shape kwargs for
                # spec-form ops like ttl.block.broadcast(..., shape=y_blk.shape).
                # Resolved before the chained-call and module-attribute branches
                # so it also works on call expressions whose result is a ranked
                # tensor. Non-tensor receivers fall through to the existing
                # handlers and surface their normal diagnostic.
                if not func_args and not kwargs and node.attr == "shape":
                    value = self.visit(node.value)
                    if value is not None and hasattr(value, "type"):
                        tensor_ty = RankedTensorType.maybe_downcast(value.type)
                        if tensor_ty is not None:
                            return tuple(tensor_ty.shape)
                # Handle chained method calls: expr().method()
                if isinstance(node.value, ast.Call):
                    return self._resolve_chained_method_call(node, func_args, kwargs)
                # When a module attribute like `torch.float32` appears as an
                # *argument value* (e.g., `ttl.math.typecast(x, torch.float32)`),
                # parent `visit_Call` visits the argument node and dispatches
                # here with empty func_args/kwargs. In that case we return the
                # underlying Python object so downstream syntax handlers
                # receive the real dtype instead of tripping the base
                # visitor's "expression does not produce a value" diagnostic.
                #
                # Restricted to non-callable globals so that no-arg call
                # targets like `torch.zeros()` still fall through to the base
                # visitor and get evaluated as actual calls. Treating any
                # module attribute as a value would silently substitute the
                # function reference for its result.
                if (
                    not func_args
                    and not kwargs
                    and isinstance(node.value, ast.Name)
                    and node.value.id in self.fn_globals
                    and hasattr(self.fn_globals[node.value.id], node.attr)
                ):
                    candidate = getattr(self.fn_globals[node.value.id], node.attr)
                    if not callable(candidate):
                        return candidate
                return super().visit_Attribute(node, func_args, kwargs)
            except (ValueError, TypeError, NotImplementedError) as e:
                if isinstance(e, TTLangCompileError):
                    raise
                self._raise_error(node, str(e))

    def visit_Subscript(self, node):
        """Handle tensor[row, col] or tensor[r0:r1, c0:c1] indexing."""
        tbl = self._var_exists(node.value.id)
        if not tbl:
            self._raise_error(node, f"Unknown variable: {node.value.id}")

        tensor = tbl[node.value.id]
        if not isinstance(getattr(tensor, "type", None), RankedTensorType):
            self._raise_error(node, "TTL only supports subscripting tensors")

        if isinstance(node.slice, ast.Tuple):
            indices = [self._build_index_or_range(elt) for elt in node.slice.elts]
        else:
            indices = [self._build_index_or_range(node.slice)]

        return (tensor, indices)

    def _to_index_value(self, node):
        """Convert AST node to MLIR index Value."""
        if isinstance(node, ast.Constant):
            return arith.ConstantOp(IndexType.get(self.ctx), node.value)
        val = self.visit(node)
        if isinstance(val.type, IndexType):
            return val
        return arith.IndexCastOp(IndexType.get(self.ctx), val)

    def _build_index_or_range(self, node):
        """Convert AST node to (start_value, is_range) tuple.

        For slice syntax (start:end), returns (start_value, True).
        For index syntax (value), returns (value, False).
        """
        if isinstance(node, ast.Slice):
            if node.lower is None:
                self._raise_error(node, "Slice must have explicit start index")
            if node.upper is None:
                self._raise_error(node, "Slice must have explicit stop index")
            if node.step is not None:
                self._raise_error(node, "Slice step is not supported")
            start_val = self._to_index_value(node.lower)
            return (start_val, True)
        else:
            return (self._to_index_value(node), False)

    # Override to use i64 for all integer constants (attributes or not)
    # TTL/TTKernel ops require i64, and this reduces casts throughout the pipeline
    def visit_Constant(self, node):
        as_attr = getattr(node, "_ttkernel_as_attr", False)
        op_constructor = IntegerAttr.get if as_attr else arith.ConstantOp
        if callable(as_attr):
            return as_attr(node)
        elif isinstance(node.value, bool):
            return op_constructor(IntegerType.get_signless(1, self.ctx), node.value)
        elif isinstance(node.value, int):
            return op_constructor(IntegerType.get_signless(64, self.ctx), node.value)
        elif isinstance(node.value, float):
            f32 = F32Type.get(self.ctx)
            if as_attr:
                return FloatAttr.get(f32, node.value)
            return arith.ConstantOp(f32, node.value)
        elif isinstance(node.value, str):
            return node.value
        else:
            self._raise_error(
                node, f"constant type {type(node.value).__name__} not implemented"
            )

    def visit_UnaryOp(self, node):
        # Fold -float_literal to a negative float constant instead of emitting
        # emitc.unary_minus on a positive constant.
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, float):
                neg_node = ast.copy_location(
                    ast.Constant(value=-node.operand.value), node
                )
                return self.visit_Constant(neg_node)
        return super().visit_UnaryOp(node)

    def _signed_int_literal(self, elt: ast.AST) -> Optional[int]:
        """Fold a signed integer literal (e.g. ``-1`` in ``dims=[-1]``).

        ``dims=[-1]`` parses as ``UnaryOp(USub, Constant(1))``, not ``Constant(-1)``.
        Uses structural pattern matching so nested unary (e.g. ``-(-1)``) folds too.
        """
        match elt:
            case ast.Constant(value=v) if type(v) is int:
                return v
            case ast.UnaryOp(op=ast.USub(), operand=inner):
                n = self._signed_int_literal(inner)
                return None if n is None else -n
            case ast.UnaryOp(op=ast.UAdd(), operand=inner):
                return self._signed_int_literal(inner)
            case _:
                return None

    def visit_List(self, node):
        """Parse a list of constants. Returns a Python list, not MLIR values."""
        result = []
        for elt in node.elts:
            v = self._signed_int_literal(elt)
            if v is None:
                self._raise_error(elt, "list elements must be constants")
            result.append(v)
        return result

    def _emit_cb_from_capture(self, cb):
        """Emit ttl.bind_cb for a captured DataflowBuffer instance."""
        ttcore_dtype = tensor_dtype_to_ttcore_datatype(cb.dtype)
        element_type = ttcore.ir.TileType.get(
            self.ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
        )
        cb_type = ttl.CircularBufferType.get(
            self.ctx,
            list(cb.shape),
            element_type,
            cb.block_count,
        )
        # Emit: %cb = ttl.bind_cb {cb_index = N, block_count = M} : !ttl.cb<...>
        return ttl.bind_cb(cb_type, cb._cb_index, block_count=cb.block_count)

    def _emit_pipe_from_capture(
        self, pipe, pipe_net_name=None, source_file=None, source_line=None
    ):
        """Emit ttl.create_pipe for a captured Pipe instance.

        `pipe_net_name`, when provided, becomes the `pipeNetName` attr
        on `ttl.create_pipe` and renders in verifier diagnostics
        verbatim. Callers pass the user's Python variable name
        (e.g. `a_pipe_net`) recovered from `_pipe_net_names`.

        `source_file` / `source_line` come from the `PipeNet([...])`
        construction site captured by `PipeNet.__init__`. When set, the
        op carries that location so the verifier's "PipeNet declared
        here" note points at the user's declaration rather than the
        first `if_src`/`if_dst` call site.
        """
        pipe_type = ttl.PipeType.get(
            self.ctx,
            pipe.src[0],
            pipe.src[1],
            pipe.dst_start[0],
            pipe.dst_start[1],
            pipe.dst_end[0],
            pipe.dst_end[1],
            pipe.pipe_net_id,
        )
        kwargs = {}
        if pipe_net_name:
            kwargs["pipe_net_name"] = pipe_net_name
        if pipe.is_collective:
            kwargs["is_collective"] = True
        if source_file and source_line is not None:
            kwargs["loc"] = Location.file(source_file, source_line, 1, self.ctx)
        return ttl.create_pipe(
            pipe_type,
            pipe.src[0],
            pipe.src[1],
            pipe.dst_start[0],
            pipe.dst_start[1],
            pipe.dst_end[0],
            pipe.dst_end[1],
            pipe.pipe_net_id,
            **kwargs,
        )

    def _emit_entry(self, node):
        assert not self.func_entry, "Cannot declare function within a function"

        if node.args.args:
            self._raise_error(
                node,
                "Thread functions must have no parameters. "
                "Use make_dataflow_buffer_like() in kernel body and capture CBs in closures.",
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

        # Add ttl module to symbol table.
        self._set_var("ttl", ttl)

        # Ensure TTL dialect is registered for type parsing
        ttl.ensure_dialects_registered(self.ctx)

        self.module_symbol_table = SymbolTable(self.module.operation)

        # Emit function body
        with InsertionPoint(func_bb):
            # Map TensorAccessor function arguments to symbol table.
            for i, name in enumerate(self._tensor_accessor_names):
                self._set_var(name, func_bb.arguments[i])
                self.streams.add(name)

            # Prepopulate other captures (non-tensor).
            from ..dataflow_buffer import DataflowBuffer
            from ..pipe import Pipe, PipeNet

            for name, val in self.captures.items():
                if is_ttnn_tensor(val):
                    continue  # Already handled via function arguments
                assert isinstance(name, str)
                if isinstance(val, int):
                    self._set_var(name, arith.ConstantOp(IndexType.get(self.ctx), val))
                elif isinstance(val, float):
                    self._set_var(name, arith.ConstantOp(F32Type.get(self.ctx), val))
                elif isinstance(val, DataflowBuffer):
                    self._set_var(name, self._emit_cb_from_capture(val))
                elif isinstance(val, Pipe):
                    pipe_val = self._emit_pipe_from_capture(val)
                    self._set_var(name, pipe_val)
                    val._mlir_value = pipe_val
                elif isinstance(val, PipeNet):
                    self._set_var(name, val)
                    # Stamp variable name (first-seen wins) so the
                    # compiler can use it in diagnostics.
                    self._pipe_net_names.setdefault(id(val), name)
                else:
                    self._raise_error(
                        node, f"Invalid capture type for var {name}: {type(val)}"
                    )

            # Module-scope PipeNets satisfy the spec's enclosing-scope rule
            # (the module is an enclosing scope of the @ttl.operation
            # function). Pre-bind them so `NAME.if_src(...)` resolves.
            # Captures take precedence: a closure cell shadows a global
            # of the same name.
            for name, val in self.fn_globals.items():
                if not isinstance(val, PipeNet):
                    continue
                if any(name in tbl for tbl in self.symbol_tables):
                    continue
                self._set_var(name, val)
                self._pipe_net_names.setdefault(id(val), name)

            for target in node.body:
                self.visit(target)

            self._close_final_signpost()
            func.ReturnOp([])

        self.symbol_tables.pop()

    def visit_FunctionDef(self, node):
        with self._loc_for_node(node):
            # Nested function defs are stored as callback ASTs for PipeNet
            if self._is_nested_function_def():
                self._store_callback_def(node)
                return
            return self._emit_entry(node)

    def visit_AsyncFunctionDef(self, node):
        with self._loc_for_node(node):
            return self._emit_entry(node)

    # Thread required by each dprint mode in compute context.
    # TileSlice errors on math; dst register reads require math.
    # Tensor mode is not available in compute (uses get_read_ptr).
    _COMPUTE_THREAD_FOR_MODE = {
        "scalar": "math",
        "cb": "pack",
        "tile": "pack",
        "dst": "math",
    }

    def _resolve_print_thread(self, mode, thread):
        """Pick the correct thread for a dprint in compute context.

        Returns the thread unchanged for datamovement kernels or when
        the user provided an explicit thread kwarg.
        """
        if thread is not None or self.kernel_type != "compute":
            return thread
        resolved = self._COMPUTE_THREAD_FOR_MODE.get(mode)
        if resolved is None:
            raise ValueError(f"unknown dprint mode '{mode}' for thread resolution")
        return resolved

    def _extract_print_kwargs(self, keywords):
        kwargs = {}
        for kw in keywords:
            if not isinstance(kw.value, ast.Constant):
                raise ValueError(f"print() keyword '{kw.arg}' must be a constant")
            kwargs[kw.arg] = kw.value.value
        return kwargs

    def visit_Print(self, args, keywords=None):
        keywords = keywords or []
        kwargs = self._extract_print_kwargs(keywords)

        thread = kwargs.get("thread")
        if thread is not None and thread not in ("math", "pack", "unpack"):
            raise ValueError(
                f"print() thread must be 'math', 'pack', or 'unpack', "
                f"got '{thread}'"
            )

        num_pages = kwargs.get("num_pages")
        if num_pages is not None and not isinstance(num_pages, int):
            raise ValueError(
                f"print() num_pages must be an integer, "
                f"got {type(num_pages).__name__}"
            )

        # DST mode: print(_dump_dst_registers=True, label="after exp")
        if kwargs.get("_dump_dst_registers"):
            if args:
                raise ValueError(
                    "print(_dump_dst_registers=True) takes no positional arguments"
                )
            label = kwargs.get("label", "")
            ttl.dprint(
                fmt=label,
                mode="dst",
                argv=[],
                thread=self._resolve_print_thread("dst", thread),
                num_pages=None,
            )
            return

        if not args:
            raise ValueError(
                "print() requires at least one argument "
                "(or _dump_dst_registers=True)"
            )

        # Visit all args once to determine types.
        # Each entry is (kind, const_val, mlir_val, name).
        visited = []
        for arg in args:
            if isinstance(arg, ast.Constant):
                visited.append(("const", arg.value, None, None))
            elif isinstance(arg, ast.Name):
                val = self.visit(arg)
                visited.append(("var", None, val, arg.id))
            else:
                raise ValueError(
                    f"print() argument type {type(arg).__name__} " f"not supported"
                )

        # Check if the last variable arg is a TT-Lang object (CB, block,
        # or tensor). If so, emit a scalar label for any preceding args
        # then the appropriate object print. This supports patterns like
        # print("C: ", C, num_pages=2) from the spec.
        last_var_idx = None
        for i in range(len(visited) - 1, -1, -1):
            if visited[i][0] == "var":
                last_var_idx = i
                break

        if last_var_idx is not None:
            _, _, last_var, last_name = visited[last_var_idx]
            is_tensor_accessor = last_name is not None and last_name in self.streams
            if self._is_object_printable(last_var, num_pages):
                prefix = visited[:last_var_idx]
                if prefix:
                    self._emit_scalar_print(prefix, thread)
                self._emit_object_print(last_var, thread, num_pages, is_tensor_accessor)
                return

        # Scalar mode: string/int/float constants and integer variables.
        self._emit_scalar_print(visited, thread)

    def _is_object_printable(self, val, num_pages):
        """Check if val is a CB, block/tile, or tensor suitable for
        object-mode dprint."""
        if ttl.CircularBufferType.maybe_downcast(val.type) is not None:
            return True
        if isinstance(val.type, RankedTensorType):
            return True
        return False

    def _emit_object_print(self, val, thread, num_pages, is_tensor_accessor=False):
        """Emit the appropriate object-mode dprint for val."""
        cb_type = ttl.CircularBufferType.maybe_downcast(val.type)
        if cb_type is not None:
            ttl.dprint(
                fmt="",
                mode="cb",
                argv=[val],
                thread=self._resolve_print_thread("cb", thread),
                num_pages=None,
            )
            return

        if isinstance(val.type, RankedTensorType):
            if is_tensor_accessor:
                # Tensor accessors use page-based printing (spec: num_pages
                # defaults to 1). TileSlice is not available for raw tensors.
                if self.kernel_type == "compute":
                    raise ValueError(
                        "print(tensor) is only supported in " "datamovement kernels"
                    )
                ttl.dprint(
                    fmt="",
                    mode="tensor",
                    argv=[val],
                    thread=self._resolve_print_thread("tensor", thread),
                    num_pages=num_pages if num_pages is not None else 1,
                )
            elif num_pages is not None:
                # CB-backed block with explicit num_pages: page-based printing.
                if self.kernel_type == "compute":
                    raise ValueError(
                        "print(block, num_pages=N) is only supported in "
                        "datamovement kernels"
                    )
                ttl.dprint(
                    fmt="",
                    mode="tensor",
                    argv=[val],
                    thread=self._resolve_print_thread("tensor", thread),
                    num_pages=num_pages,
                )
            else:
                # CB-backed block without num_pages: tile-based printing.
                ttl.dprint(
                    fmt="",
                    mode="tile",
                    argv=[val],
                    thread=self._resolve_print_thread("tile", thread),
                    num_pages=None,
                )

    def _emit_scalar_print(self, visited, thread):
        """Emit a scalar-mode dprint from a list of visited args."""
        fmt = ""
        argv = []
        for kind, const_val, val, _name in visited:
            if kind == "const":
                if not isinstance(const_val, (str, int, float)):
                    raise ValueError(
                        f"print() supports string, integer, and float "
                        f"constants, got {type(const_val).__name__}"
                    )
                fmt += str(const_val) + " "
            else:
                if not (
                    isinstance(val.type, IndexType) or isinstance(val.type, IntegerType)
                ):
                    raise ValueError(
                        f"print() scalar mode supports integer variables, "
                        f"got {val.type}"
                    )
                fmt += "{} "
                argv.append(val)

        fmt = fmt.strip()
        ttl.dprint(
            fmt=fmt,
            mode="scalar",
            argv=argv,
            thread=self._resolve_print_thread("scalar", thread),
            num_pages=None,
        )

    def _is_nested_function_def(self):
        """Check if we're inside a function body (nested def, not entry)."""
        return self.func_entry is not None

    def _store_callback_def(self, node):
        """Store a nested function def AST for use as a PipeNet callback."""
        self.symbol_tables[-1][node.name] = node

    def _get_cb_tensor_type(self, cb_val, node=None):
        """Extract the tensor type from a TTL CB type."""
        cb_type = ttl.CircularBufferType.maybe_downcast(cb_val.type)
        if cb_type is None:
            msg = f"Expected CircularBufferType, got {cb_val.type}"
            if node is not None:
                self._raise_error(node, msg)
            raise ValueError(msg)
        return RankedTensorType.get(cb_type.shape, cb_type.element_type)

    def _is_signpost_call(self, context_expr):
        """Check if a with-item context expression is a signpost call."""
        if not isinstance(context_expr, ast.Call):
            return False
        func = context_expr.func
        # with signpost("name"):
        if isinstance(func, ast.Name) and func.id == "signpost":
            return True
        # with ttl.signpost("name"):
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "signpost"
            and isinstance(func.value, ast.Name)
            and func.value.id == "ttl"
        ):
            return True
        return False

    def _extract_signpost_name(self, context_expr):
        """Extract and validate the string name from a signpost call."""
        if len(context_expr.args) != 1 or context_expr.keywords:
            self._raise_error(
                context_expr, "signpost() requires exactly one string argument"
            )
        name_arg = context_expr.args[0]
        if not isinstance(name_arg, ast.Constant) or not isinstance(
            name_arg.value, str
        ):
            self._raise_error(
                context_expr, "signpost() argument must be a string literal"
            )
        name = name_arg.value
        if not name.replace("_", "").replace("-", "").isalnum():
            self._raise_error(
                context_expr,
                f"signpost name must contain only alphanumeric characters, "
                f"underscores, or hyphens, got: '{name}'",
            )
        return name

    def _collect_pipenet_roles_in_body(self, body):
        """Return PipeNet role requirements referenced by if_src/if_dst calls."""
        from ..pipe import PipeNet

        roles = []
        seen = set()
        for stmt in body:
            for child in ast.walk(stmt):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if not isinstance(func, ast.Attribute):
                    continue
                if func.attr not in ("if_src", "if_dst"):
                    continue
                if not isinstance(func.value, ast.Name):
                    continue
                table = self._var_exists(func.value.id)
                if not table:
                    continue
                pipenet = table[func.value.id]
                if not isinstance(pipenet, PipeNet):
                    continue
                role = 0 if func.attr == "if_src" else 1
                item = (pipenet.pipe_net_id, role)
                if item in seen:
                    continue
                seen.add(item)
                roles.append(item)
        return roles

    def _emit_pipenet_scope(self, roles):
        """Create a ttl.pipenet_scope op with role attributes."""
        scope_op = ttl.pipenet_scope()
        ids = [pipe_net_id for pipe_net_id, _ in roles]
        role_values = [role for _, role in roles]
        scope_op.operation.attributes["ttl.pipe_net_ids"] = ArrayAttr.get(
            [
                IntegerAttr.get(IntegerType.get_signless(64, self.ctx), value)
                for value in ids
            ],
            self.ctx,
        )
        scope_op.operation.attributes["ttl.pipe_net_roles"] = ArrayAttr.get(
            [
                IntegerAttr.get(IntegerType.get_signless(64, self.ctx), value)
                for value in role_values
            ],
            self.ctx,
        )
        return scope_op

    def _emit_cb_with_body(self, node):
        """Emit CB acquire/release ops for a with statement body."""
        # Process each with-item: acquire resources and track for release
        releases = []  # [(release_op, cb_val), ...] in acquisition order

        self._on_scope_exit()

        for item in node.items:
            context_expr = item.context_expr
            optional_vars = item.optional_vars

            if not isinstance(context_expr, ast.Call):
                self._raise_error(
                    context_expr,
                    "'with' requires a method call (e.g., cb.reserve())",
                )

            if not isinstance(context_expr.func, ast.Attribute):
                self._raise_error(
                    context_expr, "'with' requires a method call on an object"
                )

            method_name = context_expr.func.attr
            cb_node = context_expr.func.value

            if method_name not in ("reserve", "wait"):
                self._raise_error(
                    context_expr,
                    f"'with' only supports 'reserve()' or 'wait()', got '{method_name}'",
                )

            if not isinstance(cb_node, ast.Name):
                self._raise_error(
                    context_expr,
                    "'with' requires a simple variable (e.g., cb.reserve())",
                )

            cb_table = self._var_exists(cb_node.id)
            if not cb_table:
                self._raise_error(cb_node, f"'{cb_node.id}' not found in scope")
            cb_val = cb_table[cb_node.id]

            # Get tensor type from CB for reserve/wait result
            tensor_type = self._get_cb_tensor_type(cb_val, node=context_expr)
            if method_name == "reserve":
                tensor = self._emit_op_signposts(
                    "cb_reserve",
                    context_expr,
                    lambda tt=tensor_type, cv=cb_val: ttl.cb_reserve(tt, cv),
                )
                releases.append(("cb_push", ttl.cb_push, cb_val, context_expr))
            else:  # wait
                tensor = self._emit_op_signposts(
                    "cb_wait",
                    context_expr,
                    lambda tt=tensor_type, cv=cb_val: ttl.cb_wait(tt, cv),
                )
                releases.append(("cb_pop", ttl.cb_pop, cb_val, context_expr))

            # Attach CB to tensor so store() can find the CB association
            acquire_result = ttl.attach_cb(tensor.type, tensor, cb_val)

            if optional_vars is not None:
                if not isinstance(optional_vars, ast.Name):
                    self._raise_error(
                        optional_vars,
                        "'with ... as var' requires a simple variable name",
                    )
                self._set_var(optional_vars.id, acquire_result)

        for stmt in node.body:
            self.visit(stmt)

        self._on_scope_exit()

        # Release in reverse order (implicit ops from with statement)
        for op_name, release_op, cb_val, expr_node in reversed(releases):
            self._emit_op_signposts(
                op_name,
                expr_node,
                lambda ro=release_op, cv=cb_val: ro(cv),
                implicit=True,
            )

    def visit_With(self, node):
        """
        Handle 'with' for DataflowBuffer acquire/release or signpost scopes.

        Signpost scopes:
            with ttl.signpost("my_region"):
                ...  # emits _before/_after signpost pair

        CB acquire/release:
            with lhs_cb.wait() as l, rhs_cb.wait() as r, out_cb.reserve() as o:
                ...
                # releases in reverse order: push(out), pop(rhs), pop(lhs)
        """
        with self._loc_for_node(node):
            # Check for signpost scope
            first_item = node.items[0]
            if self._is_signpost_call(first_item.context_expr):
                if len(node.items) > 1:
                    self._raise_error(
                        node,
                        "signpost() cannot be combined with other with-items",
                    )
                if first_item.optional_vars is not None:
                    self._raise_error(
                        node, "signpost() does not produce a value ('as' not supported)"
                    )
                name = self._extract_signpost_name(first_item.context_expr)
                if self.auto_profile_enabled:
                    import warnings

                    warnings.warn(
                        f"signpost('{name}') ignored: user-defined signposts "
                        "are disabled when TTLANG_AUTO_PROFILE=1. "
                        "Run one profiling mode at a time.",
                        stacklevel=2,
                    )
                    for stmt in node.body:
                        self.visit(stmt)
                    return
                self._on_scope_exit()
                self._emit_signpost(f"ttl_{name}")
                for stmt in node.body:
                    self.visit(stmt)
                self._on_scope_exit()
                self._emit_signpost(f"ttl_{name}", is_end=True)
                return

            # Only `reserve()` blocks pipe-couple their CB: the body fills
            # the reserved block on the role-gated nodes (sender writes
            # locally then `if_src(send)`; receiver does `if_dst(recv)`
            # which writes from the pipe). `wait()` blocks consume a CB
            # filled by some other thread and may sit unguarded next to
            # ancillary pipe ops, so wrapping them over-constrains.
            has_reserve = any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "reserve"
                for item in node.items
            )
            roles = (
                self._collect_pipenet_roles_in_body(node.body) if has_reserve else []
            )
            if roles:
                scope_op = self._emit_pipenet_scope(roles)
                block = Block.create_at_start(scope_op.body)
                with InsertionPoint(block):
                    self._emit_cb_with_body(node)
                    ttl.yield_()
                return

            self._emit_cb_with_body(node)


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
