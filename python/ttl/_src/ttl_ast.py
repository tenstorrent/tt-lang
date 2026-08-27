# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import struct
from dataclasses import dataclass
from typing import List, Optional, Set

from ttl.pykernel._src.kernel_ast import (
    TTCompilerBase,
    _get_single_result,
    _require_mlir_value_type,
)
from ttl.pykernel._src.utils import _get_type_str
from ttl.dialects import arith, func, memref, scf, ttcore, ttkernel
from ttl.ir import *

from ..constants import DEFAULT_TILE_SIZE
from ..condition import DispatchCondition, _BoundDispatchCondition
from ..dfb_reset import DFBReset, _BoundDFBReset
from ..dfb_reconfiguration import (
    DFBReconfiguration,
    _BoundDFBReconfiguration,
)
from ..diagnostics import TTLangCompileError
from ttl.dialects import ttl
from ..dtype_utils import is_ttnn_tensor, tensor_dtype_to_ttcore_datatype
from ..layouts import (
    LayoutConfig,
    create_layout,
    detect_memory_layout,
    TENSOR_MEMORY_LAYOUT_INTERLEAVED,
)
from ..kernel import (
    Kernel,
    KernelKind,
    _DFB_RELEASE_METHODS,
    _selector_implicit_role,
    _selector_kind,
    _selector_sort_key,
)
from ..scalar import ScalarType
from ..ttl_utils import get_thread_type_string
from .auto_profile import (
    get_line_mapper,
    is_auto_profile_enabled,
)
from .atom_inline import _DFB_SOURCE_OCCURRENCE
from .global_semaphore import (
    get_ttnn_global_semaphore_address,
    is_ttnn_global_semaphore,
)
from .tensor_registry import get_tensor_global_index, get_tensor_source

# Use the same 4096-item scale as other bounded static enumerations in the
# compiler. External protocol summaries are expected to be much shorter; this
# policy limit bounds frontend memory and downstream per-effect analysis rather
# than hardware behavior.
_MAX_EXPANDED_EXTERNAL_DFB_EFFECTS = 4096


def _saturating_add_expanded_dfb_effect_count(
    current_effect_count: int, body_effect_count: int, repeat_count: int
) -> int:
    """Return the cumulative count, saturated at one past the limit.

    Args:
        current_effect_count: Actions preceding this repeat.
        body_effect_count: Flattened actions in one copy of the repeat body.
        repeat_count: Number of body copies.

    Returns:
        The exact cumulative count within the limit, otherwise the fixed
        over-limit sentinel.

    Saturation prevents nested repeats from multiplying arbitrarily large
    Python integers before the final over-limit diagnostic.
    """
    if current_effect_count > _MAX_EXPANDED_EXTERNAL_DFB_EFFECTS:
        return _MAX_EXPANDED_EXTERNAL_DFB_EFFECTS + 1
    if repeat_count == 0:
        return current_effect_count
    remaining_effect_count = _MAX_EXPANDED_EXTERNAL_DFB_EFFECTS - current_effect_count
    if body_effect_count > remaining_effect_count // repeat_count:
        return _MAX_EXPANDED_EXTERNAL_DFB_EFFECTS + 1
    return current_effect_count + body_effect_count * repeat_count


_MISSING_STATIC_VALUE = object()


@dataclass(frozen=True)
class _ExternalTemplateArg:
    """Separate compile-time payloads from DFB values needed by allocation."""

    kind: object
    value: object
    dfb_source_occurrence: Optional[str] = None


@dataclass(frozen=True)
class _ExternalDFBDependency:
    dfb: object
    source_occurrence: Optional[str]


@dataclass(frozen=True)
class _ExternalDFBEffect:
    """One parsed external-call DFB effect before dependency indexing."""

    kind: object
    dfb: object
    num_tiles: int
    source_occurrence: Optional[str]


@dataclass(frozen=True)
class _ExternalDFBEffectRepeat:
    """One parsed repeat whose body has not been materialized."""

    count: int
    effects: tuple[object, ...]


@dataclass(frozen=True)
class _ExternalDFBAccess:
    """One parsed external-call non-transactional DFB access."""

    kind: object
    dfb: object
    source_occurrence: Optional[str]


def _make_file_loc(ctx, source_file: str, node, line_offset: int = 0) -> Location:
    """Create an MLIR file location from an AST node."""
    if not hasattr(node, "lineno"):
        raise ValueError(f"AST node {type(node).__name__} has no line number")
    return Location.file(
        source_file, node.lineno + line_offset, node.col_offset + 1, ctx
    )


@dataclass(frozen=True)
class _GuardedDFBBlock:
    """DFB block value defined only when `guard` is true."""

    value: object
    dfb: object
    guard: object
    guard_description: str
    acquire_method: str

    _ttlang_guarded_dfb_block = True


@dataclass(frozen=True)
class _GuardedDFBAssignment:
    """Branch-local DFB acquire that becomes a guarded outer binding."""

    name: str
    dfb: object
    tensor_type: object
    acquire_method: str
    node: ast.Assign


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

    tile = (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)
    if is_ttnn_tensor(tensor) and hasattr(tensor, "get_tile"):
        tile = tuple(tensor.get_tile().tile_shape)

    layout = create_layout(
        ctx,
        LayoutConfig(
            logical_shape=shape,
            grid=grid,
            dtype=tensor.dtype,
            memory_layout=mem_layout,
            tile=tile,
        ),
    )

    ttcore_dtype = tensor_dtype_to_ttcore_datatype(tensor.dtype)
    element_type = ttcore.ir.TileType.get(ctx, tile[0], tile[1], ttcore_dtype)

    # Device shape: batch dims preserved, last 2 dims converted to tile counts
    batch_dims = shape[:-2]
    tensor_rows, tensor_cols = shape[-2], shape[-1]
    total_row_tiles = _ceil_div(tensor_rows, tile[0])
    total_col_tiles = _ceil_div(tensor_cols, tile[1])
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
        # from body-local PipeNet assignments. The name is stored on emitted
        # pipe declarations so diagnostics use the user's identifier.
        self._pipe_net_names: dict[int, str] = {}

        # Include paths collected from ttl.call_extern_func invocations,
        # forwarded to the JIT compiler as -I flags.
        self._opaque_include_paths: list[str] = []
        self._active_guards = []

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

    def visit_AnnAssign(self, node):
        """Keep expressions derived from external scalar results as SSA."""
        if node.value is not None and self._contains_external_scalar_result(node.value):
            if not isinstance(node.target, ast.Name):
                self._raise_error(
                    node.target,
                    "an annotated external scalar result requires a name target",
                )
            value = self.visit(node.value)
            if value is None:
                self._raise_error(
                    node.value,
                    "an annotated external scalar expression must produce a value",
                )
            self._set_var(node.target.id, value)
            return
        return super().visit_AnnAssign(node)

    def _contains_external_scalar_result(self, root):
        for candidate in ast.walk(root):
            if not isinstance(candidate, ast.Call) or not self._is_ttl_api_call(
                candidate, "call_extern_func"
            ):
                continue
            result_type_node = next(
                (
                    keyword.value
                    for keyword in candidate.keywords
                    if keyword.arg == "result_type"
                ),
                None,
            )
            condition_result_node = next(
                (
                    keyword.value
                    for keyword in candidate.keywords
                    if keyword.arg == "condition_result"
                ),
                None,
            )
            if condition_result_node is not None or (
                result_type_node is not None
                and self._resolve_scalar_type(result_type_node) is not None
            ):
                return True
        return False

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

    def _visit_module_helper_call(self, node, helper):
        """Evaluate a module-level Python helper while tracing a thread."""
        args = [self._load_func_arg(self.visit(arg), arg, node) for arg in node.args]
        kwargs = {
            kw.arg: self._load_func_arg(self.visit(kw.value), kw.value, node)
            for kw in node.keywords
        }
        return helper(*args, **kwargs)

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
                self._strip_explicit_release_kernel_selector(node)

                # Intercept print() to handle keyword arguments.
                if (
                    not isinstance(node.func, ast.Attribute)
                    and hasattr(node.func, "id")
                    and node.func.id == "print"
                ):
                    return self.visit_Print(node.args, node.keywords)

                if self._is_ttl_api_call(node, "call_extern_func"):
                    return self.visit_Call_Extern_Func(node, node.args, node.keywords)

                if self._is_ttl_api_call(node, "reset_dfbs"):
                    return self._visit_reset_dfbs(node, reset_all=False)

                if self._is_ttl_api_call(node, "reset_all_dfbs"):
                    return self._visit_reset_dfbs(node, reset_all=True)

                if self._is_ttl_api_call(node, "reconfigure_dfbs"):
                    return self._visit_dfb_reconfiguration(node)

                if self._is_ttl_api_call(node, "raw_addr"):
                    return self._visit_raw_addr(node)

                if self._is_ttl_api_call(node, "get_dfb_id"):
                    return self._visit_get_dfb_id(node)

                # Check for PipeNet.if_src/if_dst calls
                if self._is_pipenet_callback_call(node):
                    return self._handle_pipenet_callback(node)

                # Check for PipeNet.is_src/is_dst/is_active predicate calls
                if self._is_pipenet_predicate_call(node):
                    return self._handle_pipenet_predicate(node)

                # Module-level helpers are useful for small compatibility
                # wrappers around TT-Lang syntax, such as selecting an
                # optional keyword based on an introspected API signature.
                # Evaluate those helpers while tracing, after resolving their
                # arguments through the same path as built-in syntax calls.
                if isinstance(node.func, ast.Name):
                    helper = self.fn_globals.get(node.func.id)
                    module_name = self.fn_globals.get("__name__")
                    if inspect.isfunction(helper) and helper.__module__ == module_name:
                        return self._try_emit_auto_signposts(
                            node,
                            lambda: self._visit_module_helper_call(node, helper),
                        )

                return self._try_emit_auto_signposts(
                    node, lambda: super(TTLGenericCompiler, self).visit_Call(node)
                )
            except (ValueError, TypeError, NotImplementedError) as e:
                if isinstance(e, TTLangCompileError):
                    raise
                self._raise_error(node, str(e))

    def _strip_explicit_release_kernel_selector(self, node: ast.Call) -> None:
        """Remove release placement because the thread decorator owns it."""
        if not isinstance(node.func, ast.Attribute):
            return
        if node.func.attr not in _DFB_RELEASE_METHODS:
            return
        if not isinstance(node.func.value, ast.Name):
            return
        receiver_table = self._var_exists(node.func.value.id)
        if not receiver_table:
            return
        from ..operators import _is_block

        if not _is_block(receiver_table[node.func.value.id]):
            return
        node.keywords = [
            keyword for keyword in node.keywords if keyword.arg != "kernel"
        ]

    def visit_AugAssign(self, node):
        """Handle augmented assignment on tensor values.

        `+=` on a DFB-attached block emits an accumulating store through
        `__iadd__`. Other tensor targets are rewritten to an ordinary
        assignment so loop-carried SSA values can be represented by `scf.for`
        iter_args; accumulation lowering handles recognized additive
        recurrences.
        """
        with self._loc_for_node(node):
            target = self.visit(node.target)
            if hasattr(target, "type") and isinstance(target.type, RankedTensorType):
                from ..operators import _is_block

                if isinstance(node.op, ast.Add) and _is_block(target):
                    rhs = self.visit(node.value)
                    mlir_type = _get_type_str(target.type)
                    iadd_fn = self._fn_map.get(f"{mlir_type}.__iadd__")
                    if iadd_fn:
                        result = iadd_fn(target, rhs)
                        self._set_var(node.target.id, result)
                        return
                if isinstance(node.target, ast.Name):
                    load_target = ast.copy_location(
                        ast.Name(id=node.target.id, ctx=ast.Load()), node.target
                    )
                    store_target = ast.copy_location(
                        ast.Name(id=node.target.id, ctx=ast.Store()), node.target
                    )
                    synthetic = ast.copy_location(
                        ast.Assign(
                            targets=[store_target],
                            value=ast.copy_location(
                                ast.BinOp(
                                    left=load_target, op=node.op, right=node.value
                                ),
                                node.value,
                            ),
                        ),
                        node,
                    )
                    return self.visit(synthetic)
            return super().visit_AugAssign(node)

    def _coerce_if_condition(self, condition):
        if hasattr(condition, "result"):
            condition = condition.result

        condition_type = None
        if hasattr(condition, "type") and isinstance(condition.type, memref.MemRefType):
            condition = memref.LoadOp(
                condition, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result
            condition_type = condition.type
        elif hasattr(condition, "type") and isinstance(condition.type, IntegerType):
            condition_type = condition.type
        elif isinstance(condition, arith.ConstantOp):
            condition_type = condition.type

        if condition_type is None or not isinstance(condition_type, IntegerType):
            raise ValueError("Cannot Compare Non-Integer Values")

        if condition_type.width != 1:
            condition = arith.cmpi(
                arith.CmpIPredicate.ne,
                condition,
                arith.ConstantOp(condition_type, 0),
            )
        return condition

    def _format_guard_description(self, node) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return type(node).__name__

    def _get_guarded_dfb_assignment(self, stmt):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return None
        target = stmt.targets[0]
        if not isinstance(target, ast.Name) or self._var_exists(target.id):
            return None
        value = stmt.value
        if not isinstance(value, ast.Call) or value.args or value.keywords:
            return None
        if not isinstance(value.func, ast.Attribute):
            return None
        acquire_method = value.func.attr
        if acquire_method not in ("reserve", "wait"):
            return None
        if not isinstance(value.func.value, ast.Name):
            return None
        dfb_table = self._var_exists(value.func.value.id)
        if not dfb_table:
            return None
        dfb = dfb_table[value.func.value.id]
        if not hasattr(dfb, "type"):
            return None
        if ttl.CircularBufferType.maybe_downcast(dfb.type) is None:
            return None
        return _GuardedDFBAssignment(
            target.id,
            dfb,
            self._get_cb_tensor_type(dfb, node=value),
            acquire_method,
            stmt,
        )

    def _get_guarded_dfb_assignments(self, node):
        if node.orelse:
            return []
        assignments = []
        seen_names = set()
        for stmt in node.body:
            assignment = self._get_guarded_dfb_assignment(stmt)
            if assignment is None or assignment.name in seen_names:
                continue
            assignments.append(assignment)
            seen_names.add(assignment.name)
        return assignments

    def _emit_inactive_guarded_dfb_value(self, assignment):
        return Operation.create(
            "builtin.unrealized_conversion_cast",
            results=[assignment.tensor_type],
            operands=[],
            attributes={"ttl.inactive_guarded_dfb": UnitAttr.get()},
        ).result

    def _visit_guarded_if_region(
        self,
        statements,
        carried_var_names,
        carried_initial_values,
        guarded_assignments,
        active_guard=None,
    ):
        self.symbol_tables.append({})
        if active_guard is not None:
            self._active_guards.append(active_guard)
        try:
            for stmt in statements:
                self.visit(stmt)
            self._on_scope_exit()

            yield_values = []
            for var_name, initial_value in zip(
                carried_var_names, carried_initial_values
            ):
                final_value = self.symbol_tables[-1].get(var_name, initial_value)
                initial_type = _require_mlir_value_type(
                    initial_value, var_name, "an if statement"
                )
                final_type = _require_mlir_value_type(
                    final_value, var_name, "an if statement"
                )
                if final_type != initial_type:
                    raise ValueError(
                        f"Variable '{var_name}' changes type across an if "
                        f"statement from {initial_type} to {final_type}"
                    )
                yield_values.append(_get_single_result(final_value))

            for assignment in guarded_assignments:
                final_value = self.symbol_tables[-1].get(assignment.name)
                if final_value is None:
                    self._raise_error(
                        assignment.node,
                        f"guarded DFB block '{assignment.name}' was not "
                        "defined in the active branch",
                    )
                final_value = _get_single_result(final_value)
                if final_value.type != assignment.tensor_type:
                    self._raise_error(
                        assignment.node,
                        f"guarded DFB block '{assignment.name}' changes type "
                        f"from {assignment.tensor_type} to {final_value.type}",
                    )
                yield_values.append(final_value)
            scf.YieldOp(yield_values)
        finally:
            if active_guard is not None:
                self._active_guards.pop()
            self.symbol_tables.pop()

    def _visit_inactive_guarded_if_region(
        self, statements, carried_var_names, carried_initial_values, guarded_assignments
    ):
        self.symbol_tables.append({})
        try:
            for stmt in statements:
                self.visit(stmt)
            self._on_scope_exit()

            yield_values = []
            for var_name, initial_value in zip(
                carried_var_names, carried_initial_values
            ):
                final_value = self.symbol_tables[-1].get(var_name, initial_value)
                initial_type = _require_mlir_value_type(
                    initial_value, var_name, "an if statement"
                )
                final_type = _require_mlir_value_type(
                    final_value, var_name, "an if statement"
                )
                if final_type != initial_type:
                    raise ValueError(
                        f"Variable '{var_name}' changes type across an if "
                        f"statement from {initial_type} to {final_type}"
                    )
                yield_values.append(_get_single_result(final_value))

            for assignment in guarded_assignments:
                yield_values.append(self._emit_inactive_guarded_dfb_value(assignment))
            scf.YieldOp(yield_values)
        finally:
            self.symbol_tables.pop()

    def visit_If(self, node):
        self._reject_unsupported_language_constructs([node])

        condition = self._coerce_if_condition(self.visit(node.test))
        carried_var_names = self._get_if_carried_var_names(node)
        carried_initial_values = [
            _get_single_result(self._var_exists(var_name)[var_name])
            for var_name in carried_var_names
        ]
        carried_types = [
            _require_mlir_value_type(value, var_name, "an if statement")
            for var_name, value in zip(carried_var_names, carried_initial_values)
        ]

        guarded_assignments = self._get_guarded_dfb_assignments(node)
        result_types = carried_types + [
            assignment.tensor_type for assignment in guarded_assignments
        ]
        if_op = scf.IfOp(
            cond=condition,
            results_=result_types,
            has_else=bool(node.orelse) or bool(result_types),
        )

        self._on_scope_exit()
        with InsertionPoint(if_op.then_block), Location.unknown():
            self._visit_guarded_if_region(
                node.body,
                carried_var_names,
                carried_initial_values,
                guarded_assignments,
                active_guard=condition,
            )

        if node.orelse or result_types:
            with InsertionPoint(if_op.else_block), Location.unknown():
                self._visit_inactive_guarded_if_region(
                    node.orelse,
                    carried_var_names,
                    carried_initial_values,
                    guarded_assignments,
                )

        result_index = 0
        for var_name in carried_var_names:
            self._set_var(var_name, if_op.results[result_index])
            result_index += 1
        guard_description = self._format_guard_description(node.test)
        for assignment in guarded_assignments:
            self._set_var(
                assignment.name,
                _GuardedDFBBlock(
                    if_op.results[result_index],
                    assignment.dfb,
                    condition,
                    guard_description,
                    assignment.acquire_method,
                ),
            )
            result_index += 1

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
        from ..pipe import PipeNet

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

        pipe_records = [
            ttl.PipeRecordAttr.get(
                self.ctx,
                pipe.src[0],
                pipe.src[1],
                pipe.dst_start[0],
                pipe.dst_start[1],
                pipe.dst_end[0],
                pipe.dst_end[1],
                pipe.is_collective,
            )
            for pipe in pipenet.pipes
        ]
        records_attr = ttl.PipeNetRecordsAttr.get(
            self.ctx,
            pipenet.pipe_net_id,
            pipe_net_name=pipe_net_name,
            pipes=pipe_records,
        )
        decl_file = getattr(pipenet, "_source_file", None)
        decl_line = getattr(pipenet, "_source_line", None)
        loc = None
        if decl_file and decl_line is not None:
            loc = Location.file(decl_file, decl_line, 1, self.ctx)

        if method_name == "if_src":
            op = ttl.pipenet_foreach_src(records_attr, loc=loc)
            pipe_type = ttl.SelectedPipeSrcType.get(self.ctx)
        else:
            op = ttl.pipenet_foreach_dst(records_attr, loc=loc)
            pipe_type = ttl.SelectedPipeDstType.get(self.ctx)

        block = Block.create_at_start(op.body, [pipe_type])
        with InsertionPoint(block):
            self.symbol_tables.append({})
            self.symbol_tables[-1][pipe_param_name] = block.arguments[0]

            if isinstance(callback_body, list):
                for stmt in callback_body:
                    self.visit(stmt)
            else:
                self.visit(callback_body)

            self.symbol_tables.pop()
            ttl.yield_([])

        return None  # Statement, no return value

    def _coerce_binary_operands(self, left_value, right_value, left_node, right_node):
        if (
            left_value.type != right_value.type
            and isinstance(left_value.type, IntegerType)
            and isinstance(right_value.type, IntegerType)
        ):
            raise TypeError(
                "integer operands require matching widths, got "
                f"{left_value.type} and {right_value.type}"
            )
        return super()._coerce_binary_operands(
            left_value, right_value, left_node, right_node
        )

    def _materialize_integer_literal(self, node, value: int, integer_type: IntegerType):
        bit_width = integer_type.width
        minimum = -(1 << (bit_width - 1))
        maximum = (1 << (bit_width - 1)) - 1
        if not minimum <= value <= maximum:
            self._raise_error(
                node,
                f"integer literal {value} does not fit in signed i{bit_width}",
            )
        return arith.ConstantOp(integer_type, value).result

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
            if isinstance(result, _GuardedDFBBlock):
                if result.guard not in self._active_guards:
                    self._raise_error(
                        node,
                        f"DFB block '{node.id}' is only defined when "
                        f"{result.guard_description} is true",
                    )
                return ttl.attach_cb(result.value.type, result.value, result.dfb)
            return result

        # Check if it's a module-level constant
        var_name = node.id
        if var_name in self.fn_globals:
            val = self.fn_globals[var_name]
            if type(val) is bool:
                return arith.ConstantOp(
                    IntegerType.get_signless(1, self.ctx), int(val)
                ).result
            if type(val) is int:
                return arith.ConstantOp(
                    IntegerType.get_signless(64, self.ctx), val
                ).result
            if isinstance(val, float):
                return arith.ConstantOp(F32Type.get(self.ctx), val).result
            if is_ttnn_global_semaphore(val):
                self._raise_error(
                    node,
                    "ttnn.GlobalSemaphore must be captured by an operation "
                    "factory; module-global semaphores are not supported",
                )

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

    @staticmethod
    def _is_ttl_api_call(node, name):
        """Match both ``name(...)`` and ``ttl.name(...)`` call forms."""
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr == name
            and isinstance(func.value, ast.Name)
            and func.value.id == "ttl"
        ):
            return True
        return False

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
                # Tensor-typed attributes are resolved from the SSA value type
                # so spec-form ops can construct result types without reading a
                # destination DFB during lowering.
                if (
                    not func_args
                    and not kwargs
                    and node.attr
                    in {
                        "shape",
                        "dtype",
                        "tile",
                    }
                ):
                    value = self.visit(node.value)
                    if value is not None and hasattr(value, "type"):
                        cb_ty = ttl.CircularBufferType.maybe_downcast(value.type)
                        if cb_ty is not None:
                            if node.attr == "shape":
                                return tuple(cb_ty.shape)
                            tile_ty = ttcore.ir.TileType.maybe_downcast(
                                cb_ty.element_type
                            )
                            if tile_ty is not None:
                                if node.attr == "tile":
                                    return tuple(tile_ty.shape)
                                return ttcore.DataType(tile_ty.data_type_as_int)
                        tensor_ty = RankedTensorType.maybe_downcast(value.type)
                        if tensor_ty is not None:
                            if node.attr == "shape":
                                return tuple(tensor_ty.shape)
                            tile_ty = ttcore.ir.TileType.maybe_downcast(
                                tensor_ty.element_type
                            )
                            if tile_ty is not None:
                                if node.attr == "tile":
                                    return tuple(tile_ty.shape)
                                return ttcore.DataType(tile_ty.data_type_as_int)
                            if tensor_ty.element_type == F32Type.get(self.ctx):
                                return ttcore.DataType.Float32
                            if tensor_ty.element_type == BF16Type.get(self.ctx):
                                return ttcore.DataType.BFloat16
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
            self.ctx, cb.tile[0], cb.tile[1], ttcore_dtype
        )
        cb_type = ttl.CircularBufferType.get(
            self.ctx,
            list(cb.shape),
            element_type,
            cb.block_count,
        )
        # The frontend index identifies the logical DFB; finalization may
        # replace cb_index when reusing physical storage.
        tensor_backing = None
        if cb.tensor_backing is not None:
            tensor_backing = ttl.TensorBackingAttr.get(
                self.ctx,
                get_tensor_global_index(cb.tensor_backing),
                cb.byte_offset,
                cb.byte_size,
            )
        bind_attributes = {
            "block_count": cb.block_count,
            "dfb_id": cb._cb_index,
        }
        if tensor_backing is not None:
            bind_attributes["tensor_backing"] = tensor_backing
        if cb.allocation_group is not None:
            bind_attributes["allocation_group"] = ttl.DFBAllocationGroupAttr.get(
                self.ctx, cb.allocation_group.ordinal
            )
        return ttl.bind_cb(cb_type, cb._cb_index, **bind_attributes)

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
                if val is None:
                    continue
                if type(val) is bool:
                    self._set_var(
                        name,
                        arith.ConstantOp(
                            IntegerType.get_signless(1, self.ctx), int(val)
                        ),
                    )
                elif type(val) is int:
                    self._set_var(name, arith.ConstantOp(IndexType.get(self.ctx), val))
                elif isinstance(val, float):
                    self._set_var(name, arith.ConstantOp(F32Type.get(self.ctx), val))
                elif isinstance(val, (tuple, list)):
                    # Shape and axis lists are consumed by the Python-level API,
                    # exactly as an inline literal would be.
                    self._set_var(name, val)
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
                elif (
                    val is ScalarType
                    or isinstance(val, ScalarType)
                    or isinstance(val, _BoundDispatchCondition)
                    or isinstance(val, _BoundDFBReset)
                    or isinstance(val, _BoundDFBReconfiguration)
                ):
                    continue
                elif is_ttnn_global_semaphore(val):
                    sem_addr = get_ttnn_global_semaphore_address(val)
                    i32_ty = IntegerType.get_signless(32, self.ctx)
                    self._set_var(name, arith.ConstantOp(i32_ty, sem_addr).result)
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
                # TODO(#645): Parse reserve(accumulation_strategy=...) here
                # once source-level accumulation strategy hints are specified.
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

    def _resolve_template_arg_value(self, node):
        """Resolve one external template argument without creating static SSA.

        Accepts:
        - ``ttl.dfb_descriptor(dfb)`` -- typed allocation descriptor
        - ``ttl.get_dfb_id(dfb)`` -- compatibility integer index
        - ``int`` literals / module-level ints -- signed 32-bit payload
        - ``bool`` literals / module-level bools -- boolean payload
        - ``float`` literals / module-level floats -- binary32 bit payload
        """
        arg_kind = ttl.ir.ExternalTemplateArgKind

        def _signed_integer(py_int: int):
            if not -(1 << 31) <= py_int < (1 << 31):
                self._raise_error(
                    node,
                    "ttl.call_extern_func() signed integer template argument "
                    "must fit in 32 bits",
                )
            return _ExternalTemplateArg(arg_kind.SignedInteger, py_int)

        def _unsigned_integer(py_int: int):
            if not 0 <= py_int < (1 << 32):
                self._raise_error(
                    node,
                    "ttl.call_extern_func() unsigned template argument must "
                    "fit in 32 bits",
                )
            return _ExternalTemplateArg(arg_kind.UnsignedInteger, py_int)

        def _boolean(py_bool: bool):
            return _ExternalTemplateArg(arg_kind.Boolean, int(py_bool))

        def _float_bits(py_float: float) -> int:
            try:
                return struct.unpack("<I", struct.pack("<f", py_float))[0]
            except OverflowError:
                self._raise_error(
                    node,
                    "ttl.call_extern_func() float template argument must be "
                    "representable as binary32",
                )

        def _dfb_reference(kind):
            if len(node.args) != 1 or node.keywords:
                wrapper_name = (
                    "ttl.get_dfb_id()"
                    if kind == arg_kind.DFBIndex
                    else "ttl.dfb_descriptor()"
                )
                self._raise_error(node, f"{wrapper_name} requires exactly 1 argument")
            dfb_value = self.visit(node.args[0])
            if (
                ttl.CircularBufferType.maybe_downcast(getattr(dfb_value, "type", None))
                is None
            ):
                wrapper_name = (
                    "ttl.get_dfb_id()"
                    if kind == arg_kind.DFBIndex
                    else "ttl.dfb_descriptor()"
                )
                self._raise_error(
                    node.args[0], f"{wrapper_name} argument must be a DFB"
                )
            return _ExternalTemplateArg(
                kind,
                dfb_value,
                getattr(node.args[0], _DFB_SOURCE_OCCURRENCE, None),
            )

        if isinstance(node, ast.Call) and self._is_ttl_api_call(node, "get_dfb_id"):
            return _dfb_reference(arg_kind.DFBIndex)
        if isinstance(node, ast.Call) and self._is_ttl_api_call(node, "dfb_descriptor"):
            return _dfb_reference(arg_kind.DFBDescriptor)

        if isinstance(node, ast.Constant):
            # bool is a subclass of int; check explicitly first.
            if type(node.value) is bool:
                return _boolean(node.value)
            if type(node.value) is int:
                return _signed_integer(node.value)
            if isinstance(node.value, float):
                return _unsigned_integer(_float_bits(node.value))

        int_val = self._signed_int_literal(node)
        if int_val is not None:
            return _signed_integer(int_val)

        # Fold unary-minus float literals (e.g. ``-1.5``).
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, float)
        ):
            return _unsigned_integer(_float_bits(-node.operand.value))

        if isinstance(node, ast.Name) and node.id in self.captures:
            val = self.captures[node.id]
            if type(val) is bool:
                return _boolean(val)
            if type(val) is int:
                return _signed_integer(val)
            if isinstance(val, float):
                return _unsigned_integer(_float_bits(val))
            if is_ttnn_global_semaphore(val):
                sem_addr = get_ttnn_global_semaphore_address(val)
                return _unsigned_integer(sem_addr)

        if isinstance(node, ast.Name) and node.id in self.fn_globals:
            val = self.fn_globals[node.id]
            if type(val) is bool:
                return _boolean(val)
            if type(val) is int:
                return _signed_integer(val)
            if isinstance(val, float):
                return _unsigned_integer(_float_bits(val))
            if is_ttnn_global_semaphore(val):
                self._raise_error(
                    node,
                    "ttnn.GlobalSemaphore must be captured by an operation "
                    "factory; module-global semaphores are not supported",
                )

        resolved = self.visit(node)
        if isinstance(resolved, tuple):
            self._raise_error(
                node,
                "ttl.call_extern_func() does not support tensor slices/views in "
                "extern arguments yet; pass the base tensor or "
                "ttl.raw_addr(base_tensor)",
            )
        resolved_type = getattr(resolved, "type", None)
        cb_type = ttl.CircularBufferType.maybe_downcast(resolved_type)
        if cb_type is not None:
            self._raise_error(
                node,
                "bare DFB template arguments are ambiguous; use "
                "ttl.dfb_descriptor(dfb) for allocation metadata or "
                "ttl.get_dfb_id(dfb) for an integer index",
            )
        def_op = resolved
        if not isinstance(def_op, arith.ConstantOp):
            def_op = getattr(resolved, "owner", None)
        if isinstance(resolved_type, IntegerType):
            if isinstance(def_op, arith.ConstantOp):
                value_attr = def_op.value
                if isinstance(value_attr, IntegerAttr):
                    if resolved_type.width == 1:
                        return _boolean(bool(int(value_attr.value)))
                    return _signed_integer(int(value_attr.value))
            self._raise_error(
                node,
                "ttl.call_extern_func() template_args integer values must be "
                "compile-time constants",
            )
        if isinstance(resolved_type, IndexType):
            if isinstance(def_op, arith.ConstantOp):
                value_attr = def_op.value
                if isinstance(value_attr, IntegerAttr):
                    return _signed_integer(int(value_attr.value))
            self._raise_error(
                node,
                "ttl.call_extern_func() template_args index values must be "
                "compile-time constants",
            )

        self._raise_error(
            node,
            "ttl.call_extern_func() template_args element must be an int, "
            "bool, float, ttl.dfb_descriptor(dfb), ttl.get_dfb_id(dfb), "
            "or an integer/index value",
        )

    def _resolve_string_value(self, node, param_name):
        """Resolve an AST node to a Python string.

        Accepts ``ast.Constant(str)`` or ``ast.Name`` that maps to a ``str``
        in ``self.fn_globals`` (module-level variables / closure captures).
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.fn_globals:
            val = self.fn_globals[node.id]
            if isinstance(val, str):
                return val
        self._raise_error(
            node,
            f"ttl.call_extern_func() {param_name} must be a string "
            f"literal or a module-level string variable",
        )

    def _resolve_string_list(self, node, param_name):
        """Resolve an AST list node to a Python list of strings."""
        if not isinstance(node, ast.List):
            self._raise_error(
                node,
                f"ttl.call_extern_func() {param_name} must be a list",
            )
        return [self._resolve_string_value(elt, param_name) for elt in node.elts]

    def _resolve_static_int(self, node, param_name):
        """Resolve a statically known Python integer without emitting SSA."""
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            for namespace in (self.captures, self.fn_globals):
                if node.id in namespace:
                    value = namespace[node.id]
                    if type(value) is int:
                        return value
                    break
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._resolve_static_int(node.operand, param_name)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
        ):
            lhs = self._resolve_static_int(node.left, param_name)
            rhs = self._resolve_static_int(node.right, param_name)
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and rhs == 0:
                self._raise_error(
                    node.right,
                    f"ttl.call_extern_func() {param_name} divisor must be nonzero",
                )
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.FloorDiv):
                return lhs // rhs
            return lhs % rhs
        self._raise_error(
            node,
            f"ttl.call_extern_func() {param_name} must be a statically "
            "resolvable integer",
        )

    def _resolve_static_bool(self, node, param_name):
        """Resolve a statically known Python boolean without emitting SSA."""
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return node.value
        if isinstance(node, ast.Name):
            for namespace in (self.captures, self.fn_globals):
                value = namespace.get(node.id)
                if type(value) is bool:
                    return value
        self._raise_error(
            node,
            f"ttl.call_extern_func() {param_name} must be a statically "
            "resolvable bool",
        )

    def _resolve_static_reference(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            for namespace in (self.captures, self.fn_globals):
                if node.id in namespace:
                    return namespace[node.id]
            return _MISSING_STATIC_VALUE
        if not isinstance(node, ast.Attribute):
            return _MISSING_STATIC_VALUE
        owner = self._resolve_static_reference(node.value)
        if owner is _MISSING_STATIC_VALUE:
            return _MISSING_STATIC_VALUE
        return inspect.getattr_static(owner, node.attr, _MISSING_STATIC_VALUE)

    def _resolve_scalar_type(self, node):
        """Resolve a statically known ScalarType without emitting SSA."""
        result_type = self._resolve_static_reference(node)
        if result_type is None:
            return None
        if not isinstance(result_type, ScalarType):
            type_detail = (
                ""
                if result_type is _MISSING_STATIC_VALUE
                else f", got {type(result_type).__name__}"
            )
            self._raise_error(
                node,
                "ttl.call_extern_func() result_type must be "
                "ttl.ScalarType.I32 or ttl.ScalarType.I64" + type_detail,
            )
        return result_type

    def _resolve_dispatch_condition(self, node):
        """Resolve a module-local dispatch condition declaration."""
        condition = self._resolve_static_reference(node)
        if isinstance(condition, DispatchCondition):
            self._raise_error(
                node,
                "ttl.call_extern_func() condition_result must be captured by "
                "an enclosing @ttl.operation factory",
            )
        if not isinstance(condition, _BoundDispatchCondition):
            type_detail = (
                ""
                if condition is _MISSING_STATIC_VALUE
                else f", got {type(condition).__name__}"
            )
            self._raise_error(
                node,
                "ttl.call_extern_func() condition_result must be a "
                "ttl.DispatchCondition" + type_detail,
            )
        return condition

    def _resolve_dfb_reset(self, node, api_name):
        """Resolve an operation-local synchronized reset declaration."""
        reset = self._resolve_static_reference(node)
        if isinstance(reset, DFBReset):
            self._raise_error(
                node,
                f"ttl.{api_name}() reset must be captured by an enclosing "
                "@ttl.operation factory",
            )
        if not isinstance(reset, _BoundDFBReset):
            type_detail = (
                ""
                if reset is _MISSING_STATIC_VALUE
                else f", got {type(reset).__name__}"
            )
            self._raise_error(
                node,
                f"ttl.{api_name}() reset must be a ttl.DFBReset" + type_detail,
            )
        return reset

    def _resolve_dfb_reconfiguration(self, node):
        """Resolve an operation-local DFB reconfiguration declaration."""
        boundary = self._resolve_static_reference(node)
        if isinstance(boundary, DFBReconfiguration):
            self._raise_error(
                node,
                "ttl.reconfigure_dfbs() boundary must be captured by an "
                "enclosing @ttl.operation factory",
            )
        if not isinstance(boundary, _BoundDFBReconfiguration):
            type_detail = (
                ""
                if boundary is _MISSING_STATIC_VALUE
                else f", got {type(boundary).__name__}"
            )
            self._raise_error(
                node,
                "ttl.reconfigure_dfbs() boundary must be a "
                "ttl.DFBReconfiguration" + type_detail,
            )
        return boundary

    def _visit_dfb_reconfiguration(self, node):
        if len(node.args) != 1 or node.keywords:
            self._raise_error(
                node,
                "ttl.reconfigure_dfbs() requires exactly one positional "
                "DFBReconfiguration argument",
            )
        boundary = self._resolve_dfb_reconfiguration(node.args[0])
        participant_attrs = [
            self._logical_kernel_attr(participant)
            for participant in sorted(boundary.participants, key=_selector_sort_key)
        ]
        boundary_attr = ttl.ir.DFBReconfigurationAttr.get(
            self.ctx, boundary.ordinal, participant_attrs
        )
        return ttl.dfb_reconfiguration(boundary_attr)

    def _logical_kernel_attr(self, participant):
        participant_kind = _selector_kind(participant)
        ir_kind = {
            KernelKind.COMPUTE: ttl.ir.LogicalKernelKind.Compute,
            KernelKind.DATA_MOVEMENT: ttl.ir.LogicalKernelKind.DataMovement,
        }[participant_kind]
        if isinstance(participant, KernelKind):
            return ttl.ir.LogicalKernelAttr.get(
                self.ctx,
                ir_kind,
                None,
                None,
                None,
            )
        if participant._identity is None:
            raise TypeError(
                "DFB synchronization participant Kernel must be captured by the enclosing "
                "@ttl.operation"
            )
        return ttl.ir.LogicalKernelAttr.get(
            self.ctx,
            ir_kind,
            participant.identity,
            participant._operation_identity,
            _selector_implicit_role(participant),
        )

    def _resolve_dfb_value(self, node, param_name, api_name="call_extern_func"):
        """Resolve one DFB expression and reject other SSA values."""
        value = self.visit(node)
        if ttl.CircularBufferType.maybe_downcast(getattr(value, "type", None)) is None:
            self._raise_error(
                node, f"ttl.{api_name}() {param_name} element must be a DFB"
            )
        return value

    def _resolve_external_dfb_reference(self, node, param_name):
        return _ExternalDFBDependency(
            self._resolve_dfb_value(node, param_name),
            getattr(node, _DFB_SOURCE_OCCURRENCE, None),
        )

    def _resolve_external_dfb_dependency_index(
        self, reference, ordered_dependencies, diagnostic_node, summary_name
    ):
        value_matches = [
            dependency_index
            for dependency_index, dependency in enumerate(ordered_dependencies)
            if dependency.dfb == reference.dfb
        ]
        if reference.source_occurrence is not None:
            exact_matches = [
                dependency_index
                for dependency_index in value_matches
                if ordered_dependencies[dependency_index].source_occurrence
                == reference.source_occurrence
            ]
            if len(exact_matches) == 1:
                return exact_matches[0]
            if len(exact_matches) > 1:
                value_matches = exact_matches
        if len(value_matches) == 1:
            return value_matches[0]
        if len(value_matches) > 1:
            self._raise_error(
                diagnostic_node,
                f"ttl.call_extern_func() DFB {summary_name} reference is "
                "ambiguous because the DFB appears in multiple dependency "
                "positions; use a distinct composed-operation DFB parameter "
                "for each position and reference it directly",
            )
        self._raise_error(
            diagnostic_node,
            f"ttl.call_extern_func() DFB {summary_name} references a DFB that "
            "is not a function argument, descriptor, or dependency",
        )

    def _visit_reset_dfbs(self, node, reset_all):
        api_name = "reset_all_dfbs" if reset_all else "reset_dfbs"
        if len(node.args) != 1:
            self._raise_error(
                node,
                f"ttl.{api_name}() requires one positional DFBReset argument",
            )
        reset = self._resolve_dfb_reset(node.args[0], api_name)
        participant_attrs = [
            self._logical_kernel_attr(participant)
            for participant in sorted(reset.participants, key=_selector_sort_key)
        ]
        reset_attr = ttl.ir.SynchronizedDFBResetAttr.get(
            self.ctx, reset.ordinal, participant_attrs
        )

        keyword_values = {keyword.arg: keyword.value for keyword in node.keywords}
        if reset_all:
            if keyword_values:
                self._raise_error(
                    node,
                    "ttl.reset_all_dfbs() does not accept keyword arguments",
                )
            return ttl.reset_all_dfbs(reset=reset_attr)

        if set(keyword_values) != {"dfbs"}:
            self._raise_error(
                node,
                "ttl.reset_dfbs() requires the dfbs keyword argument",
            )
        dfbs_node = keyword_values["dfbs"]
        if not isinstance(dfbs_node, ast.List) or not dfbs_node.elts:
            self._raise_error(
                dfbs_node, "ttl.reset_dfbs() dfbs must be a nonempty list"
            )
        dfbs = [
            self._resolve_dfb_value(element, "dfbs", api_name)
            for element in dfbs_node.elts
        ]
        if any(dfb in dfbs[:dfb_index] for dfb_index, dfb in enumerate(dfbs)):
            self._raise_error(dfbs_node, "ttl.reset_dfbs() dfbs must be distinct")
        return ttl.reset_dfbs(reset=reset_attr, dfbs=dfbs)

    def _resolve_dfb_effect(self, node):
        """Resolve ``DFBEffect.<kind>(dfb, tiles=N)`` to typed facts."""
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            self._raise_error(
                node,
                "ttl.call_extern_func() dfb_effects element must be a "
                "ttl.DFBEffect reserve, push, wait, or pop",
            )

        effect_owner = node.func.value
        is_qualified_owner = (
            isinstance(effect_owner, ast.Attribute)
            and effect_owner.attr == "DFBEffect"
            and isinstance(effect_owner.value, ast.Name)
            and effect_owner.value.id == "ttl"
        )
        is_direct_owner = (
            isinstance(effect_owner, ast.Name) and effect_owner.id == "DFBEffect"
        )
        effect_name = node.func.attr
        effect_kinds = {
            "reserve": ttl.ir.DFBProtocolEffectKind.Reserve,
            "push": ttl.ir.DFBProtocolEffectKind.Push,
            "wait": ttl.ir.DFBProtocolEffectKind.Wait,
            "pop": ttl.ir.DFBProtocolEffectKind.Pop,
        }
        if (
            not (is_qualified_owner or is_direct_owner)
            or effect_name not in effect_kinds
        ):
            self._raise_error(
                node,
                "ttl.call_extern_func() dfb_effects element must be a "
                "ttl.DFBEffect reserve, push, wait, or pop",
            )
        if len(node.args) != 1:
            self._raise_error(
                node,
                f"ttl.DFBEffect.{effect_name}() requires exactly one DFB argument",
            )
        keyword_values = {keyword.arg: keyword.value for keyword in node.keywords}
        if set(keyword_values) != {"tiles"}:
            self._raise_error(
                node,
                f"ttl.DFBEffect.{effect_name}() requires one tiles= keyword",
            )
        num_tiles = self._resolve_static_int(keyword_values["tiles"], "effect tiles")
        if num_tiles <= 0:
            self._raise_error(
                keyword_values["tiles"],
                "ttl.call_extern_func() effect tiles must be positive",
            )
        dfb = self._resolve_external_dfb_reference(node.args[0], "dfb_effects")
        return _ExternalDFBEffect(
            effect_kinds[effect_name],
            dfb.dfb,
            num_tiles,
            dfb.source_occurrence,
        )

    def _is_dfb_effect_repeat(self, node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return False
        effect_owner = node.func.value
        is_qualified_owner = (
            isinstance(effect_owner, ast.Attribute)
            and effect_owner.attr == "DFBEffect"
            and isinstance(effect_owner.value, ast.Name)
            and effect_owner.value.id == "ttl"
        )
        is_direct_owner = (
            isinstance(effect_owner, ast.Name) and effect_owner.id == "DFBEffect"
        )
        return (is_qualified_owner or is_direct_owner) and node.func.attr == "repeat"

    def _parse_dfb_effect_sequence(self, node):
        """Parse an ordered DFB-effect sequence without expanding repeats."""
        if not isinstance(node, ast.List):
            self._raise_error(
                node,
                "ttl.call_extern_func() dfb_effects and "
                "ttl.DFBEffect.repeat() effects must be lists",
            )

        parsed_effects = []
        expanded_effect_count = 0
        for element in node.elts:
            if not self._is_dfb_effect_repeat(element):
                parsed_effects.append(self._resolve_dfb_effect(element))
                expanded_effect_count = _saturating_add_expanded_dfb_effect_count(
                    expanded_effect_count, 1, 1
                )
                continue

            if len(element.args) != 2 or element.keywords:
                self._raise_error(
                    element,
                    "ttl.DFBEffect.repeat() requires count and effects arguments",
                )
            repeat_count = self._resolve_static_int(
                element.args[0], "effect repeat count"
            )
            if repeat_count < 0:
                self._raise_error(
                    element.args[0],
                    "ttl.DFBEffect.repeat() count must be nonnegative",
                )
            repeated_effects, repeated_effect_count = self._parse_dfb_effect_sequence(
                element.args[1]
            )
            if repeated_effect_count == 0:
                self._raise_error(
                    element.args[1],
                    "ttl.DFBEffect.repeat() effects must not be empty",
                )
            parsed_effects.append(
                _ExternalDFBEffectRepeat(repeat_count, repeated_effects)
            )
            expanded_effect_count = _saturating_add_expanded_dfb_effect_count(
                expanded_effect_count, repeated_effect_count, repeat_count
            )

        return tuple(parsed_effects), expanded_effect_count

    def _append_dfb_effect_sequence(self, parsed_effects, resolved_effects):
        """Materialize a parsed sequence after its expanded size is validated."""
        for effect in parsed_effects:
            if isinstance(effect, _ExternalDFBEffectRepeat):
                for _repeat_index in range(effect.count):
                    self._append_dfb_effect_sequence(effect.effects, resolved_effects)
                continue
            resolved_effects.append(effect)

    def _resolve_dfb_effect_sequence(self, node):
        """Resolve and flatten a literal ordered DFB-effect sequence."""
        parsed_effects, expanded_effect_count = self._parse_dfb_effect_sequence(node)
        if expanded_effect_count > _MAX_EXPANDED_EXTERNAL_DFB_EFFECTS:
            self._raise_error(
                node,
                "ttl.call_extern_func() dfb_effects may contain at most "
                f"{_MAX_EXPANDED_EXTERNAL_DFB_EFFECTS} expanded effects",
            )

        resolved_effects = []
        self._append_dfb_effect_sequence(parsed_effects, resolved_effects)
        return resolved_effects

    def _resolve_dfb_access(self, node):
        """Resolve ``DFBAccess.inspect(dfb)`` to a typed fact."""
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            self._raise_error(
                node,
                "ttl.call_extern_func() dfb_accesses element must be "
                "ttl.DFBAccess.inspect",
            )

        access_owner = node.func.value
        is_qualified_owner = (
            isinstance(access_owner, ast.Attribute)
            and access_owner.attr == "DFBAccess"
            and isinstance(access_owner.value, ast.Name)
            and access_owner.value.id == "ttl"
        )
        is_direct_owner = (
            isinstance(access_owner, ast.Name) and access_owner.id == "DFBAccess"
        )
        if not (is_qualified_owner or is_direct_owner) or (node.func.attr != "inspect"):
            self._raise_error(
                node,
                "ttl.call_extern_func() dfb_accesses element must be "
                "ttl.DFBAccess.inspect",
            )
        if len(node.args) != 1 or node.keywords:
            self._raise_error(
                node,
                "ttl.DFBAccess.inspect() requires exactly one DFB argument",
            )
        dfb = self._resolve_external_dfb_reference(node.args[0], "dfb_accesses")
        return _ExternalDFBAccess(
            ttl.ir.DFBNonTransactionalAccessKind.Inspect,
            dfb.dfb,
            dfb.source_occurrence,
        )

    def _visit_get_dfb_id(self, node):
        """Emit ttl.get_dfb_id for the DFB argument, return the i32 MLIR result."""
        if len(node.args) != 1 or node.keywords:
            self._raise_error(node, "ttl.get_dfb_id() requires exactly 1 argument")
        dfb_val = self.visit(node.args[0])
        if (
            ttl.CircularBufferType.maybe_downcast(getattr(dfb_val, "type", None))
            is None
        ):
            self._raise_error(node.args[0], "ttl.get_dfb_id() argument must be a DFB")
        return ttl.get_dfb_id(dfb_val)

    def _visit_raw_addr(self, node):
        """Emit ttl.raw_addr for the tensor argument, return the i32 base address."""
        if len(node.args) != 1 or node.keywords:
            self._raise_error(node, "ttl.raw_addr() requires exactly 1 argument")
        tensor_val = self.visit(node.args[0])
        if isinstance(tensor_val, tuple):
            self._raise_error(
                node,
                "ttl.raw_addr() does not support slices/views; pass the base tensor",
            )
        tensor_ty = getattr(tensor_val, "type", None)
        if not isinstance(tensor_ty, RankedTensorType):
            self._raise_error(node, "ttl.raw_addr() argument must be a tensor value")
        return ttl.raw_addr(tensor_val)

    def visit_Call_Extern_Func(self, node, args, keywords=None):
        """Handle ttl.call_extern_func(header, callee, ...) by emitting
        ttl.opaque_call.

        Signature::

            ttl.call_extern_func(
                header_path,                    # string (literal or variable)
                callee_name,                    # string (literal or variable)
                template_args=[1, ttl.dfb_descriptor(dfb)],
                func_args=[a, b],               # C++ function arguments
                dfb_dependencies=[scratch_dfb], # not a C++ argument
                dfb_effects=[
                    ttl.DFBEffect.wait(dfb, tiles=1),
                    ttl.DFBEffect.pop(dfb, tiles=1),
                ],
                dfb_accesses=[ttl.DFBAccess.inspect(descriptor_dfb)],
                unknown_dfb_access=False,
                include_paths=["/path/to/inc"], # -I flags for JIT compiler
                result_type=ttl.ScalarType.I64, # optional scalar result
                condition_result=active,        # dispatch-stable condition
            )

        DFBs use explicit forms in template_args and may appear directly in
        func_args:

        - ``template_args=[ttl.dfb_descriptor(dfb)]`` -- allocation metadata
          becomes a C++ template type.
        - ``template_args=[ttl.get_dfb_id(dfb)]`` -- the physical index becomes
          an integer template argument; the DFB must also be a function
          argument, descriptor, or dependency-only DFB.
        - ``func_args=[dfb]`` -- the DFB is passed as a runtime
          ``get_compile_time_arg_val(N)`` call, providing the DFB index as a
          function argument.

        Template args accept ``int``, ``bool`` (as 0/1), and ``float`` (as
        IEEE-754 bit pattern). Func args accept those scalars plus DFBs.
        """
        if len(args) < 2:
            self._raise_error(
                node,
                "ttl.call_extern_func() requires at least 2 positional arguments: "
                "header path and callee name",
            )
        if len(args) > 2:
            self._raise_error(
                node,
                "ttl.call_extern_func() accepts only 2 positional arguments "
                "(header, callee). Use template_args=[] and func_args=[] "
                "keyword arguments for call arguments.",
            )

        header = self._resolve_string_value(args[0], "header path")
        callee = self._resolve_string_value(args[1], "callee name")

        kw_map = {}
        if keywords:
            for kw in keywords:
                kw_map[kw.arg] = kw.value

        _valid_kwargs = {
            "template_args",
            "func_args",
            "dfb_dependencies",
            "dfb_effects",
            "dfb_accesses",
            "unknown_dfb_access",
            "include_paths",
            "result_type",
            "condition_result",
        }
        unexpected = set(kw_map) - _valid_kwargs
        if unexpected:
            self._raise_error(
                node,
                f"ttl.call_extern_func() got unexpected keyword argument(s): "
                f"{', '.join(sorted(unexpected))}. "
                f"Valid keywords are: {', '.join(sorted(_valid_kwargs))}",
            )

        resolved_template_args = []
        if "template_args" in kw_map:
            ta_node = kw_map["template_args"]
            if not isinstance(ta_node, ast.List):
                self._raise_error(
                    ta_node, "ttl.call_extern_func() template_args must be a list"
                )
            for elt in ta_node.elts:
                resolved_template_args.append(self._resolve_template_arg_value(elt))

        func_args = []
        func_arg_nodes = []
        unsigned_arg_indices = []
        if "func_args" in kw_map:
            fa_node = kw_map["func_args"]
            if not isinstance(fa_node, ast.List):
                self._raise_error(
                    fa_node, "ttl.call_extern_func() func_args must be a list"
                )
            for elt in fa_node.elts:
                requires_unsigned_cast = (
                    isinstance(elt, ast.Call) and self._is_ttl_api_call(elt, "raw_addr")
                ) or (
                    isinstance(elt, ast.Name)
                    and elt.id in self.captures
                    and is_ttnn_global_semaphore(self.captures[elt.id])
                )
                arg = self.visit(elt)
                if isinstance(arg, tuple):
                    self._raise_error(
                        elt,
                        "ttl.call_extern_func() does not support tensor "
                        "slices/views in extern arguments yet; pass the base "
                        "tensor or ttl.raw_addr(base_tensor)",
                    )
                if requires_unsigned_cast:
                    unsigned_arg_indices.append(len(func_args))
                func_args.append(arg)
                func_arg_nodes.append(elt)

        dependency_dfb_operands = []
        dependency_dfb_references = []
        if "dfb_dependencies" in kw_map:
            dependency_node = kw_map["dfb_dependencies"]
            if not isinstance(dependency_node, ast.List):
                self._raise_error(
                    dependency_node,
                    "ttl.call_extern_func() dfb_dependencies must be a list",
                )
            dependency_dfb_references = [
                self._resolve_external_dfb_reference(element, "dfb_dependencies")
                for element in dependency_node.elts
            ]
            dependency_dfb_operands = [
                dependency.dfb for dependency in dependency_dfb_references
            ]

        resolved_dfb_effects = []
        if "dfb_effects" in kw_map:
            effects_node = kw_map["dfb_effects"]
            resolved_dfb_effects = self._resolve_dfb_effect_sequence(effects_node)

        resolved_dfb_accesses = []
        if "dfb_accesses" in kw_map:
            accesses_node = kw_map["dfb_accesses"]
            if not isinstance(accesses_node, ast.List):
                self._raise_error(
                    accesses_node,
                    "ttl.call_extern_func() dfb_accesses must be a list",
                )
            resolved_dfb_accesses = [
                self._resolve_dfb_access(element) for element in accesses_node.elts
            ]

        unknown_dfb_access = False
        if "unknown_dfb_access" in kw_map:
            unknown_dfb_access = self._resolve_static_bool(
                kw_map["unknown_dfb_access"], "unknown_dfb_access"
            )

        if "include_paths" in kw_map:
            paths = self._resolve_string_list(kw_map["include_paths"], "include_paths")
            self._opaque_include_paths.extend(paths)

        if "result_type" in kw_map and "condition_result" in kw_map:
            self._raise_error(
                node,
                "ttl.call_extern_func() cannot combine result_type and "
                "condition_result",
            )
        result_types = []
        condition_result_attr = None
        if "result_type" in kw_map:
            result_type = self._resolve_scalar_type(kw_map["result_type"])
            if result_type is not None:
                result_types.append(
                    IntegerType.get_signless(result_type.bit_width, self.ctx)
                )
        elif "condition_result" in kw_map:
            condition = self._resolve_dispatch_condition(kw_map["condition_result"])
            result_type = IntegerType.get_signless(
                condition.scalar_type.bit_width, self.ctx
            )
            result_types.append(result_type)
            condition_result_attr = ttl.ir.DispatchConditionAttr.get(
                self.ctx, condition.ordinal, result_type
            )

        template_dfb_operands = []
        template_arg_attrs = []
        dfb_kinds = {
            ttl.ir.ExternalTemplateArgKind.DFBIndex,
            ttl.ir.ExternalTemplateArgKind.DFBDescriptor,
        }
        for template_arg in resolved_template_args:
            payload = template_arg.value
            if template_arg.kind in dfb_kinds:
                payload = len(template_dfb_operands)
                template_dfb_operands.append(template_arg.value)
            template_arg_attrs.append(
                ttl.ir.ExternalTemplateArgAttr.get(self.ctx, template_arg.kind, payload)
            )

        automatic_dependencies = [
            _ExternalDFBDependency(
                func_arg,
                getattr(func_arg_node, _DFB_SOURCE_OCCURRENCE, None),
            )
            for func_arg, func_arg_node in zip(func_args, func_arg_nodes)
            if ttl.CircularBufferType.maybe_downcast(getattr(func_arg, "type", None))
            is not None
        ]
        automatic_dependencies.extend(
            _ExternalDFBDependency(
                template_arg.value, template_arg.dfb_source_occurrence
            )
            for template_arg in resolved_template_args
            if template_arg.kind == ttl.ir.ExternalTemplateArgKind.DFBDescriptor
        )

        def is_same_source_occurrence(lhs, rhs):
            if lhs.dfb != rhs.dfb:
                return False
            if lhs.source_occurrence is not None and rhs.source_occurrence is not None:
                return lhs.source_occurrence == rhs.source_occurrence
            return True

        def has_repeated_source_occurrence(dependencies, prior_dependencies=()):
            previous_dependencies = list(prior_dependencies)
            for dependency in dependencies:
                if any(
                    is_same_source_occurrence(dependency, previous)
                    for previous in previous_dependencies
                ):
                    return True
                previous_dependencies.append(dependency)
            return False

        if has_repeated_source_occurrence(
            dependency_dfb_references, automatic_dependencies
        ):
            self._raise_error(
                kw_map["dfb_dependencies"],
                "ttl.call_extern_func() dfb_dependencies must contain only "
                "distinct dependency-only DFBs",
            )
        ordered_dependencies = automatic_dependencies + dependency_dfb_references

        if condition_result_attr is not None and (
            template_dfb_operands
            or ordered_dependencies
            or resolved_dfb_effects
            or resolved_dfb_accesses
            or unknown_dfb_access
        ):
            self._raise_error(
                node,
                "ttl.call_extern_func() condition_result call cannot access "
                "DFB state",
            )

        effect_attrs = []
        for effect in resolved_dfb_effects:
            dependency_index = self._resolve_external_dfb_dependency_index(
                effect,
                ordered_dependencies,
                kw_map["dfb_effects"],
                "effect",
            )
            effect_attrs.append(
                ttl.ir.DFBProtocolEffectAttr.get(
                    self.ctx,
                    effect.kind,
                    dependency_index,
                    effect.num_tiles,
                )
            )
        access_attrs = []
        for access in resolved_dfb_accesses:
            dependency_index = self._resolve_external_dfb_dependency_index(
                access,
                ordered_dependencies,
                kw_map["dfb_accesses"],
                "access",
            )
            access_attrs.append(
                ttl.ir.DFBNonTransactionalAccessAttr.get(
                    self.ctx,
                    access.kind,
                    dependency_index,
                )
            )
        template_args_attr = (
            ArrayAttr.get(template_arg_attrs) if template_arg_attrs else None
        )
        unsigned_arg_indices_attr = (
            DenseI32ArrayAttr.get(unsigned_arg_indices, context=self.ctx)
            if unsigned_arg_indices
            else None
        )
        effects_attr = ArrayAttr.get(effect_attrs) if effect_attrs else None
        accesses_attr = ArrayAttr.get(access_attrs) if access_attrs else None
        unknown_dfb_access_attr = UnitAttr.get(self.ctx) if unknown_dfb_access else None

        opaque_call = ttl.opaque_call(
            result_types[0] if result_types else None,
            callee,
            header,
            func_args,
            template_dfb_operands,
            dependency_dfb_operands,
            template_args=template_args_attr,
            unsigned_arg_indices=unsigned_arg_indices_attr,
            dfb_effects=effects_attr,
            dfb_accesses=accesses_attr,
            unknown_dfb_access=unknown_dfb_access_attr,
            condition_result=condition_result_attr,
        )
        if result_types:
            return opaque_call

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
                    ttl.yield_([])
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
