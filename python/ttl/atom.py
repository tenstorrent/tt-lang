# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unified-body ``@ttl.operation`` kernels with automatic thread splitting.

The unified form is a single function body instead of one @ttl.compute and
two @ttl.datamovement thread functions. At decoration time, statement-level
calls to other unified operations are inlined; at compile time the body is
split into compute (TRISC) and data-movement
(NCRISC / BRISC) threads, which then flow through the same MLIR pipeline
as @ttl.operation.

A unified operation may take TT-NN tensors, compile-time captures, and --
when intended for composition -- ttl.DFB or ttl.PipeNet parameters. Dataflow
buffers used within a top-level operation are declared in its body. An
operation with resource parameters is expand-only and cannot be called as a
TT-NN operation.

DFB declarations sit inline with the compute/copy work in a unified
body, so after the split they land inside each thread body. The existing
per-thread compiler is capture-based (thread functions take no
parameters; DFBs arrive as closure captures), so before splitting we
lift the top-level ``name = make_dfb(...)`` assigns out of the body,
evaluate them to DataflowBuffer objects, and pass them as captures --
the same shape the @ttl.operation flow produces by running its setup
body. After that the per-thread compile, CB sizing, pass pipeline, and
runner are identical to @ttl.operation.
"""

from __future__ import annotations

import ast
import copy
import functools
import inspect
import os
import textwrap
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ttl as _ttl
from ttl.pykernel._src.utils import _cleanup_source_code

from ._src.atom_inline import inline_atom_calls
from ._src.atom_split import split_function_body
from ._src.tensor_registry import register_tensor_name
from .compiler_options import CompilerOptions
from .dataflow_buffer import (
    DataflowBuffer,
    _reset_cb_counter,
    make_dataflow_buffer_like,
    make_dfb,
)
from .dtype_utils import is_ttnn_tensor
from .operators import _set_current_grid
from .pipe import PipeNet
from .ttl_api import (
    Program,
    _build_pipenet_graph,
    _canonical_tensor_args,
    _detect_memory_space_from_tensor,
    _lower_program_to_kernel,
    _make_operation_wrapper,
    _require_device,
    _run_thread_compiler,
    _validate_operation_options,
    get_min_remaining_l1_for_device,
    pykernel_gen,
)


# Names whose top-level ``x = <name>(...)`` assigns are lifted out of the body
# and evaluated to capture objects (DataflowBuffer / Pipe / PipeNet) before the
# split, the same way @ttl.operation constructs them in its setup body.
_DFB_FACTORY_NAMES = {"make_dfb", "make_dataflow_buffer_like"}
_PIPE_FACTORY_NAMES = {"Pipe", "PipeNet"}
_SETUP_FACTORY_NAMES = _DFB_FACTORY_NAMES | _PIPE_FACTORY_NAMES


class DFB:
    """Marker annotation for a DataFlow buffer parameter.

    Only meaningful on an operation that is expanded into another operation: the
    caller declares the buffer (ttl.make_dfb / make_dataflow_buffer_like)
    and passes it in, and the inliner substitutes it at the call site.
    """


@dataclass
class _ParamInfo:
    name: str
    kind: str  # "dfb" | "pipenet" | "value"
    is_keyword_only: bool


@dataclass
class _AtomSpec:
    name: str
    fn: Callable
    source: str
    source_file: str
    line_offset: int
    fn_ast: ast.FunctionDef  # post-inline
    params: List[_ParamInfo]
    dfb_param_names: List[str]
    compile_time_captures: Dict[str, Any]
    frozen_scope: Dict[str, Any]
    external_pipenets: Dict[str, PipeNet]


class _ReturnFinder(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Return(self, node):
        self.found = True

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_Lambda(self, node):
        return


def _parse_function_definition(fn: Callable) -> Optional[ast.FunctionDef]:
    try:
        module = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return None
    if len(module.body) != 1:
        return None
    if not isinstance(module.body[0], ast.FunctionDef):
        return None
    return module.body[0]


def _validate_operation_interface(fn: Callable) -> None:
    signature = inspect.signature(fn)
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                "@ttl.operation does not support *args or **kwargs "
                f"(parameter {parameter.name!r})"
            )
        if parameter.default is not inspect.Parameter.empty:
            raise ValueError(
                "@ttl.operation parameters cannot have default values "
                f"(parameter {parameter.name!r})"
            )

    function_definition = _parse_function_definition(fn)
    if function_definition is None:
        return
    finder = _ReturnFinder()
    for statement in function_definition.body:
        finder.visit(statement)
    if finder.found:
        raise ValueError(
            "@ttl.operation functions cannot return a value or use return statements"
        )


def _decorator_name(decorator: ast.expr) -> Optional[str]:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Name):
        return decorator.id
    return None


def _has_explicit_kernels(fn: Callable) -> bool:
    function_definition = _parse_function_definition(fn)
    if function_definition is None:
        return True
    for node in ast.walk(function_definition):
        if node is function_definition:
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if _decorator_name(decorator) in {"compute", "datamovement"}:
                return True
    return False


def _function_scope(fn: Callable) -> Dict[str, Any]:
    """Globals plus resolved closure cells for a function, used as the
    scope for inlining lookups and for evaluating lifted DFB assigns."""
    scope = dict(getattr(fn, "__globals__", {}) or {})
    closure = getattr(fn, "__closure__", None)
    freevars = getattr(fn.__code__, "co_freevars", ())
    if closure:
        for name, cell in zip(freevars, closure):
            try:
                scope[name] = cell.cell_contents
            except ValueError:
                continue
    return scope


def _captured_values(fn: Callable) -> Dict[str, Any]:
    closure = inspect.getclosurevars(fn)
    captures = dict(closure.globals)
    captures.update(closure.nonlocals)
    return captures


def _classify_params(fn: Callable) -> List[_ParamInfo]:
    info: List[_ParamInfo] = []
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                f"@ttl.operation: *args / **kwargs are not allowed (param {pname!r})"
            )
        ann = p.annotation
        if ann is DFB or ann in ("DFB", "ttl.DFB"):
            kind = "dfb"
        elif ann is PipeNet or ann in ("PipeNet", "ttl.PipeNet"):
            kind = "pipenet"
        else:
            kind = "value"
        info.append(
            _ParamInfo(
                name=pname,
                kind=kind,
                is_keyword_only=(p.kind == inspect.Parameter.KEYWORD_ONLY),
            )
        )
    return info


def _build_atom_spec(fn: Callable) -> _AtomSpec:
    name = fn.__name__
    try:
        source_file = inspect.getfile(fn)
    except (TypeError, OSError):
        source_file = "<unknown>"

    raw_lines, start_lineno = inspect.getsourcelines(fn)
    num_decorator_lines = 0
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("@"):
            num_decorator_lines += 1
        elif stripped.startswith("def ") or stripped.startswith("async def "):
            break
    line_offset = start_lineno + num_decorator_lines - 1

    module = ast.parse(_cleanup_source_code(fn))
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise ValueError(
            f"@ttl.operation: expected a single function definition for {name!r}"
        )
    fn_def: ast.FunctionDef = module.body[0]
    scope = _function_scope(fn)

    # Inline statement-level calls to other unified operations, then keep
    # the post-inline AST + source.
    inlined_pipenets = inline_atom_calls(fn_def, scope, caller_name=name)
    _validate_resource_declarations(fn_def, name)

    loaded_names = set()
    for node in ast.walk(fn_def):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded_names.add(node.id)

    captured_values = _captured_values(fn)
    external_pipenets = dict(inlined_pipenets)
    compile_time_captures: Dict[str, Any] = {}
    for capture_name in loaded_names & captured_values.keys():
        value = captured_values[capture_name]
        if isinstance(value, DataflowBuffer):
            raise ValueError(
                f"@ttl.operation {name!r}: external DFB {capture_name!r} is "
                "not supported; declare it as a top-level operation resource "
                "or pass it to an expand-only composed operation"
            )
        if isinstance(value, PipeNet):
            external_pipenets[capture_name] = value
        elif _is_compile_time_literal(value):
            compile_time_captures[capture_name] = copy.deepcopy(value)
        elif not isinstance(value, types.ModuleType) and not callable(value):
            raise TypeError(
                f"@ttl.operation {name!r}: compile-time capture "
                f"{capture_name!r} has unsupported type "
                f"{type(value).__name__}"
            )

    frozen_scope = dict(scope)
    frozen_scope.update(compile_time_captures)
    source = ast.unparse(fn_def)

    params = _classify_params(fn)
    return _AtomSpec(
        name=name,
        fn=fn,
        source=source,
        source_file=source_file,
        line_offset=line_offset,
        fn_ast=fn_def,
        params=params,
        dfb_param_names=[p.name for p in params if p.kind == "dfb"],
        compile_time_captures=compile_time_captures,
        frozen_scope=frozen_scope,
        external_pipenets=external_pipenets,
    )


def _is_compile_time_literal(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_compile_time_literal(element) for element in value)
    return False


def _call_name(node: ast.expr) -> Optional[str]:
    """The callee name of a Call node (``ttl.Pipe`` -> ``Pipe``), else None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_pipe_list_expr(node: ast.expr) -> bool:
    """A list/tuple/comprehension whose elements are all ``ttl.Pipe(...)``.

    Lets a PipeNet be built from a separately-named pipe list
    (``ps = [ttl.Pipe(...) for ...]; net = ttl.PipeNet(ps)``), the natural
    way to express multicast/reduce fan-out.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        return bool(node.elts) and all(_call_name(e) == "Pipe" for e in node.elts)
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return _call_name(node.elt) == "Pipe"
    return False


def _setup_assign_target(stmt: ast.stmt) -> Optional[str]:
    """If ``stmt`` is a DFB/Pipe/PipeNet construction assign, return its name.

    Recognizes ``name = <dfb/pipe-factory>(...)`` and ``name = [<pipes>]``
    (a pipe list feeding a later PipeNet)."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    if not isinstance(stmt.targets[0], ast.Name):
        return None
    if _call_name(stmt.value) in _SETUP_FACTORY_NAMES or _is_pipe_list_expr(stmt.value):
        return stmt.targets[0].id
    return None


def _collect_assignment_targets(target: ast.expr, names: set) -> None:
    if isinstance(target, ast.Name):
        names.add(target.id)
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            _collect_assignment_targets(element, names)


def _non_resource_assignment_names(fn_def: ast.FunctionDef) -> set:
    names = set()
    for statement in fn_def.body:
        if _setup_assign_target(statement) is not None:
            continue
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                _collect_assignment_targets(target, names)
        elif isinstance(statement, ast.AnnAssign):
            _collect_assignment_targets(statement.target, names)
    return names


def _loaded_names_in(node: ast.AST) -> set:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _validate_resource_declarations(
    fn_def: ast.FunctionDef, operation_name: str
) -> None:
    """Require resource construction to use simple top-level assignments."""
    allowed_calls = set()
    resource_statements = []
    for statement in fn_def.body:
        if _setup_assign_target(statement) is None:
            continue
        resource_statements.append(statement)
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) in _SETUP_FACTORY_NAMES:
                allowed_calls.add(id(node))

    local_values = _non_resource_assignment_names(fn_def)
    for statement in resource_statements:
        dependencies = _loaded_names_in(statement) & local_values
        if dependencies:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: resource declarations "
                "cannot depend on operation-local values "
                f"{sorted(dependencies)}"
            )

    for node in ast.walk(fn_def):
        if not isinstance(node, ast.Call):
            continue
        factory_name = _call_name(node)
        if factory_name not in _SETUP_FACTORY_NAMES or id(node) in allowed_calls:
            continue
        raise ValueError(
            f"@ttl.operation {operation_name!r}: resource declaration "
            f"{factory_name!r} must be a simple top-level assignment in the "
            "operation body; declarations inside control flow, callbacks, or "
            "nested scopes are not supported"
        )


def _lift_setup(
    fn_def: ast.FunctionDef,
    scope: Dict[str, Any],
) -> Tuple[ast.FunctionDef, Dict[str, DataflowBuffer], Dict[str, PipeNet]]:
    """Strip and evaluate the top-level DFB / Pipe / PipeNet construction
    assigns.

    Returns the kernel body with those statements removed, plus the
    DataflowBuffer and PipeNet objects keyed by name. Each construction
    expression is evaluated in source order in a controlled namespace, with
    results threaded back in so a later ``PipeNet([p0])`` sees an earlier
    ``p0`` and CB indices are assigned in source order. This runs just the
    setup statements that @ttl.operation runs as part of executing its whole
    body; the per-thread compiler consumes the results as captures, not as
    in-body calls.
    """
    ns = dict(scope)
    ns.setdefault("make_dfb", make_dfb)
    ns.setdefault("make_dataflow_buffer_like", make_dataflow_buffer_like)
    ns.setdefault("ttl", _ttl)

    dfbs: Dict[str, DataflowBuffer] = {}
    nets: Dict[str, PipeNet] = {}
    kept: List[ast.stmt] = []
    for stmt in fn_def.body:
        name = _setup_assign_target(stmt)
        if name is None:
            kept.append(stmt)
            continue
        value = eval(ast.unparse(stmt.value), ns)  # noqa: S307
        ns[name] = value
        if isinstance(value, DataflowBuffer):
            dfbs[name] = value
        elif isinstance(value, PipeNet):
            nets[name] = value
        # A bare Pipe stays in ns for a later PipeNet reference but is not a
        # capture; only DataflowBuffers and PipeNets are bound on threads.

    new_fn = copy.copy(fn_def)
    new_fn.body = kept
    return new_fn, dfbs, nets


def _has_real_work(body: List[ast.stmt]) -> bool:
    """A pruned body is empty if it holds only the ``pass`` placeholder."""
    return any(not isinstance(s, ast.Pass) for s in body)


def _synthesize_thread_module(fn_name: str, body: List[ast.stmt]) -> ast.Module:
    """A module holding one no-arg thread function for TTLGenericCompiler."""
    fn = ast.FunctionDef(
        name=fn_name,
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=copy.deepcopy(body) or [ast.Pass()],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


def _make_thread_callable(spec, kernel_type, fn_name, body, captures):
    def _compile_thread(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["_source_file"] = spec.source_file
        kwargs["_source_lines"] = spec.source.splitlines()
        kwargs["_line_offset"] = spec.line_offset
        return _run_thread_compiler(
            fn_name,
            kernel_type,
            captures,
            spec.frozen_scope,
            (),
            kwargs,
            _synthesize_thread_module(fn_name, body),
            kwargs["_source_lines"],
            spec.source_file,
        )

    return _compile_thread


def _compile_atom(
    spec: _AtomSpec,
    args: tuple,
    kwargs: dict,
    grid,
    num_outs: int,
    memory_space: str,
    tiled: bool,
    program_hash: int,
    fp32_dest_acc_en: Optional[bool],
    dst_full_sync_en: Optional[bool],
    target_arch: Optional[str],
    compiler_options: CompilerOptions,
):

    # The shared operation wrapper supplies values in signature order.
    bound_arguments = {param.name: value for param, value in zip(spec.params, args)}
    eval_scope = dict(spec.frozen_scope)
    eval_scope.update(bound_arguments)

    # Register ttnn tensors so the per-thread compiler can resolve global
    # tensor indices for its tensor accessors.
    for idx, (pname, val) in enumerate(bound_arguments.items()):
        if is_ttnn_tensor(val):
            register_tensor_name(val, pname, index=idx)

    # Detect L1 vs DRAM addressing from the first tensor (matching
    # @ttl.operation), since the tensor accessor type depends on it.
    first_tensor = next(
        (v for v in bound_arguments.values() if is_ttnn_tensor(v)), None
    )
    if first_tensor is not None:
        memory_space = _detect_memory_space_from_tensor(first_tensor, memory_space)

    has_ttnn_tensors = any(is_ttnn_tensor(v) for v in bound_arguments.values())
    l1_budget_override = compiler_options.l1_budget
    if l1_budget_override == 0 and has_ttnn_tensors:
        try:
            l1_budget_override = get_min_remaining_l1_for_device(_require_device(args))
        except ValueError:
            pass

    _reset_cb_counter()
    _set_current_grid(grid)

    stripped_fn, dfbs, nets = _lift_setup(copy.deepcopy(spec.fn_ast), eval_scope)

    # Assign each PipeNet a distinct operation-local id (and validate), the
    # same graph @ttl.operation builds; it also yields the runner's pipe
    # semaphore count.
    all_nets = {}
    for net in spec.external_pipenets.values():
        all_nets[id(net)] = net
    for net in nets.values():
        all_nets[id(net)] = net
    pipe_graph = _build_pipenet_graph(all_nets.values())

    split = split_function_body(
        fn_def=stripped_fn,
        dfb_param_names=set(spec.dfb_param_names),
        local_dfb_names=set(dfbs),
    )

    if os.environ.get("TTLANG_ATOM_DUMP_SPLIT"):
        for _thread in ("trisc", "ncrisc", "brisc"):
            _dbg = _synthesize_thread_module(
                f"{spec.name}__{_thread}", split.body_for(_thread)
            )
            print(f"\n===== @ttl.operation split: {_thread} =====")
            print(ast.unparse(_dbg))

    # Captures shared by every thread: ttnn tensors and scalars (bound
    # values), the lifted DFBs, and the lifted PipeNets. Tensor/DFB captures
    # are included on all threads to keep the tensor-accessor and CB layout
    # stable; unused ones are removed by MLIR DCE.
    captures: Dict[str, Any] = {}
    for pname, val in bound_arguments.items():
        if is_ttnn_tensor(val) or isinstance(val, (int, float)):
            captures[pname] = val
    captures.update(dfbs)
    captures.update(nets)
    captures.update(spec.external_pipenets)

    # TTNN interop requires exactly 3 kernels (1 compute + 2 data movement);
    # emit all three even when a thread has no work, filling it with a pass
    # body, the same shape @ttl.operation produces.
    threads = []
    any_real_work = False
    for kernel_type, thread in (
        ("compute", "trisc"),
        ("datamovement", "ncrisc"),
        ("datamovement", "brisc"),
    ):
        body = split.body_for(thread)
        any_real_work = any_real_work or _has_real_work(body)
        fn_name = f"{spec.name}__{thread}"
        threads.append(
            _make_thread_callable(spec, kernel_type, fn_name, body, captures)
        )

    if not any_real_work:
        raise ValueError(
            f"@ttl.operation '{spec.name}': body contained no compute or data "
            f"movement work after classification"
        )

    injected_program_kwargs = {
        "grid": grid,
        "memory_space": memory_space,
        "tiled": tiled,
        "debug_locations": True,
    }
    program = Program(*threads, args=args, kwargs=injected_program_kwargs)

    return _lower_program_to_kernel(
        program=program,
        args=args,
        launch_grid=grid,
        num_outs=num_outs,
        pipenets=pipe_graph,
        target_arch=target_arch,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
        compiler_options=compiler_options,
        program_hash=program_hash,
        l1_budget_override=l1_budget_override,
        kernel_source_file=spec.source_file,
        kernel_line_offset=spec.line_offset,
    )


def _compile_unified_operation(
    spec,
    decorator_options,
    runtime_args,
    _runtime_kwargs,
    resolved_grid,
    program_hash,
    target_arch,
    compiler_options,
):
    return _compile_atom(
        spec,
        runtime_args,
        {},
        resolved_grid,
        decorator_options["num_outs"],
        decorator_options["memory_space"],
        decorator_options["tiled"],
        program_hash,
        fp32_dest_acc_en=decorator_options["fp32_dest_acc_en"],
        dst_full_sync_en=decorator_options["dst_full_sync_en"],
        target_arch=target_arch,
        compiler_options=compiler_options,
    )


class Atom:
    """Internal representation of a composable unified operation."""

    def __init__(self, spec: _AtomSpec, decorator_options: dict):
        self._spec = spec
        self._grid = decorator_options["grid"]
        self._ttl_operation_kind = "unified"

        expand_only_params = [
            param.name for param in spec.params if param.kind in {"dfb", "pipenet"}
        ]
        compile_callback = functools.partial(
            _compile_unified_operation, spec, decorator_options
        )
        prepare_call = functools.partial(
            _canonical_tensor_args,
            spec.fn,
            expand_only_params=expand_only_params,
        )
        self._wrapper = _make_operation_wrapper(
            spec.fn,
            compile_callback,
            grid=decorator_options["grid"],
            fp32_dest_acc_en=decorator_options["fp32_dest_acc_en"],
            dst_full_sync_en=decorator_options["dst_full_sync_en"],
            options=decorator_options["options"],
            prepare_call=prepare_call,
        )
        functools.update_wrapper(self, spec.fn)

    @property
    def name(self) -> str:
        return self._spec.name

    def __call__(self, *args, **kwargs):
        if self._grid is None:
            raise ValueError(
                f"@ttl.operation {self.name!r} has no grid and is expand-only; "
                "it cannot be called directly"
            )
        return self._wrapper(*args, **kwargs)


def _unified_operation(
    grid: Optional[Union[tuple, Callable]] = None,
    num_outs: int = 1,
    memory_space: str = "L1",
    tiled: bool = True,
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    options: Optional[str] = None,
) -> Callable:
    """Build the unified-body form selected by ``@ttl.operation``.

    Accepts the same compile parameters as @ttl.operation (grid, the fp32
    / dst-sync overrides, compiler options). A grid is required for a
    top-level operation; a composed operation used only for expansion needs none.
    """
    _validate_operation_options(num_outs, memory_space, tiled)

    def _decorator(f):
        spec = _build_atom_spec(f)
        return Atom(
            spec,
            {
                "grid": grid,
                "num_outs": num_outs,
                "memory_space": memory_space,
                "tiled": tiled,
                "fp32_dest_acc_en": fp32_dest_acc_en,
                "dst_full_sync_en": dst_full_sync_en,
                "options": options,
            },
        )

    return _decorator


def operation(
    grid: Optional[Union[tuple, Callable]] = None,
    indexing_maps: Optional[List[Callable]] = None,
    iterator_types: Optional[List[str]] = None,
    num_outs: int = 1,
    memory_space: str = "L1",
    tiled: bool = True,
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    options: Optional[str] = None,
) -> Callable:
    """Define a unified-body or explicit multi-kernel operation."""

    def _decorator(fn):
        _validate_operation_interface(fn)
        explicit_options = indexing_maps is not None or iterator_types is not None
        if explicit_options or _has_explicit_kernels(fn):
            prepare_call = functools.partial(_canonical_tensor_args, fn)
            wrapped = pykernel_gen(
                grid=grid,
                indexing_maps=indexing_maps,
                iterator_types=iterator_types,
                num_outs=num_outs,
                memory_space=memory_space,
                tiled=tiled,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=dst_full_sync_en,
                options=options,
                _prepare_call=prepare_call,
            )(fn)
            wrapped._ttl_operation_kind = "multi_kernel"
            return wrapped

        return _unified_operation(
            grid=grid,
            num_outs=num_outs,
            memory_space=memory_space,
            tiled=tiled,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            options=options,
        )(fn)

    return _decorator
