# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unified-body ``@ttl.operation`` kernels with logical-kernel splitting.

The unified form is a single function body instead of one @ttl.compute and
two @ttl.datamovement functions. At decoration time, statement-level calls to
other unified operations are inlined. At compile time, the body is split into
target-independent compute and data-movement kernels. Backend assignment then
maps those logical kernels to the target's supported kernel slots.

A unified operation may take TT-NN tensors, compile-time captures, and --
when intended for composition -- ttl.DFB or ttl.PipeNet parameters. Dataflow
buffers used within a top-level operation are declared in its body. An
operation with resource parameters is expand-only and cannot be called as a
TT-NN operation.

DFB declarations sit inline with the compute/copy work in a unified body. The
existing per-kernel compiler is capture-based, so top-level static resource
assignments are lifted before splitting and supplied as captures. The
per-kernel compile, DFB sizing, pass pipeline, and runner remain identical to
the explicit multi-kernel form.
"""

from __future__ import annotations

import ast
import copy
import functools
import inspect
import os
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import ttl as _ttl
from ttl.pykernel._src.utils import _cleanup_source_code

from ._src.atom_inline import inline_atom_calls
from ._src.atom_rules import (
    defines_kernels_by_spelling,
    function_scope,
    parse_function_definition,
    setup_assign_target,
    validate_operation_interface,
    validate_resource_declarations,
)
from ._src.atom_split import split_function_body
from ._src.tensor_registry import register_tensor_name
from .compiler_options import CompilerOptions
from .dataflow_buffer import (
    DataflowBuffer,
    _reset_cb_counter,
    make_dataflow_buffer_like,
    make_dfb,
    make_tensor_backed_dfb,
)
from .dtype_utils import is_ttnn_tensor
from .kernel import (
    Kernel,
    KernelSelector,
    _bind_kernel_declarations,
    _operation_identity,
    _selector_implicit_role,
    _selector_kind,
)
from .operators import _set_current_grid
from .pipe import PipeNet
from .ttl_api import (
    Program,
    _BackendKernelSlot,
    _backend_kernel_capacities,
    _backend_kernel_slots,
    _build_pipenet_graph,
    _canonical_tensor_args,
    _detect_memory_space_from_tensor,
    _lower_program_to_kernel,
    _make_operation_wrapper,
    _run_thread_compiler,
    _slot_idle_kernel,
    _validate_operation_options,
    pykernel_gen,
)


def _assign_backend_kernel_slots(
    split, target_arch: Optional[str] = None
) -> Dict[_BackendKernelSlot, KernelSelector]:
    assignments: Dict[_BackendKernelSlot, KernelSelector] = {}
    remaining = list(split.kernels)
    backend_slots = _backend_kernel_slots(target_arch)

    for slot in backend_slots:
        if slot.implicit_role is None:
            selector: KernelSelector = slot.kind
            if selector in remaining:
                assignments[slot] = selector
                remaining.remove(selector)
            continue
        selector = next(
            (
                kernel
                for kernel in remaining
                if _selector_implicit_role(kernel) == slot.implicit_role
            ),
            None,
        )
        if selector is not None:
            assignments[slot] = selector
            remaining.remove(selector)

    for selector in remaining:
        slot = next(
            (
                candidate
                for candidate in backend_slots
                if candidate not in assignments
                and candidate.kind == _selector_kind(selector)
            ),
            None,
        )
        if slot is None:
            raise AssertionError(
                f"no backend slot for planned {selector!r}; capacity validation "
                "must reject this before backend assignment"
            )
        assignments[slot] = selector
    return assignments


def _backend_kernel_bodies(
    split,
    assignments: Mapping[_BackendKernelSlot, KernelSelector],
    target_arch: Optional[str],
) -> Tuple[Tuple[_BackendKernelSlot, KernelSelector, List[ast.stmt]], ...]:
    """Pair every backend slot with a logical kernel and the body it emits.

    A slot the plan left unassigned still produces a kernel, so it takes its idle
    logical identity rather than none; runtime resources can then select every
    emitted kernel by identity.
    """
    bodies = []
    for slot in _backend_kernel_slots(target_arch):
        logical_kernel = assignments.get(slot)
        if logical_kernel is None:
            bodies.append((slot, _slot_idle_kernel(slot), [ast.Pass()]))
            continue
        bodies.append((slot, logical_kernel, split.body_for(logical_kernel)))

    selectors = [logical_kernel for _, logical_kernel, _ in bodies]
    if len(set(selectors)) != len(selectors):
        raise AssertionError(
            f"backend slots produced duplicate logical identities {selectors!r}; "
            "each emitted kernel must be selectable by identity"
        )
    return tuple(bodies)


class DFB:
    """Marker annotation for a DataFlow buffer parameter.

    Only meaningful on an operation that is expanded into another operation: the
    caller declares the buffer with a DFB factory and passes it in, and the
    inliner substitutes it at the call site.
    """


@dataclass
class _ParamInfo:
    name: str
    kind: str  # "dfb" | "pipenet" | "value"
    is_keyword_only: bool


@dataclass
class _AtomSpec:
    name: str
    operation_identity: str
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
    logical_kernels: Dict[str, Kernel]


def _has_explicit_kernels(fn: Callable) -> bool:
    function_definition = parse_function_definition(fn)
    if function_definition is None:
        return True
    return defines_kernels_by_spelling(function_definition)


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
    scope = function_scope(fn)

    # Inline statement-level calls to other unified operations, then keep
    # the post-inline AST + source.
    inlined_pipenets, inlined_logical_kernels = inline_atom_calls(
        fn_def, scope, caller_name=name
    )
    validate_resource_declarations(fn_def, name)

    loaded_names = set()
    for node in ast.walk(fn_def):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded_names.add(node.id)

    captured_values = _captured_values(fn)
    external_pipenets = dict(inlined_pipenets)
    compile_time_captures: Dict[str, Any] = {}
    logical_kernels: Dict[str, Kernel] = dict(inlined_logical_kernels)
    captured_logical_kernels: Dict[str, Kernel] = {}
    for capture_name in sorted(loaded_names & captured_values.keys()):
        value = captured_values[capture_name]
        if isinstance(value, DataflowBuffer):
            raise ValueError(
                f"@ttl.operation {name!r}: external DFB {capture_name!r} is "
                "not supported; declare it as a top-level operation resource "
                "or pass it to an expand-only composed operation"
            )
        if isinstance(value, PipeNet):
            external_pipenets[capture_name] = value
        elif isinstance(value, Kernel):
            if not any(value is kernel for kernel in logical_kernels.values()):
                captured_logical_kernels[capture_name] = value
        elif _is_compile_time_literal(value):
            compile_time_captures[capture_name] = copy.deepcopy(value)
        elif not isinstance(value, types.ModuleType) and not callable(value):
            raise TypeError(
                f"@ttl.operation {name!r}: compile-time capture "
                f"{capture_name!r} has unsupported type "
                f"{type(value).__name__}"
            )

    operation_identity = _operation_identity(fn)
    _bind_logical_kernels(captured_logical_kernels, operation_identity)
    logical_kernels.update(captured_logical_kernels)

    frozen_scope = dict(scope)
    frozen_scope.update(compile_time_captures)
    frozen_scope.update(logical_kernels)
    source = ast.unparse(fn_def)

    params = _classify_params(fn)
    return _AtomSpec(
        name=name,
        operation_identity=operation_identity,
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
        logical_kernels=logical_kernels,
    )


def _bind_logical_kernels(
    logical_kernels: Dict[str, Kernel], operation_identity: str
) -> None:
    """Bind captured declarations in place during operation registration."""
    _bind_kernel_declarations(logical_kernels, operation_identity)


def _is_compile_time_literal(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_compile_time_literal(element) for element in value)
    return False


def _lift_setup(
    fn_def: ast.FunctionDef,
    scope: Dict[str, Any],
    operation_identity: str,
) -> Tuple[
    ast.FunctionDef,
    Dict[str, DataflowBuffer],
    Dict[str, PipeNet],
    Dict[str, Kernel],
]:
    """Strip and evaluate top-level static operation-resource assignments.

    Returns the kernel body with those statements removed, plus the
    DataflowBuffer, PipeNet, and bound logical Kernel objects keyed by name.
    Construction expressions are evaluated in source order so dependencies and
    DFB indices remain deterministic.
    """
    ns = dict(scope)
    ns.setdefault("make_dfb", make_dfb)
    ns.setdefault("make_dataflow_buffer_like", make_dataflow_buffer_like)
    ns.setdefault("make_tensor_backed_dfb", make_tensor_backed_dfb)
    ns.setdefault("Kernel", Kernel)
    ns.setdefault("ttl", _ttl)

    dfbs: Dict[str, DataflowBuffer] = {}
    nets: Dict[str, PipeNet] = {}
    kernels: Dict[str, Kernel] = {}
    kept: List[ast.stmt] = []
    for stmt in fn_def.body:
        name = setup_assign_target(stmt)
        if name is None:
            kept.append(stmt)
            continue
        value = eval(ast.unparse(stmt.value), ns)  # noqa: S307
        if isinstance(value, DataflowBuffer):
            dfbs[name] = value
        elif isinstance(value, PipeNet):
            nets[name] = value
        elif isinstance(value, Kernel):
            value._bind(name, operation_identity)
            kernels[name] = value
        ns[name] = value
        # A bare Pipe remains in ns for a later PipeNet reference.

    new_fn = copy.copy(fn_def)
    new_fn.body = kept
    return new_fn, dfbs, nets, kernels


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
    math_fidelity: Optional[str],
    target_arch: Optional[str],
    compiler_options: CompilerOptions,
    l1_budget_override: int,
):

    # The shared operation wrapper supplies values in signature order.
    bound_arguments = {param.name: value for param, value in zip(spec.params, args)}
    logical_kernels = dict(spec.logical_kernels)
    eval_scope = dict(spec.frozen_scope)
    eval_scope.update(logical_kernels)
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

    _reset_cb_counter()
    _set_current_grid(grid)

    stripped_fn, dfbs, nets, lifted_logical_kernels = _lift_setup(
        copy.deepcopy(spec.fn_ast),
        eval_scope,
        operation_identity=spec.operation_identity,
    )
    logical_kernels.update(lifted_logical_kernels)
    selector_scope = dict(eval_scope)
    selector_scope.update(logical_kernels)

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
        logical_kernels=logical_kernels,
        selector_scope=selector_scope,
        kernel_capacities=_backend_kernel_capacities(target_arch),
    )
    backend_assignments = _assign_backend_kernel_slots(split, target_arch)
    backend_bodies = tuple(
        _backend_kernel_bodies(split, backend_assignments, target_arch)
    )

    if os.environ.get("TTLANG_ATOM_DUMP_SPLIT"):
        for slot, _, body in backend_bodies:
            _dbg = _synthesize_thread_module(f"{spec.name}__{slot.source_name}", body)
            print(f"\n===== @ttl.operation split: {slot.source_name} =====")
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

    # TTNN interop requires one emitted thread for every backend slot. Empty
    # slots retain a pass body so argument metadata stays aligned with slot order.
    threads = []
    thread_logical_kernels = []
    any_real_work = False
    for slot, logical_kernel, body in backend_bodies:
        any_real_work = any_real_work or _has_real_work(body)
        fn_name = f"{spec.name}__{slot.source_name}"
        threads.append(
            _make_thread_callable(spec, slot.kernel_type, fn_name, body, captures)
        )
        thread_logical_kernels.append(logical_kernel)

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
        math_fidelity=math_fidelity,
        compiler_options=compiler_options,
        program_hash=program_hash,
        l1_budget_override=l1_budget_override,
        kernel_source_file=spec.source_file,
        kernel_line_offset=spec.line_offset,
        logical_kernels=thread_logical_kernels,
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
    l1_budget_override,
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
        math_fidelity=decorator_options["math_fidelity"],
        target_arch=target_arch,
        compiler_options=compiler_options,
        l1_budget_override=l1_budget_override,
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
            math_fidelity=decorator_options["math_fidelity"],
            options=decorator_options["options"],
            prepare_call=prepare_call,
        )
        functools.update_wrapper(self, spec.fn)

    @property
    def name(self) -> str:
        return self._spec.name

    def _operation_identity_capture(self) -> tuple[str, str]:
        return ("operation", self._spec.operation_identity)

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
    math_fidelity: Optional[str] = None,
    options: Optional[str] = None,
) -> Callable:
    """Build the unified-body form selected by ``@ttl.operation``.

    Accepts the same compile parameters as @ttl.operation (grid, the fp32
    / dst-sync overrides, compiler options). A grid is required for a
    top-level operation; a composed operation used only for expansion needs none.
    """
    _validate_operation_options(num_outs, memory_space, tiled, math_fidelity)

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
                "math_fidelity": math_fidelity,
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
    math_fidelity: Optional[str] = None,
    options: Optional[str] = None,
) -> Callable:
    """Define a unified-body or explicit multi-kernel operation."""

    def _decorator(fn):
        validate_operation_interface(fn)
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
                math_fidelity=math_fidelity,
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
            math_fidelity=math_fidelity,
            options=options,
        )(fn)

    return _decorator
