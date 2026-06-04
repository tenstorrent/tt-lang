# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""@ttl.atom: unified-body kernels with automatic thread splitting.

Unlike @ttl.operation, which requires the author to write one
@ttl.compute and two @ttl.datamovement thread functions explicitly, a
@ttl.atom is a single function body. At decoration time, statement-level
calls to other @ttl.atom functions are inlined; at compile time the
unified body is split into compute (TRISC) and data-movement
(NCRISC / BRISC) threads, which then flow through the same MLIR pipeline
as @ttl.operation.

An atom may take ttnn tensors (resolved at the call site, exactly like
@ttl.operation), compile-time scalars, and -- for atoms intended to be
inlined into another atom -- ttl.DFB parameters. DataFlow buffers used
within a top-level atom are declared in the body via ttl.make_dfb /
ttl.make_dataflow_buffer_like; a DFB-parameter atom has no way to be
supplied a buffer except by being inlined, so it is never a JIT entry
point.

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
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from .pipe import PipeNet


# Names whose top-level ``x = <name>(...)`` assigns are lifted out of the body
# and evaluated to capture objects (DataflowBuffer / Pipe / PipeNet) before the
# split, the same way @ttl.operation constructs them in its setup body.
_DFB_FACTORY_NAMES = {"make_dfb", "make_dataflow_buffer_like"}
_PIPE_FACTORY_NAMES = {"Pipe", "PipeNet"}
_SETUP_FACTORY_NAMES = _DFB_FACTORY_NAMES | _PIPE_FACTORY_NAMES


class DFB:
    """Marker annotation for a DataFlow buffer parameter.

    Only meaningful on an atom that is inlined into another atom: the
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
    inlined_dfb_tags: Dict[str, int]  # inlined scratch DFB name -> inline site id


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


def _classify_params(fn: Callable) -> List[_ParamInfo]:
    from .pipe import PipeNet

    info: List[_ParamInfo] = []
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                f"@ttl.atom: *args / **kwargs are not allowed (param {pname!r})"
            )
        ann = p.annotation
        if ann is DFB:
            kind = "dfb"
        elif ann is PipeNet:
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
    from ttl.pykernel._src.utils import _cleanup_source_code

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
            f"@ttl.atom: expected a single function definition for {name!r}"
        )
    fn_def: ast.FunctionDef = module.body[0]

    # Inline statement-level calls to other @ttl.atom functions, then keep
    # the post-inline AST + source.
    inlined_dfb_tags = inline_atom_calls(
        fn_def, _function_scope(fn), caller_name=name
    )
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
        inlined_dfb_tags=inlined_dfb_tags,
    )


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
    if _call_name(stmt.value) in _SETUP_FACTORY_NAMES or _is_pipe_list_expr(
        stmt.value
    ):
        return stmt.targets[0].id
    return None


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
    import ttl as _ttl

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


def _cb_configs_from_lifted(lifted: Dict[str, DataflowBuffer]):
    """DataflowBuffer list indexed by CB index, matching _collect_cb_configs."""
    by_index = {dfb._cb_index: dfb for dfb in lifted.values()}
    if not by_index:
        return []
    return [by_index.get(i) for i in range(max(by_index) + 1)]


def _reuse_inlined_dfb_indices(
    dfbs: Dict[str, DataflowBuffer],
    inlined_dfb_tags: Dict[str, int],
) -> None:
    """Overlay the scratch DFBs of distinct inline sites onto shared CB indices.

    Each inlined callee's body is substituted as one contiguous block, so its
    scratch DFBs are confined to that inline site. Sibling sites are sequenced
    by the bridge DFBs that carry data between them (a site's outputs are only
    pushed once its scratch has quiesced, and the next site waits on those
    outputs), so identically-configured scratch from *different* sites can share
    a CB index / L1 allocation.

    This is structural, not lifetime-based: scratch within one site stays
    distinct (it is that atom's working set, live across its own async threads),
    and caller-declared (bridge) DFBs are left untouched. We never consult the
    statement order of the unified body, because after the thread split the
    compute and data-movement statements run concurrently and that order is not
    a runtime order.
    """
    if not inlined_dfb_tags:
        return

    def cfg(name: str) -> tuple:
        d = dfbs[name]
        return (tuple(d.shape), d.block_count, d.dtype)

    bridges = [n for n in dfbs if n not in inlined_dfb_tags]
    # site id -> CB config -> scratch names declared by that site.
    sites: Dict[int, Dict[tuple, List[str]]] = {}
    for name in dfbs:
        if name not in inlined_dfb_tags:
            continue
        sites.setdefault(inlined_dfb_tags[name], {}).setdefault(cfg(name), []).append(
            name
        )
    if not sites:
        return

    # Per config, the overlaid width is the most any single site needs; lay the
    # configs out contiguously above the bridge DFBs.
    cfg_width: Dict[tuple, int] = {}
    for per_cfg in sites.values():
        for c, names in per_cfg.items():
            cfg_width[c] = max(cfg_width.get(c, 0), len(names))

    base = len(bridges)
    cfg_base: Dict[tuple, int] = {}
    offset = 0
    for c, width in cfg_width.items():
        cfg_base[c] = base + offset
        offset += width

    # Bridges keep indices [0, base) in their original order; every site
    # overlays the same per-config block, so slot k of one site shares an
    # index with slot k of any sibling site of the same config.
    for new_index, name in enumerate(sorted(bridges, key=lambda n: dfbs[n]._cb_index)):
        dfbs[name]._cb_index = new_index
    for per_cfg in sites.values():
        for c, names in per_cfg.items():
            for slot, name in enumerate(names):
                dfbs[name]._cb_index = cfg_base[c] + slot


def _make_thread_callable(spec, kernel_type, fn_name, body, captures):
    from .ttl_api import _run_thread_compiler

    def _compile_thread(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["_source_file"] = spec.source_file
        kwargs["_source_lines"] = spec.source.splitlines()
        kwargs["_line_offset"] = spec.line_offset
        return _run_thread_compiler(
            fn_name,
            kernel_type,
            captures,
            _function_scope(spec.fn),
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
    from .operators import _set_current_grid
    from .ttl_api import (
        Program,
        _build_pipenet_graph,
        _detect_memory_space_from_tensor,
        _lower_program_to_kernel,
        _require_device,
        get_min_remaining_l1_for_device,
    )

    # Bind call-site values to parameter names.
    bound = inspect.signature(spec.fn).bind(*args, **kwargs)
    eval_scope = dict(_function_scope(spec.fn))
    eval_scope.update(bound.arguments)

    # Register ttnn tensors so the per-thread compiler can resolve global
    # tensor indices for its tensor accessors.
    for idx, (pname, val) in enumerate(bound.arguments.items()):
        if is_ttnn_tensor(val):
            register_tensor_name(val, pname, index=idx)

    # Detect L1 vs DRAM addressing from the first tensor (matching
    # @ttl.operation), since the tensor accessor type depends on it.
    first_tensor = next(
        (v for v in bound.arguments.values() if is_ttnn_tensor(v)), None
    )
    if first_tensor is not None:
        memory_space = _detect_memory_space_from_tensor(first_tensor, memory_space)

    has_ttnn_tensors = any(is_ttnn_tensor(v) for v in bound.arguments.values())
    l1_budget_override = compiler_options.l1_budget
    if l1_budget_override == 0 and has_ttnn_tensors:
        try:
            l1_budget_override = get_min_remaining_l1_for_device(_require_device(args))
        except ValueError:
            pass

    _reset_cb_counter()
    _set_current_grid(grid)

    stripped_fn, dfbs, nets = _lift_setup(spec.fn_ast, eval_scope)
    _reuse_inlined_dfb_indices(dfbs, spec.inlined_dfb_tags)

    # Assign each PipeNet a distinct operation-local id (and validate), the
    # same graph @ttl.operation builds; it also yields the runner's pipe
    # semaphore count.
    pipe_graph = _build_pipenet_graph(nets.values())

    split = split_function_body(
        fn_def=stripped_fn,
        dfb_param_names=set(spec.dfb_param_names),
        all_param_names={p.name for p in spec.params},
        local_dfb_names=set(dfbs),
    )

    if os.environ.get("TTLANG_ATOM_DUMP_SPLIT"):
        for _thread in ("trisc", "ncrisc", "brisc"):
            _dbg = _synthesize_thread_module(
                f"{spec.name}__{_thread}", split.body_for(_thread)
            )
            print(f"\n===== @ttl.atom split: {_thread} =====")
            print(ast.unparse(_dbg))

    # Captures shared by every thread: ttnn tensors and scalars (bound
    # values), the lifted DFBs, and the lifted PipeNets. Tensor/DFB captures
    # are included on all threads to keep the tensor-accessor and CB layout
    # stable; unused ones are removed by MLIR DCE.
    captures: Dict[str, Any] = {}
    for pname, val in bound.arguments.items():
        if is_ttnn_tensor(val) or isinstance(val, (int, float)):
            captures[pname] = val
    captures.update(dfbs)
    captures.update(nets)

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
            f"@ttl.atom '{spec.name}': body contained no compute or data "
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
        cb_configs=_cb_configs_from_lifted(dfbs),
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


class Atom:
    """A @ttl.atom kernel: inline-able as a callee, JIT-compiled at top level."""

    def __init__(self, spec: _AtomSpec, decorator_opts: dict):
        self._spec = spec
        self._opts = decorator_opts
        self._kernel_id = random.getrandbits(64)
        self._cache: Dict[tuple, Any] = {}
        functools.update_wrapper(self, spec.fn)

    @property
    def name(self) -> str:
        return self._spec.name

    def __call__(self, *args, **kwargs):
        from .ttl_api import (
            _device_target_arch,
            _make_cache_key,
            _resolve_grid,
            _should_execute,
        )

        opts = self._opts
        resolved_grid = _resolve_grid(opts["grid"], args, kwargs)

        opts_str = kwargs.pop("options", opts["options"])
        env_opts = os.environ.get("TTLANG_COMPILER_OPTIONS")
        if env_opts:
            opts_str = f"{opts_str or ''} {env_opts}".strip() or None
        compiler_options = CompilerOptions.from_string(opts_str).merge(
            CompilerOptions.from_argv()
        )
        target_arch = _device_target_arch(args)

        cache_key = _make_cache_key(
            args,
            fp32_dest_acc_en=opts["fp32_dest_acc_en"],
            dst_full_sync_en=opts["dst_full_sync_en"],
            target_arch=target_arch,
            compiler_options=compiler_options,
        )

        compiled_kernel = self._cache.get(cache_key)
        if compiled_kernel is None:
            compiled_kernel = _compile_atom(
                self._spec,
                args,
                kwargs,
                resolved_grid,
                opts["num_outs"],
                opts["memory_space"],
                opts["tiled"],
                hash((self._kernel_id, cache_key)),
                fp32_dest_acc_en=opts["fp32_dest_acc_en"],
                dst_full_sync_en=opts["dst_full_sync_en"],
                target_arch=target_arch,
                compiler_options=compiler_options,
            )
            if compiled_kernel is not None:
                self._cache[cache_key] = compiled_kernel

        if compiled_kernel is not None and _should_execute():
            return compiled_kernel(*args)
        return None


def atom(
    grid: Optional[Union[tuple, Callable]] = None,
    num_outs: int = 1,
    memory_space: str = "L1",
    tiled: bool = True,
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    options: Optional[str] = None,
) -> Callable:
    """Decorator for a unified-body, auto-split @ttl.atom kernel.

    Accepts the same compile parameters as @ttl.operation (grid, the fp32
    / dst-sync overrides, compiler options). A grid is required for a
    top-level atom; an atom used only as an inlined callee needs none.
    """
    if num_outs != 1:
        raise ValueError(f"num_outs must be 1, got {num_outs}")

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
