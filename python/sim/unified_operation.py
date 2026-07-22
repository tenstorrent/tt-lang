# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run thread-unified ``@ttl.operation`` bodies on the simulator.

A unified operation body performs data-movement and compute work directly in
the body (``dfb.reserve()``/``wait()``, ``ttl.copy(...).wait()``, block math),
leaving thread assignment to the compiler. The simulator, however, executes an
operation as three cooperating kernels (one compute + two data movement).

Rather than re-derive thread assignment, this module reuses the compiler
frontend's splitter, ``ttl._src.atom_split.split_function_body``, which returns
three statement bodies -- ``trisc`` (compute), ``ncrisc`` (default data
movement), and ``brisc`` (pipe senders). Those map onto the simulator's
compute / dm0 / dm1 kernels. The unified body is rewritten into an equivalent
multi-kernel function (shared dataflow-buffer construction hoisted into the
outer scope, three nested ``@ttl.compute`` / ``@ttl.datamovement`` kernels
capturing those buffers), which the existing multi-kernel machinery then runs
unchanged.

The splitter is loaded from its source file rather than imported as
``ttl._src.atom_split`` because the simulator shadows ``sys.modules["ttl"]`` with
its own namespace object, which has no importable submodules.
"""

from __future__ import annotations

import ast
import copy
import functools
import importlib.util
import inspect
import sys
import textwrap
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# DFB / pipe factory calls whose results are shared across threads and must be
# constructed once in the outer scope (mirrors ttl.atom's _SETUP_FACTORY_NAMES).
_DFB_FACTORY_NAMES: Set[str] = {"make_dataflow_buffer_like", "make_dfb"}
_PIPE_FACTORY_NAMES: Set[str] = {"Pipe", "PipeNet"}
_SETUP_FACTORY_NAMES: Set[str] = _DFB_FACTORY_NAMES | _PIPE_FACTORY_NAMES

_KERNEL_DECORATORS: Set[str] = {"compute", "datamovement"}


@functools.cache
def _load_atom_split() -> types.ModuleType:
    """Load ``atom_split`` from its source file (cached).

    Tries the bundled copy next to the simulator package first (installed
    ``tt-lang-sim`` wheel), then the compiler frontend location in the source
    tree (``python/ttl/_src/atom_split.py``). It is loaded by path rather than
    imported as ``ttl._src.atom_split`` because the simulator shadows
    ``sys.modules["ttl"]`` with a namespace object that has no submodules.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "atom_split.py",  # bundled into the sim package (wheel)
        here.parent / "ttl" / "_src" / "atom_split.py",  # source tree
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "ttl_sim_atom_split", str(path)
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            # Register before exec so @dataclass in the module can resolve its
            # own __module__ via sys.modules during class processing.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise RuntimeError(
        "could not locate atom_split.py to split a unified @ttl.operation body; "
        f"looked in: {', '.join(str(c) for c in candidates)}"
    )


def _parse_operation_funcdef(func: Callable[..., Any]) -> ast.FunctionDef:
    """Parse ``func``'s source and return its top-level ``FunctionDef``."""
    source = textwrap.dedent("".join(inspect.getsourcelines(func)[0]))
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError(f"could not parse @ttl.operation function {func.__name__!r}")


def _is_kernel_decorator(dec: ast.expr) -> bool:
    """True for ``@ttl.compute`` / ``@ttl.datamovement`` (with or without call)."""
    node = dec.func if isinstance(dec, ast.Call) else dec
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ttl"
        and node.attr in _KERNEL_DECORATORS
    )


def is_unified_body(func: Callable[..., Any]) -> bool:
    """True when ``func`` is a thread-unified operation (no hand-written kernels).

    A multi-kernel operation defines nested ``@ttl.compute`` / ``@ttl.datamovement``
    functions and is left on the legacy execution path. Anything whose source
    cannot be parsed is treated as multi-kernel (legacy), never split.
    """
    try:
        fn_def = _parse_operation_funcdef(func)
    except (OSError, TypeError, ValueError):
        return False
    for node in ast.walk(fn_def):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(_is_kernel_decorator(d) for d in node.decorator_list):
                return False
    return True


def _factory_name(value: ast.expr) -> Optional[str]:
    """Return the factory name if ``value`` is a ``ttl.<factory>(...)`` or bare
    ``<factory>(...)`` call for a known DFB/pipe factory, else None.
    """
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "ttl"
        and func.attr in _SETUP_FACTORY_NAMES
    ):
        return func.attr
    if isinstance(func, ast.Name) and func.id in _SETUP_FACTORY_NAMES:
        return func.id
    return None


def _is_setup_stmt(stmt: ast.stmt) -> bool:
    """True for a top-level ``name = <dfb/pipe factory>(...)`` assignment."""
    return isinstance(stmt, ast.Assign) and _factory_name(stmt.value) is not None


def _local_dfb_names(fn_def: ast.FunctionDef) -> Set[str]:
    """Names bound to ``make_dataflow_buffer_like`` / ``make_dfb`` results."""
    names: Set[str] = set()
    for stmt in fn_def.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and _factory_name(stmt.value) in _DFB_FACTORY_NAMES
        ):
            names.add(stmt.targets[0].id)
    return names


def _strip_setup(body: List[ast.stmt]) -> List[ast.stmt]:
    """Drop DFB/pipe construction from a per-thread body (hoisted to the outer
    scope); return ``[pass]`` if nothing remains.
    """
    kept = [s for s in body if not _is_setup_stmt(s)]
    return kept if kept else [ast.Pass()]


def _make_kernel_def(
    name: str, decorator_attr: str, body: List[ast.stmt]
) -> ast.FunctionDef:
    """Build ``@ttl.<decorator_attr>()\ndef <name>(): <body>``."""
    decorator = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="ttl", ctx=ast.Load()),
            attr=decorator_attr,
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[],
    )
    empty_args = ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    return ast.FunctionDef(
        name=name,
        args=empty_args,
        body=body or [ast.Pass()],
        decorator_list=[decorator],
        returns=None,
        type_comment=None,
        type_params=[],
    )


def _closure_dict(func: Callable[..., Any]) -> Dict[str, Any]:
    """Map ``func``'s captured free variables to their current values.

    Lets compile-time captures resolve by name in the synthesized function,
    which is compiled at module scope (no enclosing cells).
    """
    closure = getattr(func, "__closure__", None)
    if not closure:
        return {}
    freevars = func.__code__.co_freevars
    result: Dict[str, Any] = {}
    for name, cell in zip(freevars, closure):
        try:
            result[name] = cell.cell_contents
        except ValueError:
            pass
    return result


def build_multikernel_function(
    func: Callable[..., Any], namespace: Dict[str, Any]
) -> types.FunctionType:
    """Rewrite unified ``func`` into an equivalent multi-kernel function.

    ``namespace`` is the globals dict the result is compiled into (the
    operation's globals plus ``grid``). Raises ``ValueError`` -- surfaced by the
    caller -- for bodies the splitter rejects (unknown op, DFB acquire resolving
    to multiple threads, mixed compute/DM statement, unsupported assigned copy
    handle).
    """
    atom_split = _load_atom_split()
    fn_def = _parse_operation_funcdef(func)
    fn_def.decorator_list = []  # drop @ttl.operation; do not re-decorate

    local_dfbs = _local_dfb_names(fn_def)

    # Shared prologue: DFB/pipe construction, hoisted once so all three kernels
    # capture the same objects (identity matters for the reserve/wait handshake).
    setup_stmts = [copy.deepcopy(s) for s in fn_def.body if _is_setup_stmt(s)]

    split = atom_split.split_function_body(
        fn_def=fn_def,
        dfb_param_names=set(),
        local_dfb_names=local_dfbs,
    )

    kernels = [
        _make_kernel_def(
            "_ttl_compute", "compute", _strip_setup(split.body_for("trisc"))
        ),
        _make_kernel_def(
            "_ttl_dm0", "datamovement", _strip_setup(split.body_for("ncrisc"))
        ),
        _make_kernel_def(
            "_ttl_dm1", "datamovement", _strip_setup(split.body_for("brisc"))
        ),
    ]

    fn_def.body = setup_stmts + kernels

    module = ast.Module(body=[fn_def], type_ignores=[])
    ast.fix_missing_locations(module)

    try:
        filename = inspect.getfile(func)
    except (OSError, TypeError):
        filename = f"<ttl-unified-operation:{func.__name__}>"

    code = compile(module, filename, "exec")
    exec_ns: Dict[str, Any] = dict(namespace)
    exec_ns.update(_closure_dict(func))
    exec(code, exec_ns)
    return exec_ns[fn_def.name]
