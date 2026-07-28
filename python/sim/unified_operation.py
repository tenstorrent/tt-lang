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

Duplicated rules
----------------
Several AST-level rules here restate ones the compiler frontend already has in
``python/ttl/atom.py``, which cannot be imported: its module-level imports pull
in the MLIR bindings (``ttl.ttl_api``) and the compiler's own DataflowBuffer,
neither of which ships in the ``tt-lang-sim`` wheel. The counterparts are named
at each site below (``_SETUP_FACTORY_NAMES``, ``_symbol_table``,
``_has_kernel_decorator_spelling``, ``_factory_name``).

The rules themselves are stdlib-only, so they can become shared code that both
frontends call, parameterized on the facts that differ per frontend (the kernel
decorator objects to compare against, and the names that refer to the API). That
touches compiler-owned code with its own coverage, so it is left as a follow-up:
TODO(#779). Keep the duplicates in sync until then.
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
# constructed once in the outer scope. Duplicates ttl/atom.py's identically
# named sets (see "Duplicated rules" above).
_DFB_FACTORY_NAMES: Set[str] = {"make_dataflow_buffer_like", "make_dfb"}
_PIPE_FACTORY_NAMES: Set[str] = {"Pipe", "PipeNet"}
_SETUP_FACTORY_NAMES: Set[str] = _DFB_FACTORY_NAMES | _PIPE_FACTORY_NAMES

_KERNEL_DECORATORS: Set[str] = {"compute", "datamovement"}

# Names the synthesized kernel decorators are bound to in the generated
# function's namespace. Generated code must not depend on how the operation's
# module imported the TT-Lang API, so it references the decorator objects
# directly instead of spelling ``ttl.compute`` / ``ttl.datamovement``.
_COMPUTE_BINDING = "__ttl_sim_compute__"
_DATAMOVEMENT_BINDING = "__ttl_sim_datamovement__"

# Sentinel for "this expression does not resolve to a runtime object".
_UNRESOLVED: Any = object()


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
    """Parse ``func``'s source and return its top-level ``FunctionDef``.

    Line numbers are rebased onto the enclosing file, because the synthesized
    kernels are compiled under that file's name and the rest of the simulator
    reads them as absolute: ``analysis.py`` re-derives each kernel's source with
    ``inspect.getsourcelines`` (which starts from ``co_firstlineno``) and keys
    copy-wait injection points on absolute line numbers, and the splitter quotes
    line numbers in its error messages. Leaving the parse numbered from 1 makes
    that lookup read the top of the file instead of the operation body, so no
    injection points match and diagnostics point at unrelated lines.
    """
    source_lines, file_start_line = inspect.getsourcelines(func)
    source = textwrap.dedent("".join(source_lines))
    tree = ast.parse(source)
    ast.increment_lineno(tree, file_start_line - 1)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError(f"could not parse @ttl.operation function {func.__name__!r}")


def _symbol_table(func: Callable[..., Any]) -> Dict[str, Any]:
    """Names visible to ``func``: its globals plus its captured free variables.

    Duplicates ``ttl/atom.py::_function_scope`` (see "Duplicated rules" above).
    """
    table: Dict[str, Any] = dict(getattr(func, "__globals__", {}))
    table.update(_closure_dict(func))
    return table


def _resolve_expr(node: ast.expr, symbols: Dict[str, Any]) -> Any:
    """Resolve a dotted-name expression to its runtime object, or ``_UNRESOLVED``.

    Handles ``name`` and ``a.b.c`` forms, which covers every way a decorator or
    factory can be referenced. Anything else (subscripts, calls, ...) and any
    attribute lookup that fails resolves to ``_UNRESOLVED``.
    """
    if isinstance(node, ast.Name):
        return symbols.get(node.id, _UNRESOLVED)
    if isinstance(node, ast.Attribute):
        base = _resolve_expr(node.value, symbols)
        if base is _UNRESOLVED:
            return _UNRESOLVED
        try:
            return getattr(base, node.attr)
        except Exception:
            return _UNRESOLVED
    return _UNRESOLVED


def _kernel_decorator_objects() -> tuple[Any, ...]:
    """The kernel decorators a hand-written multi-kernel body can reference."""
    from .decorators import compute, datamovement

    return (compute, datamovement)


def _has_kernel_decorator_spelling(node: ast.expr) -> bool:
    """True when ``node`` is spelled like a kernel decorator, ignoring the receiver.

    Used only as a fallback when the decorator cannot be resolved to an object;
    matching on the attribute alone (``<anything>.compute``) errs toward
    classifying a body as multi-kernel, which is the safe direction.

    This is exactly the rule ``ttl/atom.py::_decorator_name`` /
    ``_has_explicit_kernels`` applies (see "Duplicated rules" above); resolving
    the object first, as :func:`_is_kernel_decorator` does, additionally
    distinguishes an unrelated decorator that happens to be named ``compute``.
    """
    if isinstance(node, ast.Attribute):
        return node.attr in _KERNEL_DECORATORS
    if isinstance(node, ast.Name):
        return node.id in _KERNEL_DECORATORS
    return False


def _is_kernel_decorator(dec: ast.expr, symbols: Dict[str, Any]) -> bool:
    """True when ``dec`` decorates a hand-written compute / datamovement kernel.

    Resolves the decorator to its runtime object rather than matching the source
    spelling, so ``@ttl.compute()``, ``@T.compute()`` after ``import ttl as T``,
    and ``@compute()`` after ``from ttl import compute`` are all recognized.
    Misclassifying a multi-kernel body as unified splits it and silently
    produces a wrong answer, so unresolvable decorators fall back to a
    spelling check, which errs toward "multi-kernel".
    """
    node = dec.func if isinstance(dec, ast.Call) else dec
    resolved = _resolve_expr(node, symbols)
    if resolved is _UNRESOLVED:
        return _has_kernel_decorator_spelling(node)
    if any(resolved is obj for obj in _kernel_decorator_objects()):
        return True
    # A decorator from a different build of the API (unshadowed ttl, reloaded
    # module) is not object-identical to ours but still names a kernel.
    return getattr(resolved, "__name__", None) in _KERNEL_DECORATORS


def _receiver_root_name(func: ast.expr) -> Optional[str]:
    """The leading name of an attribute call's receiver (``T.math.exp`` -> ``T``)."""
    node: ast.expr = func
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _thread_pinning_api_ops(api: Any) -> Dict[int, str]:
    """``id()`` -> op name for every API call that pins a thread.

    Read out of ``atom_split``'s registry rather than restated here, and keyed on
    object identity so a renamed binding (``cp = ttl.copy``) is still recognized.
    Control ops are excluded: the splitter replicates those onto every thread by
    design, so reaching one under another name changes nothing.
    """
    ops: Dict[int, str] = {}
    for name, thread in _load_atom_split()._TTL_OPS.items():
        if thread == "control":
            continue
        op = getattr(api, name, None)
        if op is not None:
            ops[id(op)] = name
    return ops


def _aliased_api_calls(fn_def: ast.FunctionDef, symbols: Dict[str, Any]) -> List[str]:
    """Calls in ``fn_def`` that reach the API under a name the splitter ignores.

    Returns each offending spelling with its line number, for the error message.
    """
    api = sys.modules.get("ttl")
    if api is None:
        return []
    pinning_ops = _thread_pinning_api_ops(api)
    found: List[str] = []
    for node in ast.walk(fn_def):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            root = _receiver_root_name(func)
            if root is not None and root != "ttl" and symbols.get(root) is api:
                found.append(f"'{ast.unparse(func)}(...)' on line {node.lineno}")
        elif isinstance(func, ast.Name):
            resolved = symbols.get(func.id, _UNRESOLVED)
            if resolved is not _UNRESOLVED and id(resolved) in pinning_ops:
                op = pinning_ops[id(resolved)]
                found.append(f"'{func.id}(...)' (ttl.{op}) on line {node.lineno}")
    return found


def _reject_aliased_api(fn_def: ast.FunctionDef, symbols: Dict[str, Any]) -> None:
    """Reject a unified body that reaches the API under a name the splitter ignores.

    ``atom_split`` decides which thread a statement belongs to by matching the
    receiver name ``ttl`` (``atom_split._classify_ttl_call``), and
    :func:`_factory_name` recognizes DFB construction the same way, so a call
    spelled ``T.copy(...)`` anchors nothing. The body is then split on incomplete
    information: an unanchored statement is replicated onto all three threads, or
    the split fails claiming a block has no uses when its only use is the call
    that went unrecognized. Name the spelling to fix instead.

    Not detected, because the receiver does not resolve to the API module: a name
    bound to one of its namespaces (``M = ttl.math`` and then ``M.exp(...)``).

    There is no counterpart in ``ttl/atom.py``: the compiler is exposed to the
    same gap, and to a sharper form of it, since it recognizes aliased DFB
    construction (its ``_call_name`` ignores the receiver) and so reaches the
    splitter with anchors missing. Teaching ``_classify_ttl_call`` the set of
    names that refer to the API would fix both and retire this guard -- see #779.

    Rejecting rather than resolving the alias is deliberate on two grounds. The
    compiler cannot split an aliased body either, so accepting one here would let
    the simulator pass a program that mis-splits once compiled; of the two ways to
    diverge from the compiler, the permissive one is the harmful one. And the spec
    is silent on how the API must be bound -- every example spells ``ttl.<op>``,
    but no rule requires that name -- so alias support is an extension rather than
    a conformance requirement, whereas "Thread assignment" does require that an
    operation the compiler does not recognize be an error instead of taking a
    default thread. Erroring here is the conformant behavior.

    Supporting aliases later must not rewrite the body. The synthesized kernels
    are compiled under the original file's name and line numbers precisely so that
    Python tooling shows the user's own source, and canonicalizing receivers would
    leave the displayed line disagreeing with the code that runs. Classification
    would have to read a canonicalized throwaway copy while the kernels keep the
    original statements.

    Raises:
        ValueError: If any call reaches the API under another name.
    """
    spellings = _aliased_api_calls(fn_def, symbols)
    if not spellings:
        return
    raise ValueError(
        f"the TT-Lang API is reached as {', '.join(spellings)}, but a "
        f"thread-unified body must reference it as 'ttl' (e.g. 'ttl.copy(...)'), "
        f"because thread assignment resolves calls by that name. Either spell "
        f"these calls 'ttl.<op>(...)', or write the operation as explicit "
        f"compute / datamovement kernels."
    )


def is_unified_body(func: Callable[..., Any]) -> bool:
    """True when ``func`` is a thread-unified operation (no hand-written kernels).

    A multi-kernel operation defines nested compute / datamovement kernels and is
    left on the legacy execution path. Anything whose source cannot be parsed is
    treated as multi-kernel (legacy), never split.
    """
    try:
        fn_def = _parse_operation_funcdef(func)
    except (OSError, TypeError, ValueError):
        return False
    symbols = _symbol_table(func)
    for node in ast.walk(fn_def):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(_is_kernel_decorator(d, symbols) for d in node.decorator_list):
                return False
    return True


def _factory_name(value: ast.expr) -> Optional[str]:
    """Return the factory name if ``value`` is a ``ttl.<factory>(...)`` or bare
    ``<factory>(...)`` call for a known DFB/pipe factory, else None.

    Requiring the ``ttl`` receiver matches how the splitter resolves calls, and is
    safe only because :func:`_reject_aliased_api` turns any other spelling away
    first; without that, an aliased factory would silently leave its DFB out of
    ``local_dfb_names`` and unanchor the whole body. Note ``ttl/atom.py``'s
    counterpart, ``_call_name``, ignores the receiver instead, so the two
    frontends fail differently on the same aliased body (#779).
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
    name: str, decorator_binding: str, body: List[ast.stmt]
) -> ast.FunctionDef:
    """Build ``@<decorator_binding>()\ndef <name>(): <body>``.

    ``decorator_binding`` names the decorator object injected into the generated
    function's namespace, so the result does not depend on the operation's module
    binding the API as ``ttl``.
    """
    decorator = ast.Call(
        func=ast.Name(id=decorator_binding, ctx=ast.Load()),
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
    symbols = _symbol_table(func)

    fn_def = _parse_operation_funcdef(func)
    fn_def.decorator_list = []  # drop @ttl.operation; do not re-decorate
    _reject_aliased_api(fn_def, symbols)

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
            "_ttl_compute", _COMPUTE_BINDING, _strip_setup(split.body_for("trisc"))
        ),
        _make_kernel_def(
            "_ttl_dm0", _DATAMOVEMENT_BINDING, _strip_setup(split.body_for("ncrisc"))
        ),
        _make_kernel_def(
            "_ttl_dm1", _DATAMOVEMENT_BINDING, _strip_setup(split.body_for("brisc"))
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
    compute_decorator, datamovement_decorator = _kernel_decorator_objects()
    exec_ns[_COMPUTE_BINDING] = compute_decorator
    exec_ns[_DATAMOVEMENT_BINDING] = datamovement_decorator
    exec(code, exec_ns)
    return exec_ns[fn_def.name]
