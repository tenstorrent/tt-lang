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

The syntactic rules this needs -- which statements construct the shared dataflow
buffers and pipe nets, whether a body writes its threads by hand, whether a
construction sits where it can be lifted out of the body -- come from
``ttl._src.atom_rules``, the module the compiler frontend applies them from as
well. They are shared rather than restated so the two frontends cannot answer the
same question differently, which would mean a program behaving differently in
simulation than compiled.

Both frontend modules are loaded from their source files rather than imported as
``ttl._src.*`` because the simulator shadows ``sys.modules["ttl"]`` with its own
namespace object, which has no importable submodules. Only what the simulator
decides with those answers lives here: kernel decorators are additionally
resolved to their runtime objects, aliased API access is refused, and diagnostics
are worded and located for the simulator's users.
"""

from __future__ import annotations

import ast
import copy
import functools
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# Names the synthesized kernel decorators are bound to in the generated
# function's namespace. Generated code must not depend on how the operation's
# module imported the TT-Lang API, so it references the decorator objects
# directly instead of spelling ``ttl.compute`` / ``ttl.datamovement``.
_COMPUTE_BINDING = "__ttl_sim_compute__"
_DATAMOVEMENT_BINDING = "__ttl_sim_datamovement__"

# Sentinel for "this expression does not resolve to a runtime object".
_UNRESOLVED: Any = object()


@functools.cache
def _load_frontend_module(filename: str) -> types.ModuleType:
    """Load a stdlib-only compiler frontend module from its source file (cached).

    Tries the bundled copy next to the simulator package first (installed
    ``tt-lang-sim`` wheel), then the frontend location in the source tree
    (``python/ttl/_src/``). Loading by path rather than importing
    ``ttl._src.<name>`` is necessary because the simulator shadows
    ``sys.modules["ttl"]`` with a namespace object that has no submodules.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / filename,  # bundled into the sim package (wheel)
        here.parent / "ttl" / "_src" / filename,  # source tree
    ]
    for path in candidates:
        if path.is_file():
            name = f"ttl_sim_{Path(filename).stem}"
            spec = importlib.util.spec_from_file_location(name, str(path))
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            # Register before exec so @dataclass in the module can resolve its
            # own __module__ via sys.modules during class processing.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise RuntimeError(
        f"could not locate {filename} to run a unified @ttl.operation body; "
        f"looked in: {', '.join(str(c) for c in candidates)}"
    )


def _load_atom_split() -> types.ModuleType:
    """The compiler frontend's thread-assignment splitter."""
    return _load_frontend_module("atom_split.py")


def _rules() -> types.ModuleType:
    """The syntactic rules both frontends apply to an operation body."""
    return _load_frontend_module("atom_rules.py")


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
    fn_def = _rules().parse_function_definition(func, rebase_lines=True)
    if fn_def is None:
        raise ValueError(f"could not parse @ttl.operation function {func.__name__!r}")
    return fn_def


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


def _is_kernel_decorator(dec: ast.expr, symbols: Dict[str, Any]) -> bool:
    """True when ``dec`` decorates a hand-written compute / datamovement kernel.

    Resolves the decorator to its runtime object rather than matching the source
    spelling, so ``@ttl.compute()``, ``@T.compute()`` after ``import ttl as T``,
    and ``@compute()`` after ``from ttl import compute`` are all recognized.
    Misclassifying a multi-kernel body as unified splits it and silently
    produces a wrong answer, so unresolvable decorators fall back to the shared
    spelling rule, which errs toward "multi-kernel" -- and which is all the
    compiler applies (``atom_rules.defines_kernels_by_spelling``), so resolving
    first only ever adds recognition.
    """
    rules = _rules()
    node = dec.func if isinstance(dec, ast.Call) else dec
    resolved = _resolve_expr(node, symbols)
    if resolved is _UNRESOLVED:
        return rules.is_kernel_decorator_spelling(node)
    if any(resolved is obj for obj in _kernel_decorator_objects()):
        return True
    # A decorator from a different build of the API (unshadowed ttl, reloaded
    # module) is not object-identical to ours but still names a kernel.
    return getattr(resolved, "__name__", None) in rules.KERNEL_DECORATORS


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
    receiver name ``ttl`` (``atom_split._classify_ttl_call``), so a call spelled
    ``T.copy(...)`` anchors nothing. The body is then split on incomplete
    information: an unanchored statement is replicated onto all three threads, or
    the split fails claiming a block has no uses when its only use is the call
    that went unrecognized. Name the spelling to fix instead.

    Not detected, because the receiver does not resolve to the API module: a name
    bound to one of its namespaces (``M = ttl.math`` and then ``M.exp(...)``).

    There is no counterpart in the compiler, which is exposed to the same gap and
    reaches the splitter with anchors missing. Construction is not what saves
    either of them: ``atom_rules.call_name`` ignores the receiver, so both
    frontends do recognize ``T.make_dfb(...)`` as construction; it is the ops the
    splitter classifies that go unrecognized. Teaching ``_classify_ttl_call`` the
    set of names that refer to the API would fix both and retire this guard --
    see #779.

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

    Which definitions are searched is the shared walk; only the test applied to
    each decorator is the simulator's own.
    """
    rules = _rules()
    try:
        fn_def = _parse_operation_funcdef(func)
    except (OSError, TypeError, ValueError):
        return False
    symbols = rules.function_scope(func)
    return not rules.defines_kernels(
        fn_def, lambda dec: _is_kernel_decorator(dec, symbols)
    )


def _reject_unsupported_setup(fn_def: ast.FunctionDef) -> None:
    """Reject DFB / pipe construction that cannot be hoisted out of the body.

    Only a top-level ``name = <factory>(...)`` assignment is hoisted, and hoisting
    is what gives all three kernels the same object: the reserve/wait handshake is
    keyed on identity, so a buffer each kernel builds for itself is a different
    buffer. Construction that is nested in control flow, in a callback, or in
    another scope -- or bound to something other than a single name -- is left in
    the body and duplicated into every kernel, which fails later as a dataflow
    state error against the wrong buffer, pointing nowhere near the cause.

    Hoisting also fixes when the construction runs: before the kernels, and
    outside them. A construction that reads a value the body computes for itself
    therefore cannot be hoisted either -- that value lives in the kernels the body
    becomes -- and is rejected as well, rather than left to fail as a ``NameError``
    from a statement the user never wrote in that position.

    Both conditions are decided by ``atom_rules``, the same code the compiler
    validates with (``atom_rules.validate_resource_declarations``), so the two
    frontends cannot disagree about which bodies are constructible. Only the
    wording differs: the compiler's message is pinned by its own test, while these
    quote the line to fix, which is what the simulator's users get shown.

    Neither rule comes from the spec, which places this construction alongside
    loops and index arithmetic as a compile-time construct "shared by the threads
    that need them" ("Thread assignment") and puts no condition on where it
    appears. Both frontends are narrower because they lift the construction
    textually and evaluate it ahead of the split, so it can only read names bound
    outside the body. Accepting the general form is a feature both would have to
    grow, and until then rejecting is better than mis-splitting silently.

    Raises:
        ValueError: If a construction is not a hoistable top-level assignment, or
            reads a value that only exists inside the body.
    """
    rules = _rules()

    dependency = rules.find_local_dependency(fn_def)
    if dependency is not None:
        read = ", ".join(repr(n) for n in dependency.names)
        raise ValueError(
            f"the construction of '{dependency.target}' on line "
            f"{dependency.statement.lineno} reads {read}, which the operation body "
            f"computes for itself. Construction is hoisted out of the body and "
            f"runs before the kernels, so it cannot see anything the body "
            f"computes; the value has to come from outside the operation."
        )

    unhoistable = rules.find_unhoistable_resource(fn_def)
    if unhoistable is not None:
        factory = unhoistable.factory
        raise ValueError(
            f"'{factory}(...)' on line {unhoistable.call.lineno} must be a simple "
            f"top-level assignment in the operation body "
            f"('name = {factory}(...)'), because dataflow buffers and pipes are "
            f"constructed once and shared by all three kernels. Construction "
            f"inside control flow, a callback, or a nested scope, or bound to more "
            f"than one name, cannot be shared: each kernel would build its own."
        )


def _is_setup_stmt(stmt: ast.stmt) -> bool:
    """True for a top-level ``name = <dfb/pipe factory>(...)`` assignment."""
    return _rules().setup_assign_target(stmt) is not None


def _local_dfb_names(fn_def: ast.FunctionDef) -> Set[str]:
    """Names bound to ``make_dataflow_buffer_like`` / ``make_dfb`` results."""
    rules = _rules()
    names: Set[str] = set()
    for stmt in fn_def.body:
        name = rules.setup_assign_target(stmt)
        if name is None or not isinstance(stmt, ast.Assign):
            continue
        if rules.call_name(stmt.value) in rules.DFB_FACTORY_NAMES:
            names.add(name)
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
        body=body,
        decorator_list=[decorator],
        returns=None,
        type_comment=None,
        type_params=[],
    )


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
    rules = _rules()
    atom_split = _load_atom_split()
    symbols = rules.function_scope(func)

    fn_def = _parse_operation_funcdef(func)
    fn_def.decorator_list = []  # drop @ttl.operation; do not re-decorate
    _reject_aliased_api(fn_def, symbols)
    _reject_unsupported_setup(fn_def)

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
    # The synthesized function is compiled at module scope, with no enclosing
    # cells, so what the original captured has to be passed in by name.
    exec_ns: Dict[str, Any] = dict(namespace)
    exec_ns.update(rules.closure_values(func))
    compute_decorator, datamovement_decorator = _kernel_decorator_objects()
    exec_ns[_COMPUTE_BINDING] = compute_decorator
    exec_ns[_DATAMOVEMENT_BINDING] = datamovement_decorator
    exec(code, exec_ns)
    return exec_ns[fn_def.name]
