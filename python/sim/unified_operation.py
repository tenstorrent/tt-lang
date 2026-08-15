# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run thread-unified ``@ttl.operation`` bodies on the simulator.

A unified operation body performs data-movement and compute work directly in
the body (``dfb.reserve()``/``wait()``, ``ttl.copy(...).wait()``, block math),
leaving thread assignment to the compiler. The simulator, however, executes an
operation as three cooperating kernels (one compute + two data movement).

Rather than re-derive thread assignment, this module reuses the compiler
frontend's splitter, ``ttl._src.atom_split.split_function_body``, which returns a
body per logical kernel. The three the simulator can run -- ``KernelKind.COMPUTE``,
``KernelKind.DATA_MOVEMENT``, and the implicit pipe-source data-movement kernel --
map onto the simulator's compute / dm0 / dm1 kernels. The unified body is rewritten
into an equivalent
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
import contextlib
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

_OPERATION_DECORATOR = "operation"


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
            # own __module__ via sys.modules during class processing.  A failed
            # exec must take the registration back: the entry would otherwise
            # name a half-initialized module that the next lookup accepts.
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(spec.name, None)
                raise
            return module

    raise RuntimeError(
        f"could not locate {filename} to run a unified @ttl.operation body; "
        f"looked in: {', '.join(str(c) for c in candidates)}"
    )


@contextlib.contextmanager
def _frontend_ttl_package():
    """Make ``ttl.kernel`` importable for the duration of a frontend load.

    The splitter imports the compiler's logical-kernel selectors as
    ``ttl.kernel``, but the simulator has replaced ``sys.modules["ttl"]`` with a
    namespace object carrying no submodules. Installing a real package rooted at
    the frontend source directory lets that one import resolve against the
    compiler's own definitions, so both frontends compare identical selectors.

    ``ttl.kernel`` reads the logical kernel names from the TableGen-generated
    dialect bindings, which a simulator-only install does not build. They are
    supplied here from the same ``.td`` spelling so the names stay in agreement.
    """
    frontend_root = Path(__file__).resolve().parent.parent / "ttl"
    if not (frontend_root / "kernel.py").is_file():
        yield  # bundled wheel layout: the splitter ships without the package
        return

    class _LogicalKernelKind:
        Compute = "compute"
        DataMovement = "data_movement"

    package = types.ModuleType("ttl")
    package.__path__ = [str(frontend_root)]
    dialects = types.ModuleType("ttl.dialects")
    dialects.__path__ = [str(frontend_root / "dialects")]
    enum_gen = types.ModuleType("ttl.dialects._ttl_enum_gen")
    enum_gen.LogicalKernelKind = _LogicalKernelKind

    installed = {
        "ttl": package,
        "ttl.dialects": dialects,
        "ttl.dialects._ttl_enum_gen": enum_gen,
    }
    saved = {name: sys.modules.get(name) for name in installed}
    sys.modules.update(installed)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_atom_split() -> types.ModuleType:
    """The compiler frontend's thread-assignment splitter."""
    with _frontend_ttl_package():
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
    spelling. The shared spelling rule already ignores the receiver, so it covers
    ``@ttl.compute()``, ``@T.compute()`` and a bare ``@compute()`` on its own;
    what resolution adds is the decorator bound under another name
    (``from ttl import compute as build_math``), which no spelling recognizes and
    which is why this looks at the object at all.

    Resolution can also withhold recognition the spelling would have given: a
    body's own ``compute`` that is not this API's decorator resolves to something
    else and is not read as a kernel. So this is not the compiler's rule plus
    extra recognition -- the compiler applies only the spelling
    (``atom_rules.defines_kernels_by_spelling``) -- and the two can disagree about
    a body that reuses a kernel decorator's name for something else.

    Misclassifying a multi-kernel body as unified splits it and returns a wrong
    result with no error reported, so a decorator that resolves to nothing falls
    back to the spelling rule, which errs toward "multi-kernel".
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


def _is_operation_decorator(dec: ast.expr, symbols: Dict[str, Any]) -> bool:
    """True when ``dec`` is ``@ttl.operation``, however the API is bound."""
    from .operation import operation

    node = dec.func if isinstance(dec, ast.Call) else dec
    resolved = _resolve_expr(node, symbols)
    if resolved is _UNRESOLVED:
        return _rules().decorator_name(dec) == _OPERATION_DECORATOR
    if resolved is operation:
        return True
    return getattr(resolved, "__name__", None) == _OPERATION_DECORATOR


def _clear_decorators(fn_def: ast.FunctionDef, symbols: Dict[str, Any]) -> None:
    """Strip ``fn_def``'s decorators, refusing any that would be dropped unreported.

    The synthesized function carries none of them, for a different reason on each
    side of ``@ttl.operation``. Re-applying ``@ttl.operation`` would send the
    result back through this rewrite. A decorator written above it is applied by
    Python to whatever the operation decorator returns, at the original definition
    site, so leaving it here would apply it twice.

    A decorator written below it is the one that would otherwise be lost: nothing
    else applies it, and the body it was written to wrap does not survive as a
    function -- it becomes three kernels. Reproducing it would also mean running
    something the compiler does not, since the compiler compiles an operation body
    with its decorator lines removed (``pykernel._src.utils._cleanup_source_code``).
    So it is refused, and the message points at the placement that does work.

    Note the simulator's legacy path for hand-written kernels does run such a
    decorator, because it rebuilds from the wrapper's code object rather than from
    source. That difference is not resolved here.

    Raises:
        ValueError: If a decorator is listed below ``@ttl.operation``.
    """
    decorators = fn_def.decorator_list
    fn_def.decorator_list = []

    operation_positions = [
        index
        for index, dec in enumerate(decorators)
        if _is_operation_decorator(dec, symbols)
    ]
    if not operation_positions:
        return
    below = decorators[operation_positions[-1] + 1 :]
    if not below:
        return

    spellings = ", ".join(
        f"'@{ast.unparse(dec)}' on line {dec.lineno}" for dec in below
    )
    if len(below) == 1:
        placement = f"the decorator {spellings} sits"
        pronoun, wraps = "it", "the body it wraps"
    else:
        placement = f"the decorators {spellings} sit"
        pronoun, wraps = "them", "the body they wrap"

    raise ValueError(
        f"{placement} below '@ttl.operation', where neither the simulator nor the "
        f"compiler applies {pronoun}: {wraps} is rewritten into one compute and two "
        f"data movement kernels, and the compiler compiles that body with its "
        f"decorators removed. Move {pronoun} above '@ttl.operation' to wrap the "
        f"operation as a whole."
    )


def _receiver_root_name(func: ast.expr) -> Optional[str]:
    """The leading name of an attribute call's receiver (``T.math.exp`` -> ``T``)."""
    node: ast.expr = func
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _api_ops_by_thread(api: Any) -> tuple[Dict[int, str], Dict[int, str]]:
    """``id()`` -> op name for the API's calls, split into pinning and control.

    Read out of ``atom_split``'s registry rather than restated here, and keyed on
    object identity so any binding of the same op is recognized, however it is
    spelled: a renamed one (``cp = ttl.copy``) or one reached through a name
    bound to a namespace (``m = ttl.math`` and then ``m.exp(...)``), which is why
    the namespaces are walked too.

    Pinning ops are the ones the split depends on. Control ops are separated
    rather than dropped, because they are what makes a spelling harmless: the
    splitter replicates them onto every thread whatever their receiver, so
    reaching one under another name changes nothing.
    """
    atom_split = _load_atom_split()
    control_placement = atom_split._Placement.CONTROL
    pinning: Dict[int, str] = {}
    control: Dict[int, str] = {}
    for name, placement in atom_split._TTL_OPS.items():
        op = getattr(api, name, None)
        if op is None:
            continue
        (control if placement is control_placement else pinning)[id(op)] = name
    for namespace in atom_split._TTL_NAMESPACES:
        holder = getattr(api, namespace, None)
        if holder is None:
            continue
        for name in dir(holder):
            if name.startswith("_"):
                continue
            op = getattr(holder, name, None)
            # A namespace also re-exports the helpers and types its own module
            # imported (``ttl.math.Block``, ``ttl.math.get_context``), which are
            # not ops. Keep the functions the namespace's module defines.
            if not inspect.isfunction(op):
                continue
            if getattr(op, "__module__", "").rsplit(".", 1)[-1] != namespace:
                continue
            pinning[id(op)] = f"{namespace}.{name}"
    return pinning, control


def _aliased_api_calls(fn_def: ast.FunctionDef, symbols: Dict[str, Any]) -> List[str]:
    """Calls in ``fn_def`` that reach the API under a name the splitter ignores.

    A call is offending when the splitter would have to classify it and cannot:
    it is spelled with a receiver other than ``ttl`` (or with none), and it
    either resolves to a thread-pinning op or is an attribute of the API module
    itself -- the second case catching an attribute that resolves to nothing
    recognizable, which spelled ``ttl.<name>`` would have been the splitter's own
    "unknown op" error and aliased is left unanchored with no error reported --
    assigning the call to a thread the program did not ask for.

    A call that resolves to a control op is not offending, in either spelling.
    That is the same exemption the splitter grants (control ops are replicated,
    not anchored) and it covers construction, so an aliased
    ``T.make_dataflow_buffer_like(...)`` is accepted: it is recognized as
    construction by name, without its receiver (``atom_rules.call_name``).

    Returns each offending spelling with its line number, for the error message.
    """
    api = sys.modules.get("ttl")
    if api is None:
        return []
    pinning_ops, control_ops = _api_ops_by_thread(api)
    found: List[str] = []
    for node in ast.walk(fn_def):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, (ast.Attribute, ast.Name)):
            continue
        root = _receiver_root_name(func) if isinstance(func, ast.Attribute) else None
        if root == "ttl":
            continue
        resolved = _resolve_expr(func, symbols)
        if resolved is not _UNRESOLVED and id(resolved) in control_ops:
            continue
        if resolved is not _UNRESOLVED and id(resolved) in pinning_ops:
            op = pinning_ops[id(resolved)]
            found.append(f"'{ast.unparse(func)}(...)' (ttl.{op}) on line {node.lineno}")
        elif root is not None and symbols.get(root) is api:
            found.append(f"'{ast.unparse(func)}(...)' on line {node.lineno}")
    return found


def _reject_aliased_api(fn_def: ast.FunctionDef, symbols: Dict[str, Any]) -> None:
    """Reject a unified body that reaches the API under a name the splitter ignores.

    ``atom_split`` decides which thread a statement belongs to by matching the
    receiver name ``ttl`` (``atom_split._classify_ttl_call``), so a call spelled
    ``T.copy(...)`` anchors nothing. The body is then split on incomplete
    information: an unanchored statement is replicated onto all three threads, or
    the split fails claiming a block has no uses when its only use is the call
    that went unrecognized. Name the spelling to fix instead.

    Only the ops the splitter classifies are rejected. Construction and the other
    control ops are accepted under any name, because the splitter does not
    classify them: they are replicated onto every thread whatever their receiver,
    and construction is recognized by name without one
    (``atom_rules.call_name``), so an aliased ``T.make_dfb(...)`` is hoisted and
    shared like any other. Rejecting those would turn away bodies that split
    correctly.

    There is no counterpart in the compiler, which is exposed to the same gap and
    reaches the splitter with anchors missing. Teaching ``_classify_ttl_call`` the
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


def _reject_captured_dfb(func: Callable[..., Any]) -> None:
    """Reject a body that reads a dataflow buffer built outside the operation.

    The spec draws this line where the compiler does: a dataflow buffer "is
    constructed in the scope of an operation function" ("Dataflow buffer"),
    while a pipe net may be constructed "in an enclosing scope and captured by
    the operation function" ("Pipe net"). So a captured pipe net is supported
    and a captured buffer is not, and the compiler refuses one at decoration
    time (``atom.py``, "external DFB ... is not supported").

    Without this the simulator would take the body: a captured buffer is not
    among the names the body constructs, so ``blk = outer.reserve()`` anchors no
    thread, gets replicated onto all three kernels, and fails as a dataflow
    state error inside a kernel the user never wrote. Sharing a buffer by
    hoisting it into an enclosing scope is a natural thing to try -- the spec's
    pipe-net wording invites it -- so it is worth the same answer the compiler
    gives, at the same time.

    Captured means captured, so this asks what the body actually closes over
    (``inspect.getclosurevars``, as ``atom.py`` does) rather than intersecting the
    names it reads with the enclosing scope. A body that builds its own buffer
    binds that name locally, and a local is not a capture however many
    module-level objects share its spelling.

    Raises:
        ValueError: If the body reads a captured dataflow buffer.
    """
    from .dfb import DataflowBuffer

    try:
        scopes = inspect.getclosurevars(func)
    except TypeError:
        # Not something with a code object to read; nothing is captured.
        return
    captured = sorted(
        name
        for scope in (scopes.nonlocals, scopes.globals)
        for name, value in scope.items()
        if isinstance(value, DataflowBuffer)
    )
    if not captured:
        return
    names = ", ".join(repr(name) for name in captured)
    raise ValueError(
        f"the dataflow buffer(s) {names} are constructed outside the operation. "
        f"A dataflow buffer is constructed in the scope of the operation "
        f"function that uses it, because construction is what tells the three "
        f"kernels which buffer they share; one built elsewhere reaches them as "
        f"three separate uses. Construct it in the body "
        f"('name = ttl.make_dataflow_buffer_like(...)'), or pass the tensor it "
        f"is built from as an operation argument. (A pipe net may be captured; "
        f"a dataflow buffer may not.)"
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
    grow, and until then rejecting is better than mis-splitting into a program
    that runs and computes the wrong result.

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


def validate_operation_interface(func: Callable[..., Any]) -> None:
    """Check ``func`` against the operation interface the specification states.

    Applies to any operation, kernels written by hand or not, and is the shared
    rule (``atom_rules.validate_operation_interface``) with the compiler's own
    wording, so a signature one frontend refuses is not one the other quietly
    takes: a parameter with a default value or a ``*args`` / ``**kwargs``
    signature, and a body that returns.

    Raises:
        ValueError: With the message the compiler gives.
    """
    _rules().validate_operation_interface(func)


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

    def is_kernel(dec: ast.expr) -> bool:
        return _is_kernel_decorator(dec, symbols)

    return not rules.defines_kernels(fn_def, is_kernel)


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
    _clear_decorators(fn_def, symbols)
    _reject_aliased_api(fn_def, symbols)
    _reject_captured_dfb(func)
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

    kernel_kind = atom_split.KernelKind
    compute = kernel_kind.COMPUTE
    data_movement = kernel_kind.DATA_MOVEMENT
    pipe_source = atom_split._PIPE_SOURCE_KERNEL

    # The simulator runs the fixed three-kernel core, so a body that selects a
    # user-declared logical kernel has no thread to run it on.
    unsupported = [
        k for k in split.kernels if k not in (compute, data_movement, pipe_source)
    ]
    if unsupported:
        raise ValueError(
            "the simulator runs only the implicit compute and data movement "
            "kernels; it cannot run the user-declared logical kernels "
            f"{sorted(str(k) for k in unsupported)}"
        )

    kernels = [
        _make_kernel_def(
            "_ttl_compute", _COMPUTE_BINDING, _strip_setup(split.body_for(compute))
        ),
        _make_kernel_def(
            "_ttl_dm0",
            _DATAMOVEMENT_BINDING,
            _strip_setup(split.body_for(data_movement)),
        ),
        _make_kernel_def(
            "_ttl_dm1", _DATAMOVEMENT_BINDING, _strip_setup(split.body_for(pipe_source))
        ),
    ]

    fn_def.body = setup_stmts + kernels

    module = ast.Module(body=[fn_def], type_ignores=[])
    ast.fix_missing_locations(module)

    # Reached only for a body that parsed, so it has a file to be compiled under:
    # the original one, which is what makes the kernels' line numbers resolve.
    code = compile(module, inspect.getfile(func), "exec")
    # The synthesized function is compiled at module scope, with no enclosing
    # cells, so what the original captured has to be passed in by name.
    exec_ns: Dict[str, Any] = dict(namespace)
    exec_ns.update(rules.closure_values(func))
    compute_decorator, datamovement_decorator = _kernel_decorator_objects()
    exec_ns[_COMPUTE_BINDING] = compute_decorator
    exec_ns[_DATAMOVEMENT_BINDING] = datamovement_decorator
    exec(code, exec_ns)
    return exec_ns[fn_def.name]
