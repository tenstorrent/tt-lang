# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Automatic copy-wait insertion for simulator kernel functions.

This module handles `ttl.copy()` calls that are missing the paired
`tx.wait()` call, mirroring the compiler's `ttl-insert-copy-wait` pass:

* **Bare copy calls** (`ttl.copy(src, dst)` with no assignment) — the
  `copy()` function itself calls `wait()` immediately before returning
  because the handle will be discarded and nothing else can wait on it.
* **Assigned copies with no wait** (`tx = ttl.copy(...)` with no
  `tx.wait()`) — an injection point is inserted on the very next
  statement so that `tx.wait()` fires before anything else runs.

Push/pop injection for `dfb.reserve()` / `dfb.wait()` blocks is handled
directly inside `DataflowBuffer` at runtime, not here:

* A second `dfb.reserve()` on the same buffer auto-pushes the previous block.
* A second `dfb.wait()` on the same buffer auto-pops the previous block.
* When a kernel function returns normally, any remaining pending blocks are
  auto-pushed/popped by the `_tagged` wrapper in `program.py`.

The analysis approach:

1. **AST analysis** (`collect_reachable_analyses`) — parse the source of
   each kernel function, recursively discover nested `def` helpers and
   module-scope callees referenced by simple name, and for each unwaited
   `tx = ttl.copy(...)` compute an *injection point*: the first line the
   runtime should execute after the copy call.

2. **Runtime interception** (`install_copy_wait_hooks`) — register
   `sys.monitoring` callbacks (Python 3.12+) that fire `tx.wait()` at
   the computed injection point for every discovered code object.  The
   original source is never modified; debuggers see unaltered line numbers.

Helper discovery
----------------
`collect_reachable_analyses` walks the AST of each kernel function and
recurses into:

* **Nested `def`s** — functions defined inline inside the kernel function
  body.  Their code objects are matched via `func.__code__.co_consts`.
* **Module-scope callees** — functions referenced by a bare name in the
  kernel body and resolved via `func.__globals__`.  Only plain Python
  functions (those with `__code__` and `__globals__`) are followed.

A shared visited set prevents duplicate analysis when the same helper is
called from multiple kernel functions.

Design constraints
------------------
* The original source must remain untouched (no AST rewriting, no exec of
  modified code) so that Python debuggers work on the original file.
* The analysis runs once per kernel function and the result is stored in
  `SimulatorContext.injection_points_cache` by the caller.
* `sys.monitoring` allows multiple independent tools (debugger, coverage,
  this module) to coexist without any chaining or mutual interference.

Unsupported patterns
--------------------
* **Callees referenced via attribute access** (`obj.method()`) or
  **through a container** are not followed; only simple name calls can be
  resolved statically via `__globals__`.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import types
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InjectionPoint:
    """Describes where to auto-insert a copy-wait (``tx.wait()``) call.

    ``trigger_lineno`` is the *absolute file line number* at which the
    ``sys.monitoring`` callback fires.  The callback fires *before* that line
    executes, so ``tx.wait()`` runs before the next statement after the copy.

    When ``trigger_on_return`` is ``True``, the callback fires on the
    function's ``return`` event instead (used when the copy is the last
    statement in the function).
    """

    var_name: str
    trigger_lineno: Optional[int]  # None when trigger_on_return is True
    trigger_on_return: bool = False


@dataclass
class KernelAnalysis:
    """Result of analysing one kernel function.

    ``injection_points`` covers copy-wait injection (Case B: assigned
    ``tx = ttl.copy(...)`` with no explicit ``tx.wait()``).

    ``bare_copy_linenos`` is the set of absolute file line numbers of bare
    ``ttl.copy(...)`` calls whose return value is not assigned to any
    variable (Case A).  These are forwarded to ``copy()`` via the simulator
    context so that ``copy()`` can call ``wait()`` immediately.

    ``violations`` is the set of unsupported patterns found during static
    analysis.  A non-empty set causes the simulator to print diagnostics and
    abort before running the kernel.
    """

    injection_points: tuple[InjectionPoint, ...]
    bare_copy_linenos: frozenset[int]
    violations: tuple["PatternViolation", ...] = ()


@dataclass
class PatternViolation:
    """One unsupported ``ttl.copy()`` pattern found during analysis.

    The simulator collects all violations across every kernel function before
    reporting them together, so the user sees every problem in a single run.
    """

    source_file: str
    lineno: int  # absolute file line number (1-based)
    col: int  # 1-based column number
    message: str
    func_name: str  # name of the kernel function containing the violation


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _is_ttl_copy_call(node: ast.expr) -> bool:
    """Return True if ``node`` is a ``ttl.copy(...)`` call expression."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ttl"
    )


def _find_copy_records(
    stmts: list[ast.stmt],
    file_start_line: int,
) -> tuple[list[tuple[str, int]], list[int]]:
    """Find ttl.copy() calls that need automatic wait() insertion.

    Scans ``stmts`` (flat list from ``_all_stmts_flat``) for two patterns:

    * **Case B** — ``tx = ttl.copy(...)`` with no subsequent ``tx.wait()``:
      returned as ``(var_name, abs_lineno)`` pairs in ``assigned_no_wait``.
    * **Case A** — bare ``ttl.copy(...)`` expression with no assignment:
      returned as absolute line numbers in ``bare_linenos``.

    Returns ``(assigned_no_wait, bare_linenos)``.
    """
    # Collect all assigned copy vars and their linenos.
    assigned: list[tuple[str, int]] = []  # (var_name, abs_lineno)
    bare_linenos: list[int] = []

    for stmt in stmts:
        abs_lineno = file_start_line + stmt.lineno - 1

        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and _is_ttl_copy_call(stmt.value):
                assigned.append((target.id, abs_lineno))

        elif isinstance(stmt, ast.Expr) and _is_ttl_copy_call(stmt.value):
            bare_linenos.append(abs_lineno)

    # Map each variable name to the absolute line numbers where .wait() is called.
    # A copy assignment is only disqualified if the matching .wait() appears
    # *after* it; a wait that precedes the assignment (e.g. at the top of a loop
    # body to release the previous iteration's copy) must not suppress injection.
    wait_abs_linenos: dict[str, list[int]] = {}
    for stmt in stmts:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "wait"
                and isinstance(node.value.func.value, ast.Name)
            ):
                name = node.value.func.value.id
                wait_abs_linenos.setdefault(name, []).append(
                    file_start_line + node.lineno - 1
                )

    return [
        (var, ln)
        for var, ln in assigned
        if not any(wl > ln for wl in wait_abs_linenos.get(var, []))
    ], bare_linenos


# ---------------------------------------------------------------------------
# Shared parsing helper
# ---------------------------------------------------------------------------


def _parse_func_def(
    func: types.FunctionType,
) -> tuple[ast.FunctionDef, int, str] | None:
    """Parse ``func``'s source and return ``(func_def, file_start_line, source_file)``.

    Returns ``None`` if the source is unavailable, unparseable, or contains
    no ``FunctionDef`` at the top level.
    """
    try:
        source_lines, file_start_line = inspect.getsourcelines(func)
        source_file = inspect.getfile(func)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent("".join(source_lines))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node, file_start_line, source_file
    return None


# ---------------------------------------------------------------------------
# Pattern validation
# ---------------------------------------------------------------------------


def _violations_for_func_def(
    func_def: ast.FunctionDef,
    file_start_line: int,
    source_file: str,
    func_name: str,
) -> list[PatternViolation]:
    """Return ``PatternViolation``s for any ``ttl.copy()`` calls in unsupported positions.

    Allowed positions:
    * Bare expression statement: ``ttl.copy(src, dst)``
    * Simple named assignment: ``tx = ttl.copy(src, dst)``
    * Immediate method-chain on the result: ``ttl.copy(src, dst).wait()``
    """
    allowed: set[int] = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if _is_ttl_copy_call(call):
                allowed.add(id(call))
            elif (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "wait"
                and isinstance(call.func.value, ast.Call)
                and _is_ttl_copy_call(call.func.value)
            ):
                allowed.add(id(call.func.value))
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and _is_ttl_copy_call(node.value)
        ):
            allowed.add(id(node.value))

    return [
        PatternViolation(
            source_file=source_file,
            lineno=file_start_line + node.lineno - 1,
            col=node.col_offset + 1,
            message=(
                "ttl.copy() is used in an unsupported pattern. "
                "Supported patterns: "
                "'ttl.copy(src, dst)' (bare call) or "
                "'tx = ttl.copy(src, dst)' (simple assignment)."
            ),
            func_name=func_name,
        )
        for node in ast.walk(func_def)
        if isinstance(node, ast.Call)
        and _is_ttl_copy_call(node)
        and id(node) not in allowed
    ]


def validate_kernel_function(func: types.FunctionType) -> list[PatternViolation]:
    """Check that all ``ttl.copy()`` calls use supported patterns.

    Returns a list of ``PatternViolation`` objects (one per unsupported call
    site).  An empty list means the function is valid, or the source is
    unavailable (built-in, dynamically generated, etc.).

    Supported patterns for ``ttl.copy()``:

    * ``ttl.copy(src, dst)``  (bare call, auto-waited)
    * ``tx = ttl.copy(src, dst)``  (simple assignment)
    * ``ttl.copy(src, dst).wait()``  (immediate method chain)
    """
    parsed = _parse_func_def(func)
    if parsed is None:
        return []
    func_def, file_start_line, source_file = parsed
    return _violations_for_func_def(
        func_def, file_start_line, source_file, func.__name__
    )


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _all_stmts_flat(tree: ast.FunctionDef) -> list[ast.stmt]:
    """Return every statement node in a function body in source order.

    Recurses into ``for``/``while``/``if``/``with``/``try`` bodies so that
    ``ttl.copy()`` calls inside loops and conditionals are all visible to the
    analysis.  Does not descend into nested ``def``/``class`` definitions.
    """
    result: list[ast.stmt] = []

    def _collect(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            result.append(stmt)
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
                _collect(stmt.body)
                _collect(stmt.orelse)
            elif isinstance(stmt, ast.With):
                _collect(stmt.body)
            elif isinstance(stmt, ast.Try):
                _collect(stmt.body)
                for handler in stmt.handlers:
                    _collect(handler.body)
                _collect(stmt.orelse)
                _collect(stmt.finalbody)
            # Do NOT recurse into nested def/class bodies.

    _collect(tree.body)
    # Sort by line number for deterministic ordering.
    result.sort(key=lambda s: s.lineno)
    return result


# ---------------------------------------------------------------------------
# Public API: analysis
# ---------------------------------------------------------------------------


def _analyze_func_def_node(
    func_def: ast.FunctionDef,
    file_start_line: int,
) -> KernelAnalysis:
    """Return injection points for a single ``FunctionDef`` AST node.

    ``file_start_line`` is the absolute line number of the first line of the
    *outermost parsed source*, matching the convention used by
    ``_find_copy_records``.  All AST line numbers are relative to that origin,
    so the same ``file_start_line`` applies to nested defs at any depth.
    Violations are not computed here; callers that need them should use
    ``_make_analysis_with_violations`` instead.
    """
    stmts = _all_stmts_flat(func_def)
    if not stmts:
        return KernelAnalysis(injection_points=(), bare_copy_linenos=frozenset())

    assigned_no_wait, bare_linenos = _find_copy_records(stmts, file_start_line)
    abs_linenos = [file_start_line + s.lineno - 1 for s in stmts]
    injection_points = tuple(
        InjectionPoint(
            var_name=var_name,
            trigger_lineno=(
                tl := next((ln for ln in abs_linenos if ln > copy_lineno), None)
            ),
            trigger_on_return=tl is None,
        )
        for var_name, copy_lineno in assigned_no_wait
    )
    return KernelAnalysis(
        injection_points=injection_points,
        bare_copy_linenos=frozenset(bare_linenos),
    )


def _collect_all_nested_codes(
    code: types.CodeType,
) -> dict[tuple[str, int], types.CodeType]:
    """Return all code objects nested inside ``code`` at any depth.

    Recursively walks ``co_consts`` so that defs nested inside other defs are
    included, not just immediately enclosed ones.

    Keys are ``(co_name, co_firstlineno)`` pairs where ``co_firstlineno`` is
    an absolute file line number, matching ``file_start_line + node.lineno - 1``
    for the corresponding ``ast.FunctionDef`` node.
    """
    result: dict[tuple[str, int], types.CodeType] = {}
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            result[(c.co_name, c.co_firstlineno)] = c
            result.update(_collect_all_nested_codes(c))
    return result


def _make_analysis_with_violations(
    func_def: ast.FunctionDef,
    file_start_line: int,
    source_file: str,
    func_name: str,
) -> KernelAnalysis:
    """Analyse ``func_def`` and attach pattern violations.

    Combines ``_analyze_func_def_node`` (injection points) with
    ``_violations_for_func_def`` (unsupported pattern checks) into a single
    ``KernelAnalysis``.
    """
    base = _analyze_func_def_node(func_def, file_start_line)
    return KernelAnalysis(
        injection_points=base.injection_points,
        bare_copy_linenos=base.bare_copy_linenos,
        violations=tuple(
            _violations_for_func_def(func_def, file_start_line, source_file, func_name)
        ),
    )


def _collect_reachable_from_parsed(
    func: types.FunctionType,
    func_def: ast.FunctionDef,
    file_start_line: int,
    source_file: str,
    _visited: set[int],
) -> dict[types.CodeType, KernelAnalysis]:
    """Collect analyses for ``func`` and all reachable callees.

    Called by ``collect_reachable_analyses`` with an already-parsed AST so
    that ``_parse_func_def`` is invoked exactly once per function.

    Discovers nested defs at any depth (via ``_collect_all_nested_codes``) and
    module-scope callees referenced by simple name (via ``func.__globals__``).
    """
    code = func.__code__
    result: dict[types.CodeType, KernelAnalysis] = {}

    result[code] = _make_analysis_with_violations(
        func_def, file_start_line, source_file, func.__name__
    )

    # Build a lookup of ALL descendant code objects (any nesting depth).
    all_nested = _collect_all_nested_codes(code)

    for node in ast.walk(func_def):
        if isinstance(node, ast.FunctionDef) and node is not func_def:
            # Nested def: locate its code object and analyse its body.
            abs_lineno = file_start_line + node.lineno - 1
            nested_code = all_nested.get((node.name, abs_lineno))
            if nested_code is not None and id(nested_code) not in _visited:
                _visited.add(id(nested_code))
                result[nested_code] = _analyze_func_def_node(node, file_start_line)

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # Name call: resolve the callee via __globals__ and recurse.
            callee = func.__globals__.get(node.func.id)
            if (
                callable(callee)
                and hasattr(callee, "__code__")
                and hasattr(callee, "__globals__")
            ):
                result.update(collect_reachable_analyses(callee, _visited))

    return result


def analyze_kernel_function(func: types.FunctionType) -> KernelAnalysis:
    """Analyse ``func`` and return injection points for missing ``tx.wait()`` calls.

    Returns an empty ``KernelAnalysis`` if:
    * The source is unavailable (built-in, dynamically generated, etc.).
    * All ``ttl.copy()`` calls already have explicit ``tx.wait()`` calls.
    * No ``ttl.copy()`` calls are found.
    """
    parsed = _parse_func_def(func)
    if parsed is None:
        return KernelAnalysis(injection_points=(), bare_copy_linenos=frozenset())
    func_def, file_start_line, source_file = parsed

    return _make_analysis_with_violations(
        func_def, file_start_line, source_file, func.__name__
    )


def collect_reachable_analyses(
    func: types.FunctionType,
    _visited: set[int] | None = None,
) -> dict[types.CodeType, KernelAnalysis]:
    """Return ``KernelAnalysis`` for ``func`` and all reachable callees.

    Discovers and analyses:

    * ``func`` itself (the top-level kernel function, including violations).
    * Nested ``def``s at any depth inside ``func``'s body — code objects are
      located via a full recursive walk of ``func.__code__.co_consts``.
    * Module-scope callees referenced by a simple name anywhere in the body —
      resolved via ``func.__globals__`` and recursed into.  Only plain Python
      functions (those with ``__code__`` and ``__globals__``) are followed;
      built-ins, classes, and other callables are skipped silently.

    Mutual recursion between helpers is safe: ``_visited`` tracks processed
    code objects by ``id`` and short-circuits on the second encounter.

    ``_visited`` is a set of ``id(code)`` values already processed, used to
    prevent infinite recursion through mutually-recursive helpers.  Omit it
    when calling for a single function.  Pass a shared set across multiple
    top-level calls (e.g. across the three kernel functions) so that a callee
    shared by more than one kernel is analysed only once.
    """
    if _visited is None:
        _visited = set()

    code = func.__code__
    if id(code) in _visited:
        return {}
    _visited.add(id(code))

    parsed = _parse_func_def(func)
    if parsed is None:
        return {}
    func_def, file_start_line, source_file = parsed

    return _collect_reachable_from_parsed(
        func, func_def, file_start_line, source_file, _visited
    )


# ---------------------------------------------------------------------------
# Public API: runtime interception
# ---------------------------------------------------------------------------


def _monitoring_api():
    """Return ``sys.monitoring`` or fail before copy-wait hooks are required."""
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        raise RuntimeError(
            "Automatic copy-wait injection requires Python 3.12 or newer. "
            "Add explicit tx.wait() calls or run the simulator with Python 3.12+."
        )
    return monitoring


def _fire_injection(frame: types.FrameType, ip: InjectionPoint) -> None:
    """Call ``tx.wait()`` on the CopyTransaction held in ``frame.f_locals[ip.var_name]``.

    Silently skips if the variable is absent or the transaction is already
    completed — this handles the case where the user called ``tx.wait()``
    explicitly before the injection point fired.
    """
    from .copy import CopyTransaction

    handle = frame.f_locals.get(ip.var_name)
    if not isinstance(handle, CopyTransaction) or handle.is_completed:
        return
    handle.wait()


def _report_injection_error(exc: Exception) -> None:
    """Format and print ``exc`` using the diagnostic machinery, then re-raise.

    ``find_user_code_location()`` walks the live call stack from here, skipping
    simulator frames, so it lands on the user kernel line where the injection
    fired — exactly the right location to point the error at.
    """
    from .diagnostics import find_user_code_location, print_diagnostic_error

    try:
        source_file, source_line = find_user_code_location()
    except Exception:
        raise exc from None

    print_diagnostic_error(
        name="copy-wait injection",
        message=str(exc),
        source_file=source_file,
        source_line=source_line,
    )
    # Raise a clean exception with no chained traceback so the user does not
    # see simulator-internal frames.
    raise type(exc)(str(exc)) from None


def _line_callback(code: types.CodeType, line_number: int) -> object:
    """sys.monitoring LINE callback — fires ``tx.wait()`` at trigger lines.

    Called by the interpreter before executing each instrumented line.
    ``sys._getframe(1)`` gives the frame of the monitored function, which
    still has all locals live at this point.
    """
    # Guard context lookup separately: if there is no active simulation context
    # (e.g. the callback fires during interpreter teardown) treat it as a no-op.
    try:
        from .context import get_context

        entry = get_context().active_hooks.get(id(code))
    except Exception:
        return None

    if entry is None:
        return None
    by_lineno, _ = entry
    ips = by_lineno.get(line_number)
    if ips:
        frame = sys._getframe(1)
        try:
            for ip in ips:
                _fire_injection(frame, ip)
        except Exception as exc:
            _report_injection_error(exc)
    return None


def _return_callback(
    code: types.CodeType, instruction_offset: int, retval: object
) -> object:
    """sys.monitoring PY_RETURN callback — fires ``tx.wait()`` at function exit.

    Called by the interpreter when a function returns normally.  The
    monitored frame is still on the stack at this point, so
    ``sys._getframe(1)`` gives access to its locals.
    """
    try:
        from .context import get_context

        entry = get_context().active_hooks.get(id(code))
    except Exception:
        return None

    if entry is None:
        return None
    _, return_ips = entry
    if return_ips:
        frame = sys._getframe(1)
        try:
            for ip in return_ips:
                _fire_injection(frame, ip)
        except Exception as exc:
            _report_injection_error(exc)
    return None


def install_copy_wait_hooks(
    injection_map: dict[types.CodeType, KernelAnalysis],
) -> None:
    """Register copy-wait injection hooks for the current simulation run.

    ``injection_map`` maps each kernel function's code object to its
    ``KernelAnalysis``.  Copy-wait injection points (Case B: assigned
    ``tx = ttl.copy(...)`` with no ``tx.wait()``) are stored in
    ``get_context().active_hooks``; bare-copy line numbers (Case A) are added
    to ``get_context().auto_wait_copy_lines``.

    On first call, claims a ``sys.monitoring`` tool ID and registers
    ``_line_callback`` / ``_return_callback``.  Subsequent calls only update
    the context's hooks and enable local events for new code objects.

    Because ``sys.monitoring`` supports independent tool slots, no chaining
    of existing tracers (e.g. pytest-cov, pdb) is needed.

    Clearing hooks between runs requires no monitoring reconfiguration —
    simply resetting the context (via ``reset_context()``) empties
    ``active_hooks`` so callbacks become no-ops for that code object.
    """
    # Build a map of code -> injection_points, skipping empty analyses.
    active_map = {
        code: analysis.injection_points
        for code, analysis in injection_map.items()
        if analysis.injection_points
    }
    monitoring = _monitoring_api() if active_map else None

    # Build lookup tables and store in the current context's active_hooks.
    from .context import get_context

    ctx = get_context()

    # Populate bare-copy line set (Case A) for all kernel functions.
    for code, analysis in injection_map.items():
        for lineno in analysis.bare_copy_linenos:
            ctx.auto_wait_copy_lines.add((code, lineno))

    if not active_map:
        return

    tool_id = monitoring.OPTIMIZER_ID

    # Claim the tool ID and register callbacks.  reset_context() frees the
    # slot between runs, so this always starts from a clean state.
    monitoring.use_tool_id(tool_id, "tt-lang-sim")
    monitoring.register_callback(tool_id, monitoring.events.LINE, _line_callback)
    monitoring.register_callback(tool_id, monitoring.events.PY_RETURN, _return_callback)

    # Build lookup tables, store them, and enable per-code-object events.
    # id(code) is used as the dict key so that lookup in the callbacks is
    # identity-based; two distinct code objects with identical bytecode cannot
    # collide the way they would with code-object equality as the key.
    for code, ips in active_map.items():
        by_lineno: dict[int, list[InjectionPoint]] = {}
        return_ips: list[InjectionPoint] = []
        for ip in ips:
            if ip.trigger_on_return:
                return_ips.append(ip)
            else:
                by_lineno.setdefault(ip.trigger_lineno, []).append(ip)  # type: ignore[arg-type]
        ctx.active_hooks[id(code)] = (by_lineno, return_ips)

        ev = monitoring.events.NO_EVENTS
        if by_lineno:
            ev |= monitoring.events.LINE
        if return_ips:
            ev |= monitoring.events.PY_RETURN
        monitoring.set_local_events(tool_id, code, ev)
