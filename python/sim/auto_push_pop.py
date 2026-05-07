# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Automatic push/pop and copy-wait insertion for simulator thread functions.

When a thread function acquires a block via ``dfb.reserve()`` or
``dfb.wait()`` without pairing it with an explicit ``blk.push()`` /
``blk.pop()`` call (and without using the ``with`` context manager), this
module inserts the missing release at the right point so the user does not
have to write it.

It also handles ``ttl.copy()`` calls that are missing the paired
``tx.wait()`` call, mirroring the compiler's ``ttl-insert-copy-wait`` pass:

* **Bare copy calls** (``ttl.copy(src, dst)`` with no assignment) — the
  ``copy()`` function itself calls ``wait()`` immediately before returning
  because the handle will be discarded and nothing else can wait on it.
* **Assigned copies with no wait** (``tx = ttl.copy(...)`` with no
  ``tx.wait()``) — an injection point is inserted on the very next
  statement so that ``tx.wait()`` fires before anything else runs.

Additionally, when a ``dfb.wait()`` or ``dfb.reserve()`` call is passed
*inline* as an argument to ``ttl.copy()`` without an intermediate variable
assignment (e.g. ``ttl.copy(out_dfb.wait(), out[s])``), the module
automatically inserts the corresponding ``dfb.pop_block()`` /
``dfb.push_block()`` call after the copy completes.

The approach mirrors the compiler's ``ttl-insert-cb-sync`` /
``ttl-insert-copy-wait`` passes:

1. **AST analysis** (``analyze_thread_function``) — parse the source of the
   thread function, walk it as an ordered statement list, and for each
   unmatched ``reserve``/``wait`` compute an *injection point*: the first
   line the runtime should execute after the block's last transitive use.
   The scope boundary for each acquire is the next ``reserve``/``wait`` on
   the same DFB variable in the same function body, exactly as the MLIR pass
   uses the next ``cb_reserve``/``cb_wait`` as a boundary.

2. **Runtime interception** (``install_auto_push_pop``) — register
   ``sys.monitoring`` callbacks (Python 3.12+) that fire ``push()`` /
   ``pop()`` or ``wait()`` on the live object at the computed injection point.
   The original source is never modified; debuggers see unaltered line
   numbers.

Design constraints
------------------
* The original source must remain untouched (no AST rewriting, no exec of
  modified code) so that Python debuggers work on the original file.
* The analysis runs once per thread function and the result is stored in
  ``SimulatorContext.injection_points_cache`` by the caller.
* ``sys.monitoring`` allows multiple independent tools (debugger, coverage,
  this module) to coexist without any chaining or mutual interference.

Limitations
-----------
* Only ``reserve``/``wait`` calls at the *outermost* scope of the function
  body are analysed.  Acquires inside ``with`` statements are already
  handled by the context-manager ``__exit__`` and are intentionally skipped.
* Control-flow analysis is conservative: last-use search walks the entire
  subtree of each statement, so a use inside an ``if``-branch counts even if
  that branch may not be taken.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import types
from dataclasses import dataclass
from typing import Literal, Optional, cast


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InjectionPoint:
    """Describes where and how to auto-insert a push, pop, or copy wait.

    ``trigger_lineno`` is the *absolute file line number* at which the
    ``sys.settrace`` hook should fire.  The hook fires *before* that line
    executes (i.e. at the start of the line event), so choosing the
    scope-boundary line as the trigger means push/pop runs before the next
    acquire on the same DFB.

    When ``trigger_on_return`` is ``True``, the hook fires on the function's
    ``return`` event instead (used when the last use is the last statement in
    the function).
    """

    var_name: str
    action: Literal["push", "pop", "wait", "push_dfb", "pop_dfb"]
    trigger_lineno: Optional[int]  # None when trigger_on_return is True
    trigger_on_return: bool = False


@dataclass
class ThreadAnalysis:
    """Result of analysing one thread function.

    ``injection_points`` covers DFB push/pop and copy-wait (Case B: assigned
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
    """One unsupported DFB-acquire or ttl.copy() pattern found during analysis.

    The simulator collects all violations across every thread function before
    reporting them together, so the user sees every problem in a single run.
    """

    source_file: str
    lineno: int  # absolute file line number (1-based)
    col: int  # 1-based column number
    message: str
    func_name: str  # name of the thread function containing the violation


@dataclass
class _AcquireRecord:
    """Internal: one ``reserve``/``wait`` call found during AST analysis."""

    var_name: str
    dfb_name: str
    action: Literal["reserve", "wait"]
    lineno: int  # absolute file line number
    inside_loop: bool = False  # True when inside a for/while body
    is_inline: bool = False  # True when inside a ttl.copy() arg (no assignment)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _is_method_call_on_any(
    stmt: ast.stmt,
    names: str | set[str],
    methods: str | set[str],
) -> bool:
    """Return True if ``stmt`` is ``obj.method()`` with obj in ``names`` and method in ``methods``."""
    if not isinstance(stmt, ast.Expr):
        return False
    call = stmt.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    name_set = {names} if isinstance(names, str) else names
    method_set = {methods} if isinstance(methods, str) else methods
    return (
        func.attr in method_set
        and isinstance(func.value, ast.Name)
        and func.value.id in name_set
    )


def _is_explicit_release(stmt: ast.stmt, var_name: str) -> bool:
    """Return True if ``stmt`` is ``<var_name>.push()`` or ``<var_name>.pop()``."""
    return _is_method_call_on_any(stmt, var_name, {"push", "pop"})


def _name_loaded_in(node: ast.AST, name: str) -> bool:
    """Return True if ``name`` appears with Load context anywhere under ``node``."""
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Name)
            and child.id == name
            and isinstance(child.ctx, ast.Load)
        ):
            return True
    return False


def _extract_copy_handle(stmt: ast.stmt, var_name: str) -> Optional[str]:
    """If ``stmt`` is ``handle = ttl.copy(..., var_name, ...)``, return the handle name.

    Matches ``handle = ttl.copy(...)`` where ``var_name`` appears anywhere in
    the argument list with Load context.  The ``ttl.copy`` call is identified
    by the ``.copy`` attribute name on any object (not just ``ttl``).
    """
    if not isinstance(stmt, ast.Assign):
        return None
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        return None
    call = stmt.value
    if not isinstance(call, ast.Call):
        return None
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "copy"):
        return None
    if not _name_loaded_in(call, var_name):
        return None
    return stmt.targets[0].id


def _is_handle_wait(stmt: ast.stmt, handle_names: set[str]) -> bool:
    """Return True if ``stmt`` is ``handle.wait()`` for any name in ``handle_names``."""
    return _is_method_call_on_any(stmt, handle_names, "wait")


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

    # Filter assigned copies: keep only those with no explicit wait() anywhere
    # in the function (conservative: any tx.wait() in the function disqualifies).
    assigned_no_wait: list[tuple[str, int]] = []
    waited_vars: set[str] = set()
    for stmt in stmts:
        if _is_method_call_on_any(stmt, {v for v, _ in assigned}, "wait"):
            assert isinstance(stmt, ast.Expr)
            assert isinstance(stmt.value, ast.Call)
            func = stmt.value.func
            assert isinstance(func, ast.Attribute)
            assert isinstance(func.value, ast.Name)
            waited_vars.add(func.value.id)

    for var_name, abs_lineno in assigned:
        if var_name not in waited_vars:
            assigned_no_wait.append((var_name, abs_lineno))

    return assigned_no_wait, bare_linenos


def _find_last_use_lineno(
    var_name: str,
    stmts: list[ast.stmt],
    file_start_line: int,
    search_start_lineno: int,
    upper_bound_lineno: Optional[int],
) -> Optional[int]:
    """Return the absolute line of the last use of ``var_name``.

    A "use" is any statement where ``var_name`` appears with Load context, or
    where a copy-handle derived from a ``ttl.copy(..., var_name, ...)`` call
    has ``.wait()`` called on it (since that is the true completion point of a
    copy, not the ``ttl.copy()`` call itself).

    Only statements strictly after ``search_start_lineno`` and strictly before
    ``upper_bound_lineno`` (pass ``None`` to scan to the end) are considered.
    """
    last_use: Optional[int] = None
    # Map from handle var name -> True, for handles derived from var_name.
    copy_handles: set[str] = set()

    for stmt in stmts:
        abs_lineno = file_start_line + stmt.lineno - 1
        if abs_lineno <= search_start_lineno:
            continue
        if upper_bound_lineno is not None and abs_lineno >= upper_bound_lineno:
            break

        # Track copy handles: tx = ttl.copy(..., var_name, ...)
        handle = _extract_copy_handle(stmt, var_name)
        if handle is not None:
            copy_handles.add(handle)

        # Direct use: var_name appears with Load context in this statement.
        if _name_loaded_in(stmt, var_name):
            last_use = abs_lineno

        # Copy-handle wait: tx.wait() where tx is a handle for var_name.
        if copy_handles and _is_handle_wait(stmt, copy_handles):
            last_use = abs_lineno

    return last_use


def _find_next_stmt_lineno(
    after_lineno: int,
    stmts: list[ast.stmt],
    file_start_line: int,
) -> Optional[int]:
    """Return the absolute line of the first statement strictly after ``after_lineno``.

    Returns ``None`` if ``after_lineno`` is at or past the last statement
    (caller should use ``trigger_on_return=True`` in that case).
    """
    for stmt in stmts:
        abs_lineno = file_start_line + stmt.lineno - 1
        if abs_lineno > after_lineno:
            return abs_lineno
    return None


# ---------------------------------------------------------------------------
# Pattern validation
# ---------------------------------------------------------------------------


def _collect_copy_handle_names(func_def: ast.FunctionDef) -> set[str]:
    """Return variable names assigned from any copy-like function call.

    Recognises both the high-level ``ttl.copy()`` form and bare ``copy()``
    calls (from ``from python.sim.copy import copy``), as well as any other
    call whose function name contains ``"copy"``.  These variables hold
    ``CopyTransaction`` handles whose ``.wait()`` calls should NOT be treated
    as DFB acquire calls during validation.
    """
    names: set[str] = set()
    for node in ast.walk(func_def):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            continue
        call = node.value
        func = call.func
        is_copy = (
            _is_ttl_copy_call(call)
            or (isinstance(func, ast.Name) and "copy" in func.id.lower())
            or (isinstance(func, ast.Attribute) and "copy" in func.attr.lower())
        )
        if is_copy:
            names.add(node.targets[0].id)
    return names


def _find_allowed_dfb_acquire_ids(func_def: ast.FunctionDef) -> set[int]:
    """Return the ``id()``s of all DFB acquire call nodes in allowed positions.

    Allowed positions for ``dfb.reserve()`` / ``dfb.wait()``:

    * Named assignment: ``blk = dfb.reserve()``
    * ``with`` context manager: ``with dfb.reserve() as blk:``
    * Direct positional or keyword argument of ``ttl.copy()``
    """
    allowed: set[int] = set()
    for node in ast.walk(func_def):
        # Named assignment: blk = dfb.reserve() / blk = dfb.wait()
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and _is_inline_dfb_acquire(node.value) is not None
        ):
            allowed.add(id(node.value))

        # with dfb.reserve() as blk:
        elif isinstance(node, ast.With):
            for item in node.items:
                if (
                    isinstance(item.context_expr, ast.Call)
                    and _is_inline_dfb_acquire(item.context_expr) is not None
                ):
                    allowed.add(id(item.context_expr))

        # Direct argument of ttl.copy(): ttl.copy(dfb.wait(), ...) etc.
        elif isinstance(node, ast.Call) and _is_ttl_copy_call(node):
            all_args = list(node.args) + [kw.value for kw in node.keywords]
            for arg in all_args:
                if (
                    isinstance(arg, ast.Call)
                    and _is_inline_dfb_acquire(arg) is not None
                ):
                    allowed.add(id(arg))

    return allowed


def _find_allowed_copy_ids(func_def: ast.FunctionDef) -> set[int]:
    """Return the ``id()``s of all ``ttl.copy()`` call nodes in allowed positions.

    Allowed positions for ``ttl.copy()``:

    * Bare expression statement: ``ttl.copy(src, dst)``
    * Simple named assignment: ``tx = ttl.copy(src, dst)``
    * Immediate method-chain on the result: ``ttl.copy(src, dst).wait()``
    """
    allowed: set[int] = set()
    for node in ast.walk(func_def):
        # Bare call: ttl.copy(src, dst)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if _is_ttl_copy_call(call):
                allowed.add(id(call))
            # Immediate method chain: ttl.copy(src, dst).method()
            elif (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Call)
                and _is_ttl_copy_call(call.func.value)
            ):
                allowed.add(id(call.func.value))

        # Simple assignment: tx = ttl.copy(src, dst)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and _is_ttl_copy_call(node.value)
        ):
            allowed.add(id(node.value))

    return allowed


def validate_thread_function(func: types.FunctionType) -> list[PatternViolation]:
    """Check that all DFB acquire and ``ttl.copy()`` calls use supported patterns.

    Walks the full AST of ``func`` and compares every call site against the
    set of positions the auto-injection analysis understands.  Returns a list
    of ``PatternViolation`` objects (one per unsupported call site).  An empty
    list means the function is valid.

    Returns an empty list when the function source is unavailable (built-in,
    dynamically generated, etc.).

    Supported patterns for ``dfb.reserve()`` / ``dfb.wait()``:

    * ``blk = dfb.reserve()`` / ``blk = dfb.wait()``
    * ``with dfb.reserve() as blk:`` / ``with dfb.wait() as blk:``
    * ``ttl.copy(dfb.wait(), ...)`` / ``ttl.copy(..., dfb.reserve())``

    Supported patterns for ``ttl.copy()``:

    * ``ttl.copy(src, dst)``  (bare call, auto-waited)
    * ``tx = ttl.copy(src, dst)``  (simple assignment)
    """
    try:
        source_lines, file_start_line = inspect.getsourcelines(func)
        source_file = inspect.getfile(func)
    except (OSError, TypeError):
        return []

    source = textwrap.dedent("".join(source_lines))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    func_def: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if func_def is None:
        return []

    func_name = func.__name__
    violations: list[PatternViolation] = []

    copy_handle_names = _collect_copy_handle_names(func_def)
    allowed_acquire_ids = _find_allowed_dfb_acquire_ids(func_def)
    allowed_copy_ids = _find_allowed_copy_ids(func_def)

    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call):
            continue

        # Check DFB acquire calls (reserve/wait with no args on a Name).
        result = _is_inline_dfb_acquire(node)
        if result is not None:
            dfb_name, method = result
            # Skip CopyTransaction.wait() — tx.wait() has the same AST shape
            # but the receiver is a known copy handle, not a DFB.
            if dfb_name in copy_handle_names:
                continue
            if id(node) not in allowed_acquire_ids:
                abs_lineno = file_start_line + node.lineno - 1
                col = node.col_offset + 1
                violations.append(
                    PatternViolation(
                        source_file=source_file,
                        lineno=abs_lineno,
                        col=col,
                        message=(
                            f"{dfb_name}.{method}() is used in an unsupported pattern. "
                            f"Supported patterns: "
                            f"'blk = {dfb_name}.{method}()', "
                            f"'with {dfb_name}.{method}() as blk:', or "
                            f"'ttl.copy({dfb_name}.{method}(), ...)'."
                        ),
                        func_name=func_name,
                    )
                )
            continue

        # Check ttl.copy() calls.
        if _is_ttl_copy_call(node) and id(node) not in allowed_copy_ids:
            abs_lineno = file_start_line + node.lineno - 1
            col = node.col_offset + 1
            violations.append(
                PatternViolation(
                    source_file=source_file,
                    lineno=abs_lineno,
                    col=col,
                    message=(
                        "ttl.copy() is used in an unsupported pattern. "
                        "Supported patterns: "
                        "'ttl.copy(src, dst)' (bare call) or "
                        "'tx = ttl.copy(src, dst)' (simple assignment)."
                    ),
                    func_name=func_name,
                )
            )

    return violations


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _compute_loop_body_lines(stmts: list[ast.stmt]) -> set[int]:
    """Return the set of relative line numbers that are inside a for/while body."""
    loop_body_lines: set[int] = set()

    def _mark(stmts_inner: list[ast.stmt], in_loop: bool) -> None:
        for stmt in stmts_inner:
            if in_loop:
                loop_body_lines.add(stmt.lineno)
            if isinstance(stmt, (ast.For, ast.While)):
                _mark(stmt.body, in_loop=True)
                _mark(stmt.orelse, in_loop=in_loop)
            elif isinstance(stmt, ast.If):
                _mark(stmt.body, in_loop=in_loop)
                _mark(stmt.orelse, in_loop=in_loop)
            elif isinstance(stmt, ast.With):
                _mark(stmt.body, in_loop=in_loop)
            elif isinstance(stmt, ast.Try):
                _mark(stmt.body, in_loop=in_loop)
                for h in stmt.handlers:
                    _mark(h.body, in_loop=in_loop)

    _mark(stmts, in_loop=False)
    return loop_body_lines


def _is_inline_dfb_acquire(
    node: ast.expr,
) -> Optional[tuple[str, Literal["reserve", "wait"]]]:
    """Return ``(dfb_name, method)`` if ``node`` is a bare ``dfb.reserve()`` or
    ``dfb.wait()`` call expression with no arguments, otherwise ``None``."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr in ("reserve", "wait")
        and isinstance(func.value, ast.Name)
        and not node.args
        and not node.keywords
    ):
        return None
    return func.value.id, func.attr  # type: ignore[return-value]


def _find_acquire_records(
    stmts: list[ast.stmt],
    file_start_line: int,
    loop_body_lines: set[int],
) -> list[_AcquireRecord]:
    """Find all ``blk = dfb.reserve()`` / ``blk = dfb.wait()`` calls.

    Skips assignments inside ``with`` statements (already handled by the
    context manager) and skips bare ``dfb.reserve()`` calls whose result is
    not captured (nothing to push/pop).

    Sets ``inside_loop=True`` for acquires that are nested inside a
    ``for``/``while`` body — these use the acquire's own line as the scope
    boundary so that push/pop fires at the start of every iteration.
    """
    records: list[_AcquireRecord] = []
    for stmt in stmts:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        call = stmt.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in ("reserve", "wait")
            and isinstance(func.value, ast.Name)
        ):
            continue
        var_name = stmt.targets[0].id
        dfb_name = func.value.id
        action: Literal["reserve", "wait"] = func.attr  # type: ignore[assignment]
        abs_lineno = file_start_line + stmt.lineno - 1
        records.append(
            _AcquireRecord(
                var_name=var_name,
                dfb_name=dfb_name,
                action=action,
                lineno=abs_lineno,
                inside_loop=stmt.lineno in loop_body_lines,
            )
        )
    return records


def _find_inline_acquire_records(
    stmts: list[ast.stmt],
    file_start_line: int,
    loop_body_lines: set[int],
) -> list[_AcquireRecord]:
    """Find ``dfb.wait()`` / ``dfb.reserve()`` calls passed inline as arguments
    to ``ttl.copy()``.

    Detects both positional-argument patterns:

    * ``ttl.copy(dfb.wait(), tensor_expr)``   — dfb will need ``pop_block()``
    * ``ttl.copy(tensor_expr, dfb.reserve())`` — dfb will need ``push_block()``

    The returned records have ``is_inline=True`` and use ``dfb_name`` as
    ``var_name`` so that ``_fire_injection`` can retrieve the DFB from
    ``frame.f_locals``.
    """
    records: list[_AcquireRecord] = []
    for stmt in stmts:
        # Unwrap the ttl.copy() call node from bare Expr or Assign.
        if isinstance(stmt, ast.Expr) and _is_ttl_copy_call(stmt.value):
            call_node = cast(ast.Call, stmt.value)
        elif isinstance(stmt, ast.Assign) and _is_ttl_copy_call(stmt.value):
            call_node = cast(ast.Call, stmt.value)
        else:
            continue

        abs_lineno = file_start_line + stmt.lineno - 1
        all_args = list(call_node.args) + [kw.value for kw in call_node.keywords]
        for arg in all_args:
            result = _is_inline_dfb_acquire(arg)
            if result is None:
                continue
            dfb_name, method = result
            records.append(
                _AcquireRecord(
                    var_name=dfb_name,
                    dfb_name=dfb_name,
                    action=method,
                    lineno=abs_lineno,
                    inside_loop=stmt.lineno in loop_body_lines,
                    is_inline=True,
                )
            )
    return records


def _find_scope_boundary_lineno(
    acquire: _AcquireRecord,
    all_records: list[_AcquireRecord],
    file_start_line: int,
    last_stmt_lineno: int,
) -> Optional[int]:
    """Return the absolute line of the scope boundary for this acquire.

    For a loop-body acquire (``inside_loop=True``) the boundary is the
    acquire's own line — this causes the injection to fire at the start of
    every subsequent iteration *before* the reserve/wait runs again.

    For a non-loop acquire the boundary is the line of the next
    ``reserve``/``wait`` on the same DFB variable, or ``None`` if there is
    none (inject on return).
    """
    if acquire.inside_loop:
        return acquire.lineno

    next_lineno: Optional[int] = None
    for rec in all_records:
        if rec is acquire:
            continue
        if rec.dfb_name != acquire.dfb_name:
            continue
        if rec.lineno <= acquire.lineno:
            continue
        if next_lineno is None or rec.lineno < next_lineno:
            next_lineno = rec.lineno
    return next_lineno


def _has_explicit_release(
    acquire: _AcquireRecord,
    stmts: list[ast.stmt],
    file_start_line: int,
    upper_bound_lineno: Optional[int],
) -> bool:
    """Return True if an explicit push/pop for this acquire already exists.

    Only statements between the acquire line (exclusive) and
    ``upper_bound_lineno`` (exclusive) are considered.  Pass ``None`` to
    scan to the end of the function.
    """
    abs_acquire = acquire.lineno
    for stmt in stmts:
        abs_lineno = file_start_line + stmt.lineno - 1
        if abs_lineno <= abs_acquire:
            continue
        if upper_bound_lineno is not None and abs_lineno >= upper_bound_lineno:
            break
        if _is_explicit_release(stmt, acquire.var_name):
            return True
    return False


def _all_stmts_flat(tree: ast.FunctionDef) -> list[ast.stmt]:
    """Return every statement node in a function body in source order.

    Recurses into ``for``/``while``/``if``/``with``/``try`` bodies so that
    acquires and uses inside loops and conditionals are all visible to the
    analysis.  Does not descend into nested ``def``/``class`` definitions.
    """
    result: list[ast.stmt] = []

    def _collect(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            result.append(stmt)
            if isinstance(stmt, (ast.For, ast.While)):
                _collect(stmt.body)
                _collect(stmt.orelse)
            elif isinstance(stmt, ast.If):
                _collect(stmt.body)
                _collect(stmt.orelse)
            elif isinstance(stmt, ast.With):
                _collect(stmt.body)
            elif isinstance(stmt, ast.Try):
                _collect(stmt.body)
                for handler in stmt.handlers:
                    _collect(handler.body)
                _collect(stmt.orelse)
                _collect(stmt.finalbody if hasattr(stmt, "finalbody") else [])
            # Do NOT recurse into nested def/class bodies.

    _collect(tree.body)
    # Sort by line number for deterministic ordering.
    result.sort(key=lambda s: s.lineno)
    return result


# ---------------------------------------------------------------------------
# Public API: analysis
# ---------------------------------------------------------------------------


def analyze_thread_function(func: types.FunctionType) -> ThreadAnalysis:
    """Analyse ``func`` and return injection points for missing push/pop/wait calls.

    The result is cached per function object so repeated calls (e.g. in a
    loop over cores) pay only the first analysis cost.

    Returns an empty ``ThreadAnalysis`` if:
    * The source is unavailable (built-in, dynamically generated, etc.).
    * All acquire calls already have explicit releases.
    * No acquire calls or bare/unwaited copies are found.
    """
    _empty = ThreadAnalysis(injection_points=(), bare_copy_linenos=frozenset())

    try:
        source_lines, file_start_line = inspect.getsourcelines(func)
    except (OSError, TypeError):
        return _empty

    source = textwrap.dedent("".join(source_lines))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _empty

    # The parsed source contains exactly one FunctionDef: the thread function
    # itself.  inspect.getsourcelines returns only that function's source.
    func_def: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if func_def is None:
        return _empty

    stmts = _all_stmts_flat(func_def)
    if not stmts:
        return _empty

    last_stmt_lineno = file_start_line + stmts[-1].lineno - 1

    injection_points: list[InjectionPoint] = []

    # ------------------------------------------------------------------
    # DFB push/pop analysis
    # ------------------------------------------------------------------
    loop_body_lines = _compute_loop_body_lines(stmts)
    acquire_records = _find_acquire_records(stmts, file_start_line, loop_body_lines)
    inline_records = _find_inline_acquire_records(
        stmts, file_start_line, loop_body_lines
    )
    # Combined list used for scope-boundary lookups so that named and inline
    # acquires on the same DFB correctly constrain each other.
    all_records = acquire_records + inline_records

    # Group loop acquires by DFB name so that multiple reserves/waits on the
    # same DFB within one loop body get the correct intra-iteration triggers.
    loop_groups: dict[str, list[_AcquireRecord]] = {}
    for rec in acquire_records:
        if rec.inside_loop:
            loop_groups.setdefault(rec.dfb_name, []).append(rec)
    for lst in loop_groups.values():
        lst.sort(key=lambda r: r.lineno)

    for acquire in acquire_records:
        action: Literal["push", "pop", "wait"] = (
            "push" if acquire.action == "reserve" else "pop"
        )

        if acquire.inside_loop:
            group = loop_groups[acquire.dfb_name]
            idx = group.index(acquire)
            is_last = idx == len(group) - 1

            # upper_bound for explicit-release check and last-use search:
            # non-last acquires are bounded by the next acquire's line;
            # last/single acquires have no bound (scan to end of function).
            intra_boundary: Optional[int] = (
                group[idx + 1].lineno if not is_last else None
            )

            if _has_explicit_release(acquire, stmts, file_start_line, intra_boundary):
                continue

            # Determine trigger from last-use analysis.
            # Fallback: non-last -> intra_boundary; last/single -> group[0].lineno.
            fallback: Optional[int] = intra_boundary if not is_last else group[0].lineno
            trigger_lineno: Optional[int] = _trigger_from_last_use(
                acquire.var_name,
                acquire.lineno,
                stmts,
                file_start_line,
                intra_boundary,
                fallback,
            )

            injection_points.append(
                InjectionPoint(
                    var_name=acquire.var_name,
                    action=action,
                    trigger_lineno=trigger_lineno,
                    trigger_on_return=False,
                )
            )
            # Always add a return trigger to clean up the final iteration.
            injection_points.append(
                InjectionPoint(
                    var_name=acquire.var_name,
                    action=action,
                    trigger_lineno=None,
                    trigger_on_return=True,
                )
            )

        else:
            scope_boundary = _find_scope_boundary_lineno(
                acquire, all_records, file_start_line, last_stmt_lineno
            )
            if _has_explicit_release(acquire, stmts, file_start_line, scope_boundary):
                continue

            trigger_lineno = _trigger_from_last_use(
                acquire.var_name,
                acquire.lineno,
                stmts,
                file_start_line,
                scope_boundary,
                scope_boundary,  # fallback: conservative scope boundary
            )
            on_return = trigger_lineno is None
            injection_points.append(
                InjectionPoint(
                    var_name=acquire.var_name,
                    action=action,
                    trigger_lineno=trigger_lineno,
                    trigger_on_return=on_return,
                )
            )

    # ------------------------------------------------------------------
    # Inline DFB acquire analysis
    # (dfb.wait()/dfb.reserve() passed directly as ttl.copy() arguments)
    # ------------------------------------------------------------------
    for acquire in inline_records:
        dfb_action: Literal["push_dfb", "pop_dfb"] = (
            "push_dfb" if acquire.action == "reserve" else "pop_dfb"
        )
        if acquire.inside_loop:
            # Fire just before the acquire re-runs on every subsequent
            # iteration, and also on function return for the last iteration.
            injection_points.append(
                InjectionPoint(
                    var_name=acquire.var_name,
                    action=dfb_action,
                    trigger_lineno=acquire.lineno,
                    trigger_on_return=False,
                )
            )
            injection_points.append(
                InjectionPoint(
                    var_name=acquire.var_name,
                    action=dfb_action,
                    trigger_lineno=None,
                    trigger_on_return=True,
                )
            )
        else:
            # Use next acquire on the same DFB as the scope boundary, or
            # trigger on return if this is the last acquire.
            scope_boundary = _find_scope_boundary_lineno(
                acquire, all_records, file_start_line, last_stmt_lineno
            )
            injection_points.append(
                InjectionPoint(
                    var_name=acquire.var_name,
                    action=dfb_action,
                    trigger_lineno=scope_boundary,
                    trigger_on_return=scope_boundary is None,
                )
            )

    # ------------------------------------------------------------------
    # Copy-wait analysis
    # ------------------------------------------------------------------
    assigned_no_wait, bare_linenos = _find_copy_records(stmts, file_start_line)

    # Case B: tx = ttl.copy(...) with no tx.wait() — insert wait on next line.
    for var_name, copy_lineno in assigned_no_wait:
        next_lineno = _find_next_stmt_lineno(copy_lineno, stmts, file_start_line)
        on_return = next_lineno is None
        injection_points.append(
            InjectionPoint(
                var_name=var_name,
                action="wait",
                trigger_lineno=next_lineno,
                trigger_on_return=on_return,
            )
        )

    return ThreadAnalysis(
        injection_points=tuple(injection_points),
        bare_copy_linenos=frozenset(bare_linenos),
        violations=tuple(validate_thread_function(func)),
    )


def _trigger_from_last_use(
    var_name: str,
    acquire_lineno: int,
    stmts: list[ast.stmt],
    file_start_line: int,
    search_upper_bound: Optional[int],
    fallback_trigger: Optional[int],
) -> Optional[int]:
    """Compute the trigger line for a push/pop injection point.

    Finds the last use of ``var_name`` within ``(acquire_lineno,
    search_upper_bound)`` and returns the line of the first statement after
    that use.  If no use is found, or the next-statement line equals or
    exceeds ``search_upper_bound`` (the last use is immediately adjacent to
    the scope boundary), ``fallback_trigger`` is returned — the conservative
    boundary that is at least correct even without last-use information.

    Returns ``None`` only when ``fallback_trigger`` is ``None`` (the caller
    should then use ``trigger_on_return=True``).
    """
    # Only attempt last-use improvement when there is a concrete fallback line.
    # When fallback_trigger is None the caller uses trigger_on_return=True,
    # which fires after the function exits and is always safe.  Replacing it
    # with an early line trigger risks firing push/pop before the block's last
    # operation has completed (e.g. before tx.wait() returns for a copy).
    if fallback_trigger is None:
        return None

    last_use = _find_last_use_lineno(
        var_name, stmts, file_start_line, acquire_lineno, search_upper_bound
    )
    if last_use is not None:
        next_lineno = _find_next_stmt_lineno(last_use, stmts, file_start_line)
        # Accept the earlier trigger only if it strictly improves upon the
        # fallback (i.e. it is an earlier line that is still before the
        # scope boundary).
        if next_lineno is not None and next_lineno < fallback_trigger:
            return next_lineno
    return fallback_trigger


# ---------------------------------------------------------------------------
# Public API: runtime interception
# ---------------------------------------------------------------------------

# sys.monitoring tool ID used by this module.  OPTIMIZER_ID is chosen
# because the simulator is not a debugger, coverage tool, or profiler.
# The tool is claimed once per interpreter session.
_TOOL_ID: int = sys.monitoring.OPTIMIZER_ID


def _fire_injection(frame: types.FrameType, ip: InjectionPoint) -> None:
    """Call push/pop/wait on the object held in ``frame.f_locals[ip.var_name]``.

    Silently skips if the variable is absent or already in the released/completed
    state — this handles the case where the user explicitly called push/pop/wait
    before the injection point was reached.
    """
    if ip.action == "wait":
        from .copy import CopyTransaction

        handle = frame.f_locals.get(ip.var_name)
        if not isinstance(handle, CopyTransaction):
            return
        if handle.is_completed:
            return
        handle.wait()
        return

    if ip.action in ("push_dfb", "pop_dfb"):
        from .dfb import DataflowBuffer

        dfb = frame.f_locals.get(ip.var_name)
        if not isinstance(dfb, DataflowBuffer):
            return
        if ip.action == "push_dfb":
            dfb.auto_push_block()
        else:
            dfb.auto_pop_block()
        return

    from .blockstate import AccessState
    from .dfb import Block

    block = frame.f_locals.get(ip.var_name)
    if not isinstance(block, Block):
        return
    if block._sm.access_state == AccessState.OS:
        return

    if ip.action == "push":
        block.push()
    else:
        block.pop()


def _line_callback(code: types.CodeType, line_number: int) -> object:
    """sys.monitoring LINE callback — fires push/pop at trigger lines.

    Called by the interpreter before executing each instrumented line.
    ``sys._getframe(1)`` gives the frame of the monitored function, which
    still has all locals live at this point.
    """
    try:
        from .context import get_context

        entry = get_context().active_hooks.get(code)
        if entry is None:
            return None
        by_lineno, _ = entry
        ips = by_lineno.get(line_number)
        if ips:
            frame = sys._getframe(1)
            for ip in ips:
                _fire_injection(frame, ip)
    except Exception:
        pass
    return None


def _return_callback(
    code: types.CodeType, instruction_offset: int, retval: object
) -> object:
    """sys.monitoring PY_RETURN callback — fires push/pop at function exit.

    Called by the interpreter when a function returns normally.  The
    monitored frame is still on the stack at this point, so
    ``sys._getframe(1)`` gives access to its locals.
    """
    try:
        from .context import get_context

        entry = get_context().active_hooks.get(code)
        if entry is None:
            return None
        _, return_ips = entry
        if return_ips:
            frame = sys._getframe(1)
            for ip in return_ips:
                _fire_injection(frame, ip)
    except Exception:
        pass
    return None


def install_auto_push_pop(
    injection_map: dict[types.CodeType, ThreadAnalysis],
) -> None:
    """Register injection hooks for the current simulation run.

    ``injection_map`` maps each thread function's code object to its
    ``ThreadAnalysis``.  Injection points for push/pop/wait (Case B) are
    stored in ``get_context().active_hooks``; bare-copy line numbers (Case A)
    are added to ``get_context().auto_wait_copy_lines``.

    On first call, claims ``_TOOL_ID`` from ``sys.monitoring`` and registers
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

    # Build lookup tables and store in the current context's active_hooks.
    from .context import get_context

    ctx = get_context()

    # Populate bare-copy line set (Case A) for all thread functions.
    for code, analysis in injection_map.items():
        for lineno in analysis.bare_copy_linenos:
            ctx.auto_wait_copy_lines.add((code, lineno))

    if not active_map:
        return

    for code, ips in active_map.items():
        by_lineno: dict[int, list[InjectionPoint]] = {}
        return_ips: list[InjectionPoint] = []
        for ip in ips:
            if ip.trigger_on_return:
                return_ips.append(ip)
            elif ip.trigger_lineno is not None:
                by_lineno.setdefault(ip.trigger_lineno, []).append(ip)

        # Ordering: wait -> push_dfb/pop_dfb -> push/pop.
        # tx.wait() must complete first (transitions the block state so that
        # the subsequent dfb release and push()/pop() can succeed).
        def _sort_key(ip: InjectionPoint) -> int:
            if ip.action == "wait":
                return 0
            if ip.action in ("push_dfb", "pop_dfb"):
                return 1
            return 2

        return_ips.sort(key=_sort_key)
        for lineno_ips in by_lineno.values():
            lineno_ips.sort(key=_sort_key)
        ctx.active_hooks[code] = (by_lineno, return_ips)

    # Claim the tool ID and register callbacks exactly once per session.
    if sys.monitoring.get_tool(_TOOL_ID) is None:
        sys.monitoring.use_tool_id(_TOOL_ID, "ttlang-sim")
        sys.monitoring.register_callback(
            _TOOL_ID, sys.monitoring.events.LINE, _line_callback
        )
        sys.monitoring.register_callback(
            _TOOL_ID, sys.monitoring.events.PY_RETURN, _return_callback
        )

    # Enable per-code-object events for THIS run's code objects.
    #
    # NOTE: We must iterate over active_map (the new code objects) rather than
    # ctx.active_hooks.items() here.  Python code objects compare equal
    # (and hash identically) when they have the same bytecode and constants,
    # even if they come from different source files.  When two kernels share a
    # function with an identical body (e.g. compute_fn in parameterized tests),
    # ctx.active_hooks retains the FIRST code object as the dict key.
    # sys.monitoring.set_local_events works on object identity, so we must
    # call it with the actual new code object, not the cached equal-but-distinct
    # object stored as the dict key.
    for code in active_map:
        by_lineno_v, return_ips_v = ctx.active_hooks[code]
        ev = sys.monitoring.events.NO_EVENTS
        if by_lineno_v:
            ev |= sys.monitoring.events.LINE
        if return_ips_v:
            ev |= sys.monitoring.events.PY_RETURN
            # Always co-register LINE alongside PY_RETURN.  On Linux,
            # set_local_events() with only PY_RETURN may not arm the
            # callback reliably; pairing it with LINE resolves this.
            ev |= sys.monitoring.events.LINE
        if ev:
            sys.monitoring.set_local_events(_TOOL_ID, code, ev)
