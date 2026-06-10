# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""@ttl.atom-in-@ttl.atom AST inlining.

A @ttl.atom function may invoke another @ttl.atom function as a
statement. At decoration time the caller's AST is rewritten by
substituting the callee's body in place of the call, binding the call
site's args to the callee's parameters and alpha-renaming the callee's
local variables to avoid collisions with the caller's scope.

Constraints on inlined callees:

  - The callee body must not declare its own DFBs or PipeNets. All
    DataFlow buffers and PipeNets used by the callee must be passed
    in as parameters. Operating on existing PipeNets (e.g.
    ``.if_src``/``.if_dst``) and emitting copies is fine.
  - The callee must be invoked as a bare statement (``Callee(...)``);
    using the result as an expression is not supported.
  - The callee's parameters must all be provided at the call site; no
    defaults are applied.
"""

from __future__ import annotations

import ast
import copy
import itertools
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from ttl.atom import Atom


_FORBIDDEN_CALLEE_NAMES = {"make_dataflow_buffer_like", "make_dfb", "PipeNet"}

_inline_counter = itertools.count()


def inline_atom_calls(
    fn_def: ast.FunctionDef,
    fn_globals: Dict[str, object],
    caller_name: str,
) -> None:
    """Rewrite ``fn_def`` in place: inline statement-level calls to
    @ttl.atom kernels reachable through ``fn_globals``.

    ``fn_def.body`` is replaced with the post-inline statement list.
    Nested compound bodies (if / for / while / with) are inlined
    recursively. Raises ``ValueError`` if a discovered callee's body
    declares its own DFBs or PipeNets.
    """
    fn_def.body = _inline_stmts(fn_def.body, fn_globals, caller_name)


def _inline_stmts(
    stmts: List[ast.stmt],
    fn_globals: Dict[str, object],
    caller_name: str,
) -> List[ast.stmt]:
    out: List[ast.stmt] = []
    for stmt in stmts:
        _inline_inside_compound(stmt, fn_globals, caller_name)
        match = _match_atom_call(stmt, fn_globals)
        if match is None:
            out.append(stmt)
            continue
        callee, call = match
        out.extend(_expand_call(callee, call, caller_name))
    return out


def _inline_inside_compound(
    stmt: ast.stmt,
    fn_globals: Dict[str, object],
    caller_name: str,
) -> None:
    """Recurse into compound-statement bodies so nested calls inline too."""
    for attr in ("body", "orelse", "finalbody"):
        body = getattr(stmt, attr, None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            setattr(stmt, attr, _inline_stmts(body, fn_globals, caller_name))
    # try/except handler bodies
    handlers = getattr(stmt, "handlers", None)
    if isinstance(handlers, list):
        for h in handlers:
            if isinstance(h, ast.ExceptHandler):
                h.body = _inline_stmts(h.body, fn_globals, caller_name)


def _match_atom_call(
    stmt: ast.stmt,
    fn_globals: Dict[str, object],
) -> Tuple["Atom", ast.Call] | None:
    """Return (callee_atom, call_node) if ``stmt`` is a statement-level
    call to a @ttl.atom kernel resolvable through ``fn_globals``."""
    from ttl.atom import Atom

    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    call = stmt.value
    if not isinstance(call.func, ast.Name):
        return None
    obj = fn_globals.get(call.func.id)
    if not isinstance(obj, Atom):
        return None
    return obj, call


def _expand_call(
    callee: "Atom",
    call: ast.Call,
    caller_name: str,
) -> List[ast.stmt]:
    spec = callee._spec
    _validate_callee_for_inline(spec, caller_name)

    bindings = _bind_args_to_params(spec, call, caller_name)
    locals_in_callee = _collect_local_names(spec.fn_ast) - set(bindings)
    suffix = f"__{spec.name}_inl{next(_inline_counter)}"
    rename_map = {n: n + suffix for n in locals_in_callee}

    transformer = _SubstituteTransformer(
        bindings=bindings,
        rename_map=rename_map,
        callee_name=spec.name,
        caller_name=caller_name,
    )
    inlined: List[ast.stmt] = []
    for stmt in spec.fn_ast.body:
        new_stmt = transformer.visit(copy.deepcopy(stmt))
        ast.fix_missing_locations(new_stmt)
        inlined.append(new_stmt)
    return inlined


def _validate_callee_for_inline(spec, caller_name: str) -> None:
    """Forbid DFB / PipeNet declarations in a callee that's being inlined."""
    for node in ast.walk(spec.fn_ast):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        attr = None
        if isinstance(func, ast.Attribute):
            attr = func.attr
        elif isinstance(func, ast.Name):
            attr = func.id
        if attr in _FORBIDDEN_CALLEE_NAMES:
            raise ValueError(
                f"@ttl.atom: cannot inline {spec.name!r} into "
                f"{caller_name!r}: callee body invokes {attr!r}. "
                "Inlined callees must take all DFBs / PipeNets as "
                "parameters; declare them in the caller and pass them in."
            )


def _bind_args_to_params(spec, call: ast.Call, caller_name: str) -> Dict[str, ast.expr]:
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise ValueError(
            f"@ttl.atom: inlining {spec.name!r} into {caller_name!r}: "
            "*-unpacking is not supported at the call site."
        )
    if any(k.arg is None for k in call.keywords):
        raise ValueError(
            f"@ttl.atom: inlining {spec.name!r} into {caller_name!r}: "
            "**-unpacking is not supported at the call site."
        )

    positional_params = [p for p in spec.params if not p.is_keyword_only]
    if len(call.args) > len(positional_params):
        raise ValueError(
            f"@ttl.atom: inlining {spec.name!r} into {caller_name!r}: "
            f"got {len(call.args)} positional args, expected at most "
            f"{len(positional_params)} (params: "
            f"{[p.name for p in positional_params]})"
        )

    bindings: Dict[str, ast.expr] = {}
    for p, arg in zip(positional_params, call.args):
        bindings[p.name] = arg

    kwargs = {k.arg: k.value for k in call.keywords}
    known_names = {p.name for p in spec.params}
    unknown = set(kwargs) - known_names
    if unknown:
        raise ValueError(
            f"@ttl.atom: inlining {spec.name!r} into {caller_name!r}: "
            f"unknown keyword args {sorted(unknown)} "
            f"(params: {sorted(known_names)})"
        )

    for p in spec.params:
        if p.name in bindings:
            if p.name in kwargs:
                raise ValueError(
                    f"@ttl.atom: inlining {spec.name!r} into "
                    f"{caller_name!r}: param {p.name!r} passed both "
                    "positionally and by keyword."
                )
            continue
        if p.name in kwargs:
            bindings[p.name] = kwargs[p.name]
            continue
        raise ValueError(
            f"@ttl.atom: inlining {spec.name!r} into {caller_name!r}: "
            f"missing argument for param {p.name!r}."
        )

    return bindings


def _collect_local_names(fn_def: ast.FunctionDef) -> Set[str]:
    """Names that the callee assigns to anywhere in its body.

    Used to alpha-rename locals so they don't collide with caller-scope
    names after substitution. Nested ``FunctionDef`` / ``Lambda`` scopes
    are skipped: their locals shadow the outer scope already.
    """
    names: Set[str] = set()

    class _LocalCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            return

        def visit_AsyncFunctionDef(self, node):
            return

        def visit_Lambda(self, node):
            return

        def visit_Assign(self, node):
            for tgt in node.targets:
                _walk_target(tgt, names)
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            _walk_target(node.target, names)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            _walk_target(node.target, names)
            self.generic_visit(node)

        def visit_For(self, node):
            _walk_target(node.target, names)
            self.generic_visit(node)

        def visit_AsyncFor(self, node):
            _walk_target(node.target, names)
            self.generic_visit(node)

        def visit_With(self, node):
            for item in node.items:
                if item.optional_vars is not None:
                    _walk_target(item.optional_vars, names)
            self.generic_visit(node)

        def visit_comprehension(self, node):
            _walk_target(node.target, names)
            self.generic_visit(node)

    for stmt in fn_def.body:
        _LocalCollector().visit(stmt)
    return names


def _walk_target(target: ast.expr, names: Set[str]) -> None:
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _walk_target(elt, names)
    elif isinstance(target, ast.Starred):
        _walk_target(target.value, names)


class _SubstituteTransformer(ast.NodeTransformer):
    """Replace param Names with caller's arg expressions; rename locals."""

    def __init__(
        self,
        bindings: Dict[str, ast.expr],
        rename_map: Dict[str, str],
        callee_name: str,
        caller_name: str,
    ):
        self.bindings = bindings
        self.rename_map = rename_map
        self.callee_name = callee_name
        self.caller_name = caller_name

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id in self.bindings:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                raise ValueError(
                    f"@ttl.atom: inlining {self.callee_name!r} into "
                    f"{self.caller_name!r}: cannot assign to or delete "
                    f"parameter {node.id!r} inside the callee body."
                )
            return ast.copy_location(copy.deepcopy(self.bindings[node.id]), node)
        if node.id in self.rename_map:
            new = ast.Name(id=self.rename_map[node.id], ctx=node.ctx)
            return ast.copy_location(new, node)
        return node
