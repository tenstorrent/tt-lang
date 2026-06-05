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

  - A callee may declare its own DFBs / PipeNets, but only when it is
    inlined at the caller body's top level: the decls are hoisted to the
    top level (where ``_lift_setup`` evaluates them), so a callee that
    declares buffers cannot be inlined inside a for / if / while / with.
    The buffer decls must themselves be top-level statements of the
    callee for the same reason. Operating on existing PipeNets (e.g.
    ``.if_src``/``.if_dst``) and emitting copies is always fine.
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


_inline_counter = itertools.count()


def inline_atom_calls(
    fn_def: ast.FunctionDef,
    fn_globals: Dict[str, object],
    caller_name: str,
) -> Dict[str, int]:
    """Rewrite ``fn_def`` in place: inline statement-level calls to
    @ttl.atom kernels reachable through ``fn_globals``.

    ``fn_def.body`` is replaced with the post-inline statement list.
    Nested compound bodies (if / for / while / with) are inlined
    recursively. Returns a map from each (alpha-renamed) inlined DFB name to
    the id of the inline site it came from, so the caller can overlay the
    scratch of distinct sibling sites onto shared CB indices. Raises
    ``ValueError`` if a callee that declares its own buffers is inlined
    anywhere but the body top level.
    """
    inlined_dfb_tags: Dict[str, int] = {}
    fn_def.body = _inline_stmts(
        fn_def.body, fn_globals, caller_name, inlined_dfb_tags, top_level=True
    )
    return inlined_dfb_tags


def _inline_stmts(
    stmts: List[ast.stmt],
    fn_globals: Dict[str, object],
    caller_name: str,
    inlined_dfb_tags: Dict[str, int],
    top_level: bool,
) -> List[ast.stmt]:
    out: List[ast.stmt] = []
    for stmt in stmts:
        _inline_inside_compound(stmt, fn_globals, caller_name, inlined_dfb_tags)
        match = _match_atom_call(stmt, fn_globals)
        if match is None:
            out.append(stmt)
            continue
        callee, call = match
        out.extend(
            _expand_call(callee, call, caller_name, top_level, inlined_dfb_tags)
        )
    return out


def _inline_inside_compound(
    stmt: ast.stmt,
    fn_globals: Dict[str, object],
    caller_name: str,
    inlined_dfb_tags: Dict[str, int],
) -> None:
    """Recurse into compound-statement bodies so nested calls inline too.

    Calls discovered here are not at the body top level, so a callee that
    declares its own buffers cannot be inlined into them.
    """
    for attr in ("body", "orelse", "finalbody"):
        body = getattr(stmt, attr, None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            setattr(
                stmt,
                attr,
                _inline_stmts(
                    body, fn_globals, caller_name, inlined_dfb_tags, top_level=False
                ),
            )
    # try/except handler bodies
    handlers = getattr(stmt, "handlers", None)
    if isinstance(handlers, list):
        for h in handlers:
            if isinstance(h, ast.ExceptHandler):
                h.body = _inline_stmts(
                    h.body, fn_globals, caller_name, inlined_dfb_tags, top_level=False
                )


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
    top_level: bool,
    inlined_dfb_tags: Dict[str, int],
) -> List[ast.stmt]:
    spec = callee._spec
    dfb_decl_names = _check_callee_buffers(spec, caller_name, top_level)

    bindings = _bind_args_to_params(spec, call, caller_name)
    # Fold the callee's constant closure freevars (e.g. factory tile-count
    # params) into the body: after substitution they live in the caller, whose
    # scope does not carry the callee's closure.
    for name, value in _closure_consts(spec.fn).items():
        bindings.setdefault(name, ast.Constant(value=value))
    locals_in_callee = _collect_local_names(spec.fn_ast) - set(bindings)
    site = next(_inline_counter)
    suffix = f"__{spec.name}_inl{site}"
    rename_map = {n: n + suffix for n in locals_in_callee}

    # The callee's local DFBs are hoisted into the caller; tag their renamed
    # names with this inline site so sibling sites can overlay CB indices.
    for n in dfb_decl_names:
        inlined_dfb_tags[rename_map.get(n, n)] = site

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


def _check_callee_buffers(spec, caller_name: str, top_level: bool) -> Set[str]:
    """Validate a callee's buffer declarations and return its top-level
    DFB-decl names.

    A callee may declare its own DFBs / Pipes / PipeNets only as top-level
    statements (so they can be hoisted) and only when inlined at the caller
    body top level (so the hoist target is the caller's top level too).
    Returns the names of the callee's top-level ``make_dfb`` /
    ``make_dataflow_buffer_like`` assigns (the scratch buffers eligible for
    reuse); Pipes / PipeNets are hoisted but not reused.
    """
    from ttl.atom import (
        _DFB_FACTORY_NAMES,
        _SETUP_FACTORY_NAMES,
        _call_name,
        _setup_assign_target,
    )

    # Whitelist every factory call inside a top-level setup-assign's value:
    # the whole RHS is evaluated when the assign is hoisted, so pipes nested in
    # a ``PipeNet([Pipe(...)])`` argument are fine. Factory calls anywhere else
    # (a compound block, a non-decl expression) cannot be hoisted.
    dfb_names: Set[str] = set()
    top_level_decl_calls: Set[int] = set()
    for stmt in spec.fn_ast.body:
        if _setup_assign_target(stmt) is None:
            continue
        for sub in ast.walk(stmt.value):
            if isinstance(sub, ast.Call) and _call_name(sub) in _SETUP_FACTORY_NAMES:
                top_level_decl_calls.add(id(sub))
        if _call_name(stmt.value) in _DFB_FACTORY_NAMES:
            dfb_names.add(stmt.targets[0].id)

    declares = False
    for node in ast.walk(spec.fn_ast):
        if not (isinstance(node, ast.Call) and _call_name(node) in _SETUP_FACTORY_NAMES):
            continue
        declares = True
        if id(node) not in top_level_decl_calls:
            raise ValueError(
                f"@ttl.atom: cannot inline {spec.name!r} into {caller_name!r}: "
                "buffer declarations must be top-level statements of the callee "
                "so they can be hoisted; found one nested in its body."
            )

    if declares and not top_level:
        raise ValueError(
            f"@ttl.atom: cannot inline {spec.name!r} into {caller_name!r}: it "
            "declares its own DFBs / PipeNets, so it must be called at the atom "
            "body top level, not inside a for / if / while / with."
        )

    return dfb_names


def _closure_consts(fn) -> Dict[str, object]:
    """Constant closure freevars of ``fn`` (e.g. a factory's tile-count params).

    These are folded into the body as literals when it is inlined, since the
    caller's scope does not carry the callee's closure cells. Only simple
    constants are folded; other freevars (atoms, modules) resolve through the
    caller's globals or are themselves inlined.
    """
    closure = getattr(fn, "__closure__", None) or ()
    freevars = getattr(getattr(fn, "__code__", None), "co_freevars", ())
    consts: Dict[str, object] = {}
    for name, cell in zip(freevars, closure):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if value is None or isinstance(value, (int, float, str, bool)):
            consts[name] = value
    return consts


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
