# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Syntactic rules for @ttl.operation bodies, shared by both frontends.

The compiler (``ttl/atom.py``) and the simulator (``sim/unified_operation.py``)
must answer the same questions about an operation body before either can act on
it: does the body write its threads by hand, which statements construct the
dataflow buffers and pipe nets those threads share, and is that construction in a
position where it can be lifted out of the body. The two have to answer
identically -- a body they read differently is a program that behaves differently
in simulation than it does compiled -- so each rule lives here once and is called
from both, rather than being restated on each side and kept in sync by hand.

Every rule is a predicate over syntax, and nothing outside the standard library is
imported, because the simulator loads this file by path: it shadows
``sys.modules["ttl"]`` with its own namespace object, which has no importable
submodules, and the ``tt-lang-sim`` wheel ships no compiler code. ``atom_split``
is bundled under the same constraint.

What stays with each frontend is what legitimately differs between them: how a
construction is evaluated (the compiler produces compile-time allocations, the
simulator live objects whose identity carries dataflow state), how a diagnostic is
worded and located, and what each does with a rule's answer.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set, Tuple

# Factory calls whose results are shared by every thread of an operation, so their
# construction is lifted out of the body and evaluated once.
DFB_FACTORY_NAMES: Set[str] = {
    "make_dfb",
    "make_dataflow_buffer_like",
    "make_tensor_backed_dfb",
}
PIPE_FACTORY_NAMES: Set[str] = {"Pipe", "PipeNet"}
KERNEL_FACTORY_NAMES: Set[str] = {"Kernel"}
ALLOCATION_GROUP_FACTORY_NAMES: Set[str] = {"make_dfb_allocation_group"}
SETUP_FACTORY_NAMES: Set[str] = (
    DFB_FACTORY_NAMES
    | PIPE_FACTORY_NAMES
    | KERNEL_FACTORY_NAMES
    | ALLOCATION_GROUP_FACTORY_NAMES
)

# Decorators that mark a hand-written kernel of a multi-kernel operation.
KERNEL_DECORATORS: Set[str] = {"compute", "datamovement"}


def parse_function_definition(
    fn: Callable[..., Any], *, rebase_lines: bool = False
) -> Optional[ast.FunctionDef]:
    """Parse ``fn`` and return its ``FunctionDef``, or None if it cannot be read.

    With ``rebase_lines``, line numbers are shifted onto the enclosing file rather
    than starting at 1. A caller that keeps the parsed statements needs them
    absolute -- the simulator compiles the split kernels under the original file's
    name, and its copy-wait analysis keys on absolute line numbers -- while a
    caller that only inspects structure does not care.
    """
    try:
        source_lines, start_lineno = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        return None
    try:
        module = ast.parse(textwrap.dedent("".join(source_lines)))
    except (IndentationError, SyntaxError):
        return None
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return None
    if rebase_lines:
        ast.increment_lineno(module, start_lineno - 1)
    return module.body[0]


class _ReturnFinder(ast.NodeVisitor):
    """Finds a ``return`` in an operation body, ignoring its nested functions.

    A kernel written inside the body is a function of its own and may return; the
    body itself may not.
    """

    def __init__(self) -> None:
        self.found = False

    def visit_Return(self, node: ast.Return) -> None:
        self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def validate_operation_interface(fn: Callable[..., Any]) -> None:
    """Check ``fn``'s signature and body against the operation interface rules.

    The specification states these under "Operation function": an operation takes
    only tensors as parameters, "Operation parameters have no default values, and
    the signature uses no ``*args`` or ``**kwargs``", and the function returns
    nothing -- everything else it needs is a compile-time argument captured from
    the enclosing scope.

    The rules are here rather than with either frontend because they decide which
    programs are operations at all, and a program one frontend takes and the other
    refuses is worse than either answer. The wording is the compiler's, whose
    diagnostics are pinned by ``test/python/atom/operation_boundaries_invalid.py``.

    A body whose source cannot be read is left alone: only the return rule needs
    the source, and refusing an operation because its source is unavailable (a
    REPL, an exec'd string) would refuse a program that is otherwise fine.

    Raises:
        ValueError: If a parameter is variadic or has a default, or if the body
            returns.
    """
    for parameter in inspect.signature(fn).parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                "@ttl.operation does not support *args or **kwargs "
                f"(parameter {parameter.name!r})"
            )
        if parameter.default is not inspect.Parameter.empty:
            raise ValueError(
                "@ttl.operation parameters cannot have default values "
                f"(parameter {parameter.name!r})"
            )

    function_definition = parse_function_definition(fn)
    if function_definition is None:
        return
    finder = _ReturnFinder()
    for statement in function_definition.body:
        finder.visit(statement)
    if finder.found:
        raise ValueError(
            "@ttl.operation functions cannot return a value or use return statements"
        )


def closure_values(fn: Callable[..., Any]) -> Dict[str, Any]:
    """The free variables ``fn`` captures, mapped to their current values.

    Cells still empty at call time (a recursive definition being bound) are
    skipped rather than reported, since there is no value to resolve against yet.
    """
    values: Dict[str, Any] = {}
    closure = getattr(fn, "__closure__", None)
    if not closure:
        return values
    for name, cell in zip(getattr(fn.__code__, "co_freevars", ()), closure):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            continue
    return values


def function_scope(fn: Callable[..., Any]) -> Dict[str, Any]:
    """Names visible to ``fn``: its globals plus the free variables it captures.

    This is the scope a construction lifted out of the body is evaluated in, and
    the one a decorator or factory reference is resolved through.
    """
    scope = dict(getattr(fn, "__globals__", {}) or {})
    scope.update(closure_values(fn))
    return scope


def decorator_name(decorator: ast.expr) -> Optional[str]:
    """The name a decorator expression spells, ignoring its receiver."""
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Name):
        return decorator.id
    return None


def is_kernel_decorator_spelling(decorator: ast.expr) -> bool:
    """True when ``decorator`` is spelled like a kernel decorator.

    The name alone is matched, so ``@anything.compute`` counts. A caller able to
    resolve the decorator to a runtime object should compare against that instead
    and use this only as a fallback, since spelling cannot tell an unrelated
    decorator named ``compute`` from the real one.
    """
    return decorator_name(decorator) in KERNEL_DECORATORS


def defines_kernels(
    fn_def: ast.FunctionDef, is_kernel_decorator: Callable[[ast.expr], bool]
) -> bool:
    """True when ``fn_def`` nests a function ``is_kernel_decorator`` accepts.

    The predicate is a parameter because a caller that can resolve a decorator to
    its runtime object recognizes more than spelling does, while the walk itself
    has to be the same for both: which definitions count (nested ones, sync and
    async alike, never ``fn_def`` itself) decides whether a body is thread-unified,
    and answering that differently would split a body one frontend runs as three
    hand-written kernels.
    """
    for node in ast.walk(fn_def):
        if node is fn_def:
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(is_kernel_decorator(d) for d in node.decorator_list):
            return True
    return False


def defines_kernels_by_spelling(fn_def: ast.FunctionDef) -> bool:
    """True when ``fn_def`` nests a function decorated as a kernel."""
    return defines_kernels(fn_def, is_kernel_decorator_spelling)


def call_name(node: ast.expr) -> Optional[str]:
    """The callee name of a Call node (``ttl.Pipe`` -> ``Pipe``), else None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def is_pipe_list_expr(node: ast.expr) -> bool:
    """A list/tuple/comprehension whose elements are all ``ttl.Pipe(...)``.

    Lets a PipeNet be built from a separately-named pipe list
    (``ps = [ttl.Pipe(...) for ...]; net = ttl.PipeNet(ps)``), the natural
    way to express multicast/reduce fan-out.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        return bool(node.elts) and all(call_name(e) == "Pipe" for e in node.elts)
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return call_name(node.elt) == "Pipe"
    return False


def setup_assign_target(stmt: ast.stmt) -> Optional[str]:
    """If ``stmt`` constructs a static operation resource, return its name.

    Recognizes ``name = <dfb/pipe/kernel-factory>(...)`` and ``name = [<pipes>]``
    (a pipe list feeding a later PipeNet). A single ``Name`` target is required,
    because the construction is lifted out of the body and every thread that needs
    the object refers to it by that name."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    if not isinstance(stmt.targets[0], ast.Name):
        return None
    if call_name(stmt.value) in SETUP_FACTORY_NAMES or is_pipe_list_expr(stmt.value):
        return stmt.targets[0].id
    return None


def collect_assignment_targets(target: ast.expr, names: Set[str]) -> None:
    """Collect the names an assignment target binds, tuple targets included."""
    if isinstance(target, ast.Name):
        names.add(target.id)
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            collect_assignment_targets(element, names)


def non_resource_assignment_names(fn_def: ast.FunctionDef) -> Set[str]:
    """Names the body assigns itself, excluding the lifted construction.

    These exist only where the body runs, so a lifted construction cannot read
    them.
    """
    names: Set[str] = set()
    for statement in fn_def.body:
        if setup_assign_target(statement) is not None:
            continue
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                collect_assignment_targets(target, names)
        elif isinstance(statement, ast.AnnAssign):
            collect_assignment_targets(statement.target, names)
    return names


def loaded_names_in(node: ast.AST) -> Set[str]:
    """Every name read anywhere under ``node``."""
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


@dataclass(frozen=True)
class LocalDependency:
    """A resource construction reading values the body computes for itself."""

    statement: ast.stmt
    target: str
    names: Tuple[str, ...]


@dataclass(frozen=True)
class UnhoistableResource:
    """A resource construction that is not a liftable top-level assignment."""

    call: ast.Call
    factory: str


def find_local_dependency(fn_def: ast.FunctionDef) -> Optional[LocalDependency]:
    """The first resource construction that reads an operation-local value.

    Construction is lifted out of the body and evaluated before the threads run,
    so it can only read names bound outside the body.
    """
    local_values = non_resource_assignment_names(fn_def)
    for statement in fn_def.body:
        target = setup_assign_target(statement)
        if target is None:
            continue
        dependencies = loaded_names_in(statement) & local_values
        if dependencies:
            return LocalDependency(statement, target, tuple(sorted(dependencies)))
    return None


def find_unhoistable_resource(
    fn_def: ast.FunctionDef,
) -> Optional[UnhoistableResource]:
    """The first factory call that is not part of a liftable top-level assign.

    Construction under control flow, in a callback, in a nested scope, or bound to
    anything but a single name stays where it is written and is therefore repeated
    by every thread that reaches it, instead of being shared.
    """
    liftable: Set[int] = set()
    for statement in fn_def.body:
        if setup_assign_target(statement) is None:
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and call_name(node) in SETUP_FACTORY_NAMES:
                liftable.add(id(node))

    for node in ast.walk(fn_def):
        if not isinstance(node, ast.Call):
            continue
        factory = call_name(node)
        if factory not in SETUP_FACTORY_NAMES or id(node) in liftable:
            continue
        assert factory is not None
        return UnhoistableResource(node, factory)
    return None


def validate_resource_declarations(
    fn_def: ast.FunctionDef, operation_name: str
) -> None:
    """Require resource construction to use simple top-level assignments.

    Raises:
        ValueError: If a construction reads an operation-local value, or is not a
            top-level assignment to a single name.
    """
    dependency = find_local_dependency(fn_def)
    if dependency is not None:
        raise ValueError(
            f"@ttl.operation {operation_name!r}: resource declarations "
            "cannot depend on operation-local values "
            f"{list(dependency.names)}"
        )

    unhoistable = find_unhoistable_resource(fn_def)
    if unhoistable is not None:
        raise ValueError(
            f"@ttl.operation {operation_name!r}: resource declaration "
            f"{unhoistable.factory!r} must be a simple top-level assignment in the "
            "operation body; declarations inside control flow, callbacks, or "
            "nested scopes are not supported"
        )
