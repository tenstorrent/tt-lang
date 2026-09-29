# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Composition support for unified ``@ttl.operation`` functions."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

from ttl.condition import DispatchCondition
from ttl.dfb_allocation_group import DFBAllocationGroup
from ttl.dfb_reset import DFBReset
from ttl.dfb_reconfiguration import DFBReconfiguration
from ttl.fabric import FabricManagerClaim
from ttl.kernel import Kernel, KernelKind, _selector_implicit_role
from ttl.scalar import ScalarType

_INLINED_OPERATION_STATEMENT = "_ttl_inlined_operation_statement"
_DFB_SOURCE_OCCURRENCE = "_ttl_dfb_source_occurrence"
_LOOP_TARGETS_READ_AFTER = "_ttl_loop_targets_read_after"

_NESTED_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _copy_dfb_source_occurrence(
    source: ast.AST,
    destination: ast.AST,
    default_occurrence: Optional[str] = None,
) -> None:
    occurrence = getattr(source, _DFB_SOURCE_OCCURRENCE, default_occurrence)
    if occurrence is not None:
        setattr(destination, _DFB_SOURCE_OCCURRENCE, occurrence)


class _OuterLocalCollector(ast.NodeVisitor):
    def __init__(self):
        self.names: Set[str] = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_FunctionDef(self, node):
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node):
        self.names.add(node.name)

    def visit_Lambda(self, node):
        return

    def visit_ListComp(self, node):
        return

    def visit_SetComp(self, node):
        return

    def visit_DictComp(self, node):
        return

    def visit_GeneratorExp(self, node):
        return

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)


class _NestedBindingCollector(ast.NodeVisitor):
    def __init__(self):
        self.names: Set[str] = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_FunctionDef(self, node):
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node):
        self.names.add(node.name)

    def visit_Lambda(self, node):
        return

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)


def _loop_target_paths(target: ast.expr, path: Tuple[int, ...] = ()):
    if isinstance(target, ast.Name):
        yield path, target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for index, element in enumerate(target.elts):
            yield from _loop_target_paths(element, path + (index,))


_DEFERRED_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.GeneratorExp,
)
_TRY_STATEMENTS = (ast.Try, getattr(ast, "TryStar", ast.Try))


def _read_names(node: Optional[ast.AST]) -> Set[str]:
    if node is None:
        return set()
    return {
        name.id
        for name in ast.walk(node)
        if isinstance(name, ast.Name) and isinstance(name.ctx, (ast.Load, ast.Del))
    }


def _stored_names(target: Optional[ast.expr]) -> Set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _stored_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_stored_names(element) for element in target.elts))
    return set()


class _Jumps(NamedTuple):
    """Names live where a ``break``, ``continue``, or exception transfers."""

    break_live: FrozenSet[str] = frozenset()
    continue_live: FrozenSet[str] = frozenset()
    exception_live: FrozenSet[str] = frozenset()


class _LoopExitLiveness:
    """Backward live-name analysis that records each ``For`` loop's exit.

    A name is live at a point when some path from that point reads it before
    storing it. The set recorded for a loop is the one live where the loop
    finishes, before its ``else`` suite. Unknown control flow is
    over-approximated: any loop may run zero times, an exception may leave a
    ``try`` or ``with`` body at any statement or a ``for`` loop at any
    iteration, and a ``finally`` suite continues to every exit of its ``try``
    whichever way it was entered.
    """

    def __init__(self):
        self.exit_live_by_loop: Dict[int, Set[str]] = {}
        # An enclosing loop's iterations only grow a nested loop's live sets,
        # so each loop resumes from its previous fixed point.
        self.head_live_by_loop: Dict[int, Set[str]] = {}

    def block(self, statements, live_out: Set[str], jumps: _Jumps) -> Set[str]:
        live = set(live_out)
        for statement in reversed(statements):
            live = self._statement(statement, live, jumps) | jumps.exception_live
        return live

    def _loop(self, node, live_out: Set[str], jumps: _Jumps) -> Set[str]:
        exit_live = self.block(node.orelse, live_out, jumps)
        head = set(self.head_live_by_loop.get(id(node), ()))
        while True:
            body_jumps = _Jumps(
                frozenset(live_out), frozenset(head), jumps.exception_live
            )
            body_live = self.block(node.body, head, body_jumps)
            if isinstance(node, ast.While):
                new_head = _read_names(node.test) | body_live
            else:
                new_head = (body_live - _stored_names(node.target)) | _read_names(
                    node.target
                )
            new_head |= exit_live | jumps.exception_live
            if new_head == head:
                break
            head = new_head
        self.head_live_by_loop[id(node)] = head
        if isinstance(node, ast.For):
            self.exit_live_by_loop[id(node)] = exit_live
        if isinstance(node, ast.While):
            return head
        return _read_names(node.iter) | head

    def _try(self, node, live_out: Set[str], jumps: _Jumps) -> Set[str]:
        final_live = set(live_out)
        final_jumps = jumps
        if node.finalbody:
            final_live = self.block(
                node.finalbody,
                live_out
                | jumps.break_live
                | jumps.continue_live
                | jumps.exception_live,
                jumps,
            )
            final_jumps = jumps._replace(
                exception_live=jumps.exception_live | final_live
            )
        # Every matching ``except*`` handler runs, so each one continues into
        # the handlers after it.
        chains_handlers = not isinstance(node, ast.Try)
        handler_live: Set[str] = set()
        for handler in reversed(node.handlers):
            later_live = handler_live if chains_handlers else set()
            handler_jumps = final_jumps._replace(
                exception_live=final_jumps.exception_live | later_live
            )
            handler_body_live = self.block(
                handler.body, final_live | later_live, handler_jumps
            )
            handler_live = (
                handler_live
                | _read_names(handler.type)
                | (handler_body_live - {handler.name})
            )
        orelse_live = self.block(node.orelse, final_live, final_jumps)
        body_jumps = final_jumps._replace(
            exception_live=final_jumps.exception_live | handler_live
        )
        return self.block(node.body, orelse_live, body_jumps)

    def _statement(self, node, live_out: Set[str], jumps: _Jumps) -> Set[str]:
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            return self._loop(node, live_out, jumps)
        if isinstance(node, _TRY_STATEMENTS):
            return self._try(node, live_out, jumps)
        if isinstance(node, ast.If):
            return (
                _read_names(node.test)
                | self.block(node.body, live_out, jumps)
                | self.block(node.orelse, live_out, jumps)
            )
        if isinstance(node, (ast.With, ast.AsyncWith)):
            # The context manager may suppress an exception from the body.
            body_jumps = jumps._replace(exception_live=jumps.exception_live | live_out)
            stored: Set[str] = set()
            reads: Set[str] = set()
            for item in node.items:
                stored |= _stored_names(item.optional_vars)
                reads |= _read_names(item.context_expr)
                reads |= _read_names(item.optional_vars)
            return reads | (self.block(node.body, live_out, body_jumps) - stored)
        if isinstance(node, ast.Match):
            live = _read_names(node.subject) | live_out
            for case in node.cases:
                live |= _read_names(case.pattern) | _read_names(case.guard)
                live |= self.block(case.body, live_out, jumps)
            return live
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.block(node.body, set(), _Jumps())
            # TODO: `_read_names` counts a nested scope's own
            # parameters and locals as reads of the enclosing names, so a
            # later `lambda kind: ...` makes a loop target `kind` live and a
            # loop over non-literal elements is rejected; subtract the nested
            # scope's bindings here and in the `Assign` case below.
            return (live_out - {node.name}) | _read_names(node)
        if isinstance(node, ast.Break):
            return set(jumps.break_live)
        if isinstance(node, ast.Continue):
            return set(jumps.continue_live)
        if isinstance(node, (ast.Return, ast.Raise)):
            return _read_names(node)
        if isinstance(node, ast.Assign):
            stored = set().union(*(_stored_names(target) for target in node.targets))
            return (live_out - stored) | _read_names(node)
        if isinstance(node, ast.AnnAssign):
            stored = _stored_names(node.target) if node.value is not None else set()
            return (
                (live_out - stored) | _read_names(node.value) | _read_names(node.target)
            )
        if isinstance(node, ast.AugAssign):
            return live_out | _stored_names(node.target) | _read_names(node)
        if isinstance(node, ast.Delete):
            deleted = set().union(*(_stored_names(target) for target in node.targets))
            return (live_out - deleted) | _read_names(node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return live_out - {
                alias.asname or alias.name.split(".")[0] for alias in node.names
            }
        return live_out | _read_names(node)


def _mark_loop_targets_read_after(statements: List[ast.stmt]) -> None:
    """Record on each ``For`` which target names code after it may read.

    The mark is read only when the loop is unrolled. It maps each such name's
    position in the target to its name. A name counts when it is live where the
    loop finishes, or when a nested function, lambda, or generator outside the
    loop body reads it, since those run at a point the analysis does not know.
    Unrolling substitutes the element into scopes inside the loop body, so they
    never read the name.

    TODO: a callback defined outside the loop and called
    inside it reads the target at call time in Python but sees only the
    post-loop binding here (or is rejected for non-literal elements), and a
    callback defined inside the body binds its iteration's element at
    definition time. Reject or model calls of such callbacks.
    """
    liveness = _LoopExitLiveness()
    liveness.block(statements, set(), _Jumps())
    closure_reads_by_scope = {
        id(scope): _read_names(scope) - _nested_binding_names(scope)
        for statement in statements
        for scope in ast.walk(statement)
        if isinstance(scope, _DEFERRED_SCOPES)
    }
    for statement in statements:
        for loop in ast.walk(statement):
            if not isinstance(loop, ast.For):
                continue
            body_node_ids = {
                id(node) for inner in loop.body for node in ast.walk(inner)
            }
            used_names = set(liveness.exit_live_by_loop[id(loop)])
            for scope_id, reads in closure_reads_by_scope.items():
                if scope_id not in body_node_ids:
                    used_names |= reads
            setattr(
                loop,
                _LOOP_TARGETS_READ_AFTER,
                {
                    path: name
                    for path, name in _loop_target_paths(loop.target)
                    if name in used_names
                },
            )


def _is_constant_structure(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_constant_structure(element) for element in node.elts)
    return False


def _rebound_names(statements) -> Set[str]:
    """Return names stored or deleted in ``statements`` outside nested scopes."""
    names: Set[str] = set()
    pending = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, _NESTED_SCOPES):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name)
        pending.extend(ast.iter_child_nodes(node))
    return names


def _contains_loop_control(statements) -> bool:
    """Return whether break or continue in ``statements`` targets their loop."""
    pending = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Break, ast.Continue)):
            return True
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            # Only a nested loop's else clause still belongs to the outer loop.
            pending.extend(node.orelse)
            continue
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return False


class _SubstituteTransformer(ast.NodeTransformer):
    def __init__(
        self,
        bindings: Dict[str, ast.expr],
        rename_map: Dict[str, str],
        callee_name: str,
        caller_name: str,
        dfb_parameter_names: Set[str],
        inline_suffix: str,
    ):
        self.bindings = bindings
        self.rename_map = rename_map
        self.callee_name = callee_name
        self.caller_name = caller_name
        self.inline_suffix = inline_suffix
        self.dfb_parameter_occurrences = {
            name: f"{inline_suffix}:{name}" for name in dfb_parameter_names
        }

    def visit_FunctionDef(self, node):
        node.name = self.rename_map.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        node.name = self.rename_map.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.bindings:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                raise ValueError(
                    f"@ttl.operation: composing {self.callee_name!r} into "
                    f"{self.caller_name!r} cannot assign to parameter "
                    f"{node.id!r}"
                )
            replacement = copy.deepcopy(self.bindings[node.id])
            # Preserve a nested formal dependency occurrence, or create one
            # when this substitution binds the current operation's DFB parameter.
            _copy_dfb_source_occurrence(
                node,
                replacement,
                self.dfb_parameter_occurrences.get(node.id),
            )
            return ast.copy_location(replacement, node)
        if node.id not in self.rename_map:
            return node
        replacement = ast.Name(id=self.rename_map[node.id], ctx=node.ctx)
        _copy_dfb_source_occurrence(node, replacement)
        return ast.copy_location(replacement, node)

    def visit_Subscript(self, node):
        transformed_node = self.generic_visit(node)
        if not isinstance(transformed_node.value, (ast.Tuple, ast.List)):
            return transformed_node
        try:
            sequence_index = ast.literal_eval(transformed_node.slice)
        except (TypeError, ValueError, SyntaxError):
            return transformed_node
        if not isinstance(sequence_index, int) or isinstance(sequence_index, bool):
            return transformed_node
        try:
            element = transformed_node.value.elts[sequence_index]
        except IndexError:
            return transformed_node
        _copy_dfb_source_occurrence(transformed_node, element)
        return ast.copy_location(element, transformed_node)

    def visit_For(self, node):
        # Local names are renamed below; keep the source spelling for messages.
        source_target_names = dict(_loop_target_paths(node.target))
        transformed_node = self.generic_visit(node)
        if not isinstance(transformed_node.iter, (ast.Tuple, ast.List)):
            return transformed_node
        # Unrolling removes the loop that break and continue refer to.
        if _contains_loop_control(transformed_node.body):
            raise ValueError(
                f"@ttl.operation {self.caller_name!r}: a loop over a captured "
                "sequence cannot use break or continue"
            )
        # Unrolling substitutes the element for every read of the target, so a
        # store to the target inside the body would be bypassed by later reads.
        renamed_to_source = {
            name: source_target_names[path]
            for path, name in _loop_target_paths(transformed_node.target)
        }
        rebound_targets = sorted(
            renamed_to_source[name]
            for name in _rebound_names(transformed_node.body) & set(renamed_to_source)
        )
        if rebound_targets:
            raise ValueError(
                f"@ttl.operation {self.caller_name!r}: loop target "
                f"{rebound_targets[0]!r} in {self.callee_name!r} is rebound "
                "inside its loop over a captured sequence"
            )
        targets_read_after = getattr(transformed_node, _LOOP_TARGETS_READ_AFTER)
        # TODO: an empty sequence leaves an earlier binding of
        # the target in place in Python; this rejects that program too because
        # the liveness analysis does not track definite assignment.
        if targets_read_after and not transformed_node.iter.elts:
            raise ValueError(
                f"@ttl.operation {self.caller_name!r}: a loop over an empty "
                "captured sequence leaves its target unbound for code outside "
                "the loop"
            )

        unrolled_body = []
        for element in transformed_node.iter.elts:
            loop_bindings = {}
            self._bind_loop_target(transformed_node.target, element, loop_bindings)
            loop_transformer = _SubstituteTransformer(
                loop_bindings,
                {},
                self.callee_name,
                self.caller_name,
                set(),
                self.inline_suffix,
            )
            for statement in transformed_node.body:
                transformed_statement = loop_transformer.visit(copy.deepcopy(statement))
                if isinstance(transformed_statement, list):
                    unrolled_body.extend(transformed_statement)
                else:
                    unrolled_body.append(transformed_statement)
        # As in Python, the target keeps the last element after the loop; code
        # after the loop that reads a target name needs that binding.
        for path, source_name in sorted(targets_read_after.items()):
            target = transformed_node.target
            value = transformed_node.iter.elts[-1]
            for index in path:
                target = target.elts[index]
                value = value.elts[index]
            if not _is_constant_structure(value):
                raise ValueError(
                    f"@ttl.operation {self.caller_name!r}: loop target "
                    f"{source_name!r} in {self.callee_name!r} is read after its "
                    "loop over a captured sequence, which requires literal "
                    "elements"
                )
            unrolled_body.append(
                ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=target.id, ctx=ast.Store())],
                        value=copy.deepcopy(value),
                    ),
                    transformed_node,
                )
            )
        # Without break, the else suite runs once after the last element.
        unrolled_body.extend(transformed_node.orelse)
        return unrolled_body

    def _bind_loop_target(self, target, value, bindings):
        if isinstance(target, ast.Name):
            bindings[target.id] = value
            return
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
        ):
            for target_element, value_element in zip(target.elts, value.elts):
                self._bind_loop_target(target_element, value_element, bindings)
            return
        raise ValueError(
            f"@ttl.operation: captured sequence loop in {self.callee_name!r} "
            "has incompatible target and element structures"
        )


def inline_atom_calls(
    fn_def: ast.FunctionDef,
    fn_globals: Dict[str, object],
    caller_name: str,
) -> Tuple[
    Dict[str, object],
    Dict[str, Kernel],
    Dict[str, FabricManagerClaim],
    Dict[str, DispatchCondition],
    Dict[str, DFBAllocationGroup],
    Dict[str, DFBReset],
    Dict[str, DFBReconfiguration],
]:
    reserved_names = _identifier_names(fn_def)
    external_pipenets = {}
    logical_kernels = {}
    fabric_manager_claims = {
        name: fn_globals[name]
        for name in sorted(_loaded_names(fn_def.body))
        if name in fn_globals and isinstance(fn_globals[name], FabricManagerClaim)
    }
    dispatch_conditions = {}
    allocation_groups = {}
    dfb_resets = {}
    dfb_reconfigurations = {}
    inline_discriminators = {}
    fn_def.body = _inline_statements(
        fn_def.body,
        fn_globals,
        caller_name,
        reserved_names,
        external_pipenets,
        logical_kernels,
        fabric_manager_claims,
        dispatch_conditions,
        allocation_groups,
        dfb_resets,
        dfb_reconfigurations,
        inline_discriminators,
    )
    return (
        external_pipenets,
        logical_kernels,
        fabric_manager_claims,
        dispatch_conditions,
        allocation_groups,
        dfb_resets,
        dfb_reconfigurations,
    )


def _static_boolean_value(
    expression: ast.expr,
    static_booleans: Dict[str, bool],
) -> Optional[bool]:
    if isinstance(expression, ast.Constant) and type(expression.value) is bool:
        return expression.value
    if isinstance(expression, ast.Name):
        return static_booleans.get(expression.id)
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
        operand = _static_boolean_value(expression.operand, static_booleans)
        return None if operand is None else not operand
    if isinstance(expression, ast.BoolOp):
        operands = [
            _static_boolean_value(value, static_booleans) for value in expression.values
        ]
        if any(operand is None for operand in operands):
            return None
        if isinstance(expression.op, ast.And):
            return all(operands)
        if isinstance(expression.op, ast.Or):
            return any(operands)
    return None


class _StaticBooleanBranchSpecializer(ast.NodeTransformer):
    def __init__(self, captured_values: Dict[str, object]):
        self.static_booleans = {
            name: value
            for name, value in captured_values.items()
            if type(value) is bool
        }

    def _visit_function(self, node):
        enclosing_booleans = self.static_booleans
        self.static_booleans = {
            name: value
            for name, value in enclosing_booleans.items()
            if name not in _nested_binding_names(node)
        }
        try:
            return self.generic_visit(node)
        finally:
            self.static_booleans = enclosing_booleans

    def generic_visit(self, node):
        # Removing a branch can empty the body of an enclosing statement.
        transformed = super().generic_visit(node)
        body = getattr(transformed, "body", None)
        if isinstance(body, list) and not body:
            transformed.body = [ast.copy_location(ast.Pass(), transformed)]
        # A try statement needs a handler or a nonempty finally block.
        if (
            isinstance(transformed, ast.Try)
            and not transformed.handlers
            and not transformed.finalbody
        ):
            transformed.finalbody = [ast.copy_location(ast.Pass(), transformed)]
        return transformed

    def visit_FunctionDef(self, node):
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node):
        return self._visit_function(node)

    def visit_If(self, node):
        condition = _static_boolean_value(node.test, self.static_booleans)
        if condition is None:
            return self.generic_visit(node)
        selected = node.body if condition else node.orelse
        specialized = []
        for statement in selected:
            replacement = self.visit(statement)
            if isinstance(replacement, list):
                specialized.extend(replacement)
            elif replacement is not None:
                specialized.append(replacement)
        return specialized


def specialize_static_boolean_branches(
    fn_def: ast.FunctionDef,
    captured_values: Dict[str, object],
) -> None:
    """Remove branches selected by captured or inlined boolean literals."""
    _StaticBooleanBranchSpecializer(captured_values).visit(fn_def)


def _inline_statements(
    statements: List[ast.stmt],
    scope: Dict[str, object],
    caller_name: str,
    reserved_names: Set[str],
    external_pipenets: Dict[str, object],
    logical_kernels: Dict[str, Kernel],
    fabric_manager_claims: Dict[str, FabricManagerClaim],
    dispatch_conditions: Dict[str, DispatchCondition],
    allocation_groups: Dict[str, DFBAllocationGroup],
    dfb_resets: Dict[str, DFBReset],
    dfb_reconfigurations: Dict[str, DFBReconfiguration],
    inline_discriminators: Dict[str, int],
) -> List[ast.stmt]:
    result: List[ast.stmt] = []
    for statement in statements:
        _inline_compound_bodies(
            statement,
            scope,
            caller_name,
            reserved_names,
            external_pipenets,
            logical_kernels,
            fabric_manager_claims,
            dispatch_conditions,
            allocation_groups,
            dfb_resets,
            dfb_reconfigurations,
            inline_discriminators,
        )
        match = _standalone_operation_call(statement, scope)
        if match is None:
            _reject_unsupported_operation_calls(statement, scope, caller_name)
            result.append(statement)
            continue
        callee, call = match
        result.extend(
            _expand_call(
                callee,
                call,
                caller_name,
                scope,
                reserved_names,
                external_pipenets,
                logical_kernels,
                fabric_manager_claims,
                dispatch_conditions,
                allocation_groups,
                dfb_resets,
                dfb_reconfigurations,
                inline_discriminators,
            )
        )
    return result


def _inline_compound_bodies(
    statement: ast.stmt,
    scope: Dict[str, object],
    caller_name: str,
    reserved_names: Set[str],
    external_pipenets: Dict[str, object],
    logical_kernels: Dict[str, Kernel],
    fabric_manager_claims: Dict[str, FabricManagerClaim],
    dispatch_conditions: Dict[str, DispatchCondition],
    allocation_groups: Dict[str, DFBAllocationGroup],
    dfb_resets: Dict[str, DFBReset],
    dfb_reconfigurations: Dict[str, DFBReconfiguration],
    inline_discriminators: Dict[str, int],
) -> None:
    for attribute in ("body", "orelse", "finalbody"):
        body = getattr(statement, attribute, None)
        if not isinstance(body, list):
            continue
        if not body or not isinstance(body[0], ast.stmt):
            continue
        inlined = _inline_statements(
            body,
            scope,
            caller_name,
            reserved_names,
            external_pipenets,
            logical_kernels,
            fabric_manager_claims,
            dispatch_conditions,
            allocation_groups,
            dfb_resets,
            dfb_reconfigurations,
            inline_discriminators,
        )
        setattr(statement, attribute, inlined)

    handlers = getattr(statement, "handlers", None)
    if not isinstance(handlers, list):
        return
    for handler in handlers:
        if isinstance(handler, ast.ExceptHandler):
            handler.body = _inline_statements(
                handler.body,
                scope,
                caller_name,
                reserved_names,
                external_pipenets,
                logical_kernels,
                fabric_manager_claims,
                dispatch_conditions,
                allocation_groups,
                dfb_resets,
                dfb_reconfigurations,
                inline_discriminators,
            )


def _standalone_operation_call(
    statement: ast.stmt,
    scope: Dict[str, object],
) -> Optional[Tuple[object, ast.Call]]:
    if not isinstance(statement, ast.Expr):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if not isinstance(call.func, ast.Name):
        return None
    callee = scope.get(call.func.id)
    if _operation_kind(callee) != "unified":
        return None
    if not hasattr(callee, "_spec"):
        return None
    return callee, call


def _operation_kind(value: object) -> Optional[str]:
    return getattr(value, "_ttl_operation_kind", None)


def _resolve_reference(node: ast.expr, scope: Dict[str, object]):
    if isinstance(node, ast.Name):
        return scope.get(node.id)
    if not isinstance(node, ast.Attribute):
        return None
    parent = _resolve_reference(node.value, scope)
    return inspect.getattr_static(parent, node.attr, None)


def _reject_unsupported_operation_calls(
    statement: ast.stmt,
    scope: Dict[str, object],
    caller_name: str,
) -> None:
    for node in ast.walk(statement):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == caller_name:
            raise ValueError(f"@ttl.operation {caller_name!r} cannot compose itself")
        callee = _resolve_reference(node.func, scope)
        operation_kind = _operation_kind(callee)
        if operation_kind is None:
            continue
        reference = ast.unparse(node.func)
        if operation_kind == "multi_kernel":
            raise ValueError(
                f"@ttl.operation: cannot compose multi-kernel operation "
                f"{reference!r} into {caller_name!r}"
            )
        if isinstance(node.func, ast.Attribute):
            raise ValueError(
                f"@ttl.operation: compose {reference!r} through a captured "
                "name instead of a qualified reference"
            )
        raise ValueError(
            f"@ttl.operation: composed call to {reference!r} in "
            f"{caller_name!r} must be a standalone statement"
        )


def _expand_call(
    callee: object,
    call: ast.Call,
    caller_name: str,
    scope: Dict[str, object],
    reserved_names: Set[str],
    external_pipenets: Dict[str, object],
    logical_kernels: Dict[str, Kernel],
    fabric_manager_claims: Dict[str, FabricManagerClaim],
    dispatch_conditions: Dict[str, DispatchCondition],
    allocation_groups: Dict[str, DFBAllocationGroup],
    dfb_resets: Dict[str, DFBReset],
    dfb_reconfigurations: Dict[str, DFBReconfiguration],
    inline_discriminators: Dict[str, int],
) -> List[ast.stmt]:
    spec = callee._spec
    bindings = _bind_args_to_params(spec, call, caller_name)
    suffix = _inline_suffix(spec, call, inline_discriminators)
    _add_capture_bindings(
        spec,
        bindings,
        scope,
        reserved_names,
        suffix,
    )
    _add_external_pipenet_bindings(
        spec,
        bindings,
        scope,
        reserved_names,
        external_pipenets,
        suffix,
    )
    selected_kernels = _add_logical_kernel_bindings(
        spec,
        bindings,
        scope,
        reserved_names,
        logical_kernels,
    )
    _add_fabric_manager_claim_bindings(
        spec,
        bindings,
        scope,
        reserved_names,
        fabric_manager_claims,
    )
    _add_dispatch_condition_bindings(
        spec,
        bindings,
        scope,
        reserved_names,
        dispatch_conditions,
    )
    _add_allocation_group_bindings(
        spec,
        bindings,
        scope,
        reserved_names,
        allocation_groups,
    )
    _add_dfb_reset_bindings(
        spec,
        bindings,
        scope,
        reserved_names,
        dfb_resets,
        selected_kernels,
        suffix,
    )
    _add_dfb_reconfiguration_bindings(
        spec,
        bindings,
        scope,
        reserved_names,
        dfb_reconfigurations,
        selected_kernels,
        suffix,
    )

    local_names = _collect_local_names(spec.fn_ast)
    rebound_names = local_names & set(bindings)
    if rebound_names:
        raise ValueError(
            f"@ttl.operation: composing {spec.name!r} into {caller_name!r} "
            f"would rebind {sorted(rebound_names)}"
        )
    _validate_nested_bindings(spec, bindings, local_names, caller_name)

    rename_map = _make_rename_map(local_names, suffix, reserved_names)
    transformer = _SubstituteTransformer(
        bindings,
        rename_map,
        spec.name,
        caller_name,
        set(spec.dfb_param_names),
        suffix,
    )

    cloned_body = copy.deepcopy(spec.fn_ast.body)
    _mark_loop_targets_read_after(cloned_body)
    result: List[ast.stmt] = []
    for cloned_statement in cloned_body:
        transformed_statement = transformer.visit(cloned_statement)
        inlined_statements = (
            transformed_statement
            if isinstance(transformed_statement, list)
            else [transformed_statement]
        )
        for inlined_statement in inlined_statements:
            ast.fix_missing_locations(inlined_statement)
            setattr(inlined_statement, _INLINED_OPERATION_STATEMENT, True)
            result.append(inlined_statement)
    return result


def _inline_suffix(spec, call: ast.Call, discriminators: Dict[str, int]) -> str:
    """Return a deterministic, caller-local suffix for one composed call."""
    call_text = ast.dump(call, annotate_fields=True, include_attributes=False)
    digest = hashlib.sha256(
        f"{spec.operation_identity}\0{call_text}".encode("utf-8")
    ).hexdigest()[:12]
    occurrence = discriminators.get(digest, 0)
    discriminators[digest] = occurrence + 1
    return f"__{spec.name}_inl_{digest}_{occurrence}"


def _add_capture_bindings(
    spec,
    bindings: Dict[str, ast.expr],
    scope: Dict[str, object],
    reserved_names: Set[str],
    suffix: str,
) -> None:
    loaded_names = _loaded_names(spec.fn_ast.body)
    for name in sorted(loaded_names):
        if name in bindings:
            continue
        if name in spec.compile_time_captures:
            value = spec.compile_time_captures[name]
            bindings[name] = _literal_node(
                value,
                scope=scope,
                reserved_names=reserved_names,
                suffix=suffix,
                name_hint=name,
            )
            continue
        value = spec.frozen_scope.get(name)
        if not (
            isinstance(value, KernelKind)
            or isinstance(value, Kernel)
            and _selector_implicit_role(value) is not None
        ):
            continue
        fresh_name = _fresh_name(f"{spec.name}__{name}", suffix, reserved_names)
        scope[fresh_name] = value
        bindings[name] = ast.Name(id=fresh_name, ctx=ast.Load())


def _add_external_pipenet_bindings(
    spec,
    bindings: Dict[str, ast.expr],
    scope: Dict[str, object],
    reserved_names: Set[str],
    external_pipenets: Dict[str, object],
    suffix: str,
) -> None:
    loaded_names = _loaded_names(spec.fn_ast.body)
    for name, pipenet in spec.external_pipenets.items():
        if name not in loaded_names or name in bindings:
            continue
        fresh_name = _fresh_name(name, suffix, reserved_names)
        bindings[name] = ast.Name(id=fresh_name, ctx=ast.Load())
        scope[fresh_name] = pipenet
        external_pipenets[fresh_name] = pipenet


def _add_logical_kernel_bindings(
    spec,
    bindings: Dict[str, ast.expr],
    scope: Dict[str, object],
    reserved_names: Set[str],
    logical_kernels: Dict[str, Kernel],
) -> Dict[int, Kernel]:
    loaded_names = _loaded_names(spec.fn_ast.body)
    selected_kernels: Dict[int, Kernel] = {}
    synchronization_participant_ids = {
        id(participant)
        for reset_name, reset in spec.dfb_resets.items()
        if reset_name in loaded_names
        for participant in reset.participants
    }
    synchronization_participant_ids.update(
        id(participant)
        for boundary_name, boundary in spec.dfb_reconfigurations.items()
        if boundary_name in loaded_names
        for participant in boundary.participants
        if isinstance(participant, Kernel)
    )
    for name, kernel in spec.logical_kernels.items():
        if name in bindings:
            continue
        if (
            name not in loaded_names
            and id(kernel) not in synchronization_participant_ids
        ):
            continue
        existing_name = next(
            (
                candidate_name
                for candidate_name, candidate in logical_kernels.items()
                if candidate == kernel
            ),
            None,
        )
        if existing_name is None:
            existing_name = _fresh_name(f"{spec.name}__{name}", "", reserved_names)
            scope[existing_name] = kernel
            logical_kernels[existing_name] = kernel
        selected_kernels[id(kernel)] = logical_kernels[existing_name]
        bindings[name] = ast.Name(id=existing_name, ctx=ast.Load())
    return selected_kernels


def _add_dispatch_condition_bindings(
    spec,
    bindings: Dict[str, ast.expr],
    scope: Dict[str, object],
    reserved_names: Set[str],
    dispatch_conditions: Dict[str, DispatchCondition],
) -> None:
    loaded_names = _loaded_names(spec.fn_ast.body)
    for name, condition in spec.dispatch_conditions.items():
        if name not in loaded_names or name in bindings:
            continue
        existing_name = next(
            (
                candidate_name
                for candidate_name, candidate in dispatch_conditions.items()
                if candidate is condition
            ),
            None,
        )
        if existing_name is None:
            existing_name = _fresh_name(f"{spec.name}__{name}", "", reserved_names)
            scope[existing_name] = condition
            dispatch_conditions[existing_name] = condition
        bindings[name] = ast.Name(id=existing_name, ctx=ast.Load())


def _add_fabric_manager_claim_bindings(
    spec,
    bindings: Dict[str, ast.expr],
    scope: Dict[str, object],
    reserved_names: Set[str],
    fabric_manager_claims: Dict[str, FabricManagerClaim],
) -> None:
    loaded_names = _loaded_names(spec.fn_ast.body)
    for name, claim in spec.fabric_manager_claims.items():
        if name not in loaded_names or name in bindings:
            continue
        existing_name = next(
            (
                candidate_name
                for candidate_name, candidate in fabric_manager_claims.items()
                if candidate is claim
            ),
            None,
        )
        if existing_name is None:
            existing_name = _fresh_name(f"{spec.name}__{name}", "", reserved_names)
            scope[existing_name] = claim
            fabric_manager_claims[existing_name] = claim
        bindings[name] = ast.Name(id=existing_name, ctx=ast.Load())


def _add_allocation_group_bindings(
    spec,
    bindings: Dict[str, ast.expr],
    scope: Dict[str, object],
    reserved_names: Set[str],
    allocation_groups: Dict[str, DFBAllocationGroup],
) -> None:
    loaded_names = _loaded_names(spec.fn_ast.body)
    for name, group in spec.allocation_groups.items():
        if name not in loaded_names or name in bindings:
            continue
        existing_name = next(
            (
                candidate_name
                for candidate_name, candidate in allocation_groups.items()
                if candidate is group
            ),
            None,
        )
        if existing_name is None:
            existing_name = _fresh_name(f"{spec.name}__{name}", "", reserved_names)
            scope[existing_name] = group
            allocation_groups[existing_name] = group
        bindings[name] = ast.Name(id=existing_name, ctx=ast.Load())


def _remap_composed_synchronization_participant(
    participant: Kernel | KernelKind,
    selected_kernels: Dict[int, Kernel],
) -> Kernel | KernelKind:
    if not isinstance(participant, Kernel):
        return participant
    if _selector_implicit_role(participant) is not None:
        return participant
    return selected_kernels[id(participant)]


def _add_dfb_reset_bindings(
    spec,
    bindings: Dict[str, ast.expr],
    scope: Dict[str, object],
    reserved_names: Set[str],
    dfb_resets: Dict[str, DFBReset],
    selected_kernels: Dict[int, Kernel],
    suffix: str,
) -> None:
    loaded_names = _loaded_names(spec.fn_ast.body)
    reset_instances: Dict[int, DFBReset] = {}
    for name, reset in spec.dfb_resets.items():
        if name not in loaded_names or name in bindings:
            continue
        reset_instance = reset_instances.get(id(reset))
        if reset_instance is None:
            # Each composed call executes a distinct dynamic reset. Aliases
            # within that call retain one identity across all participants.
            reset_instance = DFBReset(
                participants=tuple(
                    _remap_composed_synchronization_participant(
                        participant, selected_kernels
                    )
                    for participant in reset.participants
                ),
            )
            reset_instances[id(reset)] = reset_instance
        fresh_name = _fresh_name(f"{spec.name}__{name}", suffix, reserved_names)
        scope[fresh_name] = reset_instance
        dfb_resets[fresh_name] = reset_instance
        bindings[name] = ast.Name(id=fresh_name, ctx=ast.Load())


def _add_dfb_reconfiguration_bindings(
    spec,
    bindings: Dict[str, ast.expr],
    scope: Dict[str, object],
    reserved_names: Set[str],
    dfb_reconfigurations: Dict[str, DFBReconfiguration],
    selected_kernels: Dict[int, Kernel],
    suffix: str,
) -> None:
    loaded_names = _loaded_names(spec.fn_ast.body)
    boundary_instances: Dict[int, DFBReconfiguration] = {}
    for name, boundary in spec.dfb_reconfigurations.items():
        if name not in loaded_names or name in bindings:
            continue
        boundary_instance = boundary_instances.get(id(boundary))
        if boundary_instance is None:
            # Each composed call declares a distinct boundary site. Aliases
            # within that call retain one identity across all participants.
            boundary_instance = DFBReconfiguration(
                participants=tuple(
                    _remap_composed_synchronization_participant(
                        participant, selected_kernels
                    )
                    for participant in boundary.participants
                ),
                discard_dfb_state=boundary.discard_dfb_state,
            )
            boundary_instances[id(boundary)] = boundary_instance
        fresh_name = _fresh_name(f"{spec.name}__{name}", suffix, reserved_names)
        scope[fresh_name] = boundary_instance
        dfb_reconfigurations[fresh_name] = boundary_instance
        bindings[name] = ast.Name(id=fresh_name, ctx=ast.Load())


def _literal_node(
    value: object,
    *,
    scope: Dict[str, object],
    reserved_names: Set[str],
    suffix: str,
    name_hint: str,
) -> ast.expr:
    if value is ScalarType or isinstance(value, (ScalarType, KernelKind)):
        type_name = "class" if value is ScalarType else value.name.lower()
        category = "kernel_kind" if isinstance(value, KernelKind) else "scalar_type"
        fresh_name = _fresh_name(
            f"{name_hint}__{category}_{type_name}", suffix, reserved_names
        )
        scope[fresh_name] = value
        return ast.Name(id=fresh_name, ctx=ast.Load())
    if isinstance(value, tuple):
        elements = [
            _literal_node(
                element,
                scope=scope,
                reserved_names=reserved_names,
                suffix=suffix,
                name_hint=f"{name_hint}_{index}",
            )
            for index, element in enumerate(value)
        ]
        return ast.Tuple(elts=elements, ctx=ast.Load())
    if isinstance(value, list):
        elements = [
            _literal_node(
                element,
                scope=scope,
                reserved_names=reserved_names,
                suffix=suffix,
                name_hint=f"{name_hint}_{index}",
            )
            for index, element in enumerate(value)
        ]
        return ast.List(elts=elements, ctx=ast.Load())
    return ast.Constant(value=value)


def _make_rename_map(
    names: Set[str],
    suffix: str,
    reserved_names: Set[str],
) -> Dict[str, str]:
    rename_map = {}
    for name in sorted(names):
        rename_map[name] = _fresh_name(name, suffix, reserved_names)
    return rename_map


def _fresh_name(base: str, suffix: str, reserved_names: Set[str]) -> str:
    candidate = base + suffix
    discriminator = 0
    while candidate in reserved_names:
        discriminator += 1
        candidate = f"{base}{suffix}_{discriminator}"
    reserved_names.add(candidate)
    return candidate


def _bind_args_to_params(spec, call: ast.Call, caller_name: str) -> Dict[str, ast.expr]:
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        raise ValueError(
            f"@ttl.operation: composing {spec.name!r} into {caller_name!r} "
            "does not support *-unpacking"
        )
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError(
            f"@ttl.operation: composing {spec.name!r} into {caller_name!r} "
            "does not support **-unpacking"
        )

    positional_parameters = [
        parameter for parameter in spec.params if not parameter.is_keyword_only
    ]
    if len(call.args) > len(positional_parameters):
        raise ValueError(
            f"@ttl.operation: composing {spec.name!r} into {caller_name!r} "
            f"received too many positional arguments"
        )

    bindings: Dict[str, ast.expr] = {}
    for parameter, argument in zip(positional_parameters, call.args):
        bindings[parameter.name] = argument

    keyword_arguments = {}
    for keyword in call.keywords:
        keyword_arguments[keyword.arg] = keyword.value

    parameter_names = {parameter.name for parameter in spec.params}
    unknown_names = set(keyword_arguments) - parameter_names
    if unknown_names:
        raise ValueError(
            f"@ttl.operation: composing {spec.name!r} into {caller_name!r} "
            f"received unknown arguments {sorted(unknown_names)}"
        )

    for parameter in spec.params:
        if parameter.name in bindings:
            if parameter.name in keyword_arguments:
                raise ValueError(
                    f"@ttl.operation: argument {parameter.name!r} was passed twice"
                )
            continue
        if parameter.name not in keyword_arguments:
            raise ValueError(
                f"@ttl.operation: missing argument {parameter.name!r} while "
                f"composing {spec.name!r} into {caller_name!r}"
            )
        bindings[parameter.name] = keyword_arguments[parameter.name]

    for name, argument in bindings.items():
        if not isinstance(argument, ast.Name):
            raise TypeError(
                f"@ttl.operation: argument {name!r} while composing "
                f"{spec.name!r} into {caller_name!r} must be a tensor or "
                "resource name"
            )
    return bindings


def _collect_local_names(fn_def: ast.FunctionDef) -> Set[str]:
    collector = _OuterLocalCollector()
    for statement in fn_def.body:
        collector.visit(statement)
    return collector.names


def _identifier_names(root: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _loaded_names(roots: Iterable[ast.AST]) -> Set[str]:
    names: Set[str] = set()
    for root in roots:
        for node in ast.walk(root):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
    return names


def _scope_roots(scope: ast.AST) -> List[ast.AST]:
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return list(scope.body)
    if isinstance(scope, ast.Lambda):
        return [scope.body]

    roots: List[ast.AST] = []
    if isinstance(scope, ast.DictComp):
        roots.extend((scope.key, scope.value))
    elif isinstance(scope, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        roots.append(scope.elt)
    for generator in scope.generators:
        roots.append(generator.iter)
        roots.extend(generator.ifs)
    return roots


def _function_parameter_names(scope: ast.AST) -> Set[str]:
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return set()
    names = {argument.arg for argument in scope.args.posonlyargs}
    names.update(argument.arg for argument in scope.args.args)
    names.update(argument.arg for argument in scope.args.kwonlyargs)
    if scope.args.vararg is not None:
        names.add(scope.args.vararg.arg)
    if scope.args.kwarg is not None:
        names.add(scope.args.kwarg.arg)
    return names


def _nested_binding_names(scope: ast.AST) -> Set[str]:
    names = _function_parameter_names(scope)
    collector = _NestedBindingCollector()
    for root in _scope_roots(scope):
        collector.visit(root)
    names.update(collector.names)
    if isinstance(scope, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        for generator in scope.generators:
            _collect_target_names(generator.target, names)
    return names


def _collect_target_names(target: ast.expr, names: Set[str]) -> None:
    if isinstance(target, ast.Name):
        names.add(target.id)
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            _collect_target_names(element, names)
        return
    if isinstance(target, ast.Starred):
        _collect_target_names(target.value, names)


def _validate_nested_bindings(
    spec,
    bindings: Dict[str, ast.expr],
    local_names: Set[str],
    caller_name: str,
) -> None:
    protected_names = set(bindings)
    protected_names.update(local_names)
    for replacement in bindings.values():
        if isinstance(replacement, ast.Name):
            protected_names.add(replacement.id)

    for scope in ast.walk(spec.fn_ast):
        if scope is spec.fn_ast or not isinstance(scope, _NESTED_SCOPES):
            continue
        conflicts = _nested_binding_names(scope) & protected_names
        if conflicts:
            raise ValueError(
                f"@ttl.operation: composing {spec.name!r} into "
                f"{caller_name!r} would capture or rebind "
                f"{sorted(conflicts)}; rename the nested binding"
            )
