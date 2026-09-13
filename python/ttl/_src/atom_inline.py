# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Composition support for unified ``@ttl.operation`` functions."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ttl.condition import DispatchCondition
from ttl.dfb_allocation_group import DFBAllocationGroup
from ttl.dfb_reset import DFBReset
from ttl.dfb_reconfiguration import DFBReconfiguration
from ttl.fabric import FabricManagerClaim
from ttl.kernel import Kernel, KernelKind, _selector_implicit_role
from ttl.scalar import ScalarType

_INLINED_OPERATION_STATEMENT = "_ttl_inlined_operation_statement"
_DFB_SOURCE_OCCURRENCE = "_ttl_dfb_source_occurrence"

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
        transformed_node = self.generic_visit(node)
        if not isinstance(transformed_node.iter, (ast.Tuple, ast.List)):
            return transformed_node

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
                transformed_statement = loop_transformer.visit(
                    copy.deepcopy(statement)
                )
                if isinstance(transformed_statement, list):
                    unrolled_body.extend(transformed_statement)
                else:
                    unrolled_body.append(transformed_statement)
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
            transformed = self.generic_visit(node)
            if not transformed.body:
                transformed.body = [ast.copy_location(ast.Pass(), transformed)]
            return transformed
        finally:
            self.static_booleans = enclosing_booleans

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

    result: List[ast.stmt] = []
    for statement in spec.fn_ast.body:
        cloned_statement = copy.deepcopy(statement)
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
    if value is ScalarType or isinstance(value, ScalarType):
        type_name = "class" if value is ScalarType else value.name.lower()
        fresh_name = _fresh_name(
            f"{name_hint}__scalar_type_{type_name}", suffix, reserved_names
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
