# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Frontend emission for static TT-Lang dispatch controllers."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

from ttl.dialects import func, ttl as ttl_dialect
from ttl.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    Location,
    Module,
    TypeAttr,
    UnitAttr,
)

from ._src.ttl_ast import _build_tensor_type
from .ttl_api import _canonical_tensor_args, _resolve_grid


@dataclass(frozen=True)
class _Invocation:
    target: object
    dispatcher_argument_names: tuple[str, ...]


def _bind_invocation(dispatcher, call: ast.Call) -> _Invocation:
    if not isinstance(call.func, ast.Name):
        raise ValueError(
            f"@ttl.operation dispatcher {dispatcher.name!r} requires a direct "
            "target name"
        )
    target = dispatcher._spec.frozen_scope.get(call.func.id)
    if target is None or getattr(target, "_ttl_operation_kind", None) != "unified":
        raise ValueError(
            f"@ttl.operation dispatcher {dispatcher.name!r} cannot resolve "
            f"target {call.func.id!r}"
        )
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError(
            f"@ttl.operation dispatcher {dispatcher.name!r} does not support "
            f"expanded keyword arguments in target {target.name!r}"
        )

    signature = inspect.signature(target._spec.fn)
    try:
        bound = signature.bind(
            *call.args,
            **{keyword.arg: keyword.value for keyword in call.keywords},
        )
    except TypeError as error:
        raise ValueError(
            f"@ttl.operation dispatcher {dispatcher.name!r} has an invalid "
            f"call to target {target.name!r}: {error}"
        ) from None
    if tuple(bound.arguments) != tuple(signature.parameters):
        missing = [name for name in signature.parameters if name not in bound.arguments]
        raise ValueError(
            f"@ttl.operation dispatcher {dispatcher.name!r} must bind every "
            f"argument of target {target.name!r}; missing {missing}"
        )
    resource_parameters = [
        parameter.name for parameter in target._spec.params if parameter.kind != "value"
    ]
    if resource_parameters:
        raise ValueError(
            f"@ttl.operation dispatcher target {target.name!r} has resource "
            f"parameter(s) {resource_parameters} and cannot compile independently"
        )

    dispatcher_parameters = set(inspect.signature(dispatcher._spec.fn).parameters)
    argument_names = []
    for target_argument, expression in bound.arguments.items():
        if (
            not isinstance(expression, ast.Name)
            or expression.id not in dispatcher_parameters
        ):
            raise ValueError(
                f"@ttl.operation dispatcher {dispatcher.name!r} target "
                f"{target.name!r} argument {target_argument!r} must directly "
                "reference a dispatcher argument"
            )
        argument_names.append(expression.id)
    return _Invocation(target, tuple(argument_names))


def _collect_invocations(dispatcher) -> tuple[_Invocation, ...]:
    invocations = []
    for statement in dispatcher._spec.dispatcher_ast.body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Return) and statement.value is None:
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            invocations.append(_bind_invocation(dispatcher, statement.value))
            continue
        raise ValueError(
            f"@ttl.operation dispatcher {dispatcher.name!r} contains unsupported "
            f"straight-line statement {type(statement).__name__}"
        )
    if not invocations:
        raise ValueError(
            f"@ttl.operation dispatcher {dispatcher.name!r} has no target invocations"
        )
    return tuple(invocations)


def _target_symbol_names(dispatcher_name: str, invocations) -> dict[str, str]:
    used = {dispatcher_name}
    symbols = {}
    for invocation in invocations:
        identity = invocation.target._spec.operation_identity
        if identity in symbols:
            continue
        base = invocation.target.name
        symbol = base
        suffix = 0
        while symbol in used:
            suffix += 1
            symbol = f"{base}_{suffix}"
        used.add(symbol)
        symbols[identity] = symbol
    return symbols


def _infer_types(ctx, dispatcher, runtime_arguments, invocations):
    argument_types = {}
    target_types = {}
    for invocation in invocations:
        target = invocation.target
        values = tuple(
            runtime_arguments[name] for name in invocation.dispatcher_argument_names
        )
        grid = _resolve_grid(target._grid, values, {})
        options = target._decorator_options
        inferred = tuple(
            _build_tensor_type(
                ctx,
                value,
                grid,
                options["tiled"],
                options["memory_space"],
            )
            for value in values
        )
        identity = target._spec.operation_identity
        previous_target_types = target_types.setdefault(identity, inferred)
        if previous_target_types != inferred:
            raise ValueError(
                f"@ttl.operation dispatcher {dispatcher.name!r} invokes target "
                f"{target.name!r} with incompatible argument types"
            )
        for name, inferred_type in zip(invocation.dispatcher_argument_names, inferred):
            previous_type = argument_types.setdefault(name, inferred_type)
            if previous_type != inferred_type:
                raise ValueError(
                    f"@ttl.operation dispatcher {dispatcher.name!r} argument "
                    f"{name!r} has incompatible target ABI types"
                )

    unused = [name for name in runtime_arguments if name not in argument_types]
    if unused:
        raise ValueError(
            f"@ttl.operation dispatcher {dispatcher.name!r} cannot infer types "
            f"for unused argument(s) {unused}"
        )
    return argument_types, target_types


def emit_dispatch_ir(dispatcher, args: tuple, kwargs: dict) -> Module:
    """Emit verified straight-line dispatch IR using concrete argument types."""
    runtime_values = _canonical_tensor_args(dispatcher._spec.fn, args, kwargs)
    runtime_arguments = {
        parameter.name: value
        for parameter, value in zip(dispatcher._spec.params, runtime_values)
    }
    invocations = _collect_invocations(dispatcher)
    symbols = _target_symbol_names(dispatcher.name, invocations)

    context = Context()
    with context, Location.unknown(context):
        ttl_dialect.ensure_dialects_registered(context)
        module = Module.create()
        argument_types, target_types = _infer_types(
            context, dispatcher, runtime_arguments, invocations
        )
        with InsertionPoint(module.body):
            declared = set()
            for invocation in invocations:
                target = invocation.target
                identity = target._spec.operation_identity
                if identity in declared:
                    continue
                declared.add(identity)
                ttl_dialect.DispatchTargetOp(
                    symbols[identity],
                    TypeAttr.get(FunctionType.get(target_types[identity], [])),
                    identity,
                    [parameter.name for parameter in target._spec.params],
                )

            dispatcher_type_list = [
                argument_types[parameter.name] for parameter in dispatcher._spec.params
            ]
            dispatcher_function = func.FuncOp(
                dispatcher.name, (dispatcher_type_list, [])
            )
            dispatcher_function.attributes["ttl.dispatcher"] = UnitAttr.get()
            entry = dispatcher_function.add_entry_block()
            dispatcher_values = {
                parameter.name: value
                for parameter, value in zip(dispatcher._spec.params, entry.arguments)
            }
            with InsertionPoint(entry):
                for invocation in invocations:
                    identity = invocation.target._spec.operation_identity
                    ttl_dialect.DispatchInvokeOp(
                        symbols[identity],
                        [
                            dispatcher_values[name]
                            for name in invocation.dispatcher_argument_names
                        ],
                    )
                func.ReturnOp([])

        module.operation.verify()
        return module


__all__ = ["emit_dispatch_ir"]
