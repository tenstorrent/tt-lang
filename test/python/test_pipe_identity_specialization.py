# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for graph callback identity predicate planning."""

import ast
import textwrap

import pytest

from ttl._src.ttl_ast import (
    TTLGenericCompiler,
    _InvalidPipeIdentity,
    _NotPipeIdentity,
    _PipeIdentityPredicate,
    _PipeIdentityValue,
)


class _PipeIdentity:
    def __init__(self, source_device_index=1, destination_device_index=0):
        self.source_device_index = source_device_index
        self.destination_device_index = destination_device_index


def _predicate(source):
    return ast.parse(source, mode="eval").body


def _compiler(*, captures=None, globals_=None):
    compiler = object.__new__(TTLGenericCompiler)
    compiler.captures = captures or {}
    compiler.fn_globals = globals_ or {}
    return compiler


def _evaluate(source, identity=None):
    identity = identity or _PipeIdentity()
    return _compiler()._evaluate_pipe_identity_predicate(
        _predicate(source), {"pipe": _PipeIdentityValue(identity)}
    )


@pytest.mark.parametrize(
    "source,expected",
    [
        ("pipe.source_device_index == 1", True),
        ("pipe.source_device_index != 1", False),
        ("pipe.source_device_index < 2", True),
        ("pipe.source_device_index <= 0", False),
        ("pipe.source_device_index > 0", True),
        ("pipe.source_device_index >= 2", False),
        ("1 == pipe.source_device_index", True),
    ],
)
def test_pipe_identity_predicate_evaluates_supported_comparisons(source, expected):
    result = _evaluate(source)

    assert isinstance(result, _PipeIdentityPredicate)
    assert result.value is expected


def test_non_identity_predicate_is_not_specialized():
    result = _compiler()._evaluate_pipe_identity_predicate(
        _predicate("runtime_value == 1"), {}
    )

    assert isinstance(result, _NotPipeIdentity)


@pytest.mark.parametrize("constant_table", ["captures", "globals"])
def test_pipe_identity_predicate_resolves_named_constant(constant_table):
    compiler_args = {constant_table: {"ROOT": 1}}
    compiler = _compiler(
        captures=compiler_args.get("captures"),
        globals_=compiler_args.get("globals"),
    )

    result = compiler._evaluate_pipe_identity_predicate(
        _predicate("pipe.source_device_index == ROOT"),
        {"pipe": _PipeIdentityValue(_PipeIdentity())},
    )

    assert isinstance(result, _PipeIdentityPredicate)
    assert result.value is True


@pytest.mark.parametrize(
    "source,message",
    [
        (
            "0 < pipe.source_device_index < 2",
            "require a single comparison",
        ),
        (
            "pipe.source_device_index == runtime_value",
            "require a compile-time constant non-identity operand",
        ),
        (
            "pipe.unknown_device_index == 0",
            "has no property 'unknown_device_index'",
        ),
        (
            "pipe.source_device_index < 'zero'",
            "invalid pipe callback identity comparison",
        ),
    ],
)
def test_invalid_pipe_identity_predicate_has_typed_diagnostic(source, message):
    result = _evaluate(source)

    assert isinstance(result, _InvalidPipeIdentity)
    assert message in result.message


def test_callback_plan_tracks_alias_and_selected_nested_branch():
    callback = ast.parse(
        """
def callback(pipe):
    source_device_index = pipe.source_device_index
    if source_device_index == 1:
        if 0 < source_device_index:
            selected()
    else:
        rejected()
"""
    ).body[0]
    outer_if = callback.body[1]
    nested_if = outer_if.body[0]

    plan = _compiler()._plan_pipe_identity_callback(
        callback.body, "pipe", _PipeIdentity(source_device_index=1)
    )

    outer_predicate = plan.predicate_for(outer_if)
    nested_predicate = plan.predicate_for(nested_if)
    assert isinstance(outer_predicate, _PipeIdentityPredicate)
    assert outer_predicate.value is True
    assert isinstance(nested_predicate, _PipeIdentityPredicate)
    assert nested_predicate.value is True


@pytest.mark.parametrize(
    "mutation",
    [
        "source_device_index += 1",
        "source_device_index: int = 1",
        "del source_device_index",
        "try:\n    pass\nexcept Exception as source_device_index:\n    pass",
        "import math as source_device_index",
        "from math import pi as source_device_index",
        "def source_device_index():\n    pass",
        "async def source_device_index():\n    pass",
        "class source_device_index:\n    pass",
        "for unused in range(1):\n    pass\nelse:\n    source_device_index = 1",
    ],
)
def test_callback_plan_discards_identity_after_reassignment(mutation):
    callback = ast.parse(
        f"""
def callback(pipe):
    source_device_index = pipe.source_device_index
{textwrap.indent(mutation, "    ")}
    if source_device_index == 1:
        selected()
"""
    ).body[0]
    conditional = callback.body[-1]

    plan = _compiler()._plan_pipe_identity_callback(
        callback.body, "pipe", _PipeIdentity(source_device_index=1)
    )

    assert isinstance(plan.predicate_for(conditional), _NotPipeIdentity)


@pytest.mark.parametrize(
    "source",
    [
        "global source_device_index",
        "def nested():\n    nonlocal source_device_index",
    ],
)
def test_unmodeled_scope_declaration_discards_identity(source):
    statement = ast.parse(source).body[0]
    if isinstance(statement, ast.FunctionDef):
        statement = statement.body[0]

    assigned_names = TTLGenericCompiler._pipe_identity_statement_assigned_names(
        statement
    )

    assert assigned_names == {"source_device_index"}
