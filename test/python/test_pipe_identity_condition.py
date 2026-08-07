# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast

import pytest

from ttl._src.ttl_ast import TTLGenericCompiler
from ttl.diagnostics import TTLangCompileError


class _PipeIdentity:
    def __init__(self, source_device_index):
        self.source_device_index = source_device_index


def _compiler(source_device_index=1):
    compiler = object.__new__(TTLGenericCompiler)
    compiler.symbol_tables = [{"__pipe_identity": _PipeIdentity(source_device_index)}]
    compiler.source_file = "test_pipe_identity_condition.py"
    compiler.line_offset = 0
    return compiler


def _predicate(source):
    return ast.parse(source, mode="eval").body


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("pipe.source_device_index == 1", True),
        ("pipe.source_device_index != 1", False),
        ("pipe.source_device_index < 2", True),
        ("pipe.source_device_index <= 1", True),
        ("pipe.source_device_index > 0", True),
        ("pipe.source_device_index >= 2", False),
        ("1 == pipe.source_device_index", True),
    ],
)
def test_pipe_identity_predicate_is_evaluated_at_compile_time(source, expected):
    compiler = _compiler()

    result = compiler._evaluate_pipe_identity_predicate(_predicate(source))

    assert result is expected


@pytest.mark.parametrize(
    ("source_device_index", "expected_call"),
    [(0, "zero_branch"), (1, "nonzero_branch")],
)
def test_pipe_identity_if_visits_only_selected_branch(
    source_device_index, expected_call
):
    compiler = _compiler(source_device_index)
    conditional = ast.parse("""
if pipe.source_device_index == 0:
    zero_branch()
else:
    nonzero_branch()
""").body[0]
    visited_calls = []
    compiler.visit = lambda statement: visited_calls.append(statement.value.func.id)

    TTLGenericCompiler.visit_If(compiler, conditional)

    assert visited_calls == [expected_call]


def test_non_identity_predicate_is_not_folded():
    compiler = _compiler()

    result = compiler._evaluate_pipe_identity_predicate(_predicate("value == 1"))

    assert result is compiler._NO_PIPE_IDENTITY_VALUE


def test_pipe_identity_requires_literal_comparison_operand():
    compiler = _compiler()

    with pytest.raises(
        TTLangCompileError,
        match="pipe callback identity comparisons require a literal",
    ):
        compiler._evaluate_pipe_identity_predicate(
            _predicate("pipe.source_device_index == expected_source")
        )
