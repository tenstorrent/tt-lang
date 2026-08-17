# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""AST collector coverage for Python control-flow lowering.

These tests validate the pre-lowering analysis used to decide which Python
variables become SCF result values and which Python constructs are rejected.
"""

import ast

import pytest

from ttl.pykernel._src.kernel_ast import (
    TTCompilerBase,
    _AssignmentCollector,
    _collect_unsupported_language_constructs,
)


def _collect_assignments(source):
    parsed_module = ast.parse(source)
    collector = _AssignmentCollector()
    for statement in parsed_module.body:
        collector.visit(statement)
    return collector


def _collect_loop_results(source, outer_names):
    parsed_module = ast.parse(source)
    loop = parsed_module.body[0]
    assert isinstance(loop, ast.For)

    compiler = object.__new__(TTCompilerBase)
    compiler.symbol_tables = [{name: object() for name in outer_names}]
    return compiler._get_loop_carried_var_names(loop)


@pytest.mark.parametrize(
    ("source", "expected_assigned", "expected_loop_carried"),
    [
        (
            "accumulator = accumulator + delta",
            ["accumulator"],
            ["accumulator"],
        ),
        (
            "temporary = accumulator\naccumulator = temporary + delta",
            ["temporary", "accumulator"],
            ["accumulator"],
        ),
        (
            "updated = accumulator + delta\naccumulator = updated",
            ["updated", "accumulator"],
            ["accumulator"],
        ),
        (
            "accumulator = mirror = accumulator + delta",
            ["accumulator", "mirror"],
            ["accumulator"],
        ),
        (
            "left_accumulator, right_accumulator = "
            "left_accumulator + delta, right_accumulator + delta",
            ["left_accumulator", "right_accumulator"],
            ["left_accumulator", "right_accumulator"],
        ),
        (
            "if val > max_val:\n    max_val = val",
            ["max_val"],
            ["max_val"],
        ),
        (
            "if val > max_val:\n    max_val = val\nelse:\n    max_val = max_val",
            ["max_val"],
            ["max_val"],
        ),
        (
            "if cond1:\n    if val > max_val:\n        max_val = val",
            ["max_val"],
            ["max_val"],
        ),
        (
            "if cond:\n    tmp = acc\nelse:\n    tmp = other\nacc = tmp",
            ["tmp", "acc"],
            ["acc"],
        ),
        (
            "if c1:\n    x = acc\nelif c2:\n    x = acc + 1\nelse:\n    x = other\nacc = x",
            ["x", "acc"],
            ["acc"],
        ),
        (
            "if candidate_value > retained_value:\n"
            "    candidate_value = retained_value\n"
            "    candidate_index = retained_index\n"
            "if candidate_value == retained_value:\n"
            "    if candidate_index < retained_index:\n"
            "        candidate_value = retained_value\n"
            "        candidate_index = retained_index",
            ["candidate_value", "candidate_index"],
            ["candidate_value", "candidate_index"],
        ),
    ],
)
def test_assignment_collector_detects_loop_carried_recurrences(
    source, expected_assigned, expected_loop_carried
):
    collector = _collect_assignments(source)

    assert collector.names == expected_assigned
    assert collector.loop_carried_names == expected_loop_carried


def test_condition_read_without_assignment_is_not_loop_carried():
    collector = _collect_assignments("if x > 0:\n    y = x")

    assert collector.names == ["y"]
    assert collector.loop_carried_names == []


def test_assignment_collector_tracks_augassign_only_names():
    collector = _collect_assignments("accumulator += delta")

    assert collector.names == ["accumulator"]
    assert collector.loop_carried_names == ["accumulator"]
    assert collector.augassign_only_names == {"accumulator"}


def test_plain_assignment_clears_augassign_only_status():
    collector = _collect_assignments(
        "accumulator += delta\naccumulator = accumulator + delta"
    )

    assert collector.loop_carried_names == ["accumulator"]
    assert collector.augassign_only_names == set()


@pytest.mark.parametrize(
    ("source", "outer_names", "expected_results"),
    [
        (
            "for iteration in range(4):\n    selected_token = next_token",
            {"selected_token", "next_token"},
            ["selected_token"],
        ),
        (
            "for iteration in range(4):\n"
            "    previous_first = previous_second\n"
            "    previous_second = current_token\n"
            "    current_token = selected_token",
            {
                "previous_first",
                "previous_second",
                "current_token",
                "selected_token",
            },
            ["previous_first", "previous_second", "current_token"],
        ),
        (
            "for iteration in range(4):\n"
            "    temporary = source\n"
            "    result = temporary",
            {"source", "result"},
            ["result"],
        ),
        (
            "for iteration in range(0):\n    result = replacement",
            {"result", "replacement"},
            ["result"],
        ),
    ],
)
def test_loop_results_include_every_outer_reassignment(
    source, outer_names, expected_results
):
    assert _collect_loop_results(source, outer_names) == expected_results


@pytest.mark.parametrize(
    ("source", "expected_construct"),
    [
        (
            "while condition:\n    accumulator = delta",
            "while loops",
        ),
        (
            "accumulator = left_value if condition else right_value",
            "conditional expressions",
        ),
        (
            "if (condition_value := condition):\n    accumulator = condition_value",
            "assignment expressions",
        ),
        (
            "match selector:\n    case 0:\n        accumulator = zero_value",
            "match statements",
        ),
    ],
)
def test_unsupported_language_constructs_are_detected(source, expected_construct):
    parsed_module = ast.parse(source)

    unsupported = _collect_unsupported_language_constructs(parsed_module.body)

    assert unsupported
    assert unsupported[0][1] == expected_construct
