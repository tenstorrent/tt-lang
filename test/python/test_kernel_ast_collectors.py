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
    _AssignmentCollector,
    _collect_unsupported_language_constructs,
)


def _collect_assignments(source):
    parsed_module = ast.parse(source)
    collector = _AssignmentCollector()
    for statement in parsed_module.body:
        collector.visit(statement)
    return collector


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
    ],
)
def test_assignment_collector_detects_loop_carried_recurrences(
    source, expected_assigned, expected_loop_carried
):
    collector = _collect_assignments(source)

    assert collector.names == expected_assigned
    assert collector.loop_carried_names == expected_loop_carried


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
