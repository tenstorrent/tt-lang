#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for running thread-unified @ttl.operation bodies on the simulator.

A unified body (no hand-written @ttl.compute / @ttl.datamovement kernels) is
split into compute/dm0/dm1 kernels by reusing the compiler frontend's
thread-assignment splitter (ttl._src.atom_split). The fixture runs with
ttl/ttnn shadowed by the simulator, exactly as ``tt-lang-sim`` does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test_helpers.sim_runner import run_script_in_process

FIXTURES_DIR = Path(__file__).resolve().parent / "test_helpers"
SPEC_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "spec"


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_unified_operation_add(scheduler: str) -> None:
    """A unified-body elementwise add splits into compute/dm kernels and runs."""
    code, out = run_script_in_process(FIXTURES_DIR / "unified_add.py", scheduler)
    assert code == 0, f"unified operation failed with code {code}. Output:\n{out}"


@pytest.mark.xfail(
    reason="operation_function.py uses the assigned-handle form "
    "`tx = ttl.copy(...)` + `tx.wait()`, which the thread-assignment splitter "
    "(atom_split) rejects; it requires `ttl.copy(...).wait()` as a single "
    "statement. Expected to fail until the example is rewritten or the splitter "
    "grows support for assigned transfer handles.",
)
@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_unified_operation_spec_example(scheduler: str) -> None:
    """The spec's operation_function.py exercises the same unified-body path but
    with separate `tx = ttl.copy(...)` / `tx.wait()` statements (see xfail reason)."""
    code, out = run_script_in_process(
        SPEC_EXAMPLES_DIR / "operation_function" / "operation_function.py", scheduler
    )
    assert code == 0, f"spec operation_function.py failed with code {code}:\n{out}"
