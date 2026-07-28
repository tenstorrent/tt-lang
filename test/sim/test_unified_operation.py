#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for running thread-unified @ttl.operation bodies on the simulator.

A unified body (no hand-written @ttl.compute / @ttl.datamovement kernels) is
split into compute/dm0/dm1 kernels by reusing the compiler frontend's
thread-assignment splitter (ttl._src.atom_split). The synthesized kernels are
compiled under the original source file's name, so their line numbers must stay
absolute: the copy-wait analysis locates a kernel's source through
inspect.getsourcelines (i.e. co_firstlineno) and keys injection points on
absolute line numbers.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from sim.unified_operation import (  # type: ignore[reportPrivateUsage]
    _parse_operation_funcdef,
    build_multikernel_function,
    is_unified_body,
)
from test_helpers.sim_runner import run_script_in_process

FIXTURES_DIR = Path(__file__).resolve().parent / "test_helpers"


class _NeverCalled:
    """Stand-in for the ``ttl`` module in the sample body below.

    The sample body exists to be parsed and split, never executed; running it
    would mean a test is exercising the wrong thing, so every attribute access
    fails loudly.
    """

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"sample unified body must not run (accessed {name})")


ttl = _NeverCalled()


def _unified_body(a: Any, out: Any) -> None:
    """A minimal unified operation body, used as splitter input by the tests."""
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    blk = dfb.reserve()
    ttl.copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = dfb.wait()
    ttl.copy(out_blk, out[0:1, 0:1]).wait()
    out_blk.pop()


def test_parsed_body_uses_absolute_line_numbers() -> None:
    """The parsed body is numbered by source file, not from 1.

    ``inspect.getsourcelines`` returns a snippet starting at the decorator, so an
    un-rebased parse numbers the body from 1 while ``co_filename`` still names
    the real file -- which sends every consumer to the top of that file.
    """
    assert is_unified_body(_unified_body)

    fn_def = _parse_operation_funcdef(_unified_body)
    assert fn_def.lineno == _unified_body.__code__.co_firstlineno

    # A body statement's line number must resolve to that statement in this file.
    dfb_stmt = next(s for s in fn_def.body if isinstance(s, ast.Assign))
    own_lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    assert "make_dataflow_buffer_like" in own_lines[dfb_stmt.lineno - 1]


def test_split_function_keeps_source_line_numbers() -> None:
    """The synthesized multi-kernel function stays anchored to the original body.

    ``analysis.py`` recovers each kernel's source through ``co_firstlineno``, so
    the rewritten function must report the line of the original ``def``.
    """
    built = build_multikernel_function(_unified_body, {"ttl": ttl})
    assert built.__code__.co_firstlineno == _unified_body.__code__.co_firstlineno
    assert built.__code__.co_filename == _unified_body.__code__.co_filename


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_unified_operation_bare_copy_injects_wait(scheduler: str) -> None:
    """Bare ``ttl.copy(src, dst)`` in a unified body gets its wait injected.

    Injection points are keyed on (code object, absolute lineno), so a body whose
    line numbers do not match its file gets no injection at all and fails with a
    dataflow-state error ("expected TX_WAIT, attempted PUSH") instead.
    """
    code, out = run_script_in_process(FIXTURES_DIR / "unified_bare_copy.py", scheduler)
    assert code == 0, f"unified operation with bare copies failed:\n{out}"
