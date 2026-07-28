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
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

from sim import ttl as ttl_alias
from sim.copy import copy as renamed_copy
from sim.decorators import compute, datamovement
from sim.dfb import make_dataflow_buffer_like
from sim.unified_operation import (  # type: ignore[reportPrivateUsage]
    _parse_operation_funcdef,
    _reject_aliased_api,
    _symbol_table,
    build_multikernel_function,
    is_unified_body,
)
from test_helpers.sim_runner import run_script_in_process

FIXTURES_DIR = Path(__file__).resolve().parent / "test_helpers"

# Bound the way `from ttl import signpost` / `m = ttl.math` would bind them, for
# the sample bodies below.
signpost = ttl_alias.signpost
math_namespace = ttl_alias.math


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


def _multikernel_body_via_alias(a: Any, out: Any) -> None:
    """Hand-written kernels reached through an aliased module ("import ttl as T")."""
    dfb = ttl_alias.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

    @ttl_alias.datamovement()
    def reader() -> None:
        with dfb.reserve() as blk:
            ttl_alias.copy(a[0:1, 0:1], blk).wait()

    @ttl_alias.compute()
    def comp() -> None:
        pass

    @ttl_alias.datamovement()
    def writer() -> None:
        with dfb.wait() as blk:
            ttl_alias.copy(blk, out[0:1, 0:1]).wait()


def _multikernel_body_via_direct_import(a: Any, out: Any) -> None:
    """Hand-written kernels reached directly ("from ttl import compute")."""
    dfb = ttl_alias.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

    @datamovement()
    def reader() -> None:
        with dfb.reserve() as blk:
            ttl_alias.copy(a[0:1, 0:1], blk).wait()

    @compute()
    def comp() -> None:
        pass

    @datamovement()
    def writer() -> None:
        with dfb.wait() as blk:
            ttl_alias.copy(blk, out[0:1, 0:1]).wait()


def _multikernel_body_mixed_spelling(a: Any, out: Any) -> None:
    """Aliased kernel decorators, with the calls inside spelled ``ttl.<op>``.

    Aliasing only the decorators is the easiest spelling to get wrong, since the
    body reads exactly like a supported one.
    """
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

    @ttl_alias.datamovement()
    def reader() -> None:
        with dfb.reserve() as blk:
            ttl.copy(a[0:1, 0:1], blk).wait()

    @ttl_alias.compute()
    def comp() -> None:
        pass

    @ttl_alias.datamovement()
    def writer() -> None:
        with dfb.wait() as blk:
            ttl.copy(blk, out[0:1, 0:1]).wait()


@pytest.mark.parametrize(
    "body",
    [
        _multikernel_body_via_alias,
        _multikernel_body_via_direct_import,
        _multikernel_body_mixed_spelling,
    ],
    ids=["aliased_module", "direct_import", "aliased_decorators_only"],
)
def test_hand_written_kernels_are_not_classified_as_unified(body: Any) -> None:
    """Kernel decorators are recognized by object, not by source spelling.

    Matching only ``ttl.compute`` / ``ttl.datamovement`` text classifies these
    bodies as unified, which splits them and silently returns a wrong answer.
    """
    assert not is_unified_body(body)


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
@pytest.mark.parametrize(
    "fixture",
    ["aliased_multikernel.py", "multikernel_aliased_decorators.py"],
    ids=["aliased_calls", "aliased_decorators_only"],
)
def test_aliased_multikernel_operation_runs_correctly(
    fixture: str, scheduler: str
) -> None:
    """End-to-end multi-kernel operations written with aliased imports.

    Each fixture checks its own result against torch, so a body that gets split
    by mistake fails here rather than quietly producing zeros.
    """
    code, out = run_script_in_process(FIXTURES_DIR / fixture, scheduler)
    assert code == 0, f"aliased multi-kernel operation failed:\n{out}"


def _unified_body_via_alias(a: Any, out: Any) -> None:
    """A unified body whose calls go through an alias ("import ttl as T")."""
    dfb = ttl_alias.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    blk = dfb.reserve()
    ttl_alias.copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = dfb.wait()
    ttl_alias.copy(out_blk, out[0:1, 0:1]).wait()
    out_blk.pop()


def _unified_body_via_aliased_factory(a: Any, out: Any) -> None:
    """Only the DFB factory is aliased; every copy is spelled ``ttl.copy``.

    The factory call is what populates ``local_dfb_names``, so overlooking it
    leaves the splitter without DFB names and unanchors the whole body, even
    though the rest of the body reads as supported.
    """
    dfb = ttl_alias.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    blk = dfb.reserve()
    ttl.copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = dfb.wait()
    ttl.copy(out_blk, out[0:1, 0:1]).wait()
    out_blk.pop()


def _unified_body_via_aliased_namespace(a: Any, out: Any) -> None:
    """A namespaced compute call (``ttl.math.*``) reached through the alias."""
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    blk = dfb.reserve()
    ttl.copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = dfb.reserve()
    out_blk.store(ttl_alias.math.reduce_sum(dfb.wait(), dims=[0], shape=(1, 1)))
    ttl.copy(out_blk, out[0:1, 0:1]).wait()


def _unified_body_via_renamed_op(a: Any, out: Any) -> None:
    """A bare call to an op bound elsewhere ("from ttl import copy")."""
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    blk = dfb.reserve()
    renamed_copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = dfb.wait()
    renamed_copy(out_blk, out[0:1, 0:1]).wait()
    out_blk.pop()


def _unified_body_with_bare_factory(a: Any, out: Any) -> None:
    """A bare factory call, which the factory rules recognize by name."""
    dfb = make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    blk = dfb.reserve()
    ttl.copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = dfb.wait()
    ttl.copy(out_blk, out[0:1, 0:1]).wait()
    out_blk.pop()


def _unified_body_with_bare_control_op(a: Any, out: Any) -> None:
    """A bare control-op call, which pins no thread however it is spelled."""
    signpost("region")
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    blk = dfb.reserve()
    ttl.copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = dfb.wait()
    ttl.copy(out_blk, out[0:1, 0:1]).wait()
    out_blk.pop()


def _unified_body_via_namespace_binding(a: Any, out: Any) -> None:
    """A call through a name bound to one of the API's namespaces, not to it."""
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    blk = dfb.reserve()
    ttl.copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = dfb.reserve()
    out_blk.store(math_namespace.reduce_sum(dfb.wait(), dims=[0], shape=(1, 1)))
    ttl.copy(out_blk, out[0:1, 0:1]).wait()


@pytest.fixture
def shadowed_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shadow ``sys.modules['ttl']`` as the simulator entry point does.

    The guard compares call receivers against whatever object ``ttl`` currently
    names, so it has nothing to compare against unless the simulator namespace is
    installed there.
    """
    monkeypatch.setitem(sys.modules, "ttl", ttl_alias)  # type: ignore[arg-type]


def _rejection_reason(body: Any) -> Optional[str]:
    """The guard's complaint about ``body``, or None if it accepts it."""
    fn_def = _parse_operation_funcdef(body)
    try:
        _reject_aliased_api(fn_def, _symbol_table(body))
    except ValueError as error:
        return str(error)
    return None


@pytest.mark.parametrize(
    "body, spelling",
    [
        (_unified_body_via_alias, "ttl_alias.copy(...)"),
        (
            _unified_body_via_aliased_factory,
            "ttl_alias.make_dataflow_buffer_like(...)",
        ),
        (_unified_body_via_aliased_namespace, "ttl_alias.math.reduce_sum(...)"),
        (_unified_body_via_renamed_op, "renamed_copy(...)"),
    ],
    ids=["aliased_module", "aliased_factory", "aliased_namespace", "renamed_op"],
)
def test_unified_body_reaching_api_under_another_name_is_rejected(
    body: Any, spelling: str, shadowed_ttl: None
) -> None:
    """Each way of reaching the API under another name is turned away by name.

    Thread assignment resolves calls by the receiver name ``ttl``, so none of
    these calls anchor anything: the statement is replicated onto all three
    threads, or the split fails claiming a block has no uses. The error has to
    quote the offending spelling, since neither downstream failure points at it.
    """
    reason = _rejection_reason(body)
    assert reason is not None, f"{body.__name__} was accepted"
    assert spelling in reason, f"error does not name the spelling to fix:\n{reason}"
    assert "must reference it as 'ttl'" in reason, f"unexpected reason:\n{reason}"


@pytest.mark.parametrize(
    "body",
    [
        _unified_body,
        _unified_body_with_bare_factory,
        _unified_body_with_bare_control_op,
    ],
    ids=["literal_ttl", "bare_factory", "bare_control_op"],
)
def test_unified_body_spellings_the_splitter_resolves_are_accepted(
    body: Any, shadowed_ttl: None
) -> None:
    """The guard stays out of the way of spellings that do resolve.

    Bare factory calls are recognized by name, and control ops pin no thread
    however they are spelled, so rejecting either would only block valid code.
    """
    assert _rejection_reason(body) is None, f"{body.__name__} was rejected"


@pytest.mark.xfail(
    reason="the receiver resolves to a namespace of the API, not to the API "
    "itself, so the guard does not recognize it (#779)",
    strict=True,
)
def test_unified_body_via_namespace_binding_is_rejected(shadowed_ttl: None) -> None:
    """A known hole in the guard, recorded so it is not rediscovered.

    ``m = ttl.math`` followed by ``m.reduce_sum(...)`` reaches a thread-pinning op
    under a name the splitter ignores, exactly like the rejected spellings, but
    the receiver is a namespace object rather than the API. It mis-splits.
    """
    assert _rejection_reason(_unified_body_via_namespace_binding) is not None


@pytest.mark.parametrize(
    "fixture",
    ["unified_aliased_import.py", "unified_aliased_calls.py"],
    ids=["alias_only_import", "ttl_also_bound"],
)
def test_unified_body_with_aliased_api_is_rejected(fixture: str) -> None:
    """End to end, an aliased unified body fails at decoration with a reason.

    The second case also binds ``ttl``, so a guard that only inspects the
    module's imports lets it through; it then mis-splits onto the compute thread
    and fails at runtime with a dataflow error that never mentions the alias.
    """
    code, out = run_script_in_process(FIXTURES_DIR / fixture)
    assert code != 0, f"aliased unified body unexpectedly ran:\n{out}"
    assert "must reference it as 'ttl'" in out, f"unexpected failure mode:\n{out}"


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_unified_operation_bare_copy_injects_wait(scheduler: str) -> None:
    """Bare ``ttl.copy(src, dst)`` in a unified body gets its wait injected.

    Injection points are keyed on (code object, absolute lineno), so a body whose
    line numbers do not match its file gets no injection at all and fails with a
    dataflow-state error ("expected TX_WAIT, attempted PUSH") instead.
    """
    code, out = run_script_in_process(FIXTURES_DIR / "unified_bare_copy.py", scheduler)
    assert code == 0, f"unified operation with bare copies failed:\n{out}"
