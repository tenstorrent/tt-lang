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
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

from sim import ttl as ttl_alias
from sim import unified_operation as sim_unified_operation
from sim.copy import copy as renamed_copy
from sim.decorators import compute, datamovement
from sim.dfb import make_dataflow_buffer_like
from sim.unified_operation import (  # type: ignore[reportPrivateUsage]
    _api_ops_by_thread,
    _clear_decorators,
    _is_kernel_decorator,
    _is_operation_decorator,
    _is_setup_stmt,
    _load_frontend_module,
    _local_dfb_names,
    _parse_operation_funcdef,
    _reject_aliased_api,
    _reject_unsupported_setup,
    _rules,
    build_multikernel_function,
    is_unified_body,
)
from test_helpers.sim_runner import run_script_in_process

FIXTURES_DIR = Path(__file__).resolve().parent / "test_helpers"

# Bound the way `from ttl import signpost` / `m = ttl.math` would bind them, for
# the sample bodies below.
signpost = ttl_alias.signpost
math_namespace = ttl_alias.math

# Bound the way `from ttl import compute as build_math` would bind them: kernel
# decorators under names no spelling rule recognizes.
build_math = compute
move_data = datamovement


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


def test_a_missing_frontend_module_names_the_places_it_looked() -> None:
    """The shared rules and the splitter are compiler files, loaded by path.

    They ship with the simulator wheel rather than being imported, so a packaging
    change that stops copying one of them breaks every unified operation at
    decoration time. Neither the file nor packaging appears in what a bare lookup
    failure would report, so the error names both places it searched.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _load_frontend_module("atom_not_shipped.py")

    reason = str(excinfo.value)
    assert "atom_not_shipped.py" in reason, f"error does not name the file:\n{reason}"
    frontend_dir = Path(_rules().__file__ or "").parent
    assert (
        str(frontend_dir) in reason
    ), f"error does not name the source tree:\n{reason}"
    assert (
        str(Path(sim_unified_operation.__file__ or "").parent) in reason
    ), f"error does not name the bundled location:\n{reason}"


def test_a_frontend_module_that_fails_to_execute_leaves_no_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module is registered before it is executed, so a failure must undo that.

    The registration exists so a dataclass in the module can resolve its own
    ``__module__`` while the class body runs. If execution then raises, an entry
    naming a half-initialized module would be left behind, and the next lookup
    would take it as a loaded module and read attributes that were never bound.
    """
    module_name = "ttl_sim_atom_raises_on_exec"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        sim_unified_operation, "__file__", str(tmp_path / "unified_operation.py")
    )
    (tmp_path / "atom_raises_on_exec.py").write_text(
        'raise RuntimeError("frontend module is broken")\n'
    )

    with pytest.raises(RuntimeError, match="frontend module is broken"):
        _load_frontend_module("atom_raises_on_exec.py")

    assert module_name not in sys.modules, (
        "a module that failed to execute is still registered, so the next "
        "lookup would accept it as loaded"
    )


def _lambda_body() -> Callable[[Any, Any], Any]:
    """A body Python can call but the simulator cannot read back as a ``def``."""
    body: Callable[[Any, Any], Any] = lambda a, out: ttl.copy(a, out).wait()
    return body


@pytest.mark.parametrize(
    "body",
    [_lambda_body(), len],
    ids=["lambda", "builtin"],
)
def test_a_body_that_does_not_read_back_as_a_def_stays_on_the_legacy_path(
    body: Any,
) -> None:
    """Splitting needs the body's statements, so an unreadable body is left alone.

    ``inspect.getsourcelines`` gives a lambda's enclosing statement and a builtin
    nothing at all, neither of which parses as a ``def``. Calling such a body
    unified would hand the splitter no statements to assign; the legacy path runs
    it as written instead.
    """
    assert not is_unified_body(body)


def test_parsing_a_body_that_is_not_a_def_names_the_function() -> None:
    """The parse failure is reported against the operation, not as a stray None.

    Every caller here treats a parsed body as a given, so the one that cannot be
    parsed has to say which function it was reading.
    """
    with pytest.raises(ValueError) as excinfo:
        _parse_operation_funcdef(_lambda_body())
    assert "<lambda>" in str(excinfo.value)


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


def _multikernel_body_via_renamed_decorators(a: Any, out: Any) -> None:
    """Kernel decorators bound under other names ("from ttl import compute as ...").

    The spelling rule reads the attribute name and finds ``move_data`` /
    ``build_math``, neither of which names a kernel, so nothing but the decorator
    object identifies these as kernels.
    """
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

    @move_data()
    def reader() -> None:
        with dfb.reserve() as blk:
            ttl.copy(a[0:1, 0:1], blk).wait()

    @build_math()
    def comp() -> None:
        pass

    @move_data()
    def writer() -> None:
        with dfb.wait() as blk:
            ttl.copy(blk, out[0:1, 0:1]).wait()


@pytest.mark.parametrize(
    "body",
    [
        _multikernel_body_via_alias,
        _multikernel_body_via_direct_import,
        _multikernel_body_mixed_spelling,
        _multikernel_body_via_renamed_decorators,
    ],
    ids=[
        "aliased_module",
        "direct_import",
        "aliased_decorators_only",
        "renamed_decorators",
    ],
)
def test_hand_written_kernels_are_not_classified_as_unified(body: Any) -> None:
    """Kernel decorators are recognized by object, not by source spelling.

    Matching only ``ttl.compute`` / ``ttl.datamovement`` text classifies these
    bodies as unified, which splits them and silently returns a wrong answer.

    The first three are recognized by spelling as well, since the shared spelling
    rule reads the attribute name and ignores the receiver. The renamed case is
    the one that needs the object: its decorators spell names no rule knows, and
    it is the reason resolution is done at all.
    """
    assert not is_unified_body(body)


def _users_own_decorator(fn: Any) -> Any:
    """A decorator of the user's own that a body might bind as ``compute``."""
    return fn


def _foreign_kernel_decorator() -> Any:
    """A ``compute`` decorator from somewhere other than the running simulator."""

    def compute() -> Any:
        raise AssertionError("stands in for another build's decorator; never called")

    return compute


@pytest.mark.parametrize(
    "spelling, symbols, recognized",
    [
        ("@ttl.compute()", {}, True),
        ("@T.compute()", {}, True),
        ("@ttl.compute()", {"ttl": object()}, True),
        (
            "@other.compute()",
            {"other": SimpleNamespace(compute=_foreign_kernel_decorator())},
            True,
        ),
        ("@registry['compute']()", {"registry": {"compute": compute}}, False),
        ("@build_math()", {"build_math": compute}, True),
        ("@compute()", {"compute": _users_own_decorator}, False),
    ],
    ids=[
        "unknown_receiver",
        "unknown_alias",
        "receiver_without_the_attribute",
        "another_build_of_the_api",
        "spells_no_name",
        "renamed_decorator",
        "the_name_bound_to_something_else",
    ],
)
def test_a_decorator_that_does_not_resolve_falls_back_to_its_spelling(
    spelling: str, symbols: dict[str, Any], recognized: bool
) -> None:
    """Where the decorator object is out of reach, the name it spells decides.

    Resolution needs the body's own scope and the simulator's own decorator
    objects, and a body can be short of either: bindings this cannot follow, or a
    decorator from another build of the API. Since calling a multi-kernel body
    unified splits it and returns a wrong answer, every fallback errs toward
    "multi-kernel", which is also all the compiler does
    (``atom_rules.defines_kernels_by_spelling``).

    The last two cases are what resolution changes, in both directions: a renamed
    decorator is recognized where no spelling would, and a name bound to something
    else is refused where the spelling would have accepted it. The second is a
    disagreement with the compiler, which reads that body as kernels.

    A decorator that spells no name at all is beyond either rule: neither can find
    a kernel in ``@registry['compute']()``, so both frontends read such a body as
    unified.
    """
    fn_def = _decorated_funcdef(spelling)
    assert _is_kernel_decorator(fn_def.decorator_list[0], symbols) == recognized


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

    Construction is recognized by name without its receiver, so this body is
    hoisted, registered and split like any other: turning it away would refuse a
    program that works.
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
        _reject_aliased_api(fn_def, _rules().function_scope(body))
    except ValueError as error:
        return str(error)
    return None


@pytest.mark.parametrize(
    "body, spelling",
    [
        (_unified_body_via_alias, "ttl_alias.copy(...)"),
        (_unified_body_via_aliased_namespace, "ttl_alias.math.reduce_sum(...)"),
        (_unified_body_via_renamed_op, "renamed_copy(...)"),
        (
            _unified_body_via_namespace_binding,
            "math_namespace.reduce_sum(...)",
        ),
    ],
    ids=["aliased_module", "aliased_namespace", "renamed_op", "namespace_binding"],
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
        _unified_body_via_aliased_factory,
    ],
    ids=["literal_ttl", "bare_factory", "bare_control_op", "aliased_factory"],
)
def test_unified_body_spellings_the_splitter_resolves_are_accepted(
    body: Any, shadowed_ttl: None
) -> None:
    """The guard stays out of the way of spellings that do resolve.

    Construction is recognized by name without its receiver, and control ops pin
    no thread however they are spelled, so rejecting either would only block valid
    code. The aliased factory is the case both rules meet: it is construction, and
    construction is a control op.
    """
    assert _rejection_reason(body) is None, f"{body.__name__} was rejected"


def test_the_alias_guard_stands_down_when_the_api_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no ``ttl`` in ``sys.modules`` there is nothing to compare receivers to.

    An alias is recognized as a name resolving to the same object ``ttl`` names, so
    the guard needs the API the operation will run against. Only the simulator
    entry point installs it, and it does so before any operation is decorated;
    reached without it, accepting is the honest answer for a comparison that cannot
    be made, rather than turning away every receiver that is not spelled ``ttl``.
    """
    monkeypatch.delitem(sys.modules, "ttl", raising=False)
    assert _rejection_reason(_unified_body_via_alias) is None


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


def test_the_guard_reads_the_ops_and_only_the_ops_out_of_the_registry() -> None:
    """The two op maps hold what the splitter classifies, and nothing else.

    They decide which spellings are refused, so a stray entry refuses valid code:
    a namespace re-exports the helpers and types its module imported
    (``ttl.math.Block``, ``ttl.math.get_context``), and taking every callable
    would enter those as ops and turn away a body that merely calls one.
    """
    pinning, control = _api_ops_by_thread(ttl_alias)

    assert pinning[id(ttl_alias.copy)] == "copy"
    assert pinning[id(ttl_alias.math.reduce_sum)] == "math.reduce_sum"
    assert control[id(ttl_alias.make_dataflow_buffer_like)] == (
        "make_dataflow_buffer_like"
    )
    # Construction is a control op, which is what makes an aliased factory
    # acceptable; keeping it out of the pinning map is that decision.
    assert id(ttl_alias.make_dataflow_buffer_like) not in pinning
    for name in ("Block", "get_context"):
        helper: Any = getattr(ttl_alias.math, name)
        assert id(helper) not in pinning, f"{name} was entered as an op"


def test_unified_body_with_an_aliased_factory_runs_correctly() -> None:
    """End to end, an aliased factory is hoisted, shared and split correctly.

    The negative case above is what the guard is for; this is the line it must
    not cross. Construction carries no thread and is recognized without its
    receiver, so the only thing an alias changes here is the spelling -- and the
    run has to produce the copy, since a body that built three separate buffers
    would still decorate.
    """
    code, out = run_script_in_process(FIXTURES_DIR / "unified_aliased_factory.py")
    assert code == 0, f"aliased-factory unified body failed:\n{out}"


USE_LARGE_BUFFER = True  # compile-time switch for the sample bodies below


def _unified_body_dfb_under_if(a: Any, out: Any) -> None:
    """A DFB constructed on both arms of a compile-time condition."""
    if USE_LARGE_BUFFER:
        dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=4)
    else:
        dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    blk = dfb.reserve()
    ttl.copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = dfb.wait()
    ttl.copy(out_blk, out[0:1, 0:1]).wait()
    out_blk.pop()


def _unified_body_dfb_in_loop(a: Any, out: Any) -> None:
    """A DFB reconstructed on every iteration of a loop."""
    for i in range(2):
        dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        blk = dfb.reserve()
        ttl.copy(a[i : i + 1, 0:1], blk).wait()
        blk.push()
        out_blk = dfb.wait()
        ttl.copy(out_blk, out[i : i + 1, 0:1]).wait()
        out_blk.pop()


def _unified_body_dfb_tuple_target(a: Any, out: Any) -> None:
    """Two DFBs bound by a single tuple-target assignment."""
    in_dfb, out_dfb = (
        ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2),
        ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2),
    )
    in_blk = in_dfb.reserve()
    ttl.copy(a[0:1, 0:1], in_blk).wait()
    in_blk.push()
    out_blk = out_dfb.reserve()
    out_blk.store(in_dfb.wait())
    ttl.copy(out_blk, out[0:1, 0:1]).wait()


def _unified_body_dfb_unpacked(a: Any, out: Any) -> None:
    """One factory call unpacked into two names."""
    first, second = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    blk = first.reserve()
    ttl.copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = second.wait()
    ttl.copy(out_blk, out[0:1, 0:1]).wait()
    out_blk.pop()


def _unified_body_dfb_never_named(a: Any, out: Any) -> None:
    """A DFB used inline, so no name refers to it afterwards."""
    blk = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2).reserve()
    ttl.copy(a[0:1, 0:1], blk).wait()
    blk.push()


def _unified_body_dfb_in_callback(a: Any, out: Any) -> None:
    """A DFB constructed inside a pipe callback."""
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(0, 1))])

    def send(pipe: Any) -> None:
        dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        with dfb.reserve() as blk:
            ttl.copy(a[0:1, 0:1], blk).wait()
            ttl.copy(blk, pipe).wait()

    net.if_src(send)


def _unified_body_dfb_from_local_value(a: Any, out: Any) -> None:
    """A buffer whose depth is computed by the body."""
    depth = 2
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=depth)
    blk = dfb.reserve()
    ttl.copy(a[0:1, 0:1], blk).wait()
    blk.push()
    out_blk = dfb.wait()
    ttl.copy(out_blk, out[0:1, 0:1]).wait()
    out_blk.pop()


def _unified_body_pipe_list_from_local_grid(a: Any, out: Any) -> None:
    """A PipeNet sized by a grid query the body makes."""
    grid_x, _ = ttl.grid_size()
    net = ttl.PipeNet([ttl.Pipe(src=(x, 0), dst=(x, 1)) for x in range(grid_x)])
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

    def receive(pipe: Any) -> None:
        with dfb.reserve() as blk:
            ttl.copy(pipe, blk).wait()
            ttl.copy(blk, out[0:1, 0:1]).wait()

    net.if_dst(receive)


def _unified_body_with_inline_pipe_list(a: Any, out: Any) -> None:
    """A PipeNet built from a comprehension in the call, as the spec writes it."""
    net = ttl.PipeNet([ttl.Pipe(src=(x, 0), dst=(x, 1)) for x in range(2)])
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

    def receive(pipe: Any) -> None:
        with dfb.reserve() as blk:
            ttl.copy(pipe, blk).wait()
            ttl.copy(blk, out[0:1, 0:1]).wait()

    net.if_dst(receive)


def _unified_body_with_named_pipe_list(a: Any, out: Any) -> None:
    """A PipeNet built from a separately named pipe list.

    The list has to be hoisted along with the net that consumes it: hoisting only
    the net leaves it evaluated in a scope where the list does not exist.
    """
    pipes = [ttl.Pipe(src=(0, 0), dst=(0, 1))]
    net = ttl.PipeNet(pipes)
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

    def receive(pipe: Any) -> None:
        with dfb.reserve() as blk:
            ttl.copy(pipe, blk).wait()
            ttl.copy(blk, out[0:1, 0:1]).wait()

    net.if_dst(receive)


# Every construction shape ruled on below, accepted and rejected alike.
_SETUP_CORPUS = (
    _unified_body,
    _unified_body_with_inline_pipe_list,
    _unified_body_with_named_pipe_list,
    _unified_body_dfb_under_if,
    _unified_body_dfb_in_loop,
    _unified_body_dfb_tuple_target,
    _unified_body_dfb_unpacked,
    _unified_body_dfb_never_named,
    _unified_body_dfb_in_callback,
    _unified_body_dfb_from_local_value,
    _unified_body_pipe_list_from_local_grid,
)


def _setup_rejection_reason(body: Any) -> Optional[str]:
    """The setup guard's complaint about ``body``, or None if it accepts it."""
    fn_def = _parse_operation_funcdef(body)
    try:
        _reject_unsupported_setup(fn_def)
    except ValueError as error:
        return str(error)
    return None


@pytest.mark.parametrize(
    "body",
    [
        _unified_body_dfb_under_if,
        _unified_body_dfb_in_loop,
        _unified_body_dfb_tuple_target,
        _unified_body_dfb_unpacked,
        _unified_body_dfb_never_named,
        _unified_body_dfb_in_callback,
    ],
    ids=[
        "under_if",
        "in_loop",
        "tuple_target",
        "unpacked",
        "never_named",
        "in_callback",
    ],
)
def test_unhoistable_buffer_construction_is_rejected(body: Any) -> None:
    """Construction the simulator cannot hoist is turned away at decoration.

    Only a top-level ``name = <factory>(...)`` is lifted into the shared scope, so
    any other form is duplicated into all three kernels and each reserves a buffer
    of its own. That surfaces as a dataflow-state error against a buffer the body
    never named, so the error here has to quote the construction to fix.
    """
    reason = _setup_rejection_reason(body)
    assert reason is not None, f"{body.__name__} was accepted"
    assert "make_dataflow_buffer_like" in reason, f"unexpected reason:\n{reason}"
    assert "must be a simple top-level assignment" in reason, f"reason:\n{reason}"

    quoted = re.search(r"on line (\d+)", reason)
    assert quoted is not None, f"error does not quote a line:\n{reason}"
    own_lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    assert "make_dataflow_buffer_like" in own_lines[int(quoted.group(1)) - 1]


@pytest.mark.parametrize(
    "body, value",
    [
        (_unified_body_dfb_from_local_value, "depth"),
        (_unified_body_pipe_list_from_local_grid, "grid_x"),
    ],
    ids=["buffer_depth", "pipe_net_width"],
)
def test_construction_reading_a_body_local_value_is_rejected(
    body: Any, value: str
) -> None:
    """A construction cannot read a value the body computes for itself.

    Hoisting moves the construction ahead of the kernels, while the value stays
    behind in them, so the hoisted statement runs against a name that does not
    exist there. The compiler applies the same rule, and rejecting here reports it
    against the declaration instead of as a ``NameError`` mid-run.
    """
    reason = _setup_rejection_reason(body)
    assert reason is not None, f"{body.__name__} was accepted"
    assert repr(value) in reason, f"error does not name the value:\n{reason}"
    assert "computes for itself" in reason, f"unexpected reason:\n{reason}"


@pytest.mark.parametrize(
    "body",
    [
        _unified_body,
        _unified_body_with_inline_pipe_list,
        _unified_body_with_named_pipe_list,
    ],
    ids=["dfb_only", "inline_pipe_list", "named_pipe_list"],
)
def test_hoistable_setup_is_accepted(body: Any) -> None:
    """Callbacks and pipe lists are fine; only the construction site is ruled on.

    A pipe list bound to its own name is hoisted like a factory call, because the
    ``PipeNet(...)`` reading it is hoisted too.
    """
    assert _setup_rejection_reason(body) is None, f"{body.__name__} was rejected"


def test_setup_guard_agrees_with_the_compilers_validator() -> None:
    """The guard turns away exactly the bodies the compiler's validator does.

    Both read their rules from ``atom_rules``, so this pins the wiring rather than
    the logic: a sim-side condition added on top of the shared rules would make a
    body unrunnable in simulation while it still compiles, or the reverse. Wordings
    are compared only for their verdict, since the compiler's message is pinned by
    its own test while this one quotes the line to fix.
    """
    validate = _rules().validate_resource_declarations
    for body in _SETUP_CORPUS:
        try:
            validate(_parse_operation_funcdef(body), body.__name__)
        except ValueError as error:
            compiler_verdict: Optional[str] = str(error)
        else:
            compiler_verdict = None
        simulator_verdict = _setup_rejection_reason(body)
        assert (simulator_verdict is None) == (compiler_verdict is None), (
            f"{body.__name__}: simulator said {simulator_verdict!r}, "
            f"compiler said {compiler_verdict!r}"
        )


def test_hoisting_and_buffer_registration_agree() -> None:
    """A construction is hoisted only when its name is registered as a buffer.

    These were separate conditions -- hoisting accepted any assignment while
    registration required a single ``Name`` target -- so an unpacked factory call
    was lifted out of the body while the splitter never learned the names were
    buffers, leaving the reserve/wait calls on them anchored to no thread.
    """
    supported = _parse_operation_funcdef(_unified_body)
    assert sum(_is_setup_stmt(s) for s in supported.body) == 1
    assert _local_dfb_names(supported) == {"dfb"}

    unpacked = _parse_operation_funcdef(_unified_body_dfb_unpacked)
    assert not any(_is_setup_stmt(s) for s in unpacked.body)
    assert _local_dfb_names(unpacked) == set()


def test_unified_body_with_nested_dfb_is_rejected() -> None:
    """End to end, a DFB built under an ``if`` fails at decoration with a reason.

    Unguarded, the run reaches the dataflow protocol and dies on a state error
    against a per-kernel buffer, which names neither the construction nor its
    line.
    """
    code, out = run_script_in_process(FIXTURES_DIR / "unified_nested_dfb.py")
    assert code != 0, f"unified body with a nested DFB unexpectedly ran:\n{out}"
    assert (
        "must be a simple top-level assignment" in out
    ), f"unexpected failure mode:\n{out}"


def test_unified_body_with_a_captured_dfb_is_rejected() -> None:
    """A buffer built outside the operation is refused, as the compiler refuses it.

    The specification constructs a dataflow buffer in the scope of the operation
    that uses it, while allowing a pipe net to be captured -- so this is the one
    of the two that has to be turned away.  Unguarded, the captured name is not
    among the buffers the body constructs, so its reserve and wait anchor no
    thread and the run dies on a dataflow state error inside a synthesized
    kernel, naming neither the buffer nor the line that built it.
    """
    code, out = run_script_in_process(FIXTURES_DIR / "unified_captured_dfb.py")
    assert code != 0, f"unified body with a captured DFB unexpectedly ran:\n{out}"
    assert (
        "constructed outside the operation" in out
    ), f"unexpected failure mode:\n{out}"
    # The compiler's own refusal is at decoration time too, so the message must
    # arrive before any kernel runs.
    assert "Cannot perform" not in out, f"reached the dataflow protocol:\n{out}"


def test_a_body_that_builds_its_own_buffer_is_not_read_as_capturing_one() -> None:
    """The guard turns away captures, not names that appear twice in a file.

    A body's own buffer is a local, and a module-level buffer spelled the same way
    is a different object it never reads. Refusing this program would refuse a
    normal one -- two operations in a file, each naming its buffer ``dfb`` -- with
    a message about construction the user did exactly as asked.
    """
    code, out = run_script_in_process(FIXTURES_DIR / "unified_shadowed_dfb_name.py")
    assert code == 0, f"a body with its own buffer was turned away:\n{out}"


def _passthrough(fn: Any) -> Any:
    """A user decorator that a sample body can be stacked with."""
    return fn


# What a body's own module would have bound, for the decorator samples below.
_DECORATOR_SYMBOLS: dict[str, Any] = {
    "ttl": ttl_alias,
    "T": ttl_alias,
    "operation": ttl_alias.operation,
    "profile": _passthrough,
    "trace": _passthrough,
}


def _decorated_funcdef(*decorators: str) -> ast.FunctionDef:
    """Parse a unified body carrying ``decorators``, listed top-down."""
    lines = "\n".join(decorators)
    source = textwrap.dedent(
        """
        def copy_through(src, dst):
            dfb = ttl.make_dataflow_buffer_like(src, shape=(1, 1), block_count=2)
            blk = dfb.reserve()
            ttl.copy(src[0:1, 0:1], blk).wait()
            blk.push()
        """
    ).strip()
    parsed = ast.parse(f"{lines}\n{source}").body[0]
    assert isinstance(parsed, ast.FunctionDef)
    return parsed


@pytest.mark.parametrize(
    "spelling",
    [
        "@ttl.operation(grid=(1, 1))",
        "@T.operation(grid=(1, 1))",
        "@operation(grid=(1, 1))",
    ],
    ids=["literal", "aliased_module", "direct_import"],
)
def test_decorators_above_the_operation_are_dropped_without_complaint(
    spelling: str,
) -> None:
    """Python applies those itself, so the synthesized function must not repeat them.

    They decorate whatever ``@ttl.operation`` returns, at the original definition
    site. Leaving them on the rewritten body would run them a second time, once
    per node. The operation decorator has to be recognized under any binding for
    this to tell them apart from the case below.
    """
    fn_def = _decorated_funcdef("@profile", "@trace", spelling)
    assert _is_operation_decorator(fn_def.decorator_list[-1], _DECORATOR_SYMBOLS)

    _clear_decorators(fn_def, _DECORATOR_SYMBOLS)
    assert fn_def.decorator_list == []


def test_a_decorator_below_the_operation_is_rejected() -> None:
    """Nothing would apply it, so it is refused instead of silently ignored.

    The body it was written to wrap does not survive the rewrite as a function --
    it becomes three kernels -- and the compiler drops it too, compiling the body
    with its decorator lines removed. Being told beats being ignored.
    """
    fn_def = _decorated_funcdef("@ttl.operation(grid=(1, 1))", "@profile")
    with pytest.raises(ValueError) as excinfo:
        _clear_decorators(fn_def, _DECORATOR_SYMBOLS)

    reason = str(excinfo.value)
    assert "'@profile'" in reason, f"error does not name the decorator:\n{reason}"
    assert "Move it above" in reason, f"error offers no fix:\n{reason}"


@pytest.mark.parametrize(
    "below, named, plural",
    [
        (("@trace",), ["'@trace' on line 3"], False),
        (("@trace(level=2)",), ["'@trace(level=2)' on line 3"], False),
        (
            ("@trace", "@trace(level=2)"),
            ["'@trace' on line 3", "'@trace(level=2)' on line 4"],
            True,
        ),
    ],
    ids=["bare", "called", "two"],
)
def test_decorators_on_both_sides_report_only_the_ones_below(
    below: tuple[str, ...], named: list[str], plural: bool
) -> None:
    """With decorators on both sides, only those below are a problem.

    The ones above are applied by Python and are meant to be dropped here, so
    naming them too would send the author looking for a fault in a line that is
    working. Each offender is quoted with its own line, since a stack of them is
    otherwise hard to tell apart.
    """
    fn_def = _decorated_funcdef("@profile", "@ttl.operation(grid=(1, 1))", *below)
    with pytest.raises(ValueError) as excinfo:
        _clear_decorators(fn_def, _DECORATOR_SYMBOLS)

    reason = str(excinfo.value)
    for spelling in named:
        assert spelling in reason, f"error does not quote {spelling}:\n{reason}"
    assert "'@profile'" not in reason, f"error blames a decorator above:\n{reason}"

    # Reads as a sentence either way: "the decorator ... sits", "... decorators ... sit".
    assert ("the decorators " in reason) == plural, f"wrong agreement:\n{reason}"
    assert ("the body they wrap" in reason) == plural, f"wrong agreement:\n{reason}"
    assert ("Move them above" in reason) == plural, f"wrong agreement:\n{reason}"


def test_the_operation_decorator_is_matched_by_name_when_unresolved() -> None:
    """Placement is still judged for a body whose bindings are out of reach.

    Which decorators are below ``@ttl.operation`` can only be read off relative to
    it, so failing to locate it turns the rejection below into a silent drop -- the
    outcome the check exists to prevent. Matching the name it spells keeps that
    working, and there is nothing else a decorator named ``operation`` on an
    operation body would be.
    """
    fn_def = _decorated_funcdef("@ttl.operation(grid=(1, 1))", "@profile")
    assert _is_operation_decorator(fn_def.decorator_list[0], {})

    with pytest.raises(ValueError) as excinfo:
        _clear_decorators(fn_def, {})
    assert "'@profile'" in str(excinfo.value)


def test_decorators_are_dropped_when_the_operation_cannot_be_identified() -> None:
    """An unrecognizable decorator list is cleared rather than rejected.

    Placement can only be judged relative to ``@ttl.operation``. With nothing
    identifiable to judge against, clearing matches the behavior that has always
    applied, and is better than refusing a body that may be perfectly valid.
    """
    fn_def = _decorated_funcdef("@profile")
    _clear_decorators(fn_def, _DECORATOR_SYMBOLS)
    assert fn_def.decorator_list == []


def test_unified_operation_under_a_stacked_decorator_runs_once() -> None:
    """End to end, a decorator above ``@ttl.operation`` still wraps the call once.

    This is why the rewrite clears the whole decorator list: the entry it drops is
    already applied by Python at the definition site, and re-applying it here
    would double every wrapper the user put on an operation.
    """
    code, out = run_script_in_process(FIXTURES_DIR / "unified_stacked_decorator.py")
    assert code == 0, f"unified operation under a stacked decorator failed:\n{out}"
    assert "wrapped calls: 1" in out, f"decorator did not run exactly once:\n{out}"
    assert "copied region matches: True" in out, f"operation copied nothing:\n{out}"


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_unified_operation_bare_copy_injects_wait(scheduler: str) -> None:
    """Bare ``ttl.copy(src, dst)`` in a unified body gets its wait injected.

    Injection points are keyed on (code object, absolute lineno), so a body whose
    line numbers do not match its file gets no injection at all and fails with a
    dataflow-state error ("expected TX_WAIT, attempted PUSH") instead.
    """
    code, out = run_script_in_process(FIXTURES_DIR / "unified_bare_copy.py", scheduler)
    assert code == 0, f"unified operation with bare copies failed:\n{out}"
