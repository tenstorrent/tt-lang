# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the per-process simulator context.

The simulator owns a single ``SimulatorContext`` per process at any time.
Greenlets are cooperative (one running at a time) and pytest workers run in
separate processes, so there is no concurrent access to the context.  These
tests cover:

* lazy creation on first ``get_context()`` call
* ``set_context()`` swapping the current context
* ``reset_context()`` installing a fresh context with defaults
* greenlets sharing the same context (since there is only one)
"""

import sys

import pytest
from greenlet import greenlet

from sim.context import (
    cleanup_run_context,
    get_context,
    set_context,
    reset_context,
)
from sim.context_types import (
    SimulatorContext,
    SimulatorConfig,
    CopySystemState,
    WarningState,
)


class TestContextCreation:
    """Test context creation and retrieval."""

    def test_get_context_creates_on_first_access(self):
        """``get_context()`` lazily creates a fresh context when none exists."""
        # Clear any existing context by installing an empty one and then
        # taking it away via the internal global.  We use the public API to
        # avoid coupling to the storage detail.
        reset_context()  # installs a fresh one
        ctx1 = get_context()
        assert isinstance(ctx1, SimulatorContext)
        assert isinstance(ctx1.config, SimulatorConfig)
        assert isinstance(ctx1.copy_state, CopySystemState)
        assert isinstance(ctx1.warnings, WarningState)

        # Second access returns the same context (no auto-recreation).
        ctx2 = get_context()
        assert ctx1 is ctx2

    def test_context_has_default_values(self):
        """A freshly reset context has the documented defaults."""
        reset_context()
        ctx = get_context()

        assert ctx.config.max_dfbs == 32
        assert ctx.config.scheduler_algorithm == "fair"
        assert len(ctx.copy_state.pipe_buffer) == 0
        assert ctx.scheduler is None

    def test_set_context_replaces_current(self):
        """``set_context()`` replaces the current context object."""
        reset_context()
        original = get_context()

        new_ctx = SimulatorContext()
        new_ctx.config.max_dfbs = 64
        set_context(new_ctx)

        retrieved = get_context()
        assert retrieved is new_ctx
        assert retrieved is not original
        assert retrieved.config.max_dfbs == 64

    def test_reset_context_creates_fresh_state(self):
        """``reset_context()`` discards the existing context and starts over."""
        reset_context()
        ctx = get_context()

        ctx.config.max_dfbs = 100
        ctx.config.scheduler_algorithm = "greedy"
        ctx.copy_state.pipe_buffer["test"] = "value"

        reset_context()
        new_ctx = get_context()

        assert new_ctx is not ctx
        assert new_ctx.config.max_dfbs == 32
        assert new_ctx.config.scheduler_algorithm == "fair"
        assert len(new_ctx.copy_state.pipe_buffer) == 0

    def test_reset_context_without_sys_monitoring(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Python 3.10 cleanup has no sys.monitoring tool slot to free."""
        monkeypatch.delattr(sys, "monitoring", raising=False)

        reset_context()

        assert isinstance(get_context(), SimulatorContext)

    def test_cleanup_run_context_without_sys_monitoring(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Run cleanup is import-safe on Python 3.10."""
        monkeypatch.delattr(sys, "monitoring", raising=False)

        cleanup_run_context()

        assert get_context().active_hooks == {}


class TestGreenletSharing:
    """All greenlets within a run share the same context."""

    def test_child_greenlet_sees_same_context(self):
        """A child greenlet sees the same context as its parent."""
        reset_context()
        parent_ctx = get_context()
        parent_ctx.config.max_dfbs = 128

        child_ctx_ref = None

        def child_function():
            nonlocal child_ctx_ref
            child_ctx_ref = get_context()

        child = greenlet(child_function)
        child.switch()

        assert child_ctx_ref is parent_ctx
        assert child_ctx_ref.config.max_dfbs == 128

    def test_deeply_nested_greenlets_share_context(self):
        """Greenlets created from greenlets all see the same context."""
        reset_context()
        root_ctx = get_context()
        root_ctx.config.max_dfbs = 256

        contexts_seen = []

        def level3():
            contexts_seen.append(("level3", get_context()))

        def level2():
            contexts_seen.append(("level2", get_context()))
            child3 = greenlet(level3)
            child3.switch()

        def level1():
            contexts_seen.append(("level1", get_context()))
            child2 = greenlet(level2)
            child2.switch()

        child1 = greenlet(level1)
        child1.switch()

        assert len(contexts_seen) == 3
        for name, ctx in contexts_seen:
            assert ctx is root_ctx, f"{name} saw different context"
            assert ctx.config.max_dfbs == 256

    def test_child_can_modify_shared_context(self):
        """A child greenlet's writes to the context are visible to the parent."""
        reset_context()
        parent_ctx = get_context()
        parent_ctx.config.max_dfbs = 10

        def child_function():
            ctx = get_context()
            ctx.config.max_dfbs = 20
            ctx.config.scheduler_algorithm = "greedy"

        child = greenlet(child_function)
        child.switch()

        assert parent_ctx.config.max_dfbs == 20
        assert parent_ctx.config.scheduler_algorithm == "greedy"

    def test_child_set_context_swaps_for_everyone(self):
        """``set_context()`` from a child swaps the process-wide context.

        In the per-process model there is no greenlet-local isolation: any
        call to ``set_context()`` (or ``reset_context()``) replaces the
        single shared context, so the parent observes the change too.
        Test runs needing isolation should rely on the autouse
        ``reset_simulator_context`` fixture instead of trying to nest
        contexts inside greenlets.
        """
        reset_context()
        parent_ctx = get_context()
        parent_ctx.config.max_dfbs = 32

        def child_function():
            new_ctx = SimulatorContext()
            new_ctx.config.max_dfbs = 99
            set_context(new_ctx)

        child = greenlet(child_function)
        child.switch()

        # The parent now sees the new context the child installed.
        seen_from_parent = get_context()
        assert seen_from_parent is not parent_ctx
        assert seen_from_parent.config.max_dfbs == 99


class TestSequentialRuns:
    """``reset_context()`` is how successive runs get clean state."""

    def test_sequential_programs_with_reset(self):
        """Each reset_context() gives the next ``run'' fresh defaults."""
        results = []

        for program_id in range(3):
            reset_context()
            ctx = get_context()
            ctx.config.max_dfbs = 10 * (program_id + 1)
            ctx.config.scheduler_algorithm = "greedy" if program_id % 2 == 0 else "fair"

            results.append(
                {
                    "id": program_id,
                    "max_dfbs": ctx.config.max_dfbs,
                    "scheduler": ctx.config.scheduler_algorithm,
                }
            )

        assert results[0]["max_dfbs"] == 10
        assert results[1]["max_dfbs"] == 20
        assert results[2]["max_dfbs"] == 30
        assert results[0]["scheduler"] == "greedy"
        assert results[1]["scheduler"] == "fair"
        assert results[2]["scheduler"] == "greedy"

    def test_multiple_resets(self):
        """Multiple resets each produce a fresh context with defaults."""
        for i in range(5):
            reset_context()
            ctx = get_context()
            ctx.config.max_dfbs = i * 10

            reset_context()
            new_ctx = get_context()
            assert new_ctx is not ctx
            assert new_ctx.config.max_dfbs == 32

    def test_context_dataclass_independence(self):
        """Nested dataclasses are independent between successive contexts."""
        reset_context()
        ctx1 = get_context()
        ctx1.copy_state.pipe_buffer["key1"] = {"data": "value"}

        reset_context()
        ctx2 = get_context()

        assert "key1" not in ctx2.copy_state.pipe_buffer
        assert len(ctx2.copy_state.pipe_buffer) == 0


class TestWarningState:
    """Warning deduplication state lives in the context."""

    def test_warning_tracking(self):
        """Warnings are tracked per-context."""
        reset_context()
        ctx = get_context()

        ctx.warnings.broadcast_1d_warnings[("file.py", 10)] = {"node0", "node1"}
        ctx.warnings.block_print_warnings[("other.py", 20)] = {"node2"}

        assert len(ctx.warnings.broadcast_1d_warnings) == 1
        assert len(ctx.warnings.block_print_warnings) == 1
        assert "node0" in ctx.warnings.broadcast_1d_warnings[("file.py", 10)]

    def test_warning_state_is_reset_between_runs(self):
        """``reset_context()`` clears accumulated warning state."""
        reset_context()
        get_context().warnings.broadcast_1d_warnings[("file.py", 10)] = {"node0"}

        reset_context()
        assert len(get_context().warnings.broadcast_1d_warnings) == 0
