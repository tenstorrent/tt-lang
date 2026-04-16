# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for greenlet-local context management.

Verifies that simulator state is properly isolated per-greenlet
and that child greenlets inherit parent context correctly.
"""

import pytest
from greenlet import greenlet, getcurrent

from python.sim.context import (
    get_context,
    set_context,
    reset_context,
)
from python.sim.context_types import (
    SimulatorContext,
    SimulatorConfig,
    CopySystemState,
    WarningState,
)


class TestContextCreation:
    """Test context creation and retrieval."""

    def test_get_context_creates_on_first_access(self):
        """Test that get_context() auto-creates context on first access."""
        # Clear any existing context
        g = getcurrent()
        if hasattr(g, "_sim_context"):
            delattr(g, "_sim_context")

        # First access should create context
        ctx1 = get_context()
        assert isinstance(ctx1, SimulatorContext)
        assert isinstance(ctx1.config, SimulatorConfig)
        assert isinstance(ctx1.copy_state, CopySystemState)
        assert isinstance(ctx1.warnings, WarningState)

        # Second access should return same context
        ctx2 = get_context()
        assert ctx1 is ctx2

    def test_context_has_default_values(self):
        """Test that new context has sensible defaults."""
        reset_context()
        ctx = get_context()

        assert ctx.config.max_dfbs == 32
        assert ctx.config.scheduler_algorithm == "fair"
        assert len(ctx.copy_state.pipe_buffer) == 0
        assert ctx.scheduler is None

    def test_set_context_replaces_current(self):
        """Test that set_context() replaces the current context."""
        reset_context()
        original = get_context()

        # Create and set a new context
        new_ctx = SimulatorContext()
        new_ctx.config.max_dfbs = 64
        set_context(new_ctx)

        # Verify it replaced the original
        retrieved = get_context()
        assert retrieved is new_ctx
        assert retrieved is not original
        assert retrieved.config.max_dfbs == 64

    def test_reset_context_creates_fresh_state(self):
        """Test that reset_context() creates a fresh context."""
        reset_context()
        ctx = get_context()

        # Modify the context
        ctx.config.max_dfbs = 100
        ctx.config.scheduler_algorithm = "greedy"
        ctx.copy_state.pipe_buffer["test"] = "value"

        # Reset should create new context with defaults
        reset_context()
        new_ctx = get_context()

        assert new_ctx is not ctx
        assert new_ctx.config.max_dfbs == 32
        assert new_ctx.config.scheduler_algorithm == "fair"
        assert len(new_ctx.copy_state.pipe_buffer) == 0


class TestGreenletInheritance:
    """Test that child greenlets inherit parent context."""

    def test_child_greenlet_inherits_parent_context(self):
        """Test that child greenlet can access parent's context."""
        reset_context()
        parent_ctx = get_context()
        parent_ctx.config.max_dfbs = 128

        child_ctx_ref = None

        def child_function():
            nonlocal child_ctx_ref
            child_ctx_ref = get_context()

        # Create child greenlet
        child = greenlet(child_function)
        child.switch()

        # Child should have accessed parent's context
        assert child_ctx_ref is parent_ctx
        assert child_ctx_ref.config.max_dfbs == 128

    def test_nested_greenlets_walk_up_to_root(self):
        """Test that deeply nested greenlets can find root context."""
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

        # All levels should see the same root context
        assert len(contexts_seen) == 3
        for name, ctx in contexts_seen:
            assert ctx is root_ctx, f"{name} saw different context"
            assert ctx.config.max_dfbs == 256

    def test_child_can_modify_shared_context(self):
        """Test that child modifications affect parent's context."""
        reset_context()
        parent_ctx = get_context()
        parent_ctx.config.max_dfbs = 10

        def child_function():
            ctx = get_context()
            ctx.config.max_dfbs = 20
            ctx.config.scheduler_algorithm = "greedy"

        child = greenlet(child_function)
        child.switch()

        # Parent sees child's modifications (shared context)
        assert parent_ctx.config.max_dfbs == 20
        assert parent_ctx.config.scheduler_algorithm == "greedy"

    def test_child_with_own_context_is_isolated(self):
        """Test that child with its own context doesn't affect parent."""
        reset_context()
        parent_ctx = get_context()
        parent_ctx.config.max_dfbs = 32

        child_ctx_ref = None

        def child_function():
            nonlocal child_ctx_ref
            # Child creates its own context
            child_ctx = SimulatorContext()
            child_ctx.config.max_dfbs = 99
            set_context(child_ctx)
            child_ctx_ref = get_context()

        child = greenlet(child_function)
        child.switch()

        # Child should have different context
        assert child_ctx_ref is not parent_ctx
        assert child_ctx_ref.config.max_dfbs == 99
        # Parent context unchanged
        assert parent_ctx.config.max_dfbs == 32


class TestStateIsolation:
    """Test that state is properly isolated between execution contexts."""

    def test_config_isolation_between_resets(self):
        """Test that config is isolated per independent context."""
        reset_context()
        ctx1 = get_context()
        ctx1.config.max_dfbs = 16
        ctx1.config.scheduler_algorithm = "greedy"

        other_ctx = None

        def other_greenlet():
            nonlocal other_ctx
            reset_context()  # Create fresh context
            other_ctx = get_context()
            other_ctx.config.max_dfbs = 64
            other_ctx.config.scheduler_algorithm = "fair"

        g = greenlet(other_greenlet)
        g.switch()

        # Original context unchanged
        assert ctx1.config.max_dfbs == 16
        assert ctx1.config.scheduler_algorithm == "greedy"

        # Other context has its own config
        assert other_ctx.config.max_dfbs == 64
        assert other_ctx.config.scheduler_algorithm == "fair"

    def test_copy_buffer_isolation(self):
        """Test that pipe buffers are isolated per context."""
        reset_context()
        ctx1 = get_context()
        ctx1.copy_state.pipe_buffer["pipe1"] = {"data": "value1"}

        other_ctx = None

        def other_greenlet():
            nonlocal other_ctx
            reset_context()
            other_ctx = get_context()
            other_ctx.copy_state.pipe_buffer["pipe2"] = {"data": "value2"}

        g = greenlet(other_greenlet)
        g.switch()

        # Contexts have different buffers
        assert "pipe1" in ctx1.copy_state.pipe_buffer
        assert "pipe2" not in ctx1.copy_state.pipe_buffer
        assert "pipe2" in other_ctx.copy_state.pipe_buffer
        assert "pipe1" not in other_ctx.copy_state.pipe_buffer

    def test_trace_events_isolation(self):
        """Test that trace_events are isolated per context."""
        from python.sim.context_types import TraceEvent

        reset_context()
        ctx1 = get_context()
        ctx1.trace_events.append(TraceEvent(event="test", tick=0, kernel=None))

        other_ctx = None

        def other_greenlet():
            nonlocal other_ctx
            reset_context()
            other_ctx = get_context()
            other_ctx.trace_events.append(
                TraceEvent(event="other", tick=1, kernel=None)
            )

        g = greenlet(other_greenlet)
        g.switch()

        assert len(ctx1.trace_events) == 1
        assert ctx1.trace_events[0].event == "test"
        assert len(other_ctx.trace_events) == 1
        assert other_ctx.trace_events[0].event == "other"


class TestConcurrentExecution:
    """Test behavior with multiple concurrent greenlets."""

    def test_parallel_greenlets_with_shared_context(self):
        """Test multiple children sharing parent context."""
        reset_context()
        root_ctx = get_context()
        root_ctx.config.max_dfbs = 0

        results = []

        def worker(worker_id):
            ctx = get_context()
            # Increment counter (simulates accumulating shared state)
            for _ in range(3):
                ctx.config.max_dfbs += 1
            results.append((worker_id, ctx.config.max_dfbs))

        # Run three workers
        for i in range(3):
            g = greenlet(lambda wid=i: worker(wid))
            g.switch()

        # Counter was incremented 9 times total (3 workers x 3 increments)
        assert root_ctx.config.max_dfbs == 9
        assert len(results) == 3

    def test_sequential_programs_with_reset(self):
        """Test running multiple programs sequentially with reset between."""
        results = []

        for program_id in range(3):
            reset_context()
            ctx = get_context()
            ctx.config.max_dfbs = 10 * (program_id + 1)
            ctx.config.scheduler_algorithm = "greedy" if program_id % 2 == 0 else "fair"

            def program():
                ctx = get_context()
                results.append(
                    {
                        "id": program_id,
                        "max_dfbs": ctx.config.max_dfbs,
                        "scheduler": ctx.config.scheduler_algorithm,
                    }
                )

            g = greenlet(program)
            g.switch()

        # Each program saw its own isolated context
        assert results[0]["max_dfbs"] == 10
        assert results[1]["max_dfbs"] == 20
        assert results[2]["max_dfbs"] == 30
        assert results[0]["scheduler"] == "greedy"
        assert results[1]["scheduler"] == "fair"
        assert results[2]["scheduler"] == "greedy"


class TestWarningState:
    """Test warning deduplication state management."""

    def test_warning_tracking(self):
        """Test that warnings are tracked per-context."""
        reset_context()
        ctx = get_context()

        # Add warning locations
        ctx.warnings.broadcast_1d_warnings[("file.py", 10)] = {"core0", "core1"}
        ctx.warnings.block_print_warnings[("other.py", 20)] = {"core2"}

        assert len(ctx.warnings.broadcast_1d_warnings) == 1
        assert len(ctx.warnings.block_print_warnings) == 1
        assert "core0" in ctx.warnings.broadcast_1d_warnings[("file.py", 10)]

    def test_warning_isolation_between_contexts(self):
        """Test that warnings don't leak between contexts."""
        reset_context()
        ctx1 = get_context()
        ctx1.warnings.broadcast_1d_warnings[("file.py", 10)] = {"core0"}

        other_ctx = None

        def other_greenlet():
            nonlocal other_ctx
            reset_context()
            other_ctx = get_context()
            other_ctx.warnings.broadcast_1d_warnings[("file.py", 20)] = {"core1"}

        g = greenlet(other_greenlet)
        g.switch()

        # Warnings are isolated
        assert ("file.py", 10) in ctx1.warnings.broadcast_1d_warnings
        assert ("file.py", 20) not in ctx1.warnings.broadcast_1d_warnings
        assert ("file.py", 20) in other_ctx.warnings.broadcast_1d_warnings
        assert ("file.py", 10) not in other_ctx.warnings.broadcast_1d_warnings


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_context_without_parent_greenlet(self):
        """Test getting context when there's no parent (main greenlet)."""
        reset_context()
        ctx = get_context()

        assert isinstance(ctx, SimulatorContext)
        assert ctx.config.max_dfbs == 32

    def test_multiple_resets(self):
        """Test that multiple resets work correctly."""
        for i in range(5):
            reset_context()
            ctx = get_context()
            ctx.config.max_dfbs = i * 10

            reset_context()
            new_ctx = get_context()
            assert new_ctx is not ctx
            assert new_ctx.config.max_dfbs == 32

    def test_context_dataclass_independence(self):
        """Test that nested dataclasses are independent between contexts."""
        reset_context()
        ctx1 = get_context()
        ctx1.copy_state.pipe_buffer["key1"] = {"data": "value"}

        reset_context()
        ctx2 = get_context()

        assert "key1" not in ctx2.copy_state.pipe_buffer
        assert len(ctx2.copy_state.pipe_buffer) == 0
