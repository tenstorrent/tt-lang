# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for greenlet-based cooperative scheduler.
"""

import pytest

from python.sim.block import ThreadType
from python.sim.greenlet_scheduler import (
    GreenletScheduler,
    block_if_needed,
    get_scheduler,
    set_scheduler,
)


class MockBlockable:
    """Mock object that can be waited on or reserved."""

    def __init__(self, initially_ready: bool = True):
        self._ready = initially_ready
        self._wait_count = 0
        self._reserve_count = 0

    def can_wait(self) -> bool:
        return self._ready

    def can_reserve(self) -> bool:
        return self._ready

    def make_ready(self) -> None:
        self._ready = True

    def make_blocked(self) -> None:
        self._ready = False

    def on_wait(self) -> None:
        self._wait_count += 1

    def on_reserve(self) -> None:
        self._reserve_count += 1


class TestGreenletScheduler:
    """Tests for GreenletScheduler class."""

    def test_basic_execution(self) -> None:
        """Test basic thread execution."""
        scheduler = GreenletScheduler()
        executed = []

        def thread1() -> None:
            executed.append("thread1")

        def thread2() -> None:
            executed.append("thread2")

        scheduler.add_thread("t1", thread1, ThreadType.COMPUTE)
        scheduler.add_thread("t2", thread2, ThreadType.DM)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        assert "thread1" in executed
        assert "thread2" in executed

    def test_thread_completion_tracking(self) -> None:
        """Test that completed threads are tracked."""
        scheduler = GreenletScheduler()
        completed = []

        def thread1() -> None:
            completed.append("t1")

        def thread2() -> None:
            completed.append("t2")

        scheduler.add_thread("t1", thread1, ThreadType.COMPUTE)
        scheduler.add_thread("t2", thread2, ThreadType.DM)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        # Both threads should complete
        assert len(completed) == 2
        assert set(completed) == {"t1", "t2"}

    def test_blocking_and_unblocking(self) -> None:
        """Test that threads can block and unblock."""
        scheduler = GreenletScheduler()
        mock_obj = MockBlockable(initially_ready=False)
        execution_order = []

        def thread1() -> None:
            execution_order.append("t1-start")
            # This will block since mock_obj is not ready
            block_if_needed(mock_obj, "wait")
            execution_order.append("t1-after-block")

        def thread2() -> None:
            execution_order.append("t2-start")
            # Unblock thread1
            mock_obj.make_ready()
            execution_order.append("t2-end")

        scheduler.add_thread("t1", thread1, ThreadType.COMPUTE)
        scheduler.add_thread("t2", thread2, ThreadType.DM)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        # Thread 1 should start, block, then thread 2 runs and unblocks it
        assert execution_order == ["t1-start", "t2-start", "t2-end", "t1-after-block"]

    def test_deadlock_detection(self) -> None:
        """Test that deadlock is detected when all threads are blocked."""
        scheduler = GreenletScheduler()
        mock_obj = MockBlockable(initially_ready=False)

        def blocked_thread() -> None:
            # This will block forever
            block_if_needed(mock_obj, "wait")

        scheduler.add_thread("t1", blocked_thread, ThreadType.COMPUTE)

        set_scheduler(scheduler)
        try:
            with pytest.raises(RuntimeError, match="Deadlock detected"):
                scheduler.run()
        finally:
            set_scheduler(None)

    def test_deadlock_with_multiple_threads(self) -> None:
        """Test deadlock detection with multiple blocked threads."""
        scheduler = GreenletScheduler()
        mock1 = MockBlockable(initially_ready=False)
        mock2 = MockBlockable(initially_ready=False)

        def thread1() -> None:
            block_if_needed(mock1, "wait")

        def thread2() -> None:
            block_if_needed(mock2, "reserve")

        scheduler.add_thread("t1", thread1, ThreadType.COMPUTE)
        scheduler.add_thread("t2", thread2, ThreadType.DM)

        set_scheduler(scheduler)
        try:
            with pytest.raises(RuntimeError, match="Deadlock detected"):
                scheduler.run()
        finally:
            set_scheduler(None)

    def test_error_propagation(self) -> None:
        """Test that errors in threads are properly propagated."""
        scheduler = GreenletScheduler()

        def failing_thread() -> None:
            raise ValueError("Test error")

        scheduler.add_thread("t1", failing_thread, ThreadType.COMPUTE)

        set_scheduler(scheduler)
        try:
            with pytest.raises(RuntimeError, match="t1.*ValueError.*Test error"):
                scheduler.run()
        finally:
            set_scheduler(None)

    def test_round_robin_scheduling(self) -> None:
        """Test that threads are scheduled in round-robin fashion."""
        scheduler = GreenletScheduler()
        mock1 = MockBlockable(initially_ready=True)
        mock2 = MockBlockable(initially_ready=True)
        execution_order = []

        def thread1() -> None:
            execution_order.append("t1-1")
            block_if_needed(mock1, "wait")
            execution_order.append("t1-2")
            block_if_needed(mock1, "wait")
            execution_order.append("t1-3")

        def thread2() -> None:
            execution_order.append("t2-1")
            block_if_needed(mock2, "wait")
            execution_order.append("t2-2")
            block_if_needed(mock2, "wait")
            execution_order.append("t2-3")

        scheduler.add_thread("t1", thread1, ThreadType.COMPUTE)
        scheduler.add_thread("t2", thread2, ThreadType.DM)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        # Check that threads interleave (round-robin)
        assert execution_order[0] in ["t1-1", "t2-1"]
        # Should have all executions
        assert len(execution_order) == 6
        assert execution_order.count("t1-1") == 1
        assert execution_order.count("t2-2") == 1

    def test_no_scheduler_error(self) -> None:
        """Test that get_scheduler raises error when no scheduler is active."""
        set_scheduler(None)
        with pytest.raises(RuntimeError, match="No active scheduler"):
            get_scheduler()

    def test_block_if_needed_when_ready(self) -> None:
        """Test that block_if_needed doesn't block when ready."""
        scheduler = GreenletScheduler()
        mock_obj = MockBlockable(initially_ready=True)
        executed = []

        def thread() -> None:
            executed.append("before")
            block_if_needed(mock_obj, "wait")
            executed.append("after")

        scheduler.add_thread("t1", thread, ThreadType.COMPUTE)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        # Should execute both before and after without blocking
        assert executed == ["before", "after"]

    def test_block_if_needed_when_not_ready(self) -> None:
        """Test that block_if_needed blocks when not ready."""
        scheduler = GreenletScheduler()
        mock_obj = MockBlockable(initially_ready=False)
        executed = []

        def thread1() -> None:
            executed.append("t1-before")
            block_if_needed(mock_obj, "wait")
            executed.append("t1-after")

        def thread2() -> None:
            executed.append("t2")
            mock_obj.make_ready()

        scheduler.add_thread("t1", thread1, ThreadType.COMPUTE)
        scheduler.add_thread("t2", thread2, ThreadType.DM)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        # t1 should block, t2 runs and unblocks it
        assert "t1-before" in executed
        assert "t2" in executed
        assert "t1-after" in executed
        # t1-before should come before t2
        assert executed.index("t1-before") < executed.index("t2")

    def test_multiple_operations_on_same_object(self) -> None:
        """Test multiple blocking operations on the same object."""
        scheduler = GreenletScheduler()
        mock_obj = MockBlockable(initially_ready=True)
        count = []

        def thread() -> None:
            for i in range(3):
                block_if_needed(mock_obj, "wait")
                count.append(i)

        scheduler.add_thread("t1", thread, ThreadType.COMPUTE)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        assert count == [0, 1, 2]

    def test_scheduler_context_manager_pattern(self) -> None:
        """Test that scheduler can be used with try/finally pattern."""
        executed = []

        def thread() -> None:
            executed.append("done")

        scheduler = GreenletScheduler()
        scheduler.add_thread("t1", thread, ThreadType.COMPUTE)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        assert executed == ["done"]
        # Verify scheduler is cleared
        with pytest.raises(RuntimeError, match="No active scheduler"):
            get_scheduler()


class TestBlockIfNeeded:
    """Tests for the block_if_needed helper function."""

    def test_blocks_when_cannot_proceed(self) -> None:
        """Test that block_if_needed blocks when operation cannot proceed."""
        scheduler = GreenletScheduler()
        mock_obj = MockBlockable(initially_ready=False)
        blocked = []

        def thread1() -> None:
            blocked.append("before")
            block_if_needed(mock_obj, "wait")
            blocked.append("after")

        def thread2() -> None:
            mock_obj.make_ready()

        scheduler.add_thread("t1", thread1, ThreadType.COMPUTE)
        scheduler.add_thread("t2", thread2, ThreadType.DM)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        assert blocked == ["before", "after"]

    def test_does_not_block_when_can_proceed(self) -> None:
        """Test that block_if_needed doesn't block when operation can proceed."""
        scheduler = GreenletScheduler()
        mock_obj = MockBlockable(initially_ready=True)
        executed = []

        def thread() -> None:
            executed.append(1)
            block_if_needed(mock_obj, "wait")
            executed.append(2)
            block_if_needed(mock_obj, "reserve")
            executed.append(3)

        scheduler.add_thread("t1", thread, ThreadType.COMPUTE)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        assert executed == [1, 2, 3]
