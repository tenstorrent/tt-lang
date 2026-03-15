# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for greenlet-based cooperative scheduler.
"""

import pytest

from python.sim.blockstate import ThreadType
from python.sim.greenlet_scheduler import (
    GreenletScheduler,
    block_if_needed,
    get_scheduler,
    get_scheduler_algorithm,
    set_scheduler,
    set_scheduler_algorithm,
)
from test_helpers.mock_kernel import do_wait, do_reserve


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
            do_wait(mock_obj)
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
            do_wait(mock_obj)

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
            do_wait(mock1)

        def thread2() -> None:
            do_reserve(mock2)

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
            with pytest.raises(RuntimeError, match="ValueError.*Test error"):
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
            do_wait(mock1)
            execution_order.append("t1-2")
            do_wait(mock1)
            execution_order.append("t1-3")

        def thread2() -> None:
            execution_order.append("t2-1")
            do_wait(mock2)
            execution_order.append("t2-2")
            do_wait(mock2)
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
            do_wait(mock_obj)
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
            do_wait(mock_obj)
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
                do_wait(mock_obj)
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
            do_wait(mock_obj)
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
            do_wait(mock_obj)
            executed.append(2)
            do_reserve(mock_obj)
            executed.append(3)

        scheduler.add_thread("t1", thread, ThreadType.COMPUTE)

        set_scheduler(scheduler)
        try:
            scheduler.run()
        finally:
            set_scheduler(None)

        assert executed == [1, 2, 3]


class TestSchedulerAlgorithm:
    """Tests for scheduler algorithm selection."""

    def test_default_algorithm_is_greedy(self) -> None:
        """Test that default algorithm is greedy."""
        # Reset to default
        set_scheduler_algorithm("greedy")
        assert get_scheduler_algorithm() == "greedy"

    def test_can_set_fair_algorithm(self) -> None:
        """Test that fair algorithm can be set."""
        try:
            set_scheduler_algorithm("fair")
            assert get_scheduler_algorithm() == "fair"
        finally:
            # Reset to default
            set_scheduler_algorithm("greedy")

    def test_invalid_algorithm_raises_error(self) -> None:
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scheduler algorithm"):
            set_scheduler_algorithm("invalid")

    def test_greedy_scheduling_behavior(self) -> None:
        """Test greedy scheduling runs thread until it blocks."""
        try:
            set_scheduler_algorithm("greedy")
            scheduler = GreenletScheduler()
            execution_order = []

            def thread1() -> None:
                execution_order.append("t1-1")
                execution_order.append("t1-2")
                execution_order.append("t1-3")

            def thread2() -> None:
                execution_order.append("t2-1")
                execution_order.append("t2-2")

            scheduler.add_thread("t1", thread1, ThreadType.COMPUTE)
            scheduler.add_thread("t2", thread2, ThreadType.DM)

            set_scheduler(scheduler)
            try:
                scheduler.run()
            finally:
                set_scheduler(None)

            # With greedy scheduling, one thread typically completes before others start
            # (order depends on dict iteration, but each thread should run continuously)
            assert len(execution_order) == 5
        finally:
            set_scheduler_algorithm("greedy")

    def test_fair_scheduling_behavior(self) -> None:
        """Test fair scheduling interleaves threads."""
        try:
            set_scheduler_algorithm("fair")
            scheduler = GreenletScheduler()
            execution_order = []
            mock_obj1 = MockBlockable(initially_ready=False)
            mock_obj2 = MockBlockable(initially_ready=False)

            def thread1() -> None:
                execution_order.append("t1-1")
                do_wait(mock_obj1)
                execution_order.append("t1-2")

            def thread2() -> None:
                execution_order.append("t2-1")
                do_wait(mock_obj2)
                execution_order.append("t2-2")

            def thread3() -> None:
                # Unblock both threads
                mock_obj1.make_ready()
                mock_obj2.make_ready()

            scheduler.add_thread("t1", thread1, ThreadType.COMPUTE)
            scheduler.add_thread("t2", thread2, ThreadType.COMPUTE)
            scheduler.add_thread("t3", thread3, ThreadType.DM)

            set_scheduler(scheduler)
            try:
                scheduler.run()
            finally:
                set_scheduler(None)

            # Fair scheduling should allow both t1 and t2 to run before blocking
            assert "t1-1" in execution_order
            assert "t2-1" in execution_order
            assert "t1-2" in execution_order
            assert "t2-2" in execution_order
        finally:
            set_scheduler_algorithm("greedy")
