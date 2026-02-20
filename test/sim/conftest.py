# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for simulator tests."""

import pytest
from greenlet import greenlet
from python.sim.block import ThreadType, set_current_thread_type
from python.sim.greenlet_scheduler import (
    GreenletScheduler,
    set_scheduler,
    set_scheduler_algorithm,
)


def setup_scheduler_and_thread_context(thread_type: ThreadType) -> GreenletScheduler:
    """Set up scheduler and thread context for unit tests.

    Args:
        thread_type: Type of thread to simulate (COMPUTE or DM)

    Returns:
        Configured GreenletScheduler instance
    """
    # Use fair scheduler (the default)
    set_scheduler_algorithm("fair")

    # Create a scheduler instance for the test
    scheduler = GreenletScheduler()
    set_scheduler(scheduler)

    # Set thread context
    set_current_thread_type(thread_type)

    # Set the main greenlet to the current greenlet (for switching back)
    scheduler._main_greenlet = greenlet.getcurrent()

    # Simulate being within a thread by adding to _active
    test_greenlet = greenlet(lambda: None)
    scheduler._current_name = "test-thread"
    scheduler._active["test-thread"] = (
        test_greenlet,
        None,  # blocking_obj
        "",  # operation
        thread_type,
        "",  # location
        None,  # raw_loc
    )
    scheduler._has_made_progress["test-thread"] = False

    return scheduler


def teardown_scheduler_and_thread_context() -> None:
    """Clean up scheduler and thread context."""
    set_current_thread_type(None)
    set_scheduler(None)


@pytest.fixture
def compute_thread_context():
    """Set up scheduler context with COMPUTE thread for tests."""
    setup_scheduler_and_thread_context(ThreadType.COMPUTE)
    yield
    teardown_scheduler_and_thread_context()


@pytest.fixture
def dm_thread_context():
    """Set up scheduler context with DM thread for tests."""
    setup_scheduler_and_thread_context(ThreadType.DM)
    yield
    teardown_scheduler_and_thread_context()
