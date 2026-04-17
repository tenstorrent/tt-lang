# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for simulator tests."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-matmul-tutorial-ttnn",
        action="store_true",
        default=False,
        help="Run matmul-tutorial tests that require real ttnn (steps 0 and 7); skipped by default.",
    )
    parser.addoption(
        "--run-matmul-tutorial-no-ttnn",
        action="store_true",
        default=False,
        help="Run matmul-tutorial simulator tests that do not require ttnn (steps 2-6); skipped by default.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_ttnn = pytest.mark.skip(
        reason="matmul-tutorial test requiring ttnn; pass --run-matmul-tutorial-ttnn to enable"
    )
    skip_no_ttnn = pytest.mark.skip(
        reason="matmul-tutorial simulator test; pass --run-matmul-tutorial-no-ttnn to enable"
    )
    for item in items:
        if item.get_closest_marker("matmul_tutorial_ttnn") and not config.getoption(
            "--run-matmul-tutorial-ttnn"
        ):
            item.add_marker(skip_ttnn)
        if item.get_closest_marker("matmul_tutorial_no_ttnn") and not config.getoption(
            "--run-matmul-tutorial-no-ttnn"
        ):
            item.add_marker(skip_no_ttnn)


from greenlet import greenlet
from python.sim.blockstate import ThreadType
from python.sim.context import set_current_thread_type, reset_context
from python.sim.greenlet_scheduler import (
    GreenletScheduler,
    set_scheduler,
    set_scheduler_algorithm,
)


@pytest.fixture(autouse=True)
def reset_simulator_context():
    """Reset simulator context before each test to ensure test isolation.

    This ensures that modifications to context config (e.g., max_dfbs) or
    state in one test don't leak into other tests when running in parallel.
    """
    reset_context()
    yield


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

    # Simulate being within core 0 by using a valid core thread name so that
    # get_current_core_id() returns "core0" and shard-locality stats work in tests.
    test_greenlet = greenlet(lambda: None)
    scheduler._current_name = "core0-compute"
    scheduler._active["core0-compute"] = (
        test_greenlet,
        None,  # blocking_obj
        "",  # operation
        thread_type,
        "",  # location
        None,  # raw_loc
    )
    scheduler._has_made_progress["core0-compute"] = False

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
