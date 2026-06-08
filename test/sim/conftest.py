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
from sim.blockstate import KernelType
from sim.context import set_current_kernel_type, reset_context
from sim.greenlet_scheduler import (
    GreenletScheduler,
    KernelId,
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


def setup_scheduler_and_kernel_context(kernel_type: KernelType) -> GreenletScheduler:
    """Set up scheduler and kernel context for unit tests.

    Args:
        kernel_type: Type of kernel to simulate (COMPUTE or DM)

    Returns:
        Configured GreenletScheduler instance
    """
    # Use fair scheduler (the default)
    set_scheduler_algorithm("fair")

    # Create a scheduler instance for the test
    scheduler = GreenletScheduler()
    set_scheduler(scheduler)

    # Set kernel context
    set_current_kernel_type(kernel_type)

    # Set the main greenlet to the current greenlet (for switching back)
    scheduler._main_greenlet = greenlet.getcurrent()

    # Simulate being within node 0 with a valid KernelId so that
    # get_current_node_id() returns "node0" and shard-locality stats work in tests.
    # ``kind`` mirrors ``kernel_type``; ``func_name`` matches the chosen role
    # for readable display in any test diagnostics.
    test_greenlet = greenlet(lambda: None)
    tid = KernelId(0, kernel_type, kernel_type.name.lower())
    scheduler._current_kernel_id = tid
    scheduler._active[tid] = (
        test_greenlet,
        None,  # blocking_obj
        "",  # operation
        kernel_type,
        "",  # location
        None,  # raw_loc
    )
    scheduler._has_made_progress[tid] = False

    return scheduler


def teardown_scheduler_and_kernel_context() -> None:
    """Clean up scheduler and kernel context."""
    set_current_kernel_type(None)
    set_scheduler(None)


@pytest.fixture
def compute_kernel_context():
    """Set up scheduler context with COMPUTE kernel for tests."""
    setup_scheduler_and_kernel_context(KernelType.COMPUTE)
    yield
    teardown_scheduler_and_kernel_context()


@pytest.fixture
def dm_kernel_context():
    """Set up scheduler context with DM kernel for tests."""
    setup_scheduler_and_kernel_context(KernelType.DM)
    yield
    teardown_scheduler_and_kernel_context()
