# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for simulator tests."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-matmul-tutorial-dry",
        action="store_true",
        default=False,
        help="Run matmul-tutorial simulator tests in dry-run mode (steps 0 and 2-7; step 1 excluded as too slow); skipped by default.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_matmul_tutorial = pytest.mark.skip(
        reason="matmul-tutorial simulator test; pass --run-matmul-tutorial-dry to enable"
    )
    for item in items:
        if item.get_closest_marker("matmul_tutorial") and not config.getoption(
            "--run-matmul-tutorial-dry"
        ):
            item.add_marker(skip_matmul_tutorial)


from greenlet import greenlet
from sim.blockstate import KernelType
from sim.context import set_current_kernel_type, reset_context
from sim.greenlet_scheduler import (
    GreenletScheduler,
    KernelId,
    _KernelState,
    set_scheduler,
    set_scheduler_algorithm,
)
from sim.trace import set_tracing


@pytest.fixture(autouse=True)
def reset_simulator_context():
    """Reset simulator context before each test to ensure test isolation.

    Resets both the simulator context (config, scheduler, registry, trace
    events) and the trace module's process-wide configuration so that
    modifications in one test do not leak into others when running in
    parallel.  Trace state is reset here rather than inside
    ``reset_context()`` to keep ``context.py`` free of trace imports
    (see the docstring on :func:`sim.context.reset_context`).
    """
    reset_context()
    set_tracing(frozenset())
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

    # Set the main greenlet to the current greenlet (for switching back) and
    # cache its bound ``.switch`` method the same way ``GreenletScheduler.run``
    # does, so any test that drives ``block_current_kernel`` directly has the
    # fast-path slot populated.
    scheduler._main_greenlet = greenlet.getcurrent()
    scheduler._main_switch = scheduler._main_greenlet.switch

    # Simulate being within node 0 with a valid KernelId so that
    # get_current_node_id() returns "node0" and shard-locality stats work in tests.
    # ``kind`` mirrors ``kernel_type``; ``func_name`` matches the chosen role
    # for readable display in any test diagnostics.
    test_greenlet = greenlet(lambda: None)
    tid = KernelId(0, kernel_type, kernel_type.name.lower())
    state = _KernelState(test_greenlet, kernel_type)
    scheduler._current_kernel_id = tid
    scheduler._current_state = state
    scheduler._active[tid] = state
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
