# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simulator context management.

The simulator owns a single ``SimulatorContext`` per process at any time.
Each simulation run (a ``Program`` invocation, a ``ttlang-sim`` command, or a
single pytest test) gets its own fresh context, set up at the start of the
run.

A simulator run is single-threaded from the host's perspective: greenlets are
cooperative, so at most one is executing at a time and they all share the
same context.  Pytest workers are separate processes, so tests in different
workers each have their own module state.  This is why a single module-level
reference is sufficient -- there is no concurrent reader/writer that would
require per-greenlet (or per-thread) isolation.

``reset_context()`` exists mainly to give tests a clean slate between runs:
it discards the current context and installs a fresh one, equivalent to
starting a new run.
"""

from __future__ import annotations

from typing import Optional

from .context_types import SimulatorContext
from .blockstate import ThreadType


# Single per-process simulator context.  Created lazily by ``get_context()``
# and swapped wholesale by ``set_context()`` / ``reset_context()``.  See the
# module docstring for why a plain module global is the right abstraction.
_current_context: Optional[SimulatorContext] = None


def get_context() -> SimulatorContext:
    """Return the current simulator context, creating one on first access.

    Auto-creation makes simulator APIs usable from any thread/greenlet
    without explicit setup, which keeps ad-hoc scripts and the existing
    test surface simple.  Production callers (the ``ttlang-sim`` CLI, the
    pytest fixture, and ``Program.__call__``) explicitly install a fresh
    context at the start of each run via ``reset_context()``.
    """
    global _current_context
    if _current_context is None:
        _current_context = SimulatorContext()
    return _current_context


def set_context(ctx: SimulatorContext) -> None:
    """Install ``ctx`` as the current simulator context.

    Primarily a testing hook for injecting a specific context; production
    code should use ``reset_context()`` to install a fresh one.
    """
    global _current_context
    _current_context = ctx


def reset_context() -> None:
    """Discard the current context and install a fresh one.

    Called at the start of every test (via the autouse fixture) and by
    ``ttlang-sim`` at process startup so each run begins with default
    state.  Also releases the ``sys.monitoring`` tool slot used for
    copy-wait injection so the next run can re-register its callbacks.
    """
    import sys

    if sys.monitoring.get_tool(sys.monitoring.OPTIMIZER_ID) is not None:
        sys.monitoring.free_tool_id(sys.monitoring.OPTIMIZER_ID)
    set_context(SimulatorContext())


def cleanup_run_context() -> None:
    """Clear execution-specific state inside the current context.

    Called by the ``@ttl.operation`` wrapper after each ``Program`` run.
    Unlike ``reset_context()``, this preserves persistent session state
    such as ``trace_events`` and ``config`` so that callers can read
    trace output after the run completes; it only zeroes the
    per-run scratch state (scheduler, thread registry, monitoring hooks,
    auto-wait caches, DFB and L1 counters).
    """
    import sys

    ctx = get_context()
    ctx.scheduler = None
    ctx.current_thread_type = None
    ctx.thread_registry.clear()
    ctx.kernel_dfb_count = 0
    ctx.kernel_l1_bytes = 0
    ctx.active_hooks.clear()
    ctx.injection_points_cache.clear()
    ctx.auto_wait_copy_lines.clear()
    if sys.monitoring.get_tool(sys.monitoring.OPTIMIZER_ID) is not None:
        sys.monitoring.free_tool_id(sys.monitoring.OPTIMIZER_ID)


def set_dry_run(enabled: bool) -> None:
    """Enable or disable dry-run mode for the current simulator context.

    In dry-run mode the simulator skips the computational payload of
    simulator-managed objects: ``ttnn.Tensor`` arithmetic operators return
    zero tensors of the correct shape, ``ttl.math`` block operations return
    dummy blocks, and ``ttl.copy()`` transfers complete without moving any
    bytes.  The full DFB sequencing, block state machine, deadlock detection,
    and copy-wait injection still run unchanged.  This makes it safe to
    validate kernel structure without needing meaningful input data.

    **Scope:** dry-run only intercepts calls that go through the simulator
    APIs listed above.  All other Python code -- plain arithmetic on scalars,
    standard-library calls, user-defined data structures, and any control
    flow that does not branch on a simulated tensor value -- executes
    normally.  Kernels that derive loop bounds or branch conditions from
    computed tile values will therefore not be structurally validated by
    dry-run (the simulator assumes computation results do not affect control
    flow).

    Args:
        enabled: True to enable dry-run, False to disable.
    """
    get_context().config.dry_run = enabled


def get_current_thread_type() -> ThreadType:
    """Get the current kernel role (compute vs datamovement).

    Returns:
        ThreadType

    Raises:
        RuntimeError: If kernel role is not set (not within a running compute/DM kernel)
    """
    current_thread_type = get_context().current_thread_type
    if current_thread_type is None:
        raise RuntimeError(
            "Compute/DM kernel context is not set. Use this only while a compute or "
            "datamovement kernel is running, or after calling set_current_thread_type()."
        )
    return current_thread_type


def set_current_thread_type(thread_type: Optional[ThreadType]) -> None:
    """Set the current thread type.

    Args:
        thread_type: The thread type to set, or None to clear the context
    """
    get_context().current_thread_type = thread_type


def clear_current_thread_type() -> None:
    """Clear the current thread type."""
    get_context().current_thread_type = None
