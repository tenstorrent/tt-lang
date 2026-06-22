# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simulator context management.

The simulator owns a single ``SimulatorContext`` per process at any time.
Each simulation run (a ``Program`` invocation, a ``tt-lang-sim`` command, or a
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

import sys
from typing import Optional

from .context_types import SimulatorContext
from .blockstate import KernelType


# Single per-process simulator context.  Created lazily by ``get_context()``
# and swapped wholesale by ``set_context()`` / ``reset_context()``.  See the
# module docstring for why a plain module global is the right abstraction.
_current_context: Optional[SimulatorContext] = None


def _free_monitoring_tool_id() -> None:
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        return
    tool_id = monitoring.OPTIMIZER_ID
    if monitoring.get_tool(tool_id) is not None:
        monitoring.free_tool_id(tool_id)


def get_context() -> SimulatorContext:
    """Return the current simulator context, creating one on first access.

    Auto-creation makes simulator APIs usable from any thread/greenlet
    without explicit setup, which keeps ad-hoc scripts and the existing
    test surface simple.  Production callers (the ``tt-lang-sim`` CLI, the
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
    ``tt-lang-sim`` at process startup so each run begins with default
    state.  Also releases the ``sys.monitoring`` tool slot used for
    copy-wait injection so the next run can re-register its callbacks.

    Trace configuration lives on the ``trace`` module's own singleton
    (see :mod:`.trace`) and is *not* reset here, both because it has no
    place on the context and because importing trace from here would
    introduce a module-level cycle.  Callers that want a clean trace
    slate between runs (the pytest autouse fixture; the CLI when
    bootstrapping a fresh process) call :func:`trace.set_tracing`
    explicitly.
    """
    _free_monitoring_tool_id()
    set_context(SimulatorContext())


def cleanup_run_context() -> None:
    """Clear execution-specific state inside the current context.

    Called by the ``@ttl.operation`` wrapper after each ``Program`` run.
    Unlike ``reset_context()``, this preserves persistent session state
    such as ``trace_events`` and ``config`` so that callers can read
    trace output after the run completes; it only zeroes the
    per-run scratch state (scheduler, kernel registry, monitoring hooks,
    auto-wait caches, DFB and L1 counters).
    """
    ctx = get_context()
    ctx.scheduler = None
    ctx.current_kernel_type = None
    ctx.kernel_registry.clear()
    ctx.kernel_dfb_count = 0
    ctx.kernel_l1_bytes = 0
    ctx.active_hooks.clear()
    ctx.injection_points_cache.clear()
    ctx.auto_wait_copy_lines.clear()
    _free_monitoring_tool_id()


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


def get_current_kernel_type() -> KernelType:
    """Get the current kernel role (compute vs datamovement).

    Returns:
        KernelType

    Raises:
        RuntimeError: If kernel role is not set (not within a running compute/DM kernel)
    """
    current_kernel_type = get_context().current_kernel_type
    if current_kernel_type is None:
        raise RuntimeError(
            "Compute/DM kernel context is not set. Use this only while a compute or "
            "datamovement kernel is running, or after calling set_current_kernel_type()."
        )
    return current_kernel_type


def set_current_kernel_type(kernel_type: Optional[KernelType]) -> None:
    """Set the current kernel role (compute vs datamovement).

    Args:
        kernel_type: The kernel role to set, or None to clear the context
    """
    get_context().current_kernel_type = kernel_type


def clear_current_kernel_type() -> None:
    """Clear the current kernel role."""
    get_context().current_kernel_type = None
