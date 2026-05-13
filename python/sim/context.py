# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simulator context management using greenlet-local storage.

All simulator state is stored in the current greenlet's attributes,
eliminating the need for module-level globals.
"""

from __future__ import annotations

from typing import Optional

from greenlet import getcurrent

from .context_types import SimulatorContext
from .blockstate import ThreadType


def get_context() -> SimulatorContext:
    """Get simulator context from current greenlet or its parents.

    Context is stored as an attribute on greenlet objects. This function
    walks up the greenlet parent chain to find the context, eliminating
    the need for module-level globals.

    In production code, this is the only context function you need - it
    auto-creates contexts on first access. The set/reset functions are
    primarily for testing scenarios.

    Returns:
        SimulatorContext for the current greenlet tree
    """
    greenlet = getcurrent()

    # Walk up the greenlet parent chain to find context
    while greenlet is not None:
        if hasattr(greenlet, "_sim_context"):
            return greenlet._sim_context  # type: ignore
        # Move to parent greenlet
        greenlet = getattr(greenlet, "parent", None)

    # No context found in any parent - create one on the root greenlet
    # This happens when called outside of any Program execution
    root = getcurrent()
    root._sim_context = SimulatorContext()  # type: ignore
    return root._sim_context  # type: ignore


def set_context(ctx: SimulatorContext) -> None:
    """Set simulator context for current greenlet.

    Mainly useful for testing when you want to inject a specific context.
    Production code typically doesn't need this - use get_context() instead.

    Args:
        ctx: Context to set
    """
    getcurrent()._sim_context = ctx  # type: ignore


def reset_context() -> None:
    """Reset context for current greenlet to defaults.

    Creates a fresh context, discarding any previous state.
    Also frees the sys.monitoring tool slot used for copy-wait injection so
    the next simulation run can re-register its callbacks from a clean state.
    Primarily useful for test cleanup.
    """
    import sys

    if sys.monitoring.get_tool(sys.monitoring.OPTIMIZER_ID) is not None:
        sys.monitoring.free_tool_id(sys.monitoring.OPTIMIZER_ID)
    getcurrent()._sim_context = SimulatorContext()  # type: ignore


def cleanup_run_context() -> None:
    """Clean up execution-specific state after a single Program run.

    Clears scheduler, monitoring hooks, and per-run caches so that a
    subsequent operation starts cleanly.  Unlike ``reset_context()``, this
    preserves persistent session state such as ``trace_events`` and ``config``
    so that callers can read trace output after the run completes.

    Called by the ``@ttl.operation`` wrapper after each ``Program`` run.
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
