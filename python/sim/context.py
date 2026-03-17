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
    Primarily useful for test cleanup.
    """
    getcurrent()._sim_context = SimulatorContext()  # type: ignore


def get_current_thread_type() -> ThreadType:
    """Get the current thread type.

    Returns:
        ThreadType

    Raises:
        RuntimeError: If thread type is not set (not within a thread context)
    """
    current_thread_type = get_context().current_thread_type
    if current_thread_type is None:
        raise RuntimeError(
            "Thread context not set. Must be called within a kernel thread or after "
            "calling set_current_thread_type()."
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
