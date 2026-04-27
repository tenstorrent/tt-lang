# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simulator tracing system.

Provides a single trace() primitive that records named events to the current
SimulatorContext. Node, kernel, and tick are read automatically; only
event-specific data needs to be passed by the call site.

Event categories and their events:
    operation : operation_start, operation_end
    kernel    : kernel_start, kernel_end, kernel_block, kernel_unblock
    dfb       : dfb_reserve_begin, dfb_reserve_end, dfb_push,
                dfb_wait_begin, dfb_wait_end, dfb_pop
    copy      : copy_start, copy_end
"""

from typing import Any

from .context import get_context
from .context_types import TraceEvent

# All defined event categories. Assigning ALL_CATEGORIES to trace_set enables
# full tracing; an empty frozenset disables it entirely.
ALL_CATEGORIES: frozenset[str] = frozenset(
    {"operation", "kernel", "dfb", "copy", "pipe"}
)

# Map from event name to its category for filtering.
_EVENT_CATEGORY: dict[str, str] = {
    "operation_start": "operation",
    "operation_end": "operation",
    "kernel_start": "kernel",
    "kernel_end": "kernel",
    "kernel_block": "kernel",
    "kernel_unblock": "kernel",
    "dfb_reserve_begin": "dfb",
    "dfb_reserve_end": "dfb",
    "dfb_push": "dfb",
    "dfb_wait_begin": "dfb",
    "dfb_wait_end": "dfb",
    "dfb_pop": "dfb",
    "copy_start": "copy",
    "copy_end": "copy",
    "pipe_send": "pipe",
    "pipe_recv": "pipe",
}


def trace(event: str, **data: Any) -> None:
    """Record a named trace event.

    Node, kernel name, and tick are read automatically from the simulator
    context and the active scheduler. The caller passes only event-specific
    data that cannot be derived from context (e.g. occupied slot count).

    This function is a no-op when tracing is disabled (empty trace_set) or
    when the event's category is not in trace_set, so instrumented call sites
    add no overhead in untraced runs.

    Args:
        event: Event name (e.g. "dfb_push", "kernel_block").
        **data: Event-specific key-value pairs to include in the record.
    """
    ctx = get_context()
    trace_set = ctx.config.trace_set
    if not trace_set:
        return

    category = _EVENT_CATEGORY.get(event)
    if category not in trace_set:
        return

    scheduler = ctx.scheduler
    assert scheduler is not None, (
        f"trace('{event}') called without an active scheduler. "
        "Tracing must only be used inside a scheduled operation."
    )

    ctx.trace_events.append(
        TraceEvent(
            event=event,
            tick=scheduler.tick,
            kernel=scheduler.get_current_thread_name(),
            data=data,
        )
    )


def get_dfb_name(dfb: Any) -> str:
    """Return a stable display name for a DataflowBuffer.

    Checks for a registered stats name, then a numeric ID, then falls back
    to a hex object-identity suffix. Does not mutate any counters.

    Args:
        dfb: A DataflowBuffer instance.

    Returns:
        A short string identifying the DFB.
    """
    name = getattr(dfb, "_stats_name", None)
    if name:
        return name
    name = getattr(dfb, "_name", None)
    if name:
        return name
    dfb_id = getattr(dfb, "_dfb_id", None)
    if dfb_id is not None:
        return f"dfb_{dfb_id}"
    return f"dfb_{id(dfb) & 0xFFFF:04x}"


def get_pipe_name(pipe: Any) -> str:
    """Return a stable display name for a Pipe, matching the stats naming convention.

    Args:
        pipe: A Pipe instance (or an object with src / dst attributes).

    Returns:
        A string of the form 'pipe_<src>_to_<dst>'.
    """
    src = getattr(pipe, "src", "?")
    dst = getattr(pipe, "dst", "?")

    def _fmt(coord: Any) -> str:
        match coord:
            case tuple():
                parts = []
                for x in coord:
                    match x:
                        case slice():
                            start = x.start if x.start is not None else 0
                            stop = x.stop if x.stop is not None else "?"
                            parts.append(f"{start}:{stop}")
                        case _:
                            parts.append(str(x))
                return f"({',' .join(parts)})"
            case _:
                return str(coord)

    return f"pipe_{_fmt(src)}_to_{_fmt(dst)}"
