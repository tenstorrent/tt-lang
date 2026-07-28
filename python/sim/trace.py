# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simulator tracing system.

Provides a single :func:`trace` primitive that records named events into
``SimulatorContext.trace_events``.  Node, kernel, and tick are read
automatically from context; only event-specific data needs to be passed
by the call site.

Trace configuration (whether tracing is on, which categories to record)
lives on the module-level :data:`TRACE` singleton -- not on the simulator
context -- because tracing is a cross-cutting facility, analogous to
Python's ``logging`` module.  The recorded events themselves remain on
the context since they are per-run output.

Call this from outside (CLI, tests) to turn tracing on/off:

    from .trace import set_tracing, ALL_CATEGORIES
    set_tracing(ALL_CATEGORIES)        # record everything
    set_tracing(frozenset({"dfb"}))    # only dfb events
    set_tracing(frozenset())           # disable

Event categories and their events:
    operation : operation_start, operation_end
    kernel    : kernel_start, kernel_end, kernel_block, kernel_unblock
    dfb       : dfb_reserve_begin, dfb_reserve_end, dfb_push,
                dfb_wait_begin, dfb_wait_end, dfb_pop
    copy      : copy_start, copy_end
    pipe      : pipe_send, pipe_recv
"""

from typing import Any, Optional

from .context import get_context
from .context_types import TraceEvent
from .nodecontext import node_mesh_coord

# All defined event categories.  Pass ALL_CATEGORIES to :func:`set_tracing`
# to enable everything; pass an empty frozenset to disable.
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


class _TraceState:
    """Process-wide trace configuration.

    Trace state lives on this singleton (rather than on ``SimulatorContext``)
    for the same reason Python's ``logging`` module owns its own state: trace
    is a cross-cutting facility, not per-run configuration.  Only the recorded
    events themselves live on the context, since those are per-run output.

    The fast-path convention: every trace call site is guarded by
    ``if TRACE.enabled:``, and :func:`trace` only inspects ``.categories``
    when invoked.  Slot access on this singleton is ~10 ns vs. ~125 ns for
    the function call + context lookup it replaces; with >80 M trace calls
    in a dry-run, that's several seconds of avoided overhead per run.

    Anyone adding a new ``trace()`` call site MUST wrap it in the guard,
    e.g.::

        if TRACE.enabled:
            trace("my_event", payload=expensive_to_compute())
    """

    __slots__ = ("enabled", "categories")

    def __init__(self) -> None:
        self.enabled: bool = False
        self.categories: frozenset[str] = frozenset()


TRACE = _TraceState()


def set_tracing(categories: frozenset[str]) -> None:
    """Set the active trace categories (single source of truth).

    Empty ``categories`` disables tracing entirely; any non-empty value
    enables the fast-path flag so call sites invoke :func:`trace`.
    """
    TRACE.categories = categories
    TRACE.enabled = bool(categories)


def trace(event: str, **data: Any) -> None:
    """Record a named trace event.

    Node, kernel name, and tick are read automatically from the simulator
    context and the active scheduler. The caller passes only event-specific
    data that cannot be derived from context (e.g. occupied slot count).

    This function is a no-op when the event's category is not in
    ``TRACE.categories``.  The fast path is the ``if TRACE.enabled:``
    guard at every call site -- this function itself is only reached when
    tracing is on, and exists only to perform the per-category filter and
    record the event.

    Args:
        event: Event name (e.g. "dfb_push", "kernel_block").
        **data: Event-specific key-value pairs to include in the record.
    """
    category = _EVENT_CATEGORY.get(event)
    if category not in TRACE.categories:
        return

    ctx = get_context()
    scheduler = ctx.scheduler
    assert scheduler is not None, (
        f"trace('{event}') called without an active scheduler. "
        "Tracing must only be used inside a scheduled operation."
    )

    # Resolve the linear node: prefer the currently scheduled kernel; fall back
    # to an explicit ``node`` in the event data (operation_start/operation_end
    # are emitted by the program loop with no kernel currently scheduled).
    explicit = data.get("node")
    explicit_node = explicit if isinstance(explicit, int) else None
    linear_node: Optional[int] = None
    kid = scheduler.get_current_kernel_id()
    if kid is not None:
        linear_node = kid.linear_node
        # If a call site also passes an explicit ``node``, it must match the
        # currently scheduled kernel. A mismatch means the event is being
        # attributed to the wrong node (an instrumentation bug), so fail loudly
        # rather than silently preferring the kernel's node. Raise unconditionally
        # (not ``assert``) so the check survives running under ``-O``.
        if explicit_node is not None and explicit_node != linear_node:
            raise RuntimeError(
                f"trace('{event}') node mismatch: explicit node {explicit_node} "
                f"does not match the scheduled kernel "
                f"'{scheduler.get_current_kernel_name()}' (node {linear_node})."
            )
    else:
        linear_node = explicit_node

    node_label: Optional[str] = None
    device_coord: Optional[list[int]] = None
    if linear_node is not None:
        node_label = f"node{linear_node}"
        grid = getattr(scheduler, "grid", ()) or ()
        device_coord = list(node_mesh_coord(linear_node, grid))

    ctx.trace_events.append(
        TraceEvent(
            event=event,
            tick=scheduler.tick,
            kernel=scheduler.get_current_kernel_name(),
            node=node_label,
            device=device_coord,
            data=data,
        )
    )


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
