# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simulator context management using greenlet-local storage.

All simulator state is stored in the current greenlet's attributes,
eliminating the need for module-level globals.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, Optional, Set, Tuple, TypedDict

from greenlet import getcurrent

if TYPE_CHECKING:
    from .greenlet_scheduler import GreenletScheduler
    from .pipe import AnyPipe
    from .ttnnsim import Tensor
    from .typedefs import Count


@dataclass
class SimulatorConfig:
    """Simulator configuration settings."""

    max_dfbs: int = 32
    scheduler_algorithm: str = "fair"


@dataclass
class SimulatorStats:
    """Statistics collection state."""

    enabled: bool = False
    stats_by_name: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"reads": 0, "writes": 0, "tiles_read": 0, "tiles_written": 0}
        )
    )
    pipe_stats_by_name: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"reads": 0, "writes": 0, "tiles_read": 0, "tiles_written": 0}
        )
    )
    dfb_stats_by_name: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"reserves": 0, "waits": 0, "tiles_reserved": 0, "tiles_waited": 0}
        )
    )
    dfb_name_counter: int = 0


class _PipeEntry(TypedDict):
    """Pipe buffer entry for NoC pipe communication simulation.

    Each entry holds a queue of messages and a message-ID counter.
    No locking needed because greenlet scheduler is cooperative.
    """

    queue: "Deque[Tuple[Tensor, Count, int, set[int]]]"
    next_msg_id: int


@dataclass
class CopySystemState:
    """Copy system runtime state (per-greenlet)."""

    pipe_buffer: "Dict[AnyPipe, _PipeEntry]" = field(default_factory=dict)


@dataclass
class WarningState:
    """Warning deduplication tracking."""

    broadcast_1d_warnings: Dict[tuple[str, int], Set[str]] = field(default_factory=dict)
    block_print_warnings: Dict[tuple[str, int], Set[str]] = field(default_factory=dict)


@dataclass
class SimulatorContext:
    """Complete simulator runtime context stored per-greenlet."""

    config: SimulatorConfig = field(default_factory=SimulatorConfig)
    stats: SimulatorStats = field(default_factory=SimulatorStats)
    copy_state: CopySystemState = field(default_factory=CopySystemState)
    warnings: WarningState = field(default_factory=WarningState)
    scheduler: Optional["GreenletScheduler"] = None


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
    g = getcurrent()

    # Walk up the greenlet parent chain to find context
    while g is not None:
        if hasattr(g, "_sim_context"):
            return g._sim_context  # type: ignore
        # Move to parent greenlet
        g = getattr(g, "parent", None)

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
