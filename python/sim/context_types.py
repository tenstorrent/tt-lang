# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simulator context type definitions.

This module contains only the dataclass definitions for simulator context,
separated from the context management functions to avoid import cycles.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Set, Tuple, TypedDict
from .pipe import AnyPipe
from .ttnnsim import Tensor
from .typedefs import Count, Shape, BindableTemplate
from .blockstate import ThreadType


@dataclass
class SimulatorConfig:
    """Simulator configuration settings."""

    max_dfbs: int = 32
    scheduler_algorithm: str = "fair"
    default_auto_grid: Shape = (8, 8)


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


class PipeEntry(TypedDict):
    """Pipe buffer entry for NoC pipe communication simulation.

    Each entry holds a queue of messages and a message-ID counter.
    No locking needed because greenlet scheduler is cooperative.
    """

    queue: Deque[Tuple[Tensor, Count, int, set[int]]]
    next_msg_id: int


@dataclass
class CopySystemState:
    """Copy system runtime state (per-greenlet)."""

    pipe_buffer: Dict[AnyPipe, PipeEntry] = field(
        default_factory=dict[AnyPipe, PipeEntry]
    )  # pyright: ignore[reportUnknownVariableType]


@dataclass
class WarningState:
    """Warning deduplication tracking."""

    broadcast_1d_warnings: Dict[tuple[str, int], Set[str]] = field(
        default_factory=dict[tuple[str, int], Set[str]]
    )  # pyright: ignore[reportUnknownVariableType]
    block_print_warnings: Dict[tuple[str, int], Set[str]] = field(
        default_factory=dict[tuple[str, int], Set[str]]
    )  # pyright: ignore[reportUnknownVariableType]


@dataclass
class SimulatorContext:
    """Complete simulator runtime context stored per-greenlet."""

    config: SimulatorConfig = field(default_factory=SimulatorConfig)
    stats: SimulatorStats = field(default_factory=SimulatorStats)
    copy_state: CopySystemState = field(default_factory=CopySystemState)
    warnings: WarningState = field(default_factory=WarningState)
    scheduler: Any = None  # Optional[GreenletScheduler] - avoid import cycle
    current_thread_type: Optional[ThreadType] = None
    thread_registry: list[BindableTemplate] = field(
        default_factory=list[BindableTemplate]
    )  # pyright: ignore[reportUnknownVariableType]
