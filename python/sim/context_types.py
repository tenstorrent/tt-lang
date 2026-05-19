# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simulator context type definitions.

This module contains only the dataclass definitions for simulator context,
separated from the context management functions to avoid import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Deque, Dict, FrozenSet, Optional, Set, Tuple, TypedDict
from .pipe import AnyPipe
from .ttnnsim import Tensor
from .typedefs import Count, Shape, BindableTemplate
from .blockstate import KernelType

# Default L1 memory limit per node (simulator)
# keep in sync with ttl.constants.DEFAULT_L1_CB_BUDGET_BYTES
DEFAULT_MAX_L1_BYTES: int = 1432 * 1024


@dataclass
class SimulatorConfig:
    """Simulator configuration settings."""

    max_dfbs: int = 32
    scheduler_algorithm: str = "fair"
    default_auto_grid: Shape = (8, 8)
    max_l1_bytes: int = DEFAULT_MAX_L1_BYTES
    num_devices: int = 4
    # Set of event categories to record. Empty means tracing is disabled.
    # Use trace.ALL_CATEGORIES to enable all categories.
    trace_set: FrozenSet[str] = field(default_factory=frozenset)


@dataclass
class TraceEvent:
    """A single recorded trace event."""

    event: str
    tick: int
    kernel: Optional[str]
    data: Dict[str, Any] = field(default_factory=dict)


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
    copy_state: CopySystemState = field(default_factory=CopySystemState)
    warnings: WarningState = field(default_factory=WarningState)
    scheduler: Any = None  # Optional[GreenletScheduler] - avoid import cycle
    current_kernel_type: Optional[KernelType] = None
    kernel_registry: list[BindableTemplate] = field(
        default_factory=list[BindableTemplate]
    )  # pyright: ignore[reportUnknownVariableType]
    kernel_dfb_count: int = 0  # DFBs created in the current kernel body
    kernel_l1_bytes: int = (
        0  # Total L1 capacity of DFBs created in the current kernel body
    )
    trace_events: list[TraceEvent] = field(default_factory=list)
    # Maps kernel function objects to their precomputed InjectionPoint tuples.
    # Populated once per kernel invocation before the node loop runs.
    # Typed as Any to avoid importing analysis (which imports dfb -> context).
    injection_points_cache: Dict[Any, Any] = field(default_factory=dict)
    # Active copy-wait injection hooks for this simulation run.
    # Maps id(CodeType) -> (by_lineno, return_ips) lookup tables used by
    # callbacks in analysis.py.  Cleared automatically when the context is
    # replaced by reset_context(), requiring no sys.monitoring reconfiguration.
    # Typed as Any to avoid importing analysis.
    active_hooks: Dict[Any, Any] = field(default_factory=dict)
    # Set of (code_object, abs_lineno) pairs identifying bare ttl.copy() calls
    # (i.e. calls whose return value is not assigned).  The copy() function
    # checks this set to immediately call wait() for those call sites.
    # Typed as Any to avoid importing types.CodeType here.
    auto_wait_copy_lines: Set[Any] = field(default_factory=set)
