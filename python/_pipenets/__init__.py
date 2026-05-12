# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Operation-level PipeNet graph: a data type owned by an operation
invocation and consumed by both the simulator and the compiler frontend
without either depending on the other.

The graph is the single source of truth for which PipeNets an operation
uses. It is built from the operation's closure (captured PipeNets) plus
its body (PipeNets constructed in-line). The compiler and the simulator
both compute the PipeNet work extent and run validation against this graph.

Multi-device readiness: NodeCoord is intra-chip. Inter-chip pipes would
be a separate type wrapping NodeCoord plus a mesh coordinate, and
OperationPipeNets would hold both lists.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple, Union


@dataclass(frozen=True)
class NodeCoord:
    """Logical node coordinate within one device's grid.

    `coords` is a tuple of length matching the operation's grid rank.
    """

    coords: Tuple[int, ...]


@dataclass(frozen=True)
class NodeRange:
    """Half-open hyperrectangle of node coordinates: lo[i] <= x < hi[i]."""

    lo: Tuple[int, ...]
    hi: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.lo) != len(self.hi):
            raise ValueError(
                f"NodeRange lo and hi must have the same rank, "
                f"got {self.lo} and {self.hi}"
            )
        for axis, (lo, hi) in enumerate(zip(self.lo, self.hi)):
            if lo >= hi:
                raise ValueError(
                    f"NodeRange axis {axis} requires lo < hi, " f"got lo={lo}, hi={hi}"
                )


@dataclass(frozen=True)
class PipeUse:
    """One pipe within a PipeNet: a source node and one or more destinations."""

    src: NodeCoord
    dst: Union[NodeCoord, NodeRange]


@dataclass(frozen=True)
class PipeNetUse:
    """One PipeNet consumed by one operation invocation.

    `id` is operation-local: allocated 0..N-1 per OperationPipeNets and
    reset on each operation invocation.
    """

    id: int
    pipes: Tuple[PipeUse, ...]


@dataclass
class OperationPipeNets:
    """All PipeNets used by one operation invocation."""

    pipe_nets: List[PipeNetUse] = field(default_factory=list)

    def add_pipe_net(self, pipes: Iterable[PipeUse]) -> PipeNetUse:
        """Append a new PipeNetUse with the next operation-local id."""
        use = PipeNetUse(id=len(self.pipe_nets), pipes=tuple(pipes))
        self.pipe_nets.append(use)
        return use

    def work_extent(self) -> Optional[Tuple[int, ...]]:
        """Per-axis bounding box of every pipe coordinate in the graph.

        For each axis, returns the smallest grid size that contains every
        src and dst coordinate of every pipe. Source coordinates contribute
        `coord + 1`; multicast destinations contribute `hi` (already an
        exclusive upper bound). Returns None when the graph is empty.

        The rank is the maximum coordinate rank seen across pipes; pipes
        that omit higher axes contribute 1 there.
        """
        if not self.pipe_nets:
            return None
        rank = 0
        for net in self.pipe_nets:
            for pipe in net.pipes:
                rank = max(rank, len(pipe.src.coords))
                if isinstance(pipe.dst, NodeRange):
                    rank = max(rank, len(pipe.dst.hi))
                else:
                    rank = max(rank, len(pipe.dst.coords))
        if rank == 0:
            return None
        extent = [1] * rank
        for net in self.pipe_nets:
            for pipe in net.pipes:
                for axis, c in enumerate(pipe.src.coords):
                    extent[axis] = max(extent[axis], c + 1)
                if isinstance(pipe.dst, NodeRange):
                    for axis, hi in enumerate(pipe.dst.hi):
                        extent[axis] = max(extent[axis], hi)
                else:
                    for axis, c in enumerate(pipe.dst.coords):
                        extent[axis] = max(extent[axis], c + 1)
        return tuple(extent)

    def validate(self) -> None:
        """Run cross-pipe validation: empty PipeNets, mixed pipe kinds,
        multicast destination overlap."""
        for net in self.pipe_nets:
            if not net.pipes:
                raise ValueError("PipeNet requires at least one pipe")
            _validate_homogeneous_pipe_kinds(net.pipes)
            _validate_no_overlapping_destinations(net.pipes)


def _validate_homogeneous_pipe_kinds(pipes: Tuple[PipeUse, ...]) -> None:
    # Spec: `ttl.PipeNet[DstT](pipes: List[ttl.Pipe[DstT]])`. The shared
    # type variable means every pipe in a PipeNet has the same destination
    # type — all unicast or all multicast.
    has_unicast = any(isinstance(p.dst, NodeCoord) for p in pipes)
    has_multicast = any(isinstance(p.dst, NodeRange) for p in pipes)
    if has_unicast and has_multicast:
        raise ValueError(
            "PipeNet may not mix unicast and multicast pipes "
            "(spec: PipeNet[DstT] requires all pipes to share DstT); "
            "use separate PipeNets."
        )


def _validate_no_overlapping_destinations(pipes: Tuple[PipeUse, ...]) -> None:
    """Reject two multicast pipes within one PipeNet that share any destination.

    All pipes in a PipeNet share a single semaphore pair, so a node that
    receives from multiple multicast sources cannot disambiguate the
    handshake. Unicast gather (multiple unicast pipes to the same dst) is
    allowed because the receiver uses cumulative semaphore waits.

    TODO[spec]: the spec does not constrain within-PipeNet multicast
    destination overlap; this rejection is an implementation constraint
    tied to issue #505. Can be lifted once the lowering switches to
    `noc_semaphore_inc_multicast`.
    """
    mcast = [(i, p) for i, p in enumerate(pipes) if isinstance(p.dst, NodeRange)]
    if len(mcast) < 2:
        return
    seen: dict = {}
    for i, pipe in mcast:
        rng: NodeRange = pipe.dst  # type: ignore[assignment]
        for coord in itertools.product(
            *(range(lo, hi) for lo, hi in zip(rng.lo, rng.hi))
        ):
            if coord in seen:
                j = seen[coord]
                raise ValueError(
                    f"PipeNet has overlapping multicast destinations: "
                    f"pipe {j} (src={pipes[j].src.coords}) and "
                    f"pipe {i} (src={pipe.src.coords}) both target "
                    f"node {coord}. Use separate PipeNets for patterns "
                    f"where a node receives from multiple multicast "
                    f"sources."
                )
            seen[coord] = i


__all__ = [
    "NodeCoord",
    "NodeRange",
    "PipeUse",
    "PipeNetUse",
    "OperationPipeNets",
]
