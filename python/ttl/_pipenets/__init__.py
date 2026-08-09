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
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Set, Tuple, Union

from ttl.domains import DeviceRange


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


@dataclass(frozen=True)
class GraphPipeNetUse:
    """One graph-based PipeNet consumed by one operation invocation."""

    pipe_net_id: int
    edges: Tuple[Any, ...]
    transfer_graph: Any


@dataclass
class OperationPipeNets:
    """All PipeNets used by one operation invocation."""

    pipe_nets: List[PipeNetUse] = field(default_factory=list)
    graph_pipe_nets: List[GraphPipeNetUse] = field(default_factory=list)
    _next_id: int = 0

    def _next_pipe_net_id(self) -> int:
        pipe_net_id = self._next_id
        self._next_id += 1
        return pipe_net_id

    def add_pipe_net(self, pipes: Iterable[PipeUse]) -> PipeNetUse:
        """Append a new PipeNetUse with the next operation-local id."""
        use = PipeNetUse(id=self._next_pipe_net_id(), pipes=tuple(pipes))
        self.pipe_nets.append(use)
        return use

    def add_graph_pipe_net(self, transfer_graph: Any) -> GraphPipeNetUse:
        """Append a graph PipeNet with one ordered device-edge record set."""
        edges = tuple(transfer_graph.iter_edges())
        if not edges:
            raise ValueError("graph-based PipeNet requires at least one transfer edge")
        if any(isinstance(edge.destination, DeviceRange) for edge in edges):
            raise ValueError(
                "graph-based PipeNet DeviceRange destinations require multicast "
                "transport lowering"
            )
        use = GraphPipeNetUse(
            pipe_net_id=self._next_pipe_net_id(),
            edges=edges,
            transfer_graph=transfer_graph,
        )
        self.graph_pipe_nets.append(use)
        return use

    def active_node_set(self, grid: Tuple[int, ...]) -> Optional[Set[int]]:
        """Linearized active node set across every PipeNet in the graph.

        Returns None when the graph is empty, signaling that no active-set
        filtering should be applied (every node participates).
        """
        if self.graph_pipe_nets or not self.pipe_nets:
            return None
        active: Set[int] = set()
        for net in self.pipe_nets:
            for pipe in net.pipes:
                active.add(_linearize(pipe.src.coords, grid))
                for coord in _expand_dst(pipe.dst):
                    active.add(_linearize(coord, grid))
        return active

    def validate(self) -> None:
        """Run cross-pipe validation: empty PipeNets, mixed pipe kinds,
        consistent coord rank across the graph."""
        for net in self.pipe_nets:
            if not net.pipes:
                raise ValueError("PipeNet requires at least one pipe")
            _validate_no_mixed_kinds(net.pipes)
        _validate_consistent_coord_rank(self.pipe_nets)

    def resolve_device_domain(self, operation_domain: Any) -> Any:
        """Reconcile graph transfer domains with the operation domain."""
        if not self.graph_pipe_nets:
            return operation_domain

        graph_domain = self.graph_pipe_nets[0].transfer_graph.domain
        for graph_pipe_net in self.graph_pipe_nets[1:]:
            if graph_pipe_net.transfer_graph.domain != graph_domain:
                raise ValueError(
                    "graph-based PipeNets in one operation must use the same "
                    "DeviceDomain"
                )

        if operation_domain is None:
            return graph_domain
        if operation_domain != graph_domain:
            raise ValueError(
                "operation device_domain must match the DeviceDomain used by "
                "its graph-based PipeNets"
            )
        return operation_domain


def _linearize(coords: Tuple[int, ...], grid: Tuple[int, ...]) -> int:
    """Row-major linearization matching sim's flatten_node_index.

    A 1D coord on a 2D grid is treated as an already-linear node index
    (see `flatten_node_index`): the loop body uses `grid[i]` only for the
    dims the coord actually has.
    """
    if len(coords) > len(grid):
        raise ValueError(
            f"coord rank {len(coords)} exceeds grid rank {len(grid)}: "
            f"coords={coords}, grid={grid}"
        )
    linear = coords[0]
    for i in range(1, len(coords)):
        linear = linear * grid[i] + coords[i]
    return linear


def _expand_dst(dst: Union[NodeCoord, NodeRange]) -> Iterable[Tuple[int, ...]]:
    """Yield each node coordinate covered by a pipe destination."""
    if isinstance(dst, NodeCoord):
        yield dst.coords
        return
    yield from itertools.product(*(range(lo, hi) for lo, hi in zip(dst.lo, dst.hi)))


def _validate_consistent_coord_rank(pipe_nets: List[PipeNetUse]) -> None:
    # _linearize treats a rank-1 coord as already-linear (matching sim's
    # `flatten_node_index`), so mixing rank-1 and rank-2 srcs/dsts in one
    # graph would alias distinct nodes in `active_node_set`. Force a
    # single rank across the whole graph to make that aliasing impossible.
    ranks: Set[int] = set()
    for net in pipe_nets:
        for pipe in net.pipes:
            ranks.add(len(pipe.src.coords))
            if isinstance(pipe.dst, NodeRange):
                ranks.add(len(pipe.dst.lo))
            else:
                ranks.add(len(pipe.dst.coords))
    if len(ranks) > 1:
        raise ValueError(
            f"pipe coordinate ranks must be consistent across the graph, "
            f"got {sorted(ranks)}"
        )


def _validate_no_mixed_kinds(pipes: Tuple[PipeUse, ...]) -> None:
    # Spec: `ttl.PipeNet[DstT](pipes: List[ttl.Pipe[DstT]])`. The shared
    # type variable means every pipe in a PipeNet has the same destination
    # type, which also fixes the current transfer contract.
    has_point_to_point = any(isinstance(p.dst, NodeCoord) for p in pipes)
    has_collective = any(isinstance(p.dst, NodeRange) for p in pipes)
    if has_point_to_point and has_collective:
        raise ValueError(
            "PipeNet may not mix point-to-point and collective pipes "
            "(spec: PipeNet[DstT] requires all pipes to share DstT); "
            "use separate PipeNets."
        )


__all__ = [
    "NodeCoord",
    "NodeRange",
    "PipeUse",
    "PipeNetUse",
    "GraphPipeNetUse",
    "OperationPipeNets",
]
