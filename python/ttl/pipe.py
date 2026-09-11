# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pipe operations for core-to-core data transfer.

This module provides Python classes for the Pipe and PipeNet abstractions
as defined in the TT-Lang specification. PipeNet callbacks lower through
`ttl.pipenet_foreach_src` and `ttl.pipenet_foreach_dst` to TTKernel.

PipeNet supports the spec's callback API:
    net.if_src(lambda pipe: ttl.copy(blk, pipe))
    net.if_dst(lambda pipe: ttl.copy(pipe, blk))
"""

import inspect
import warnings
from typing import Callable, Iterable, List, Optional, Tuple, Union

from .domains import (
    DeviceDomain,
    DevicePoint,
    DeviceRef,
    DeviceView,
    NodeSelection,
    TransferGraph,
)

# Type aliases matching the spec
CoreCoord = Tuple[int, int]
CoreRange = Tuple[Union[int, slice], Union[int, slice]]


class SrcPipeIdentity:
    """
    Pipe identity for source-side callbacks.

    Passed to if_src callbacks to provide access to destination info.
    Used with ttl.copy(block, pipe) to send data.
    """

    def __init__(self, pipe: "Pipe"):
        self._pipe = pipe

    @property
    def dst(self) -> Union[CoreCoord, Tuple[CoreCoord, CoreCoord]]:
        """Get destination: coord for point-to-point, range for collective."""
        if self._pipe.is_point_to_point:
            return self._pipe.dst_start
        return (self._pipe.dst_start, self._pipe.dst_end)

    @property
    def destination_device_index(self) -> int:
        """Return the destination's row-major order in the device domain."""
        from .domains import DeviceRange

        if not hasattr(self._pipe, "_device_edge"):
            raise ValueError(
                "destination_device_index is available only for graph-based PipeNets"
            )
        destination = self._pipe._device_edge.destination
        if isinstance(destination, DeviceRange):
            raise ValueError(
                "destination_device_index is unavailable for device ranges"
            )
        return self._pipe._device_domain.index_order(destination)


class DstPipeIdentity:
    """
    Pipe identity for destination-side callbacks.

    Passed to if_dst callbacks to provide access to source info.
    Used with ttl.copy(pipe, block) to receive data.
    """

    def __init__(self, pipe: "Pipe"):
        self._pipe = pipe

    @property
    def src(self) -> CoreCoord:
        """Get source core coordinate."""
        return self._pipe.src

    @property
    def source_device_index(self) -> int:
        """Return the source's row-major order in the device domain."""
        if not hasattr(self._pipe, "_device_edge"):
            raise ValueError(
                "source_device_index is available only for graph-based PipeNets"
            )
        return self._pipe._device_domain.index_order(self._pipe._device_edge.source)


class Pipe:
    """
    A pipe for core-to-core data transfer.

    A pipe defines a communication edge from a source core to one or more
    destination cores. When dst is a single coordinate, the transfer contract
    is point-to-point. When dst is a range, the transfer contract is
    collective.

    Args:
        src: Source core coordinate (x, y)
        dst: Destination - either CoreCoord for point-to-point or CoreRange for
            collective

    Example:
        # Point-to-point transfer from (0, 0) to (1, 0)
        pipe = ttl.Pipe(src=(0, 0), dst=(1, 0))

        # Collective transfer from (0, 0) to column 1, rows 0-3
        pipe = ttl.Pipe(src=(0, 0), dst=(1, slice(0, 4)))
    """

    def __init__(
        self,
        src: Union[CoreCoord, NodeSelection],
        dst: Union[CoreCoord, CoreRange, NodeSelection],
    ):
        self._device_graphs = ()
        if isinstance(src, NodeSelection) or isinstance(dst, NodeSelection):
            if not isinstance(src, NodeSelection) or not isinstance(dst, NodeSelection):
                raise TypeError("both Pipe endpoints must specify devices and nodes")
            if src.devices.domain != dst.devices.domain:
                raise ValueError("Pipe endpoints must share a parent device domain")
            if not isinstance(src.devices, DevicePoint) or not isinstance(
                dst.devices, DevicePoint
            ):
                raise ValueError(
                    "Pipe requires point endpoints; use a relation constructor for device selections"
                )
            self._device_graphs = self._partition_edges_by_transport(
                src.devices.domain,
                ((src.devices.reference, dst.devices.reference),),
            )
            src, dst = src.node, dst.node
        if len(src) != 2:
            raise ValueError(f"src must be a 2-tuple, got {src}")

        self.src = src
        self.dst = dst
        # Operation-local id assigned by the OperationPipeNets builder
        # before AST emission (see _build_operation_pipenets).
        self.pipe_net_id = 0
        self._parse_dst()

    @classmethod
    def pairwise(cls, *, src: NodeSelection, dst: NodeSelection) -> "Pipe":
        """Connect equal view coordinates using the specified source/destination nodes."""
        if not isinstance(src, NodeSelection) or not isinstance(dst, NodeSelection):
            raise TypeError("pairwise requires device and node selections")
        if src.devices.domain != dst.devices.domain:
            raise ValueError("Pipe endpoints must share a parent device domain")
        if not isinstance(src.devices, DeviceView) or not isinstance(
            dst.devices, DeviceView
        ):
            raise TypeError("pairwise requires coordinate-structured device views")
        if src.devices.shape != dst.devices.shape:
            raise ValueError("pairwise device views must have equal extents")
        pipe = cls(src.node, dst.node)
        pipe._device_graphs = cls._partition_edges_by_transport(
            src.devices.domain,
            zip(src.devices.iter_device_refs(), dst.devices.iter_device_refs()),
        )
        return pipe

    @classmethod
    def all_to_all(
        cls,
        *,
        src: NodeSelection,
        dst: NodeSelection,
        include_self: bool = False,
    ) -> "Pipe":
        """Connect every selected source device to every selected destination."""
        if not isinstance(src, NodeSelection) or not isinstance(dst, NodeSelection):
            raise TypeError("all_to_all requires device and node selections")
        if src.devices.domain != dst.devices.domain:
            raise ValueError("Pipe endpoints must share a parent device domain")
        if not isinstance(include_self, bool):
            raise TypeError("include_self must be a boolean")
        edges = (
            (source, destination)
            for source in src.devices.iter_device_refs()
            for destination in dst.devices.iter_device_refs()
            if include_self or source != destination
        )
        pipe = cls(src.node, dst.node)
        pipe._device_graphs = cls._partition_edges_by_transport(
            src.devices.domain, edges
        )
        return pipe

    @staticmethod
    def _partition_edges_by_transport(
        domain: DeviceDomain,
        edges: Iterable[Tuple[DeviceRef, DeviceRef]],
    ) -> Tuple[TransferGraph, ...]:
        """Partition ``edges`` so each graph uses one synchronization protocol."""
        local_edges = []
        remote_edges = []
        for source, destination in edges:
            target = local_edges if source == destination else remote_edges
            target.append((source, destination))
        graphs = []
        if local_edges:
            graphs.append(TransferGraph.edges(domain, local_edges))
        if remote_edges:
            graphs.append(TransferGraph.edges(domain, remote_edges))
        if not graphs:
            raise ValueError("Pipe relation must contain at least one transfer")
        return tuple(graphs)

    @staticmethod
    def _validate_slice(s: slice, name: str):
        """Validate a slice has explicit int start and stop with start < stop."""
        if s.start is None or s.stop is None:
            raise ValueError(
                f"dst {name} slice must have explicit start and stop, "
                f"got slice({s.start}, {s.stop})"
            )
        if not isinstance(s.start, int) or not isinstance(s.stop, int):
            raise ValueError(
                f"dst {name} slice bounds must be integers, "
                f"got slice({s.start}, {s.stop})"
            )
        if s.start >= s.stop:
            raise ValueError(
                f"dst {name} slice start must be < stop, "
                f"got slice({s.start}, {s.stop})"
            )
        if s.step is not None and s.step != 1:
            raise ValueError(
                f"dst {name} slice step must be 1 or None "
                f"(strided collective destinations are not supported), got step={s.step}"
            )

    def _parse_dst(self):
        """Parse destination into start/end coordinates."""
        dst = self.dst

        if isinstance(dst, tuple) and len(dst) == 2:
            x, y = dst
            if isinstance(x, int) and isinstance(y, int):
                self.dst_start = (x, y)
                self.dst_end = (x, y)
                self._is_collective = False
            elif isinstance(x, int) and isinstance(y, slice):
                self._validate_slice(y, "y")
                self.dst_start = (x, y.start)
                self.dst_end = (x, y.stop - 1)
                self._is_collective = True
            elif isinstance(x, slice) and isinstance(y, int):
                self._validate_slice(x, "x")
                self.dst_start = (x.start, y)
                self.dst_end = (x.stop - 1, y)
                self._is_collective = True
            elif isinstance(x, slice) and isinstance(y, slice):
                self._validate_slice(x, "x")
                self._validate_slice(y, "y")
                self.dst_start = (x.start, y.start)
                self.dst_end = (x.stop - 1, y.stop - 1)
                self._is_collective = True
            else:
                raise ValueError(f"Invalid dst format: {dst}")
        else:
            raise ValueError(f"dst must be a 2-tuple, got {dst}")

    @property
    def is_point_to_point(self) -> bool:
        return not self._is_collective

    @property
    def is_collective(self) -> bool:
        return self._is_collective

    @property
    def is_unicast(self) -> bool:
        warnings.warn(
            "Pipe.is_unicast is deprecated; use Pipe.is_point_to_point",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_point_to_point

    @property
    def is_multicast(self) -> bool:
        warnings.warn(
            "Pipe.is_multicast is deprecated; use Pipe.is_collective",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_collective

    def _operation_identity_capture(self) -> tuple:
        identity = (
            "pipe",
            tuple(self.src),
            tuple(self.dst_start),
            tuple(self.dst_end),
            self.is_collective,
        )
        if self._device_graphs:
            return (
                "device-pipe",
                identity,
                tuple(
                    graph._operation_identity_capture() for graph in self._device_graphs
                ),
            )
        return identity


def _pipe_to_pipe_use(pipe: Pipe):
    """Convert a ttl.Pipe to a PipeUse for OperationPipeNets validation/build."""
    from ._pipenets import NodeCoord, NodeRange, PipeUse

    src = NodeCoord(coords=tuple(pipe.src))
    if pipe.is_point_to_point:
        dst = NodeCoord(coords=tuple(pipe.dst_start))
    else:
        dst = NodeRange(
            lo=(pipe.dst_start[0], pipe.dst_start[1]),
            hi=(pipe.dst_end[0] + 1, pipe.dst_end[1] + 1),
        )
    return PipeUse(src=src, dst=dst)


class PipeNet:
    """
    A local or multi-device communication relation.

    A local PipeNet contains node-level pipes. A graph PipeNet combines every
    logical-device edge with every corresponding node-level pipe. ``if_src``
    and ``if_dst`` execute once for each matching transfer.

    The launch-node active set is the union of every node pipe's source
    coordinate and destination range. Nodes outside the active set do not
    participate in pipe communication; under grid="full" or any explicit
    launch wider than the work extent, the program must guard pipe-coupled
    regions with
    `if net.is_src()`, `if net.is_dst()`, or `if net.is_active()` so the
    `ttl-verify-pipenet-guards` pass accepts the program. Pipe coordinates
    should be sized from the operation's work extent, not the launch extent.

    A PipeNet's pipes must all have the same transfer contract: all
    point-to-point or all collective. The two contracts use the PipeNet's
    synchronization state with incompatible semantics; mixing them in one
    PipeNet can race when the same node participates in both. Use separate
    PipeNets.

    Args:
        pipes: Ordered node-level pipes. Without ``graph``, these define a
            local PipeNet. With ``graph``, each graph edge uses every pipe.
        graph: Logical-device transfer relation. When ``pipes`` is omitted,
            every graph edge connects matching source and destination node
            coordinates throughout the launch grid.

    Example:
        # Gather pattern from work extent ROWS x COLS:
        net = ttl.PipeNet([
            ttl.Pipe(src=(x, y), dst=(0, y))
            for x in range(1, COLS)
            for y in range(ROWS)
        ])

        # In datamovement thread:
        net.if_src(lambda pipe: ttl.copy(blk, pipe).wait())
        net.if_dst(lambda pipe: ttl.copy(pipe, blk).wait())
    """

    def __init__(
        self,
        pipes: Optional[Iterable[Pipe]] = None,
        *,
        graph: Optional["TransferGraph"] = None,
    ):
        from ._pipenets import OperationPipeNets
        from .domains import TransferGraph

        normalized_pipes = tuple(pipes) if pipes is not None else None
        if normalized_pipes is not None:
            if not normalized_pipes:
                raise ValueError("PipeNet requires at least one pipe")
            if any(not isinstance(pipe, Pipe) for pipe in normalized_pipes):
                raise TypeError("PipeNet pipes must contain only Pipe values")
        if normalized_pipes and any(pipe._device_graphs for pipe in normalized_pipes):
            if graph is not None:
                raise ValueError(
                    "device-selected Pipe endpoints cannot be combined with a "
                    "PipeNet graph"
                )
            if any(not pipe._device_graphs for pipe in normalized_pipes):
                raise ValueError(
                    "PipeNet cannot mix device-selected Pipes with node-only Pipes"
                )
            selected_pipe_identities = [
                pipe._operation_identity_capture() for pipe in normalized_pipes
            ]
            if len(set(selected_pipe_identities)) != len(selected_pipe_identities):
                raise ValueError("PipeNet contains a duplicate device-selected Pipe")
            grouped_relations: List[Tuple[TransferGraph, List[Pipe]]] = []
            for pipe in normalized_pipes:
                for device_graph in pipe._device_graphs:
                    relative_pipe = Pipe(src=pipe.src, dst=pipe.dst)
                    combine_with_previous = (
                        bool(grouped_relations)
                        and grouped_relations[-1][0] == device_graph
                        and device_graph.explicit_edge_count == 1
                    )
                    if combine_with_previous:
                        grouped_relations[-1][1].append(relative_pipe)
                    else:
                        # Combining a multi-edge graph would change callbacks from
                        # Pipe declaration order to graph-edge order.
                        grouped_relations.append((device_graph, [relative_pipe]))
            normalized_relations = tuple(
                (device_graph, tuple(relation_pipes))
                for device_graph, relation_pipes in grouped_relations
            )
            device_selected_pipes = normalized_pipes
        elif graph is not None:
            if not isinstance(graph, TransferGraph):
                raise TypeError(
                    f"PipeNet graph must be a TransferGraph, "
                    f"got {type(graph).__name__}"
                )
            if normalized_pipes is not None and any(
                pipe.is_collective for pipe in normalized_pipes
            ):
                raise ValueError(
                    "graph PipeNet node pipes must be point-to-point; "
                    "node collective destinations require graph multicast lowering"
                )
            normalized_relations = (
                ((graph, normalized_pipes),) if normalized_pipes is not None else ()
            )
            device_selected_pipes = ()
        else:
            normalized_relations = ()
            device_selected_pipes = ()
            if normalized_pipes is None:
                raise ValueError("PipeNet requires pipes or graph")
        # Operation-local id assigned by the OperationPipeNets builder
        # before AST emission (see _build_operation_pipenets).
        self.pipe_net_id = 0
        self.pipes: List[Pipe] = []
        self.graph: Optional["TransferGraph"] = None
        self._device_relations: Tuple[Tuple["TransferGraph", Tuple[Pipe, ...]], ...] = (
            normalized_relations
        )
        self._uses_matching_node_coordinates = graph is not None and pipes is None
        if device_selected_pipes:
            self.pipes = list(device_selected_pipes)
        elif graph is not None:
            self.graph = graph
            if normalized_pipes is not None:
                self.pipes = list(normalized_pipes)
        else:
            assert normalized_pipes is not None
            self.pipes = list(normalized_pipes)

        validation_graph = OperationPipeNets()
        if self._uses_matching_node_coordinates:
            assert self.graph is not None
            validation_graph.add_graph_pipe_net(
                ((self.graph, None),), uses_matching_node_coordinates=True
            )
        elif self.is_graph:
            validation_graph.add_graph_pipe_net(
                (
                    (
                        relation_graph,
                        tuple(_pipe_to_pipe_use(pipe) for pipe in relation_pipes),
                    )
                    for relation_graph, relation_pipes in self._device_relations
                )
            )
        else:
            validation_graph.add_pipe_net(
                _pipe_to_pipe_use(pipe) for pipe in self.pipes
            )
        validation_graph.validate()

        # Preserve the user's call site so diagnostics identify the PipeNet
        # declaration instead of frontend implementation code.
        self._source_file: Optional[str] = None
        self._source_line: Optional[int] = None
        try:
            frame = inspect.stack()[1]
            self._source_file = frame.filename
            self._source_line = frame.lineno
        except (IndexError, AttributeError):
            pass

    @property
    def is_graph(self) -> bool:
        return self.graph is not None or bool(self._device_relations)

    def _operation_identity_capture(self) -> tuple:
        if not self.is_graph:
            return (
                "pipenet",
                tuple(pipe._operation_identity_capture() for pipe in self.pipes),
            )
        if self._uses_matching_node_coordinates:
            assert self.graph is not None
            return (
                "graph-pipenet-matching-node-coordinates",
                self.graph._operation_identity_capture(),
            )
        return (
            "graph-pipenet-relations",
            tuple(
                (
                    graph._operation_identity_capture(),
                    tuple(pipe._operation_identity_capture() for pipe in pipes),
                )
                for graph, pipes in self._device_relations
            ),
        )

    def if_src(self, callback: Callable[["SrcPipeIdentity"], None]) -> None:
        """
        Execute callback for each pipe where current core is source.

        The frontend compiles the callback once into a PipeNet region. The
        generated kernel executes it once per matching PipeNet record, in
        construction order.

        Args:
            callback: Function taking SrcPipeIdentity, called for matching pipes

        Note:
            This method should only be called inside a @ttl.datamovement thread.
            The callback body is compiled once and executes on the device.
        """
        # This is a marker method. The actual implementation is in ttl_ast.py
        # which detects calls to this method and handles them specially.
        raise RuntimeError(
            "PipeNet.if_src() should only be called inside a TTL kernel. "
            "The compiler handles this method specially."
        )

    def if_dst(self, callback: Callable[["DstPipeIdentity"], None]) -> None:
        """
        Execute callback for each pipe where current core is destination.

        The frontend compiles the callback once into a PipeNet region. The
        generated kernel executes it once per matching PipeNet record, in
        construction order.

        Args:
            callback: Function taking DstPipeIdentity, called for matching pipes

        Note:
            This method should only be called inside a @ttl.datamovement thread.
            The callback body is compiled once and executes on the device.
        """
        # This is a marker method. The actual implementation is in ttl_ast.py
        # which detects calls to this method and handles them specially.
        raise RuntimeError(
            "PipeNet.if_dst() should only be called inside a TTL kernel. "
            "The compiler handles this method specially."
        )

    def is_src(self) -> bool:
        """Return whether the current node is a source of any pipe in this PipeNet.

        The compiler recognizes the result as a source-role condition when it
        verifies guarded PipeNet traffic.
        """
        raise RuntimeError(
            "PipeNet.is_src() should only be called inside a TTL kernel. "
            "The compiler handles this method specially."
        )

    def is_dst(self) -> bool:
        """Return whether the current node is in a destination range in this PipeNet."""
        raise RuntimeError(
            "PipeNet.is_dst() should only be called inside a TTL kernel. "
            "The compiler handles this method specially."
        )

    def destination_count(self) -> int:
        """Return the number of records targeting the current node.

        The result equals the number of times ``if_dst`` executes its callback
        on the current node.
        """
        raise RuntimeError(
            "PipeNet.destination_count() should only be called inside a TTL "
            "kernel. The compiler handles this method specially."
        )

    def is_active(self) -> bool:
        """Return whether this node is a source or destination in this PipeNet."""
        raise RuntimeError(
            "PipeNet.is_active() should only be called inside a TTL kernel. "
            "The compiler handles this method specially."
        )
