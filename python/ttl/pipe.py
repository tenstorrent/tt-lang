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
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Union

from .domains import DevicePoint, DeviceView, NodeSelection, TransferGraph

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
        self._device_graph = None
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
            self._device_graph = TransferGraph.edges(
                src.devices.domain,
                [(src.devices.reference, dst.devices.reference)],
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
            raise TypeError("pairwise requires complete node endpoint selections")
        if src.devices.domain != dst.devices.domain:
            raise ValueError("Pipe endpoints must share a parent device domain")
        if not isinstance(src.devices, DeviceView) or not isinstance(
            dst.devices, DeviceView
        ):
            raise TypeError("pairwise requires coordinate-structured device views")
        if src.devices.shape != dst.devices.shape:
            raise ValueError("pairwise device views must have equal extents")
        pipe = cls(src.node, dst.node)
        pipe._device_graph = TransferGraph.edges(
            src.devices.domain,
            zip(src.devices.iter_device_refs(), dst.devices.iter_device_refs()),
        )
        return pipe

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
        if self._device_graph is not None:
            return (
                "device-pipe",
                identity,
                self._device_graph._operation_identity_capture(),
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


@dataclass(frozen=True)
class PipeMapping:
    """Factorized device and launch-node relations for a graph PipeNet.

    Every logical device edge in ``graph`` is combined with every node-level
    pipe in ``pipes``. Multiple mappings in one PipeNet form an ordered union.
    """

    graph: "TransferGraph"
    pipes: Tuple[Pipe, ...]

    def __init__(self, *, graph: "TransferGraph", pipes: Iterable[Pipe]):
        from .domains import TransferGraph

        if not isinstance(graph, TransferGraph):
            raise TypeError(
                f"PipeMapping graph must be a TransferGraph, "
                f"got {type(graph).__name__}"
            )
        normalized_pipes = tuple(pipes)
        if not normalized_pipes:
            raise ValueError("PipeMapping requires at least one pipe")
        if any(not isinstance(pipe, Pipe) for pipe in normalized_pipes):
            raise TypeError("PipeMapping pipes must contain only Pipe values")
        if any(pipe.is_collective for pipe in normalized_pipes):
            raise ValueError(
                "graph PipeNet node pipes must be point-to-point; "
                "node collective destinations require graph multicast lowering"
            )
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "pipes", normalized_pipes)

    def _operation_identity_capture(self) -> tuple:
        """Return the graph and node-pipe relation used by operation caching."""
        return (
            "pipe-mapping",
            self.graph._operation_identity_capture(),
            tuple(pipe._operation_identity_capture() for pipe in self.pipes),
        )


class PipeNet:
    """
    A network of pipes for multi-core communication patterns.

    PipeNet groups multiple pipes and provides if_src/if_dst methods
    for conditional execution based on core coordinates.

    Active set: the union of every pipe's source coordinate and destination
    range. Cores outside the active set do not participate in pipe
    communication; under grid="full" or any explicit launch wider than the
    work extent, the user must guard pipe-coupled regions with
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
        graph: Logical-device transfer relation. Omitting ``pipes`` applies an
            identity node pipe to every launch node.
        mappings: Ordered graph and node-pipe relations. This form cannot be
            combined with top-level ``graph`` or ``pipes`` arguments.

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
        mappings: Optional[Iterable[PipeMapping]] = None,
    ):
        from ._pipenets import OperationPipeNets
        from .domains import TransferGraph

        normalized_pipes = tuple(pipes) if pipes is not None else None
        if normalized_pipes and any(
            isinstance(pipe, Pipe) and pipe._device_graph is not None
            for pipe in normalized_pipes
        ):
            if graph is not None or mappings is not None:
                raise ValueError(
                    "complete Pipe endpoints cannot be combined with graph or mappings"
                )
            if any(
                not isinstance(pipe, Pipe) or pipe._device_graph is None
                for pipe in normalized_pipes
            ):
                raise ValueError(
                    "PipeNet cannot mix complete endpoints with device-relative pipes"
                )
            mappings = tuple(
                PipeMapping(
                    graph=pipe._device_graph,
                    pipes=[Pipe(src=pipe.src, dst=pipe.dst)],
                )
                for pipe in normalized_pipes
            )
            normalized_pipes = None
            pipes = None
        if mappings is not None:
            if graph is not None or pipes is not None:
                raise ValueError(
                    "PipeNet mappings cannot be combined with graph or pipes"
                )
            normalized_mappings = tuple(mappings)
            if not normalized_mappings:
                raise ValueError("PipeNet mappings require at least one mapping")
            if any(
                not isinstance(mapping, PipeMapping) for mapping in normalized_mappings
            ):
                raise TypeError("PipeNet mappings must contain only PipeMapping values")
        elif graph is not None:
            if not isinstance(graph, TransferGraph):
                raise TypeError(
                    f"PipeNet graph must be a TransferGraph, "
                    f"got {type(graph).__name__}"
                )
            normalized_mappings = (
                (PipeMapping(graph=graph, pipes=normalized_pipes),)
                if normalized_pipes is not None
                else ()
            )
        else:
            normalized_mappings = ()
            if normalized_pipes is None:
                raise ValueError("PipeNet requires pipes, graph, or mappings")
        # Operation-local id assigned by the OperationPipeNets builder
        # before AST emission (see _build_operation_pipenets).
        self.pipe_net_id = 0
        self.pipes: List[Pipe] = []
        self.graph: Optional["TransferGraph"] = None
        self.mappings: Tuple[PipeMapping, ...] = ()
        self._uses_grid_identity = graph is not None and pipes is None
        if mappings is not None:
            self.mappings = normalized_mappings
            if len(normalized_mappings) == 1:
                self.graph = normalized_mappings[0].graph
                self.pipes = list(normalized_mappings[0].pipes)
        elif graph is not None:
            self.graph = graph
            self.mappings = normalized_mappings
            if normalized_mappings:
                self.pipes = list(normalized_mappings[0].pipes)
        else:
            assert normalized_pipes is not None
            if not normalized_pipes:
                raise ValueError("PipeNet requires at least one pipe")
            self.pipes = list(normalized_pipes)

        validation_graph = OperationPipeNets()
        if self._uses_grid_identity:
            assert self.graph is not None
            validation_graph.add_graph_pipe_net(
                ((self.graph, None),), uses_grid_identity=True
            )
        elif self.is_graph:
            validation_graph.add_graph_pipe_net(
                (
                    (
                        mapping.graph,
                        tuple(_pipe_to_pipe_use(pipe) for pipe in mapping.pipes),
                    )
                    for mapping in self.mappings
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
        return self.graph is not None or bool(self.mappings)

    def _operation_identity_capture(self) -> tuple:
        if not self.is_graph:
            return (
                "pipenet",
                tuple(pipe._operation_identity_capture() for pipe in self.pipes),
            )
        if self._uses_grid_identity:
            assert self.graph is not None
            return (
                "graph-pipenet-grid-identity",
                self.graph._operation_identity_capture(),
            )
        return (
            "graph-pipenet-mappings",
            tuple(mapping._operation_identity_capture() for mapping in self.mappings),
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
