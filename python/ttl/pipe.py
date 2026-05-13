# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pipe operations for core-to-core data transfer.

This module provides Python classes for the Pipe and PipeNet abstractions
as defined in the TT-Lang specification. The MLIR ops (ttl.create_pipe,
ttl.if_src, ttl.if_dst) are implemented and lower to TTKernel.

PipeNet supports the spec's callback API:
    net.if_src(lambda pipe: ttl.copy(blk, pipe))
    net.if_dst(lambda pipe: ttl.copy(pipe, blk))
"""

import inspect
from typing import Callable, List, Optional, Tuple, Union

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
        """Get destination: single coord for unicast, (start, end) for multicast."""
        if self._pipe.is_unicast:
            return self._pipe.dst_start
        return (self._pipe.dst_start, self._pipe.dst_end)


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


class Pipe:
    """
    A pipe for core-to-core data transfer.

    A pipe defines a communication channel from a source core to one or more
    destination cores. When dst is a single coordinate, it's unicast.
    When dst is a range (using slices), it's multicast.

    Args:
        src: Source core coordinate (x, y)
        dst: Destination - either CoreCoord for unicast or CoreRange for multicast

    Example:
        # Unicast from (0, 0) to (1, 0)
        pipe = ttl.Pipe(src=(0, 0), dst=(1, 0))

        # Multicast from (0, 0) to column 1, rows 0-3
        pipe = ttl.Pipe(src=(0, 0), dst=(1, slice(0, 4)))
    """

    def __init__(self, src: CoreCoord, dst: Union[CoreCoord, CoreRange]):
        if len(src) != 2:
            raise ValueError(f"src must be a 2-tuple, got {src}")

        self.src = src
        self.dst = dst
        # Operation-local id assigned by the OperationPipeNets builder
        # before AST emission (see _build_operation_pipenets).
        self.pipe_net_id = 0
        self._parse_dst()

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
                f"(strided multicast is not supported), got step={s.step}"
            )

    def _parse_dst(self):
        """Parse destination into start/end coordinates."""
        dst = self.dst

        if isinstance(dst, tuple) and len(dst) == 2:
            x, y = dst
            if isinstance(x, int) and isinstance(y, int):
                # Unicast: dst is (x, y)
                self.dst_start = (x, y)
                self.dst_end = (x, y)
                self._is_multicast = False
            elif isinstance(x, int) and isinstance(y, slice):
                self._validate_slice(y, "y")
                self.dst_start = (x, y.start)
                self.dst_end = (x, y.stop - 1)
                self._is_multicast = True
            elif isinstance(x, slice) and isinstance(y, int):
                self._validate_slice(x, "x")
                self.dst_start = (x.start, y)
                self.dst_end = (x.stop - 1, y)
                self._is_multicast = True
            elif isinstance(x, slice) and isinstance(y, slice):
                self._validate_slice(x, "x")
                self._validate_slice(y, "y")
                self.dst_start = (x.start, y.start)
                self.dst_end = (x.stop - 1, y.stop - 1)
                self._is_multicast = True
            else:
                raise ValueError(f"Invalid dst format: {dst}")
        else:
            raise ValueError(f"dst must be a 2-tuple, got {dst}")

    @property
    def is_unicast(self) -> bool:
        return not self._is_multicast

    @property
    def is_multicast(self) -> bool:
        return self._is_multicast


def _pipe_to_pipe_use(pipe: Pipe):
    """Convert a ttl.Pipe to a PipeUse for OperationPipeNets validation/build."""
    from ._pipenets import NodeCoord, NodeRange, PipeUse

    src = NodeCoord(coords=tuple(pipe.src))
    if pipe.is_unicast:
        dst = NodeCoord(coords=tuple(pipe.dst_start))
    else:
        dst = NodeRange(
            lo=(pipe.dst_start[0], pipe.dst_start[1]),
            hi=(pipe.dst_end[0] + 1, pipe.dst_end[1] + 1),
        )
    return PipeUse(src=src, dst=dst)


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

    A PipeNet's pipes must all be the same kind (all unicast or all
    multicast). The TTKernel lowering allocates one semaphore pair per
    PipeNet, and the unicast and multicast handshakes use the pair's
    bits with incompatible semantics; mixing them in one PipeNet races
    when the same node participates in both. Use separate PipeNets.

    Args:
        pipes: List of Pipe objects defining the network

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

    def __init__(self, pipes: List[Pipe]):
        # Validate at construction time by building a one-net graph and
        # delegating to OperationPipeNets.validate(). Single source of
        # truth for empty/overlap/mixed-kind rules; the same graph is
        # rebuilt and re-validated at operation build time.
        from ._pipenets import OperationPipeNets

        if not pipes:
            raise ValueError("PipeNet requires at least one pipe")
        graph = OperationPipeNets()
        graph.add_pipe_net(_pipe_to_pipe_use(p) for p in pipes)
        graph.validate()
        # Operation-local id assigned by the OperationPipeNets builder
        # before AST emission (see _build_operation_pipenets).
        self.pipe_net_id = 0
        self.pipes = pipes
        # Capture the user's call site so `ttl.create_pipe` ops can carry
        # the declaration location.
        self._source_file: Optional[str] = None
        self._source_line: Optional[int] = None
        try:
            frame = inspect.stack()[1]
            self._source_file = frame.filename
            self._source_line = frame.lineno
        except (IndexError, AttributeError):
            pass

    def if_src(self, callback: Callable[["SrcPipeIdentity"], None]) -> None:
        """
        Execute callback for each pipe where current core is source.

        This method is compiled specially by the TTL compiler. At compile time,
        it iterates over all pipes and emits conditional blocks for each pipe
        where the current core matches the source coordinates.

        Args:
            callback: Function taking SrcPipeIdentity, called for matching pipes

        Note:
            This method should only be called inside a @ttl.datamovement thread.
            The callback is invoked at compile time, not runtime.
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

        This method is compiled specially by the TTL compiler. At compile time,
        it iterates over all pipes and emits conditional blocks for each pipe
        where the current core falls within the destination range.

        Args:
            callback: Function taking DstPipeIdentity, called for matching pipes

        Note:
            This method should only be called inside a @ttl.datamovement thread.
            The callback is invoked at compile time, not runtime.
        """
        # This is a marker method. The actual implementation is in ttl_ast.py
        # which detects calls to this method and handles them specially.
        raise RuntimeError(
            "PipeNet.if_dst() should only be called inside a TTL kernel. "
            "The compiler handles this method specially."
        )

    def is_src(self) -> bool:
        """Boolean predicate: current node is a source of any pipe in this net.

        Lowers to `ttl.is_src` and is recognized structurally by the
        `ttl-verify-pipenet-guards` pass, so it can be used as the condition
        of an `if` to gate pipe-coupled work."""
        raise RuntimeError(
            "PipeNet.is_src() should only be called inside a TTL kernel. "
            "The compiler handles this method specially."
        )

    def is_dst(self) -> bool:
        """Boolean predicate: current node is a destination of any pipe in
        this net. Lowers to `ttl.is_dst`."""
        raise RuntimeError(
            "PipeNet.is_dst() should only be called inside a TTL kernel. "
            "The compiler handles this method specially."
        )

    def is_active(self) -> bool:
        """Boolean predicate: current node is either a source or a
        destination of any pipe in this net. Lowers to `ttl.is_active`."""
        raise RuntimeError(
            "PipeNet.is_active() should only be called inside a TTL kernel. "
            "The compiler handles this method specially."
        )
