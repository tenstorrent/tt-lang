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

from typing import Callable, List, Tuple, Union

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


class PipeNet:
    """
    A network of pipes for multi-core communication patterns.

    PipeNet groups multiple pipes and provides if_src/if_dst methods
    for conditional execution based on core coordinates.

    Limitation: overlapping multicast destinations (a core receiving
    from multiple multicast sources) within a single PipeNet are not
    yet supported. This will be fixed once noc_semaphore_inc_multicast
    is available in the TTKernel dialect. See:
    https://github.com/tenstorrent/tt-lang/issues/505

    Args:
        pipes: List of Pipe objects defining the network

    Example:
        # Gather pattern: all cores send to (0, y)
        net = ttl.PipeNet([
            ttl.Pipe(src=(x, y), dst=(0, y))
            for x in range(1, grid_x)
            for y in range(grid_y)
        ])

        # In datamovement thread:
        net.if_src(lambda pipe: ttl.copy(blk, pipe).wait())
        net.if_dst(lambda pipe: ttl.copy(pipe, blk).wait())
    """

    _next_id = 0

    def __init__(self, pipes: List[Pipe]):
        if not pipes:
            raise ValueError("PipeNet requires at least one pipe")
        self._validate_no_overlapping_destinations(pipes)
        self.pipe_net_id = PipeNet._next_id
        PipeNet._next_id += 1
        self.pipes = pipes
        for pipe in self.pipes:
            pipe.pipe_net_id = self.pipe_net_id

    @staticmethod
    def _validate_no_overlapping_destinations(pipes: List[Pipe]):
        """Check that no core is the destination of more than one multicast pipe.

        All pipes in a PipeNet share a single semaphore pair. For multicast
        pipes, the handshake protocol cannot handle a core receiving from
        multiple sources. Use separate PipeNets for patterns where multicast
        destinations overlap (e.g., scatter-gather/all-to-all).

        Unicast gather (multiple unicast pipes to one destination) is allowed
        because the receiver uses cumulative semaphore waits.
        """
        mcast_pipes = [(i, p) for i, p in enumerate(pipes) if p.is_multicast]
        if len(mcast_pipes) < 2:
            return
        seen = {}  # (x, y) -> pipe index
        for i, pipe in mcast_pipes:
            sx, sy = pipe.dst_start
            ex, ey = pipe.dst_end
            min_x, max_x = min(sx, ex), max(sx, ex)
            min_y, max_y = min(sy, ey), max(sy, ey)
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    if (x, y) in seen:
                        j = seen[(x, y)]
                        raise ValueError(
                            f"PipeNet has overlapping multicast destinations: "
                            f"pipe {j} (src={pipes[j].src}) and "
                            f"pipe {i} (src={pipe.src}) both target "
                            f"core ({x}, {y}). Use separate PipeNets "
                            f"for patterns where a core receives from "
                            f"multiple multicast sources."
                        )
                    seen[(x, y)] = i

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
