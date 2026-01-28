# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pipe operations for core-to-core data transfer.

This module provides Python classes for the Pipe and PipeNet abstractions
as defined in the TT-Lang specification. The MLIR ops (ttl.create_pipe,
ttl.if_src, ttl.if_dst) are implemented and lower to TTKernel.

NOTE: Full DSL integration with PipeNet.if_src/if_dst callback API
requires additional compiler support. The Python classes and MLIR ops
are ready for use.
"""

from typing import List, Tuple, Union

# Type aliases matching the spec
CoreCoord = Tuple[int, int]
CoreRange = Tuple[Union[int, slice], Union[int, slice]]


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
        self._parse_dst()

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
                # Multicast over y: dst is (x, slice(y_start, y_end))
                self.dst_start = (x, y.start)
                self.dst_end = (x, y.stop - 1)
                self._is_multicast = True
            elif isinstance(x, slice) and isinstance(y, int):
                # Multicast over x: dst is (slice(x_start, x_end), y)
                self.dst_start = (x.start, y)
                self.dst_end = (x.stop - 1, y)
                self._is_multicast = True
            elif isinstance(x, slice) and isinstance(y, slice):
                # Multicast over x and y
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

    PipeNet groups multiple pipes and provides iteration methods
    to access pipes for conditional execution.

    Args:
        pipes: List of Pipe objects defining the network

    Example:
        # Gather pattern: all cores send to (0, y)
        net = ttl.PipeNet([
            ttl.Pipe(src=(x, y), dst=(0, y))
            for x in range(1, grid_x)
            for y in range(grid_y)
        ])
    """

    def __init__(self, pipes: List[Pipe]):
        if not pipes:
            raise ValueError("PipeNet requires at least one pipe")
        self.pipes = pipes

    def __iter__(self):
        """Iterate over all pipes in the network."""
        return iter(self.pipes)
