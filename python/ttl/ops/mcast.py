# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multicast: broadcast a block from one source core to many destinations.

`mcast` is a factory that builds an inlinable @ttl.atom wiring both sides of a
PipeNet: the source core stages its block and sends it; every destination
core receives into its own DFB. Call the factory with a grid, then inline the
returned atom into a composing atom that declares the net and buffers. The
mcast_rows / mcast_cols helpers build the PipeNet topology.
"""

from typing import List

import ttl

from ..pipe import Pipe


@ttl.atom()
def mcast(net: ttl.PipeNet, src, stage: ttl.DFB, dst: ttl.DFB):
    """Broadcast one block over `net`. The source core copies `src` (a tensor
    slice) into `stage` and sends it to the pipe; every destination core
    receives into `dst`. The source is included in the destination range, so
    it also receives its own block into `dst`.
    """

    def _send(pipe):
        s = stage.reserve()
        ttl.copy(src, s)
        sr = stage.wait()
        ttl.copy(sr, pipe)

    net.if_src(_send)

    def _recv(pipe):
        d = dst.reserve()
        ttl.copy(pipe, d)

    net.if_dst(_recv)


@ttl.atom()
def mcast_block(net: ttl.PipeNet, stage: ttl.DFB, dst: ttl.DFB):
    """Broadcast one staged block over `net`. The source consumes the next
    block from `stage` (produced by earlier compute); every destination
    receives into `dst`."""

    def _send(pipe):
        sr = stage.wait()
        ttl.copy(sr, pipe)

    net.if_src(_send)

    def _recv(pipe):
        d = dst.reserve()
        ttl.copy(pipe, d)

    net.if_dst(_recv)


def mcast_rows(rows: int, cols: int) -> List[Pipe]:
    """Row-broadcast pipes: in each of `rows` process rows, the source core in
    column 0 fans out to all `cols` columns. One pipe per row.
    """
    return [Pipe(src=(0, r), dst=(slice(0, cols), r)) for r in range(rows)]


def mcast_cols(rows: int, cols: int) -> List[Pipe]:
    """Column-broadcast pipes: in each of `cols` process columns, the source
    core in row 0 fans out to all `rows` rows. One pipe per column.
    """
    return [Pipe(src=(c, 0), dst=(c, slice(0, rows))) for c in range(cols)]
