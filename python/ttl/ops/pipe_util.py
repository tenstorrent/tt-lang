# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Bare point-to-point send / receive over a PipeNet.

The two halves of moving one block between cores: every source core of ``net``
ships a staged block to its pipe; every destination core receives a peer block
into its own buffer. These are the wire copy only -- staging, combine, and
buffer management stay with the caller -- so the same pattern composes into a
multicast, a K-reduce sum, or a flash tree-reduce merge.
"""

import ttl


@ttl.atom()
def pipe_send(net: ttl.PipeNet, src: ttl.DFB):
    """Each source core of ``net`` ships its ``src`` block to the pipe."""

    def send(pipe):
        b = src.wait()
        ttl.copy(b, pipe)

    net.if_src(send)


@ttl.atom()
def pipe_recv(net: ttl.PipeNet, dst: ttl.DFB):
    """Each destination core of ``net`` receives a peer block into ``dst``."""

    def recv(pipe):
        d = dst.reserve()
        ttl.copy(pipe, d)

    net.if_dst(recv)
