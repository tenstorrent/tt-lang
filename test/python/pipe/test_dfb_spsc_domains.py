# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for launch-domain-aware DFB SPSC verification.

The program below uses one DFB with two consumer kernel threads. The data
movement thread waits on the DFB only on the PipeNet source node, while the
compute thread waits on the same DFB only on the destination node. The verifier
must accept this because the consumer launch-node domains are disjoint.

Issue #663: SPSC checking must compare DFB users per launched node, not only per
kernel thread.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose

TILE = 32


@ttl.operation(grid=(2, 1))
def shared_dfb_disjoint_consumers(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])

    local_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    remote_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_dst():
        if net.is_dst():
            with (
                local_cb.wait() as local_blk,
                remote_cb.wait() as remote_blk,
                out_cb.reserve() as out_blk,
            ):
                out_blk.store(local_blk + remote_blk)

    @ttl.datamovement()
    def fill_and_pipe_local():
        node_x, _ = ttl.node(dims=2)
        with local_cb.reserve() as local_blk:
            ttl.copy(inp[0, node_x], local_blk).wait()

        def send(pipe):
            with local_cb.wait() as send_blk:
                ttl.copy(send_blk, pipe).wait()

        net.if_src(send)

        def recv(pipe):
            with remote_cb.reserve() as remote_blk:
                ttl.copy(pipe, remote_blk).wait()

        net.if_dst(recv)

    @ttl.datamovement()
    def write_dst():
        if net.is_dst():
            with out_cb.wait() as out_blk:
                ttl.copy(out_blk, out[0, 0]).wait()


def test_shared_dfb_disjoint_consumers(device):
    base = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    dst_tile = (base % 13) + 1
    src_tile = ((base // TILE) % 11) + 3
    inp_torch = torch.cat([dst_tile, src_tile], dim=1).to(torch.bfloat16)
    out_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    shared_dfb_disjoint_consumers(inp, out)

    result = ttnn.to_torch(out)
    expected = dst_tile + src_tile
    assert_allclose(expected.float(), result.float(), rtol=1e-2, atol=1e-2)
