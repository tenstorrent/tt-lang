# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Computed vs receiver-published addressing produce the same result.

The point-to-point pipe is eligible for computed receiver addresses by default;
--no-ttl-pipe-computed-addresses forces the receiver-published fallback. Both
must match torch and each other, across dtypes -- the fallback path is otherwise
only covered in bf16.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32


def _point_to_point(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)

        def send(pipe):
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 0], send_blk).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe).wait()

        net.if_src(send)

        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 0]).wait()

        if node_x == 1:
            net.if_dst(recv)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(2, 1))
def _computed_point_to_point(inp, out):
    _point_to_point(inp, out)


@ttl.operation(grid=(2, 1), options="--no-ttl-pipe-computed-addresses")
def _published_point_to_point(inp, out):
    _point_to_point(inp, out)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_computed_matches_published(device, dtype):
    inp_torch = torch.randn(TILE, TILE, dtype=dtype)

    computed_out = to_dram(torch.zeros(TILE, TILE, dtype=dtype), device)
    _computed_point_to_point(to_dram(inp_torch, device), computed_out)
    ttnn.synchronize_device(device)
    computed = ttnn.to_torch(computed_out).float()

    published_out = to_dram(torch.zeros(TILE, TILE, dtype=dtype), device)
    _published_point_to_point(to_dram(inp_torch, device), published_out)
    ttnn.synchronize_device(device)
    published = ttnn.to_torch(published_out).float()

    assert_pcc(computed, inp_torch.float())
    assert_pcc(published, inp_torch.float())
    assert_pcc(computed, published)
