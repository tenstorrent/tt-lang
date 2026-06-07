# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for indexed PipeNet receiver expressions."""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32


MODULE_PIPE_NET_GROUPS = {
    "module": [ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])]
}


def make_indexed_pipenet_unicast():
    captured_pipe_net_groups = {
        "captured": (ttl.PipeNet([ttl.Pipe(src=(3, 0), dst=(2, 0))]),)
    }

    @ttl.operation(grid=(4, 1))
    def indexed_pipenet_unicast(inp, out):
        local_pipe_net_groups = {
            "indexed": [
                MODULE_PIPE_NET_GROUPS["module"][0],
                captured_pipe_net_groups["captured"][0],
            ]
        }

        send_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        recv_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            for pipe_index in range(len(local_pipe_net_groups["indexed"])):
                indexed_net = local_pipe_net_groups["indexed"][pipe_index]
                if indexed_net.is_active():
                    if indexed_net.is_dst():
                        with recv_cb.wait() as tile_in, out_cb.reserve() as tile_out:
                            tile_out.store(ttl.math.abs(tile_in))

        @ttl.datamovement()
        def dm_read():
            node_x, _node_y = ttl.node(dims=2)
            indexed_nets = local_pipe_net_groups["indexed"]
            for pipe_index in range(len(indexed_nets)):
                if indexed_nets[pipe_index].is_src():
                    with send_cb.reserve() as send_blk:
                        ttl.copy(inp[0, node_x], send_blk).wait()

                        def send(pipe):
                            ttl.copy(send_blk, pipe).wait()

                        indexed_nets[pipe_index].if_src(send)

                indexed_nets[pipe_index].if_dst(
                    lambda pipe: ttl.copy(pipe, recv_cb.reserve()).wait()
                )

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            indexed_nets = local_pipe_net_groups["indexed"]
            for pipe_index in range(len(indexed_nets)):
                if indexed_nets[pipe_index].is_dst():
                    with out_cb.wait() as out_blk:
                        if node_x == 0:
                            ttl.copy(out_blk, out[0, 0]).wait()
                        if node_x == 2:
                            ttl.copy(out_blk, out[0, 1]).wait()

    return indexed_pipenet_unicast


INDEXED_PIPENET_UNICAST = make_indexed_pipenet_unicast()


def test_nested_indexed_pipenet_receiver_expression(device):
    inp_torch = torch.randn(TILE, 4 * TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, 2 * TILE, dtype=torch.bfloat16), device)

    INDEXED_PIPENET_UNICAST(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    expected = torch.cat(
        [
            torch.abs(inp_torch[:, TILE : 2 * TILE]),
            torch.abs(inp_torch[:, 3 * TILE : 4 * TILE]),
        ],
        dim=1,
    )
    assert_pcc(expected.float(), result.float())
