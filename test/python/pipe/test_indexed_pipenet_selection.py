# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for indexed PipeNet selection."""

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
MULTICAST_ROW_COUNT = 2
MULTICAST_SOURCE_COUNT_PER_ROW = 4
MULTICAST_GRID_WIDTH = 8
# Per row, sources 0/1 multicast to destination columns 4/5 and sources 2/3
# multicast to destination columns 6/7. Each destination sums two received
# tiles. Source and destination callbacks both select PipeNets by index.
INDEXED_MULTICAST_NETS = [
    ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(4, 6), 0))]),
    ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(slice(4, 6), 0))]),
    ttl.PipeNet([ttl.Pipe(src=(2, 0), dst=(slice(6, 8), 0))]),
    ttl.PipeNet([ttl.Pipe(src=(3, 0), dst=(slice(6, 8), 0))]),
    ttl.PipeNet([ttl.Pipe(src=(0, 1), dst=(slice(4, 6), 1))]),
    ttl.PipeNet([ttl.Pipe(src=(1, 1), dst=(slice(4, 6), 1))]),
    ttl.PipeNet([ttl.Pipe(src=(2, 1), dst=(slice(6, 8), 1))]),
    ttl.PipeNet([ttl.Pipe(src=(3, 1), dst=(slice(6, 8), 1))]),
]


MODULE_PIPE_NET_GROUPS = {"module": [ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])]}
CYCLIC_PIPE_NET_GROUP = [MODULE_PIPE_NET_GROUPS["module"][0]]
CYCLIC_PIPE_NET_GROUP.append(CYCLIC_PIPE_NET_GROUP)


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
            indexed_nets = local_pipe_net_groups["indexed"]
            for pipe_index in range(len(indexed_nets)):
                net_index = pipe_index + 0
                if indexed_nets[net_index].is_src():
                    with send_cb.reserve() as send_blk:
                        ttl.copy(inp[0, net_index], send_blk).wait()

                        def send(pipe):
                            ttl.copy(send_blk, pipe).wait()

                        indexed_nets[net_index].if_src(send)

                indexed_nets[net_index].if_dst(
                    lambda pipe: ttl.copy(pipe, recv_cb.reserve()).wait()
                )

        @ttl.datamovement()
        def dm_write():
            indexed_nets = local_pipe_net_groups["indexed"]
            for pipe_index in range(len(indexed_nets)):
                if indexed_nets[pipe_index].is_dst():
                    with out_cb.wait() as out_blk:
                        ttl.copy(out_blk, out[0, pipe_index]).wait()

    return indexed_pipenet_unicast


INDEXED_PIPENET_UNICAST = make_indexed_pipenet_unicast()


@ttl.operation(grid=(MULTICAST_GRID_WIDTH, MULTICAST_ROW_COUNT))
def indexed_pipenet_multicast(inp, out):
    local_pipe_net_groups = {"multicast": INDEXED_MULTICAST_NETS}

    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    first_recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    second_recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pipe_nets = local_pipe_net_groups["multicast"]
        for net_index in range(0, len(pipe_nets), 2):
            if pipe_nets[net_index].is_dst():
                with (
                    first_recv_dfb.wait() as first_blk,
                    second_recv_dfb.wait() as second_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    out_blk.store(first_blk + second_blk)

    @ttl.datamovement()
    def dm_read():
        pipe_nets = local_pipe_net_groups["multicast"]
        for net_index in range(len(pipe_nets)):
            row_index = net_index // MULTICAST_SOURCE_COUNT_PER_ROW
            source_col = net_index % MULTICAST_SOURCE_COUNT_PER_ROW
            if pipe_nets[net_index].is_src():
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[row_index, source_col], send_blk).wait()
                    pipe_nets[net_index].if_src(
                        lambda pipe: ttl.copy(send_blk, pipe).wait()
                    )

        for net_index in range(0, len(pipe_nets), 2):
            if pipe_nets[net_index].is_dst():
                with first_recv_dfb.reserve() as recv_blk:
                    pipe_nets[net_index].if_dst(
                        lambda pipe: ttl.copy(pipe, recv_blk).wait()
                    )

        for net_index in range(1, len(pipe_nets), 2):
            if pipe_nets[net_index].is_dst():
                with second_recv_dfb.reserve() as recv_blk:
                    pipe_nets[net_index].if_dst(
                        lambda pipe: ttl.copy(pipe, recv_blk).wait()
                    )

    @ttl.datamovement()
    def dm_write():
        node_col, node_row = ttl.node(dims=2)
        pipe_nets = local_pipe_net_groups["multicast"]
        for net_index in range(0, len(pipe_nets), 2):
            if pipe_nets[net_index].is_dst():
                with out_dfb.wait() as out_blk:
                    ttl.copy(out_blk, out[node_row, node_col]).wait()


def test_nested_indexed_pipenet_selection(device):
    """Two unicast PipeNets are selected for source and destination callbacks."""
    inp_torch = torch.randn(TILE, 4 * TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, 2 * TILE, dtype=torch.bfloat16), device)

    INDEXED_PIPENET_UNICAST(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    expected = torch.cat(
        [
            torch.abs(inp_torch[:, 0:TILE]),
            torch.abs(inp_torch[:, TILE : 2 * TILE]),
        ],
        dim=1,
    )
    assert_pcc(expected.float(), result.float())


def test_indexed_multicast_pipenet_selection(device):
    """Eight multicast PipeNets use indexed source and destination callbacks."""
    assert len(INDEXED_MULTICAST_NETS) == 8

    inp_torch = torch.randn(
        MULTICAST_ROW_COUNT * TILE,
        MULTICAST_SOURCE_COUNT_PER_ROW * TILE,
        dtype=torch.bfloat16,
    )
    out_torch = torch.zeros(
        MULTICAST_ROW_COUNT * TILE,
        MULTICAST_GRID_WIDTH * TILE,
        dtype=torch.bfloat16,
    )
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(out_torch, device)

    indexed_pipenet_multicast(inp_tt, out_tt)

    expected = torch.zeros_like(out_torch)
    for row_index in range(MULTICAST_ROW_COUNT):
        row_slice = slice(row_index * TILE, (row_index + 1) * TILE)
        left_sum = inp_torch[row_slice, 0:TILE] + inp_torch[row_slice, TILE : 2 * TILE]
        right_sum = (
            inp_torch[row_slice, 2 * TILE : 3 * TILE]
            + inp_torch[row_slice, 3 * TILE : 4 * TILE]
        )
        for dest_col in (4, 5):
            col_slice = slice(dest_col * TILE, (dest_col + 1) * TILE)
            expected[row_slice, col_slice] = left_sum
        for dest_col in (6, 7):
            col_slice = slice(dest_col * TILE, (dest_col + 1) * TILE)
            expected[row_slice, col_slice] = right_sum

    result = ttnn.to_torch(out_tt)
    assert_pcc(expected.float(), result.float())
