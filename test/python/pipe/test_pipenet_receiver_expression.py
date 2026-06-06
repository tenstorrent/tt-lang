# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for indexed multicast PipeNet receiver expressions.

The compile-only lit test verifies initial IR for several receiver expression
forms. This pytest exercises the same frontend behavior through hardware
execution with a larger indexed list of multicast PipeNets.
"""

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
ROW_COUNT = 2
SOURCE_COUNT_PER_ROW = 4
GRID_WIDTH = 8
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


@ttl.operation(grid=(GRID_WIDTH, ROW_COUNT))
def indexed_multicast_receiver_kernel(inp, out):
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    first_recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    second_recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if INDEXED_MULTICAST_NETS[0].is_dst():
            with (
                first_recv_dfb.wait() as first_blk,
                second_recv_dfb.wait() as second_blk,
                out_dfb.reserve() as out_blk,
            ):
                out_blk.store(first_blk + second_blk)
        if INDEXED_MULTICAST_NETS[2].is_dst():
            with (
                first_recv_dfb.wait() as first_blk,
                second_recv_dfb.wait() as second_blk,
                out_dfb.reserve() as out_blk,
            ):
                out_blk.store(first_blk + second_blk)
        if INDEXED_MULTICAST_NETS[4].is_dst():
            with (
                first_recv_dfb.wait() as first_blk,
                second_recv_dfb.wait() as second_blk,
                out_dfb.reserve() as out_blk,
            ):
                out_blk.store(first_blk + second_blk)
        if INDEXED_MULTICAST_NETS[6].is_dst():
            with (
                first_recv_dfb.wait() as first_blk,
                second_recv_dfb.wait() as second_blk,
                out_dfb.reserve() as out_blk,
            ):
                out_blk.store(first_blk + second_blk)

    @ttl.datamovement()
    def dm_read():
        if INDEXED_MULTICAST_NETS[0].is_src():
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 0], send_blk).wait()
                INDEXED_MULTICAST_NETS[0].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if INDEXED_MULTICAST_NETS[1].is_src():
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 1], send_blk).wait()
                INDEXED_MULTICAST_NETS[1].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if INDEXED_MULTICAST_NETS[2].is_src():
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 2], send_blk).wait()
                INDEXED_MULTICAST_NETS[2].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if INDEXED_MULTICAST_NETS[3].is_src():
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 3], send_blk).wait()
                INDEXED_MULTICAST_NETS[3].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if INDEXED_MULTICAST_NETS[4].is_src():
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[1, 0], send_blk).wait()
                INDEXED_MULTICAST_NETS[4].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if INDEXED_MULTICAST_NETS[5].is_src():
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[1, 1], send_blk).wait()
                INDEXED_MULTICAST_NETS[5].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if INDEXED_MULTICAST_NETS[6].is_src():
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[1, 2], send_blk).wait()
                INDEXED_MULTICAST_NETS[6].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if INDEXED_MULTICAST_NETS[7].is_src():
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[1, 3], send_blk).wait()
                INDEXED_MULTICAST_NETS[7].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )

        if INDEXED_MULTICAST_NETS[0].is_dst():
            with first_recv_dfb.reserve() as recv_blk:
                INDEXED_MULTICAST_NETS[0].if_dst(
                    lambda pipe: ttl.copy(pipe, recv_blk).wait()
                )
        if INDEXED_MULTICAST_NETS[1].is_dst():
            with second_recv_dfb.reserve() as recv_blk:
                INDEXED_MULTICAST_NETS[1].if_dst(
                    lambda pipe: ttl.copy(pipe, recv_blk).wait()
                )
        if INDEXED_MULTICAST_NETS[2].is_dst():
            with first_recv_dfb.reserve() as recv_blk:
                INDEXED_MULTICAST_NETS[2].if_dst(
                    lambda pipe: ttl.copy(pipe, recv_blk).wait()
                )
        if INDEXED_MULTICAST_NETS[3].is_dst():
            with second_recv_dfb.reserve() as recv_blk:
                INDEXED_MULTICAST_NETS[3].if_dst(
                    lambda pipe: ttl.copy(pipe, recv_blk).wait()
                )
        if INDEXED_MULTICAST_NETS[4].is_dst():
            with first_recv_dfb.reserve() as recv_blk:
                INDEXED_MULTICAST_NETS[4].if_dst(
                    lambda pipe: ttl.copy(pipe, recv_blk).wait()
                )
        if INDEXED_MULTICAST_NETS[5].is_dst():
            with second_recv_dfb.reserve() as recv_blk:
                INDEXED_MULTICAST_NETS[5].if_dst(
                    lambda pipe: ttl.copy(pipe, recv_blk).wait()
                )
        if INDEXED_MULTICAST_NETS[6].is_dst():
            with first_recv_dfb.reserve() as recv_blk:
                INDEXED_MULTICAST_NETS[6].if_dst(
                    lambda pipe: ttl.copy(pipe, recv_blk).wait()
                )
        if INDEXED_MULTICAST_NETS[7].is_dst():
            with second_recv_dfb.reserve() as recv_blk:
                INDEXED_MULTICAST_NETS[7].if_dst(
                    lambda pipe: ttl.copy(pipe, recv_blk).wait()
                )

    @ttl.datamovement()
    def dm_write():
        node_x, node_y = ttl.node(dims=2)
        if INDEXED_MULTICAST_NETS[0].is_dst():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[node_y, node_x]).wait()
        if INDEXED_MULTICAST_NETS[2].is_dst():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[node_y, node_x]).wait()
        if INDEXED_MULTICAST_NETS[4].is_dst():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[node_y, node_x]).wait()
        if INDEXED_MULTICAST_NETS[6].is_dst():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[node_y, node_x]).wait()


def test_indexed_multicast_receiver_list(device):
    assert len(INDEXED_MULTICAST_NETS) == 8

    inp_torch = torch.randn(
        ROW_COUNT * TILE, SOURCE_COUNT_PER_ROW * TILE, dtype=torch.bfloat16
    )
    out_torch = torch.zeros(ROW_COUNT * TILE, GRID_WIDTH * TILE, dtype=torch.bfloat16)
    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    indexed_multicast_receiver_kernel(inp, out)

    expected = torch.zeros_like(out_torch)
    for row_index in range(ROW_COUNT):
        row_slice = slice(row_index * TILE, (row_index + 1) * TILE)
        left_sum = (
            inp_torch[row_slice, 0:TILE] + inp_torch[row_slice, TILE : 2 * TILE]
        )
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

    result = ttnn.to_torch(out)
    assert_pcc(expected.float(), result.float())
