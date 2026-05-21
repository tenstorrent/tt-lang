# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unicast forwarding-chain PipeNet patterns.

This covers the hop-by-hop row and column broadcast shape used by tuned
tt-metal kernels such as minimal matmul. Each chain forwards multi-tile
blocks across several loop iterations, so the test also exercises pipe
write/read pointer tracking beyond the DFB depth.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32
CHAIN_X = 4
CHAIN_Y = 3
CHAIN_ITERS = 5
BLOCK_TILES = 2


@ttl.operation(grid=(CHAIN_X, CHAIN_Y))
def row_column_unicast_chains(row_in, col_in, out):
    row_net = ttl.PipeNet(
        [
            ttl.Pipe(src=(x, y), dst=(x + 1, y))
            for y in range(CHAIN_Y)
            for x in range(CHAIN_X - 1)
        ]
    )
    col_net = ttl.PipeNet(
        [
            ttl.Pipe(src=(x, y), dst=(x, y + 1))
            for x in range(CHAIN_X)
            for y in range(CHAIN_Y - 1)
        ]
    )

    row_cb = ttl.make_dataflow_buffer_like(
        row_in, shape=(1, BLOCK_TILES), block_count=2
    )
    col_cb = ttl.make_dataflow_buffer_like(
        col_in, shape=(1, BLOCK_TILES), block_count=2
    )
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, BLOCK_TILES), block_count=2)

    @ttl.compute()
    def compute():
        for _iter_idx in range(CHAIN_ITERS):
            with (
                row_cb.wait() as row_blk,
                col_cb.wait() as col_blk,
                out_cb.reserve() as out_blk,
            ):
                out_blk.store(row_blk + col_blk)

    @ttl.datamovement()
    def read_forward_rows_then_columns():
        node_x, node_y = ttl.node(dims=2)
        for iter_idx in range(CHAIN_ITERS):
            with row_cb.reserve() as row_pipe_blk:
                if node_x == 0:
                    ttl.copy(
                        row_in[node_y * CHAIN_ITERS + iter_idx, 0:BLOCK_TILES],
                        row_pipe_blk,
                    ).wait()
                else:

                    def recv_row(pipe):
                        ttl.copy(pipe, row_pipe_blk).wait()

                    row_net.if_dst(recv_row)

                if node_x < CHAIN_X - 1:

                    def send_row(pipe):
                        ttl.copy(row_pipe_blk, pipe).wait()

                    row_net.if_src(send_row)

            with col_cb.reserve() as col_pipe_blk:
                if node_y == 0:
                    ttl.copy(
                        col_in[
                            iter_idx,
                            node_x * BLOCK_TILES : (node_x + 1) * BLOCK_TILES,
                        ],
                        col_pipe_blk,
                    ).wait()
                else:

                    def recv_col(pipe):
                        ttl.copy(pipe, col_pipe_blk).wait()

                    col_net.if_dst(recv_col)

                if node_y < CHAIN_Y - 1:

                    def send_col(pipe):
                        ttl.copy(col_pipe_blk, pipe).wait()

                    col_net.if_src(send_col)

    @ttl.datamovement()
    def write_chain_results():
        node_x, node_y = ttl.node(dims=2)
        for iter_idx in range(CHAIN_ITERS):
            with out_cb.wait() as out_blk:
                row_idx = node_y * CHAIN_ITERS + iter_idx
                ttl.copy(
                    out_blk,
                    out[
                        row_idx : row_idx + 1,
                        node_x * BLOCK_TILES : (node_x + 1) * BLOCK_TILES,
                    ],
                ).wait()


def test_row_column_unicast_forward_chains(device):
    row_in_torch = torch.randn(
        CHAIN_Y * CHAIN_ITERS * TILE, BLOCK_TILES * TILE, dtype=torch.bfloat16
    )
    col_in_torch = torch.randn(
        CHAIN_ITERS * TILE, CHAIN_X * BLOCK_TILES * TILE, dtype=torch.bfloat16
    )
    out_torch = torch.zeros(
        CHAIN_Y * CHAIN_ITERS * TILE,
        CHAIN_X * BLOCK_TILES * TILE,
        dtype=torch.bfloat16,
    )

    row_in = to_dram(row_in_torch, device)
    col_in = to_dram(col_in_torch, device)
    out = to_dram(out_torch, device)

    row_column_unicast_chains(row_in, col_in, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out)
    expected = torch.zeros_like(out_torch)
    for node_y in range(CHAIN_Y):
        for iter_idx in range(CHAIN_ITERS):
            row_start = (node_y * CHAIN_ITERS + iter_idx) * TILE
            row_end = row_start + TILE
            row_tile = row_in_torch[row_start:row_end, :].float()
            col_start_row = iter_idx * TILE
            col_end_row = col_start_row + TILE
            for node_x in range(CHAIN_X):
                col_start = node_x * BLOCK_TILES * TILE
                col_end = col_start + BLOCK_TILES * TILE
                col_tile = col_in_torch[
                    col_start_row:col_end_row, col_start:col_end
                ].float()
                expected[row_start:row_end, col_start:col_end] = (
                    row_tile + col_tile
                ).to(torch.bfloat16)

    assert_pcc(expected.float(), result.float())
