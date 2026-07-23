# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for PipeNet receives batched into fewer DFB blocks.

These kernels post more logical receives than the receiver DFB can hold at
once. Each receiver consumes and pops a batch before posting the next batch, so
the same physical DFB block is reused by multiple logical senders.
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
GATHER_SOURCES = 3
GATHER_BLOCK_TILES = 2
ROW_ALL_GATHER_WIDTH = 4
COLUMN_ALL_GATHER_WIDTH = 2
COLUMN_ALL_GATHER_HEIGHT = 3
DTYPES = [torch.bfloat16, torch.float32]
DTYPE_IDS = ["bf16", "fp32"]


@ttl.operation(grid=(GATHER_SOURCES + 1, 1))
def gather_one_receiver_single_slot(inp, out):
    pipe1 = ttl.Pipe(src=(1, 0), dst=(0, 0))
    pipe2 = ttl.Pipe(src=(2, 0), dst=(0, 0))
    pipe3 = ttl.Pipe(src=(3, 0), dst=(0, 0))
    net = ttl.PipeNet([pipe1, pipe2, pipe3])

    send_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=(1, GATHER_BLOCK_TILES), block_count=2
    )
    recv_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, GATHER_BLOCK_TILES), block_count=1
    )

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)
        if net.is_dst():
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe1, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 0:GATHER_BLOCK_TILES]).wait()
        elif net.is_src() and node_x == 1:
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 0:GATHER_BLOCK_TILES], send_blk).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe1).wait()

        if net.is_dst():
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe2, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(
                    recv_blk, out[0, GATHER_BLOCK_TILES : 2 * GATHER_BLOCK_TILES]
                ).wait()
        elif net.is_src() and node_x == 2:
            with send_dfb.reserve() as send_blk:
                ttl.copy(
                    inp[0, GATHER_BLOCK_TILES : 2 * GATHER_BLOCK_TILES], send_blk
                ).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe2).wait()

        if net.is_dst():
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe3, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(
                    recv_blk,
                    out[0, 2 * GATHER_BLOCK_TILES : 3 * GATHER_BLOCK_TILES],
                ).wait()
        elif net.is_src() and node_x == 3:
            with send_dfb.reserve() as send_blk:
                ttl.copy(
                    inp[0, 2 * GATHER_BLOCK_TILES : 3 * GATHER_BLOCK_TILES],
                    send_blk,
                ).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe3).wait()

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(ROW_ALL_GATHER_WIDTH, 1))
def row_all_gather_two_slot_batches(inp, out):
    pipe0 = ttl.Pipe(src=(0, 0), dst=(slice(0, ROW_ALL_GATHER_WIDTH), 0))
    pipe1 = ttl.Pipe(src=(1, 0), dst=(slice(0, ROW_ALL_GATHER_WIDTH), 0))
    pipe2 = ttl.Pipe(src=(2, 0), dst=(slice(0, ROW_ALL_GATHER_WIDTH), 0))
    pipe3 = ttl.Pipe(src=(3, 0), dst=(slice(0, ROW_ALL_GATHER_WIDTH), 0))
    all_gather_net = ttl.PipeNet([pipe0, pipe1, pipe2, pipe3])

    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)
        if all_gather_net.is_dst():
            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe0, recv_blk)
                if node_x == 0:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe0).wait()
                recv_tx.wait()
            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe1, recv_blk)
                if node_x == 1:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 1], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe1).wait()
                recv_tx.wait()

            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 0]).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 1]).wait()

            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe2, recv_blk)
                if node_x == 2:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 2], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe2).wait()
                recv_tx.wait()
            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe3, recv_blk)
                if node_x == 3:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 3], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe3).wait()
                recv_tx.wait()

            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 2]).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 3]).wait()

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(ROW_ALL_GATHER_WIDTH, 1))
def row_all_gather_single_slot_batches(inp, out):
    pipe0 = ttl.Pipe(src=(0, 0), dst=(slice(0, ROW_ALL_GATHER_WIDTH), 0))
    pipe1 = ttl.Pipe(src=(1, 0), dst=(slice(0, ROW_ALL_GATHER_WIDTH), 0))
    pipe2 = ttl.Pipe(src=(2, 0), dst=(slice(0, ROW_ALL_GATHER_WIDTH), 0))
    pipe3 = ttl.Pipe(src=(3, 0), dst=(slice(0, ROW_ALL_GATHER_WIDTH), 0))
    all_gather_net = ttl.PipeNet([pipe0, pipe1, pipe2, pipe3])

    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)
        if all_gather_net.is_dst():
            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe0, recv_blk)
                if node_x == 0:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe0).wait()
                recv_tx.wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 0]).wait()

            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe1, recv_blk)
                if node_x == 1:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 1], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe1).wait()
                recv_tx.wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 1]).wait()

            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe2, recv_blk)
                if node_x == 2:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 2], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe2).wait()
                recv_tx.wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 2]).wait()

            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe3, recv_blk)
                if node_x == 3:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 3], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe3).wait()
                recv_tx.wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 3]).wait()

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(COLUMN_ALL_GATHER_WIDTH, COLUMN_ALL_GATHER_HEIGHT))
def column_all_gather_2d_single_slot_batches(inp, out):
    pipe00 = ttl.Pipe(src=(0, 0), dst=(0, slice(0, COLUMN_ALL_GATHER_HEIGHT)))
    pipe01 = ttl.Pipe(src=(0, 1), dst=(0, slice(0, COLUMN_ALL_GATHER_HEIGHT)))
    pipe02 = ttl.Pipe(src=(0, 2), dst=(0, slice(0, COLUMN_ALL_GATHER_HEIGHT)))
    pipe10 = ttl.Pipe(src=(1, 0), dst=(1, slice(0, COLUMN_ALL_GATHER_HEIGHT)))
    pipe11 = ttl.Pipe(src=(1, 1), dst=(1, slice(0, COLUMN_ALL_GATHER_HEIGHT)))
    pipe12 = ttl.Pipe(src=(1, 2), dst=(1, slice(0, COLUMN_ALL_GATHER_HEIGHT)))
    all_gather_net = ttl.PipeNet([pipe00, pipe01, pipe02, pipe10, pipe11, pipe12])

    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, node_y = ttl.node(dims=2)

        if all_gather_net.is_dst() and node_x == 0:
            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe00, recv_blk)
                if node_y == 0:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe00).wait()
                recv_tx.wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[node_y, 0]).wait()

            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe01, recv_blk)
                if node_y == 1:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[1, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe01).wait()
                recv_tx.wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[node_y, 1]).wait()

            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe02, recv_blk)
                if node_y == 2:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[2, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe02).wait()
                recv_tx.wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[node_y, 2]).wait()

        if all_gather_net.is_dst() and node_x == 1:
            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe10, recv_blk)
                if node_y == 0:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 1], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe10).wait()
                recv_tx.wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[node_y, 3]).wait()

            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe11, recv_blk)
                if node_y == 1:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[1, 1], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe11).wait()
                recv_tx.wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[node_y, 4]).wait()

            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe12, recv_blk)
                if node_y == 2:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[2, 1], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe12).wait()
                recv_tx.wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[node_y, 5]).wait()

    @ttl.datamovement()
    def dm_brisc():
        pass


def _make_input(shape, dtype):
    num_elements = 1
    for extent in shape:
        num_elements *= extent
    return torch.arange(num_elements, dtype=torch.float32).reshape(shape).to(dtype)


# Three sequential unicast receives must safely reuse one receiver DFB block.
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_gather_one_receiver_single_slot(device, dtype):
    inp_torch = _make_input((TILE, GATHER_SOURCES * GATHER_BLOCK_TILES * TILE), dtype)
    out_torch = torch.zeros_like(inp_torch)
    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    gather_one_receiver_single_slot(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out)
    assert_pcc(inp_torch.float(), result.float())


# A row all-gather must reuse a two-block receiver DFB in two batches.
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_row_all_gather_two_slot_batches(device, dtype):
    inp_torch = _make_input((TILE, ROW_ALL_GATHER_WIDTH * TILE), dtype)
    out_torch = torch.zeros_like(inp_torch)
    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    row_all_gather_two_slot_batches(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out)
    assert_pcc(inp_torch.float(), result.float())


# A row all-gather must consume each receive before reusing one DFB block.
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_row_all_gather_single_slot_batches(device, dtype):
    inp_torch = _make_input((TILE, ROW_ALL_GATHER_WIDTH * TILE), dtype)
    out_torch = torch.zeros_like(inp_torch)
    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    row_all_gather_single_slot_batches(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out)
    assert_pcc(inp_torch.float(), result.float())


# Independent columns in a 2D grid must safely reuse one receiver DFB block.
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_column_all_gather_2d_single_slot_batches(device, dtype):
    inp_torch = _make_input(
        (
            COLUMN_ALL_GATHER_HEIGHT * TILE,
            COLUMN_ALL_GATHER_WIDTH * TILE,
        ),
        dtype,
    )
    out_torch = torch.zeros(
        (
            COLUMN_ALL_GATHER_HEIGHT * TILE,
            COLUMN_ALL_GATHER_WIDTH * COLUMN_ALL_GATHER_HEIGHT * TILE,
        ),
        dtype=dtype,
    )
    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    column_all_gather_2d_single_slot_batches(inp, out)
    ttnn.synchronize_device(device)

    column0 = torch.cat(
        [
            inp_torch[0:TILE, 0:TILE],
            inp_torch[TILE : 2 * TILE, 0:TILE],
            inp_torch[2 * TILE : 3 * TILE, 0:TILE],
        ],
        dim=1,
    )
    column1 = torch.cat(
        [
            inp_torch[0:TILE, TILE : 2 * TILE],
            inp_torch[TILE : 2 * TILE, TILE : 2 * TILE],
            inp_torch[2 * TILE : 3 * TILE, TILE : 2 * TILE],
        ],
        dim=1,
    )
    expected_row = torch.cat([column0, column1], dim=1)
    expected = expected_row.repeat(COLUMN_ALL_GATHER_HEIGHT, 1)

    result = ttnn.to_torch(out)
    assert_pcc(expected.float(), result.float())
