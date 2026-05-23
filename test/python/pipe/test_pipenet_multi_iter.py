# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-iteration PipeNet gather + multicast (issue #574).

Three tests:

1. `test_gather_multi_iter` — minimal two-core gather over two stripes.
   Verifies the sender-side `cb_reserve_back` / `cb_push_back` lockstep
   fix in `PipeLowering.cpp` without depending on any other DFB
   patterns. Part of the regression test for #574.

2. `test_gather_bcast_multi_iter` — mirrors the rmsnorm-backward pattern
   surfaced by issue #574. Exercises gather + multicast inside a stripe
   loop and depends on the same lockstep fix. Each dataflow buffer is
   structured as single-producer single-consumer: partial and sum each
   have separate copies for the local-compute and pipe-send consumers,
   so the `ttl-verify-dfb-spsc` pass accepts the kernel.

   TODO(#581): replace the manual `partial_for_sum_cb` /
   `partial_for_send_cb` duplication once a compiler pass auto-splits
   user-facing DFBs whose consumers span multiple kernel threads. The
   shared compiler-allocated DFB helpers introduced by PR #540 would be
   a reasonable foundation but do not themselves perform the split.

3. `test_cross_dfb_multicast_loopback` — four-core multicast where the
   source core (0, 0) is inside the destination range (loopback) and
   the source DFB index differs from the destination DFB index. Combines
   the IR-level loopback skip (`skipSenderReserve =
   pipeType.srcInDstRange()`) with the cumulative-counter semaphore
   lowering so every stripe is delivered. Regression test for #583.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram

TILE = 32
NUM_OF_STRIPES = 2


@ttl.operation(grid=(2, 1))
def gather_multi_iter(out):
    col_cores = 2
    row_cores = 1
    row_shape = (1, 1)

    partial_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=2)
    gather_net = ttl.PipeNet(
        [
            ttl.Pipe((x, y), (0, y))
            for x in range(1, col_cores)
            for y in range(row_cores)
        ]
    )

    @ttl.compute()
    def compute():
        node_col, _node_row = ttl.node(dims=2)
        for _ri in range(NUM_OF_STRIPES):
            with partial_cb.reserve() as partial_blk:
                partial_blk.store(ttl.block.fill(1.0, shape=partial_blk.shape))
            if node_col == 0:
                blk = recv_cb.wait()
                with out_cb.reserve() as out_blk:
                    out_blk.store(blk)
            else:
                with out_cb.reserve() as out_blk:
                    out_blk.store(ttl.block.fill(1.0, shape=out_blk.shape))

    @ttl.datamovement()
    def dm_read():
        node_col, _node_row = ttl.node(dims=2)
        for _ri in range(NUM_OF_STRIPES):
            if node_col > 0:
                blk = partial_cb.wait()

                def send(pipe):
                    tx = ttl.copy(blk, pipe)
                    tx.wait()

                gather_net.if_src(send)
            else:

                def recv(pipe):
                    b = recv_cb.reserve()
                    tx = ttl.copy(pipe, b)
                    tx.wait()

                gather_net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        node_col, _node_row = ttl.node(dims=2)
        for ri in range(NUM_OF_STRIPES):
            out_blk = out_cb.wait()
            ttl.copy(out_blk, out[ri : ri + 1, node_col : node_col + 1]).wait()


@ttl.operation(grid=(2, 1))
def gather_bcast_loop(out):
    out_rows = out.shape[0] // TILE
    col_cores = min(2, out_rows)
    row_cores = 1

    rb = out_rows // NUM_OF_STRIPES
    row_shape = (rb, 1)
    # SPSC split: each cb has at most one producer and one consumer thread.
    # partial gets two copies (one for compute's local sum, one for the
    # gather sender); sum gets two copies (one for compute's out write, one
    # for the bcast sender).
    partial_for_sum_cb = ttl.make_dataflow_buffer_like(
        out, shape=row_shape, block_count=2
    )
    partial_for_send_cb = ttl.make_dataflow_buffer_like(
        out, shape=row_shape, block_count=2
    )
    recv_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=col_cores)
    sum_for_out_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=2)
    sum_for_bcast_cb = ttl.make_dataflow_buffer_like(
        out, shape=row_shape, block_count=2
    )
    bcast_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=2)
    gather_net = ttl.PipeNet(
        [
            ttl.Pipe((x, y), (0, y))
            for x in range(1, col_cores)
            for y in range(row_cores)
        ]
    )
    bcast_net = ttl.PipeNet(
        [ttl.Pipe((0, y), (slice(1, col_cores), y)) for y in range(row_cores)]
    )

    out_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=2)

    @ttl.compute()
    def compute():
        node_col, _node_row = ttl.node(dims=2)
        for _ri in range(NUM_OF_STRIPES):
            if node_col == 0:
                # Produce the local partial only on the core that consumes
                # it. An unconditional reserve here would over-push on
                # col>0 (no matching wait) and deadlock past block_count
                # iterations.
                with partial_for_sum_cb.reserve() as p_sum_blk:
                    p_sum_blk.store(ttl.block.fill(1.0, shape=p_sum_blk.shape))
                partial_blk = partial_for_sum_cb.wait()
                blk = recv_cb.wait()
                sum_val = blk + partial_blk
                with sum_for_out_cb.reserve() as sum_out_blk:
                    sum_out_blk.store(sum_val)
                with sum_for_bcast_cb.reserve() as sum_bcast_blk:
                    sum_bcast_blk.store(sum_val)
                sum_out = sum_for_out_cb.wait()
                with out_cb.reserve() as out_blk:
                    out_blk.store(sum_out)
            else:
                # Produce the send-side partial only on the cores whose
                # dm_read consumes it (col>0). Same rationale as above.
                with partial_for_send_cb.reserve() as p_send_blk:
                    p_send_blk.store(ttl.block.fill(1.0, shape=p_send_blk.shape))
                blk = bcast_cb.wait()
                with out_cb.reserve() as out_blk:
                    out_blk.store(blk)

    @ttl.datamovement()
    def dm_read():
        node_col, _node_row = ttl.node(dims=2)
        for _ri in range(NUM_OF_STRIPES):
            if node_col > 0:
                blk = partial_for_send_cb.wait()

                def send(pipe):
                    tx = ttl.copy(blk, pipe)
                    tx.wait()

                gather_net.if_src(send)

                def recv(pipe):
                    b = bcast_cb.reserve()
                    tx = ttl.copy(pipe, b)
                    tx.wait()

                bcast_net.if_dst(recv)
            else:

                def recv(pipe):
                    b = recv_cb.reserve()
                    tx = ttl.copy(pipe, b)
                    tx.wait()

                gather_net.if_dst(recv)
                blk = sum_for_bcast_cb.wait()

                def send(pipe):
                    tx = ttl.copy(blk, pipe)
                    tx.wait()

                bcast_net.if_src(send)

    @ttl.datamovement()
    def dm_write():
        node_col, node_row = ttl.node(dims=2)
        for _ri in range(NUM_OF_STRIPES):
            r0 = node_row + _ri
            out_blk = out_cb.wait()
            ttl.copy(out_blk, out[r0 : r0 + 1, node_col : node_col + 1]).wait()


def test_gather_multi_iter(device):
    # Both columns produce partial = 1.0 every iteration. With the #574 fix,
    # the gather sender advances its local fifo_wr_ptr per iter and stripe 1
    # receives fresh data (1.0); without the fix the receiver reads slot 1
    # which the sender never wrote, yielding stale L1 (typically 0).
    rows = NUM_OF_STRIPES * TILE
    cols = 2 * TILE
    out_torch = torch.full((rows, cols), -42.0, dtype=torch.bfloat16)
    out_tt = to_dram(out_torch, device)
    gather_multi_iter(out_tt)
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out_tt)
    expected = torch.full((rows, cols), 1.0, dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected)


def test_gather_bcast_multi_iter(device):
    st = 64
    out_torch = torch.full((st, st), -42.0, dtype=torch.bfloat16)
    out_tt = to_dram(out_torch, device)
    gather_bcast_loop(out_tt)
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out_tt)
    expected = torch.full((st, st), 2.0, dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected)


CROSS_DFB_COL_CORES = 4


@ttl.operation(grid=(CROSS_DFB_COL_CORES, 1))
def cross_dfb_multicast_loopback(out):
    src_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    dst_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    # Source (0, 0) is inside the destination range slice(0, 4) — loopback.
    bcast_net = ttl.PipeNet([ttl.Pipe((0, 0), (slice(0, CROSS_DFB_COL_CORES), 0))])

    @ttl.compute()
    def compute():
        node_col, _node_row = ttl.node(dims=2)
        for _ri in range(NUM_OF_STRIPES):
            if node_col == 0:
                with src_cb.reserve() as src_blk:
                    src_blk.store(ttl.block.fill(7.0, shape=src_blk.shape))

    @ttl.datamovement()
    def dm_read():
        node_col, _node_row = ttl.node(dims=2)
        for _ri in range(NUM_OF_STRIPES):
            if node_col == 0:
                blk = src_cb.wait()

                def send(pipe):
                    tx = ttl.copy(blk, pipe)
                    tx.wait()

                bcast_net.if_src(send)

    @ttl.datamovement()
    def dm_write():
        node_col, _node_row = ttl.node(dims=2)
        for ri in range(NUM_OF_STRIPES):

            def recv(pipe):
                b = dst_cb.reserve()
                tx = ttl.copy(pipe, b)
                tx.wait()

            bcast_net.if_dst(recv)
            out_blk = dst_cb.wait()
            ttl.copy(out_blk, out[ri : ri + 1, node_col : node_col + 1]).wait()


def test_cross_dfb_multicast_loopback(device):
    rows = NUM_OF_STRIPES * TILE
    cols = CROSS_DFB_COL_CORES * TILE
    out_torch = torch.full((rows, cols), -42.0, dtype=torch.bfloat16)
    out_tt = to_dram(out_torch, device)
    cross_dfb_multicast_loopback(out_tt)
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out_tt)
    expected = torch.full((rows, cols), 7.0, dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected)
