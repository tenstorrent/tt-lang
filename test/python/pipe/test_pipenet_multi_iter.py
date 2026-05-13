# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-iteration PipeNet gather + multicast (issue #574).

Two tests:

1. `test_gather_multi_iter` — minimal two-core gather over two stripes.
   Verifies the sender-side `cb_reserve_back` / `cb_push_back` lockstep
   fix in `PipeLowering.cpp` without depending on any other DFB
   patterns. This is the focused regression test for #574.

2. `test_gather_bcast_multi_iter` — mirrors issue 574 reproducer, the
   pattern surfaced by rmsnorm-backward. Exercises gather + multicast
   inside a stripe loop and depends on the same lockstep fix, but also
   uses a multi-consumer `sum_cb` (compute consumes it locally AND
   dm_read consumes it for the bcast send). The framework must emit
   matching pushes for both consumers; see issue [multi-consumer DFB]
   for the supporting framework change.
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

    partial_cb = ttl.make_dataflow_buffer_like(
        out, shape=row_shape, block_count=2
    )
    recv_cb = ttl.make_dataflow_buffer_like(
        out, shape=row_shape, block_count=2
    )
    out_cb = ttl.make_dataflow_buffer_like(
        out, shape=row_shape, block_count=2
    )
    gather_net = ttl.PipeNet([
        ttl.Pipe((x, y), (0, y))
        for x in range(1, col_cores)
        for y in range(row_cores)
    ])

    @ttl.compute()
    def compute():
        node_col, _node_row = ttl.node(dims=2)
        for _ri in range(NUM_OF_STRIPES):
            with partial_cb.reserve() as partial_blk:
                partial_blk.store(ttl.math.fill(partial_blk, 1.0))
            if node_col == 0:
                blk = recv_cb.wait()
                with out_cb.reserve() as out_blk:
                    out_blk.store(blk)
            else:
                partial_blk = partial_cb.wait()
                with out_cb.reserve() as out_blk:
                    out_blk.store(partial_blk)

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
    partial_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=col_cores)
    sum_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=2)
    bcast_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=2)
    gather_net = ttl.PipeNet([
        ttl.Pipe((x, y), (0, y))
        for x in range(1, col_cores)
        for y in range(row_cores)
    ])
    bcast_net = ttl.PipeNet([
        ttl.Pipe((0, y), (slice(1, col_cores), y))
        for y in range(row_cores)
    ])

    out_cb = ttl.make_dataflow_buffer_like(out, shape=row_shape, block_count=2)

    @ttl.compute()
    def compute():
        node_col, _node_row = ttl.node(dims=2)
        for _ri in range(NUM_OF_STRIPES):
            with partial_cb.reserve() as partial_blk:
                partial_blk.store(ttl.math.fill(partial_blk, 1.0))
            if node_col == 0:
                partial_blk = partial_cb.wait()
                blk = recv_cb.wait()
                with sum_cb.reserve() as sum_blk:
                    sum_blk.store(blk + partial_blk)
                sum_blk = sum_cb.wait()
                with out_cb.reserve() as out_blk:
                    out_blk.store(sum_blk)
            else:
                blk = bcast_cb.wait()
                with out_cb.reserve() as out_blk:
                    out_blk.store(blk)

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
                blk = sum_cb.wait()

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


@pytest.mark.xfail(
    reason="Blocked on multi-consumer DFB bug: sum_cb has 1 producer push "
    "and 2 consumer pops per iteration (compute + dm_read both consume), "
    "racing on stripe 1's bcast destination tile. Follow-up PR will scale "
    "the producer push count in ConvertTTLToTTKernel.cpp."
)
def test_gather_bcast_multi_iter(device):
    st = 64
    out_torch = torch.full((st, st), -42.0, dtype=torch.bfloat16)
    out_tt = to_dram(out_torch, device)
    gather_bcast_loop(out_tt)
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out_tt)
    expected = torch.full((st, st), 2.0, dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected)
