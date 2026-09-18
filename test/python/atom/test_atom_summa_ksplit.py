# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""SUMMA matmul with K-split reduction as a unified @ttl.operation.

Each output tile block is computed by KP cores that split the K range,
then a reduce PipeNet gathers the non-root partials onto the root band
(k_p == 0). KP is folded into the x axis so the grid is NUM_COLS*KP cols
x NUM_ROWS rows.

  - A is multicast across the NUM_COLS cores of each KP band, per M row.
  - B is multicast down the NUM_ROWS rows of each (KP band, N col).
  - Each core accumulates its K-slice partial, then non-root bands ship
    their partial to the k_p == 0 root, which sums and writes out.
"""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = ttnn.TILE_SIZE
BLOCK_M = 2
BLOCK_N = 2
BLOCK_K = 2
BLOCK_SIZE = BLOCK_M * TILE  # 64

NUM_COLS = 2  # N blocks
NUM_ROWS = 2  # M blocks
KP = 2  # K-split factor

M = NUM_ROWS * BLOCK_SIZE  # 128
N = NUM_COLS * BLOCK_SIZE  # 128
K_BLOCKS = 4
K = K_BLOCKS * BLOCK_SIZE  # 256
K_BLOCKS_PER_KP = K_BLOCKS // KP  # 2

GRID_X = NUM_COLS * KP  # 4
GRID_Y = NUM_ROWS  # 2


@ttl.operation(grid=(GRID_X, GRID_Y), fp32_dest_acc_en=True)
def atom_summa_ksplit(a, w, out):
    a_cb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_M, BLOCK_K), block_count=2)
    b_cb = ttl.make_dataflow_buffer_like(w, shape=(BLOCK_K, BLOCK_N), block_count=2)
    # Split the partial buffer per consumer: compute uses partial_for_sum_cb
    # to fold in the received contribution on k_p == 0 cores, the DM thread
    # uses partial_for_send_cb to ship the partial off k_p > 0 cores. A single
    # producer feeding both compute and DM consumers is invalid, so the matmul
    # stores the same accumulated tile into both CBs.
    partial_for_sum_cb = ttl.make_dataflow_buffer_like(
        out, shape=(BLOCK_M, BLOCK_N), block_count=2
    )
    partial_for_send_cb = ttl.make_dataflow_buffer_like(
        out, shape=(BLOCK_M, BLOCK_N), block_count=2
    )
    recv_cb = ttl.make_dataflow_buffer_like(
        out, shape=(BLOCK_M, BLOCK_N), block_count=2
    )
    # Dedicated output CB: the reduce keeps consuming partial_for_sum_cb on
    # the compute thread, then stores the final tile here for the write thread
    # to drain. CBs are single-consumer, so the accumulator and the tensor
    # write cannot share one DFB.
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(BLOCK_M, BLOCK_N), block_count=2)
    # Private BRISC staging buffers so the loopback src core does not
    # double-reserve a_cb/b_cb across BRISC (if_src) and NCRISC (if_dst).
    tmp_a_cb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_M, BLOCK_K), block_count=2)
    tmp_b_cb = ttl.make_dataflow_buffer_like(w, shape=(BLOCK_K, BLOCK_N), block_count=2)

    # A is multicast across the NUM_COLS cores of each KP band, per M row.
    a_pipes = [
        ttl.Pipe(
            src=(k_band * NUM_COLS, m),
            dst=(slice(k_band * NUM_COLS, (k_band + 1) * NUM_COLS), m),
        )
        for k_band in range(KP)
        for m in range(NUM_ROWS)
    ]
    mcast_a_net = ttl.PipeNet(a_pipes)

    # B is multicast down the NUM_ROWS rows of each (KP band, N col).
    b_pipes = [
        ttl.Pipe(
            src=(n + k_band * NUM_COLS, 0),
            dst=(n + k_band * NUM_COLS, slice(0, NUM_ROWS)),
        )
        for k_band in range(KP)
        for n in range(NUM_COLS)
    ]
    mcast_b_net = ttl.PipeNet(b_pipes)

    # K-split reduce: non-root k_p bands send their partial to the k_p == 0 root.
    reduce_pipes = [
        ttl.Pipe(src=(n + k_band * NUM_COLS, m), dst=(n, m))
        for n in range(NUM_COLS)
        for m in range(NUM_ROWS)
        for k_band in range(1, KP)
    ]
    reduce_net = ttl.PipeNet(reduce_pipes)

    node_x, node_m = ttl.node(dims=2)
    k_p = node_x // NUM_COLS
    node_n = node_x % NUM_COLS
    mr = node_m * BLOCK_M
    nc = node_n * BLOCK_N
    k_offset = k_p * K_BLOCKS_PER_KP

    p = partial_for_sum_cb.reserve()
    for _ in range(K_BLOCKS_PER_KP):
        a_blk = a_cb.wait()
        b_blk = b_cb.wait()
        p += a_blk @ b_blk
    # Finalize the partial before reading it: storing the still-reserved
    # accumulator copies it before the matmul result is packed to L1, which
    # ships garbage to the gather. Wait pins down the packed block.
    pf = partial_for_sum_cb.wait()

    # Mirror the finalized partial into the DM-side CB so the gather send has
    # its own producer/consumer pair (one DFB cannot feed both compute and the
    # send). Only k_p > 0 cores consume it.
    p_send_local = partial_for_send_cb.reserve()
    p_send_local.store(pf)

    for kb_local in range(K_BLOCKS_PER_KP):
        kb = k_offset + kb_local
        kr = kb * BLOCK_K

        def read_a(pipe):
            tmp_w = tmp_a_cb.reserve()
            ttl.copy(a[mr : mr + BLOCK_M, kr : kr + BLOCK_K], tmp_w)
            tmp_r = tmp_a_cb.wait()
            ttl.copy(tmp_r, pipe)

        mcast_a_net.if_src(read_a)

        def recv_a(pipe):
            a_blk_dm = a_cb.reserve()
            ttl.copy(pipe, a_blk_dm)

        mcast_a_net.if_dst(recv_a)

        def read_b(pipe):
            tmp_w = tmp_b_cb.reserve()
            ttl.copy(w[kr : kr + BLOCK_K, nc : nc + BLOCK_N], tmp_w)
            tmp_r = tmp_b_cb.wait()
            ttl.copy(tmp_r, pipe)

        mcast_b_net.if_src(read_b)

        def recv_b(pipe):
            b_blk_dm = b_cb.reserve()
            ttl.copy(pipe, b_blk_dm)

        mcast_b_net.if_dst(recv_b)

    if node_x < NUM_COLS:

        # Reserve the receive block inside the callback: the pipe posts the
        # block's address from here, so the reserve must be co-located with the
        # post. Hoisting it out deadlocks.
        def recv_partial(pipe):
            r_dst = recv_cb.reserve()
            ttl.copy(pipe, r_dst)

        reduce_net.if_dst(recv_partial)

        # KP == 2: one received partial. Sum it with the finalized local
        # partial into the output CB.
        r = recv_cb.wait()
        o = out_cb.reserve()
        o.store(pf + r)
        out_blk_done = out_cb.wait()
        ttl.copy(out_blk_done, out[mr : mr + BLOCK_M, nc : nc + BLOCK_N])
    else:

        def send_partial(pipe):
            p_to_send = partial_for_send_cb.wait()
            ttl.copy(p_to_send, pipe)

        reduce_net.if_src(send_partial)


def test_atom_summa_ksplit(device):
    a_torch = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    w_torch = torch.randn(K, N, dtype=torch.bfloat16) * 0.1
    expected = torch.matmul(a_torch.float(), w_torch.float()).to(torch.bfloat16)

    a_dram = to_dram(a_torch, device)
    w_dram = to_dram(w_torch, device)
    out_dram = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    atom_summa_ksplit(a_dram, w_dram, out_dram)

    got = ttnn.to_torch(out_dram).reshape(M, N).to(torch.bfloat16)
    assert_pcc(expected, got, threshold=0.99)
