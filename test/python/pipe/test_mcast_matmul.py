# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
2D multicast matmul using pipes.

A rows are multicast horizontally (one pipe per row of the grid).
B columns are multicast vertically (one pipe per column of the grid).

Uses grid="auto" to adapt to the available device grid.

Tests: multi-tile 8x8 blocks, mcast and balanced (two-NOC) patterns
with PipeNet named-function callbacks.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32
BLOCK_M = 8
BLOCK_N = 8
BLOCK_K = 8
BLOCK_SIZE = BLOCK_M * TILE  # 256


def _even_split(n_blocks, max_grid):
    """Pick blocks_per_node that divides n_blocks evenly."""
    bpn = -(-n_blocks // max_grid)
    while n_blocks % bpn != 0:
        bpn += 1
    return bpn, n_blocks // bpn


def make_mcast_kernel(M_DIM, K_DIM, N_DIM):
    M_BLOCKS = M_DIM // BLOCK_SIZE
    N_BLOCKS = N_DIM // BLOCK_SIZE
    K_BLOCKS = K_DIM // BLOCK_SIZE

    @ttl.operation(grid="auto")
    def mcast_matmul(a, w, out):
        NUM_COLS, NUM_ROWS = ttl.grid_size(dims=2)
        m_blocks_per_node, _ = _even_split(M_BLOCKS, NUM_ROWS)
        n_blocks_per_node, _ = _even_split(N_BLOCKS, NUM_COLS)

        a_pipes = [
            ttl.Pipe(src=(0, row), dst=(slice(0, NUM_COLS), row))
            for row in range(NUM_ROWS)
        ]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [
            ttl.Pipe(src=(col, 0), dst=(col, slice(0, NUM_ROWS)))
            for col in range(NUM_COLS)
        ]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_M, BLOCK_K), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(BLOCK_K, BLOCK_N), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(
            out, shape=(BLOCK_M, BLOCK_N), block_count=2
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                for local_nb in range(n_blocks_per_node):
                    nb = node_n * n_blocks_per_node + local_nb
                    out_blk = out_cb.reserve()
                    for _ in range(K_BLOCKS):
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        out_blk += a_blk @ b_blk
                        a_blk.pop()
                        b_blk.pop()
                    out_blk.push()

        @ttl.datamovement()
        def dm_read():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                mr = mb * BLOCK_M
                for local_nb in range(n_blocks_per_node):
                    nb = node_n * n_blocks_per_node + local_nb
                    nc = nb * BLOCK_N
                    for kb in range(K_BLOCKS):
                        kc = kb * BLOCK_K
                        with a_cb.reserve() as a_blk:

                            def read_a(pipe):
                                ttl.copy(
                                    a[mr : mr + BLOCK_M, kc : kc + BLOCK_K], a_blk
                                ).wait()
                                ttl.copy(a_blk, pipe).wait()

                            mcast_a_net.if_src(read_a)

                            def recv_a(pipe):
                                ttl.copy(pipe, a_blk).wait()

                            mcast_a_net.if_dst(recv_a)

                        with b_cb.reserve() as b_blk:

                            def read_b(pipe):
                                ttl.copy(
                                    w[kc : kc + BLOCK_K, nc : nc + BLOCK_N], b_blk
                                ).wait()
                                ttl.copy(b_blk, pipe).wait()

                            mcast_b_net.if_src(read_b)

                            def recv_b(pipe):
                                ttl.copy(pipe, b_blk).wait()

                            mcast_b_net.if_dst(recv_b)

        @ttl.datamovement()
        def dm_write():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                mr = mb * BLOCK_M
                for local_nb in range(n_blocks_per_node):
                    nb = node_n * n_blocks_per_node + local_nb
                    nc = nb * BLOCK_N
                    with out_cb.wait() as out_blk:
                        ttl.copy(
                            out_blk, out[mr : mr + BLOCK_M, nc : nc + BLOCK_N]
                        ).wait()

    return mcast_matmul


def make_balanced_kernel(M_DIM, K_DIM, N_DIM):
    """Balanced matmul: A on dm_read, B on dm_write (two-NOC pattern)."""
    M_BLOCKS = M_DIM // BLOCK_SIZE
    N_BLOCKS = N_DIM // BLOCK_SIZE
    K_BLOCKS = K_DIM // BLOCK_SIZE

    @ttl.operation(grid="auto")
    def balanced_matmul(a, w, out):
        NUM_COLS, NUM_ROWS = ttl.grid_size(dims=2)
        m_blocks_per_node, _ = _even_split(M_BLOCKS, NUM_ROWS)
        n_blocks_per_node, _ = _even_split(N_BLOCKS, NUM_COLS)

        a_pipes = [
            ttl.Pipe(src=(0, row), dst=(slice(0, NUM_COLS), row))
            for row in range(NUM_ROWS)
        ]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [
            ttl.Pipe(src=(col, 0), dst=(col, slice(0, NUM_ROWS)))
            for col in range(NUM_COLS)
        ]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_M, BLOCK_K), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(BLOCK_K, BLOCK_N), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(
            out, shape=(BLOCK_M, BLOCK_N), block_count=2
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                for local_nb in range(n_blocks_per_node):
                    nb = node_n * n_blocks_per_node + local_nb
                    out_blk = out_cb.reserve()
                    for _ in range(K_BLOCKS):
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        out_blk += a_blk @ b_blk
                        a_blk.pop()
                        b_blk.pop()
                    out_blk.push()

        @ttl.datamovement()
        def dm_read():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                mr = mb * BLOCK_M
                for local_nb in range(n_blocks_per_node):
                    nb = node_n * n_blocks_per_node + local_nb
                    for kb in range(K_BLOCKS):
                        kc = kb * BLOCK_K
                        with a_cb.reserve() as a_blk:

                            def read_a(pipe):
                                ttl.copy(
                                    a[mr : mr + BLOCK_M, kc : kc + BLOCK_K], a_blk
                                ).wait()
                                ttl.copy(a_blk, pipe).wait()

                            mcast_a_net.if_src(read_a)

                            def recv_a(pipe):
                                ttl.copy(pipe, a_blk).wait()

                            mcast_a_net.if_dst(recv_a)

        @ttl.datamovement()
        def dm_write():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                mr = mb * BLOCK_M
                for local_nb in range(n_blocks_per_node):
                    nb = node_n * n_blocks_per_node + local_nb
                    nc = nb * BLOCK_N
                    for kb in range(K_BLOCKS):
                        kc = kb * BLOCK_K
                        with b_cb.reserve() as b_blk:

                            def read_b(pipe):
                                ttl.copy(
                                    w[kc : kc + BLOCK_K, nc : nc + BLOCK_N], b_blk
                                ).wait()
                                ttl.copy(b_blk, pipe).wait()

                            mcast_b_net.if_src(read_b)

                            def recv_b(pipe):
                                ttl.copy(pipe, b_blk).wait()

                            mcast_b_net.if_dst(recv_b)

                    with out_cb.wait() as out_blk:
                        ttl.copy(
                            out_blk, out[mr : mr + BLOCK_M, nc : nc + BLOCK_N]
                        ).wait()

    return balanced_matmul


def make_balanced_relu_kernel(M_DIM, K_DIM, N_DIM):
    """Balanced matmul + fused relu on last K iteration."""
    M_BLOCKS = M_DIM // BLOCK_SIZE
    N_BLOCKS = N_DIM // BLOCK_SIZE
    K_BLOCKS = K_DIM // BLOCK_SIZE

    @ttl.operation(grid="auto")
    def balanced_matmul_relu(a, w, out):
        NUM_COLS, NUM_ROWS = ttl.grid_size(dims=2)
        m_blocks_per_node, _ = _even_split(M_BLOCKS, NUM_ROWS)
        n_blocks_per_node, _ = _even_split(N_BLOCKS, NUM_COLS)

        a_pipes = [
            ttl.Pipe(src=(0, row), dst=(slice(0, NUM_COLS), row))
            for row in range(NUM_ROWS)
        ]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [
            ttl.Pipe(src=(col, 0), dst=(col, slice(0, NUM_ROWS)))
            for col in range(NUM_COLS)
        ]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_M, BLOCK_K), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(BLOCK_K, BLOCK_N), block_count=2)
        acc_cb = ttl.make_dataflow_buffer_like(
            out, shape=(BLOCK_M, BLOCK_N), block_count=2
        )
        out_cb = ttl.make_dataflow_buffer_like(
            out, shape=(BLOCK_M, BLOCK_N), block_count=1
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                for local_nb in range(n_blocks_per_node):
                    nb = node_n * n_blocks_per_node + local_nb
                    with acc_cb.reserve() as init:
                        init.store(ttl.math.fill(init, 0))
                    for kb in range(K_BLOCKS):
                        with (
                            a_cb.wait() as a_blk,
                            b_cb.wait() as b_blk,
                            acc_cb.wait() as last,
                            acc_cb.reserve() as next,
                        ):
                            if kb < K_BLOCKS - 1:
                                next.store(last + a_blk @ b_blk)
                            else:
                                next.store(ttl.math.relu(last + a_blk @ b_blk))
                    with acc_cb.wait() as result, out_cb.reserve() as o:
                        o.store(result)

        @ttl.datamovement()
        def dm_read():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                mr = mb * BLOCK_M
                for local_nb in range(n_blocks_per_node):
                    nb = node_n * n_blocks_per_node + local_nb
                    for kb in range(K_BLOCKS):
                        kc = kb * BLOCK_K
                        with a_cb.reserve() as a_blk:

                            def read_a(pipe):
                                ttl.copy(
                                    a[mr : mr + BLOCK_M, kc : kc + BLOCK_K], a_blk
                                ).wait()
                                ttl.copy(a_blk, pipe).wait()

                            mcast_a_net.if_src(read_a)

                            def recv_a(pipe):
                                ttl.copy(pipe, a_blk).wait()

                            mcast_a_net.if_dst(recv_a)

        @ttl.datamovement()
        def dm_write():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                mr = mb * BLOCK_M
                for local_nb in range(n_blocks_per_node):
                    nb = node_n * n_blocks_per_node + local_nb
                    nc = nb * BLOCK_N
                    for kb in range(K_BLOCKS):
                        kc = kb * BLOCK_K
                        with b_cb.reserve() as b_blk:

                            def read_b(pipe):
                                ttl.copy(
                                    w[kc : kc + BLOCK_K, nc : nc + BLOCK_N], b_blk
                                ).wait()
                                ttl.copy(b_blk, pipe).wait()

                            mcast_b_net.if_src(read_b)

                            def recv_b(pipe):
                                ttl.copy(pipe, b_blk).wait()

                            mcast_b_net.if_dst(recv_b)

                    with out_cb.wait() as out_blk:
                        ttl.copy(
                            out_blk, out[mr : mr + BLOCK_M, nc : nc + BLOCK_N]
                        ).wait()

    return balanced_matmul_relu


def _run_matmul(make_kernel, M, K, N, device, golden_fn=None):
    a_torch = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w_torch = torch.randn(K, N, dtype=torch.bfloat16) * 0.02

    a_tt = to_dram(a_torch, device)
    w_tt = to_dram(w_torch, device)
    out_tt = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = make_kernel(M, K, N)
    kernel(a_tt, w_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    if golden_fn is not None:
        expected = golden_fn(a_tt, w_tt)
    else:
        expected = ttnn.to_torch(ttnn.matmul(a_tt, w_tt))
    assert_pcc(expected, result, threshold=0.99)


def test_mcast_matmul(device):
    """2D mcast matmul (both A+B on dm_read)."""
    _run_matmul(make_mcast_kernel, 10240, 8192, 13312, device)


def test_balanced_matmul(device):
    """Balanced matmul (A on dm_read, B on dm_write)."""
    _run_matmul(make_balanced_kernel, 10240, 8192, 13312, device)


def test_balanced_matmul_relu(device):
    """Balanced matmul + fused relu."""

    def golden_relu(a_tt, w_tt):
        return ttnn.to_torch(ttnn.relu(ttnn.matmul(a_tt, w_tt)))

    _run_matmul(
        make_balanced_relu_kernel, 10240, 8192, 13312, device, golden_fn=golden_relu
    )
