# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression for issue #541: a launch grid larger than the PipeNet work
extent must produce correct results when the user wraps pipe-coupled work
in `if net.is_active():`.

The kernels here define pipes against the work extent (M_BLOCKS, N_BLOCKS).
With `grid="full"` the launch covers the entire device compute grid; nodes
outside the active set short-circuit through the explicit `is_active()`
guard rather than executing the body with out-of-bounds tensor indices or
breaking the multicast handshake.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32
BLOCK_M = 4
BLOCK_N = 4
BLOCK_K = 4
BLOCK_SIZE = BLOCK_M * TILE  # 128


def _make_small_mcast_kernel(M_DIM, K_DIM, N_DIM):
    """2D multicast matmul where work extent (M_BLOCKS, N_BLOCKS) is smaller
    than the auto-selected launch grid.

    Each node (col, row) with row < M_BLOCKS and col < N_BLOCKS computes one
    output block. All other launched nodes must be guarded out by the
    active-set pass.
    """
    M_BLOCKS = M_DIM // BLOCK_SIZE
    N_BLOCKS = N_DIM // BLOCK_SIZE
    K_BLOCKS = K_DIM // BLOCK_SIZE

    @ttl.operation(grid="full")
    def small_mcast_matmul(a, w, out):
        a_pipes = [
            ttl.Pipe(src=(0, row), dst=(slice(1, N_BLOCKS), row))
            for row in range(M_BLOCKS)
        ]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [
            ttl.Pipe(src=(col, 0), dst=(col, slice(1, M_BLOCKS)))
            for col in range(N_BLOCKS)
        ]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_M, BLOCK_K), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(BLOCK_K, BLOCK_N), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(
            out, shape=(BLOCK_M, BLOCK_N), block_count=2
        )

        @ttl.compute()
        def compute():
            if mcast_a_net.is_active():
                with out_cb.reserve() as out_blk:
                    a_blk = a_cb.wait()
                    b_blk = b_cb.wait()
                    out_blk.store(a_blk @ b_blk)
                    a_blk.pop()
                    b_blk.pop()

        @ttl.datamovement()
        def dm_read():
            if mcast_a_net.is_active():
                node_n, node_m = ttl.node(dims=2)
                mb = node_m
                mr = mb * BLOCK_M
                nb = node_n
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
            if mcast_a_net.is_active():
                node_n, node_m = ttl.node(dims=2)
                mb = node_m
                mr = mb * BLOCK_M
                nb = node_n
                nc = nb * BLOCK_N
                with out_cb.wait() as out_blk:
                    ttl.copy(out_blk, out[mr : mr + BLOCK_M, nc : nc + BLOCK_N]).wait()

    return small_mcast_matmul


@pytest.mark.parametrize(
    "shape",
    [
        (4, 3, 1),
        (2, 2, 1),
    ],
    ids=["4x3", "2x2"],
)
def test_pipenet_active_guard_under_full_grid(shape, device):
    M_BLOCKS, N_BLOCKS, K_BLOCKS = shape
    M = M_BLOCKS * BLOCK_SIZE
    N = N_BLOCKS * BLOCK_SIZE
    K = K_BLOCKS * BLOCK_SIZE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w_torch = torch.randn(K, N, dtype=torch.bfloat16) * 0.02

    a_tt = to_dram(a_torch, device)
    w_tt = to_dram(w_torch, device)
    out_tt = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = _make_small_mcast_kernel(M, K, N)
    kernel(a_tt, w_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    expected = ttnn.to_torch(ttnn.matmul(a_tt, w_tt))
    assert_pcc(expected.float(), result.float(), threshold=0.99)
