# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multinode matmul with L1 packer accumulation via += across K iterations.
L1-only (no DRAM reads during compute). All input blocks are pre-loaded
into L1 DFBs before the K reduction loop begins.

Tests multicore configurations with a 2D grid and multiple K blocks.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32


def _make_l1_acc_multinode_kernel(block_m, block_n, grid="auto"):
    """Multinode matmul with L1 accumulation.

    All K blocks are pre-loaded into L1 before compute begins (no DRAM
    streaming during the K loop). The compute thread reserves the output
    DFB once, stores K times (triggering L1 accumulation), then pushes.
    """

    @ttl.operation(grid=grid)
    def kernel(a, b, out):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE

        M_num = Mt // block_m
        N_num = Nt // block_n

        grid_n, grid_m = ttl.grid_size(dims=2)
        m_per = -(-M_num // grid_m)
        n_per = -(-N_num // grid_n)

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(block_m, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, block_n), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(block_m, block_n), block_count=2
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for lm in range(m_per):
                mb = node_m * m_per + lm
                if mb < M_num:
                    for ln in range(n_per):
                        nb = node_n * n_per + ln
                        if nb < N_num:
                            out_blk = out_dfb.reserve()
                            for _ in range(Kt):
                                a_blk = a_dfb.wait()
                                b_blk = b_dfb.wait()
                                out_blk += a_blk @ b_blk
                                a_blk.pop()
                                b_blk.pop()
                            out_blk.push()

        @ttl.datamovement()
        def reader():
            node_n, node_m = ttl.node(dims=2)
            for lm in range(m_per):
                mb = node_m * m_per + lm
                if mb < M_num:
                    m_off = mb * block_m
                    for ln in range(n_per):
                        nb = node_n * n_per + ln
                        if nb < N_num:
                            for kt in range(Kt):
                                with a_dfb.reserve() as blk:
                                    ttl.copy(
                                        a[
                                            m_off : m_off + block_m,
                                            kt : kt + 1,
                                        ],
                                        blk,
                                    ).wait()

        @ttl.datamovement()
        def writer():
            node_n, node_m = ttl.node(dims=2)
            for lm in range(m_per):
                mb = node_m * m_per + lm
                if mb < M_num:
                    m_off = mb * block_m
                    for ln in range(n_per):
                        nb = node_n * n_per + ln
                        if nb < N_num:
                            n_off = nb * block_n
                            for kt in range(Kt):
                                with b_dfb.reserve() as blk:
                                    ttl.copy(
                                        b[
                                            kt : kt + 1,
                                            n_off : n_off + block_n,
                                        ],
                                        blk,
                                    ).wait()
                            with out_dfb.wait() as blk:
                                ttl.copy(
                                    blk,
                                    out[
                                        m_off : m_off + block_m,
                                        n_off : n_off + block_n,
                                    ],
                                ).wait()

    return kernel


PARAMS = [
    # (Mt, Kt, Nt, block_m, block_n, grid)
    (4, 2, 4, 2, 2, (2, 2)),
    (8, 4, 8, 4, 4, (2, 2)),
    (8, 4, 8, 4, 4, "auto"),
    (16, 8, 16, 8, 8, "auto"),
    # Multi-block per core: M_num=4, N_num=4 on 2x2 grid -> 2 blocks/core/axis.
    # Exercises per-block L1 acc disable/re-enable across output blocks.
    (16, 4, 16, 4, 4, (2, 2)),
]


@pytest.mark.parametrize(
    "Mt,Kt,Nt,block_m,block_n,grid",
    PARAMS,
    ids=[
        f"tiles{mt}x{kt}x{nt}_blk{bm}x{bn}_grid{g}" for mt, kt, nt, bm, bn, g in PARAMS
    ],
)
@pytest.mark.requires_device
def test_l1_acc_multinode(Mt, Kt, Nt, block_m, block_n, grid, device):
    """Multinode matmul with L1 packer accumulation across K iterations."""
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    golden = (a_torch.float() @ b_torch.float()).float()

    a_dev = to_dram(a_torch, device)
    b_dev = to_dram(b_torch, device)
    out_dev = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = _make_l1_acc_multinode_kernel(block_m, block_n, grid=grid)
    kernel(a_dev, b_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result, threshold=0.999)
