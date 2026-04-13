# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul L1 accumulation via += across K iterations.

The += operator emits ttl.store with {accumulate}, which the compiler
detects and annotates for L1 packer accumulation. Each K iteration packs
additively to L1.

Tests single-core and multicore configurations with various block sizes.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v --tb=short

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32


def _make_l1_acc_kernel(block_m, block_n, grid="auto"):
    """Matmul with L1 accumulation via += across K iterations."""

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


# Single-core tests (grid=(1,1))
SINGLE_CORE_PARAMS = [
    # (block_m, block_n, Kt)
    (2, 2, 2),  # Output 2x2=4 fits in f32 DST
    (2, 2, 4),  # K=4
    (3, 3, 2),  # Output 3x3=9 > f32 DST(4)
    (4, 4, 4),  # Output 4x4=16 > f32 DST(4)
    (8, 8, 2),  # Large output, small K
    (8, 8, 8),  # Large output, large K
]


@pytest.mark.parametrize(
    "block_m,block_n,Kt",
    SINGLE_CORE_PARAMS,
    ids=[f"blk{m}x{n}_K{k}" for m, n, k in SINGLE_CORE_PARAMS],
)
@pytest.mark.requires_device
def test_l1_acc_single_core(block_m, block_n, Kt, device):
    """L1 accumulation on single core with various block sizes."""
    M, K, N = block_m * TILE, Kt * TILE, block_n * TILE
    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    golden = (a_torch.float() @ b_torch.float()).float()

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = _make_l1_acc_kernel(block_m, block_n, grid=(1, 1))
    kernel(a, b, out)

    result = ttnn.to_torch(out).float()
    assert_pcc(golden, result, threshold=0.999)


# Multicore tests (grid="auto") with multiple output blocks
MULTI_CORE_PARAMS = [
    # (Mt, Kt, Nt, block_m, block_n)
    (16, 4, 16, 8, 8),  # 2x2 output blocks, K=4
    (32, 8, 32, 8, 8),  # 4x4 output blocks, K=8
    (128, 128, 128, 8, 8),  # 16x16 output blocks, K=128 (4096^3 shape)
]


@pytest.mark.parametrize(
    "Mt,Kt,Nt,block_m,block_n",
    MULTI_CORE_PARAMS,
    ids=[
        f"tiles{mt}x{kt}x{nt}_blk{bm}x{bn}" for mt, kt, nt, bm, bn in MULTI_CORE_PARAMS
    ],
)
@pytest.mark.requires_device
def test_l1_acc_multicore(Mt, Kt, Nt, block_m, block_n, device):
    """L1 accumulation with multicore and multiple output blocks."""
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    golden = (a_torch.float() @ b_torch.float()).float()

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = _make_l1_acc_kernel(block_m, block_n)
    kernel(a, b, out)

    result = ttnn.to_torch(out).float()
    assert_pcc(golden, result, threshold=0.999)
