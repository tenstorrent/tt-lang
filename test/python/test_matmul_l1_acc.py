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


# ---------------------------------------------------------------------------
# Non-matmul accumulation: += with a passthrough copy (sum reduction).
# ---------------------------------------------------------------------------


def _make_sum_reduction_kernel():
    """Sum K input blocks via += (no matmul)."""

    @ttl.operation(grid=(1, 1))
    def kernel(inp, out):
        Kt = inp.shape[0] // TILE
        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            for _ in range(Kt):
                inp_blk = inp_dfb.wait()
                out_blk += inp_blk
                inp_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def dm_read():
            for kt in range(Kt):
                with inp_dfb.reserve() as blk:
                    ttl.copy(inp[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.parametrize("Kt", [2, 4, 8], ids=[f"K{k}" for k in [2, 4, 8]])
@pytest.mark.requires_device
def test_l1_acc_sum_reduction(Kt, device):
    """Sum K tiles via += without matmul (passthrough accumulation)."""
    inp_torch = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    golden = inp_torch.float().reshape(Kt, TILE, TILE).sum(dim=0)

    inp_dev = to_dram(inp_torch, device)
    out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    kernel = _make_sum_reduction_kernel()
    kernel(inp_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result, threshold=0.999)


# ---------------------------------------------------------------------------
# K=1 single iteration: accumulation with one loop iteration.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_l1_acc_single_iteration(device):
    """K=1: single-iteration += loop. Semantically equivalent to plain store."""
    M, K, N = TILE, TILE, 2 * TILE
    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    golden = (a_torch.float() @ b_torch.float()).float()

    a_dev = to_dram(a_torch, device)
    b_dev = to_dram(b_torch, device)
    out_dev = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = _make_l1_acc_kernel(1, 2, grid=(1, 1))
    kernel(a_dev, b_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result, threshold=0.999)


# ---------------------------------------------------------------------------
# Consecutive += loops to the same reserve (two input streams).
# ---------------------------------------------------------------------------


def _make_consecutive_acc_kernel(K1, K2):
    """Two consecutive += loops to one output: out = (a@b summed K1) + (c@d summed K2)."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, c, d, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
        d_dfb = ttl.make_dataflow_buffer_like(d, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            for _ in range(K1):
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                out_blk += a_blk @ b_blk
                a_blk.pop()
                b_blk.pop()
            for _ in range(K2):
                c_blk = c_dfb.wait()
                d_blk = d_dfb.wait()
                out_blk += c_blk @ d_blk
                c_blk.pop()
                d_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(K1):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[0:1, kt : kt + 1], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[kt : kt + 1, 0:1], blk).wait()
            for kt in range(K2):
                with c_dfb.reserve() as blk:
                    ttl.copy(c[0:1, kt : kt + 1], blk).wait()
                with d_dfb.reserve() as blk:
                    ttl.copy(d[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
def test_l1_acc_consecutive_loops(device):
    """Two consecutive += loops to the same reserve block."""
    K1, K2 = 2, 3
    a_torch = torch.randn(TILE, K1 * TILE, dtype=torch.bfloat16)
    b_torch = torch.randn(K1 * TILE, TILE, dtype=torch.bfloat16)
    c_torch = torch.randn(TILE, K2 * TILE, dtype=torch.bfloat16)
    d_torch = torch.randn(K2 * TILE, TILE, dtype=torch.bfloat16)
    golden = (
        (a_torch.float() @ b_torch.float()) + (c_torch.float() @ d_torch.float())
    ).float()

    a_dev = to_dram(a_torch, device)
    b_dev = to_dram(b_torch, device)
    c_dev = to_dram(c_torch, device)
    d_dev = to_dram(d_torch, device)
    out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    kernel = _make_consecutive_acc_kernel(K1, K2)
    kernel(a_dev, b_dev, c_dev, d_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result, threshold=0.999)


# ---------------------------------------------------------------------------
# .store() before loop, += inside loop (overwrite then accumulate).
# ---------------------------------------------------------------------------


def _make_store_then_acc_kernel(total_k):
    """.store() before the += loop, then K-1 iterations accumulate via +=."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            a_blk = a_dfb.wait()
            b_blk = b_dfb.wait()
            out_blk.store(a_blk @ b_blk)
            a_blk.pop()
            b_blk.pop()
            for _ in range(total_k - 1):
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                out_blk += a_blk @ b_blk
                a_blk.pop()
                b_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for _ in range(total_k):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[0:1, 0:1], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[0:1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.parametrize("total_k", [2, 4], ids=[f"K{k}" for k in [2, 4]])
@pytest.mark.requires_device
def test_l1_acc_store_then_acc(total_k, device):
    """.store() before loop, += inside loop. Result = K * (a @ b)."""
    a_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    b_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    golden = (total_k * (a_torch.float() @ b_torch.float())).float()

    a_dev = to_dram(a_torch, device)
    b_dev = to_dram(b_torch, device)
    out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    kernel = _make_store_then_acc_kernel(total_k)
    kernel(a_dev, b_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result, threshold=0.999)


# ---------------------------------------------------------------------------
# Multiple += to different outputs in the same loop.
# ---------------------------------------------------------------------------


def _make_multi_output_kernel(Kt):
    """One loop with += to two independent outputs."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, c, d, out_a, out_b):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
        d_dfb = ttl.make_dataflow_buffer_like(d, shape=(1, 1), block_count=2)
        out_a_dfb = ttl.make_dataflow_buffer_like(out_a, shape=(1, 1), block_count=2)
        out_b_dfb = ttl.make_dataflow_buffer_like(out_b, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            blk_a = out_a_dfb.reserve()
            blk_b = out_b_dfb.reserve()
            for _ in range(Kt):
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                blk_a += a_blk @ b_blk
                a_blk.pop()
                b_blk.pop()
                c_blk = c_dfb.wait()
                d_blk = d_dfb.wait()
                blk_b += c_blk @ d_blk
                c_blk.pop()
                d_blk.pop()
            blk_a.push()
            blk_b.push()

        @ttl.datamovement()
        def reader():
            for kt in range(Kt):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[0:1, kt : kt + 1], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[kt : kt + 1, 0:1], blk).wait()
                with c_dfb.reserve() as blk:
                    ttl.copy(c[0:1, kt : kt + 1], blk).wait()
                with d_dfb.reserve() as blk:
                    ttl.copy(d[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_a_dfb.wait() as blk:
                ttl.copy(blk, out_a[0:1, 0:1]).wait()
            with out_b_dfb.wait() as blk:
                ttl.copy(blk, out_b[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
def test_l1_acc_multi_output(device):
    """Two independent += outputs in the same K loop."""
    Kt = 4
    a_torch = torch.randn(TILE, Kt * TILE, dtype=torch.bfloat16)
    b_torch = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    c_torch = torch.randn(TILE, Kt * TILE, dtype=torch.bfloat16)
    d_torch = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    golden_a = (a_torch.float() @ b_torch.float()).float()
    golden_b = (c_torch.float() @ d_torch.float()).float()

    a_dev = to_dram(a_torch, device)
    b_dev = to_dram(b_torch, device)
    c_dev = to_dram(c_torch, device)
    d_dev = to_dram(d_torch, device)
    out_a_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
    out_b_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    kernel = _make_multi_output_kernel(Kt)
    kernel(a_dev, b_dev, c_dev, d_dev, out_a_dev, out_b_dev)

    result_a = ttnn.to_torch(out_a_dev).float()
    result_b = ttnn.to_torch(out_b_dev).float()
    assert_pcc(golden_a, result_a, threshold=0.999)
    assert_pcc(golden_b, result_b, threshold=0.999)
