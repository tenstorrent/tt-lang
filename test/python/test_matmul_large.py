# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Large matmul tests with outer M/N/K tiling.

Three accumulation strategies for Y[M,N] = A[M,K] @ B[K,N] + C[M,N]:
  1. Kt=1 inner, fused outer K accumulation (prev + a @ b, bias preload)
  2. Kt>1 inner, no outer K loop (entire K in one DFB fill)
  3. Kt>1 inner, explicit outer K accumulation (matmul to partial, add)

TODO: update when C += A @ B is supported.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32


# =============================================================================
# Strategy 1: Kt=1 inner, fused outer K accumulation with bias.
# Modeled after examples/matmul_acc.py. Bias preloads the accumulator,
# then all K steps use prev + a @ b (fused into copy_tile + matmul_block).
# =============================================================================


def _make_matmul_bias_fused(block_m, block_n):
    @ttl.operation(grid=(1, 1))
    def kernel(a, b, c, y):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE
        num_m = Mt // block_m
        num_n = Nt // block_n

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(block_m, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, block_n), block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(c, shape=(block_m, block_n))
        acc_dfb = ttl.make_dataflow_buffer_like(
            y, shape=(block_m, block_n), block_count=2
        )
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(block_m, block_n))

        @ttl.datamovement()
        def read():
            for mi in range(num_m):
                m_off = mi * block_m
                for ni in range(num_n):
                    n_off = ni * block_n
                    with c_dfb.reserve() as blk:
                        ttl.copy(
                            c[m_off : m_off + block_m, n_off : n_off + block_n], blk
                        ).wait()
                    for kt in range(Kt):
                        with a_dfb.reserve() as ablk, b_dfb.reserve() as bblk:
                            ttl.copy(
                                a[m_off : m_off + block_m, kt : kt + 1], ablk
                            ).wait()
                            ttl.copy(
                                b[kt : kt + 1, n_off : n_off + block_n], bblk
                            ).wait()

        @ttl.compute()
        def compute():
            for _ in range(num_m):
                for _ in range(num_n):
                    with c_dfb.wait() as c_blk:
                        with acc_dfb.reserve() as acc:
                            acc.store(c_blk)
                    for _ in range(Kt):
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            acc_dfb.wait() as prev,
                        ):
                            with acc_dfb.reserve() as acc:
                                acc.store(prev + a_blk @ b_blk)
                    with acc_dfb.wait() as final:
                        with y_dfb.reserve() as out_blk:
                            out_blk.store(final)

        @ttl.datamovement()
        def write():
            for mi in range(num_m):
                m_off = mi * block_m
                for ni in range(num_n):
                    n_off = ni * block_n
                    with y_dfb.wait() as blk:
                        ttl.copy(
                            blk,
                            y[m_off : m_off + block_m, n_off : n_off + block_n],
                        ).wait()

    return kernel


# =============================================================================
# Strategy 2: Kt>1 inner, no outer K loop.
# Entire K fits in one DFB fill. Single matmul per output block.
# =============================================================================


def _make_matmul_single_k(block_m, block_n):
    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE
        num_m = Mt // block_m
        num_n = Nt // block_n

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(block_m, Kt), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(Kt, block_n), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(block_m, block_n), block_count=2
        )

        @ttl.compute()
        def compute():
            for _ in range(num_m):
                for _ in range(num_n):
                    a_blk = a_dfb.wait()
                    b_blk = b_dfb.wait()
                    with out_dfb.reserve() as out_blk:
                        out_blk.store(a_blk @ b_blk)
                    a_blk.pop()
                    b_blk.pop()

        @ttl.datamovement()
        def dm_read():
            for mi in range(num_m):
                m_off = mi * block_m
                for ni in range(num_n):
                    n_off = ni * block_n
                    with a_dfb.reserve() as blk:
                        ttl.copy(a[m_off : m_off + block_m, 0:Kt], blk).wait()
                    with b_dfb.reserve() as blk:
                        ttl.copy(b[0:Kt, n_off : n_off + block_n], blk).wait()

        @ttl.datamovement()
        def dm_write():
            for mi in range(num_m):
                m_off = mi * block_m
                for ni in range(num_n):
                    n_off = ni * block_n
                    with out_dfb.wait() as blk:
                        ttl.copy(
                            blk,
                            out[m_off : m_off + block_m, n_off : n_off + block_n],
                        ).wait()

    return kernel


# =============================================================================
# Strategy 3: Kt>1 inner, explicit outer K accumulation with bias.
# Matmul to partial DFB, then elementwise add with accumulator.
# =============================================================================


def _make_matmul_bias_explicit_k(block_m, block_k, block_n):
    @ttl.operation(grid=(1, 1))
    def kernel(a, b, c, y):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE
        num_m = Mt // block_m
        num_n = Nt // block_n
        num_k = Kt // block_k

        a_dfb = ttl.make_dataflow_buffer_like(
            a, shape=(block_m, block_k), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b, shape=(block_k, block_n), block_count=2
        )
        c_dfb = ttl.make_dataflow_buffer_like(c, shape=(block_m, block_n))
        partial_dfb = ttl.make_dataflow_buffer_like(
            y, shape=(block_m, block_n), block_count=2
        )
        acc_dfb = ttl.make_dataflow_buffer_like(
            y, shape=(block_m, block_n), block_count=2
        )
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(block_m, block_n))

        @ttl.datamovement()
        def read():
            for mi in range(num_m):
                m_off = mi * block_m
                for ni in range(num_n):
                    n_off = ni * block_n
                    with c_dfb.reserve() as blk:
                        ttl.copy(
                            c[m_off : m_off + block_m, n_off : n_off + block_n],
                            blk,
                        ).wait()
                    for kb in range(num_k):
                        k_off = kb * block_k
                        with a_dfb.reserve() as blk:
                            ttl.copy(
                                a[m_off : m_off + block_m, k_off : k_off + block_k],
                                blk,
                            ).wait()
                        with b_dfb.reserve() as blk:
                            ttl.copy(
                                b[k_off : k_off + block_k, n_off : n_off + block_n],
                                blk,
                            ).wait()

        @ttl.compute()
        def compute():
            for _ in range(num_m):
                for _ in range(num_n):
                    with c_dfb.wait() as c_blk:
                        with acc_dfb.reserve() as acc:
                            acc.store(c_blk)
                    for _ in range(num_k):
                        a_blk = a_dfb.wait()
                        b_blk = b_dfb.wait()
                        with partial_dfb.reserve() as partial:
                            partial.store(a_blk @ b_blk)
                        a_blk.pop()
                        b_blk.pop()
                        with partial_dfb.wait() as new, acc_dfb.wait() as prev:
                            with acc_dfb.reserve() as acc:
                                acc.store(prev + new)
                    with acc_dfb.wait() as final:
                        with y_dfb.reserve() as out_blk:
                            out_blk.store(final)

        @ttl.datamovement()
        def write():
            for mi in range(num_m):
                m_off = mi * block_m
                for ni in range(num_n):
                    n_off = ni * block_n
                    with y_dfb.wait() as blk:
                        ttl.copy(
                            blk,
                            y[m_off : m_off + block_m, n_off : n_off + block_n],
                        ).wait()

    return kernel


# =============================================================================
# Test parameters and harness
# =============================================================================


def _run_bias(kernel_fn, Mt, Kt, Nt, device, threshold=0.99):
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    c_torch = torch.randn(M, N, dtype=torch.bfloat16)
    y_dev = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)
    kernel_fn(
        to_dram(a_torch, device),
        to_dram(b_torch, device),
        to_dram(c_torch, device),
        y_dev,
    )
    result = ttnn.to_torch(y_dev).float()
    golden = (a_torch @ b_torch + c_torch).float()
    assert_pcc(golden, result, threshold=threshold)


def _run_matmul(kernel_fn, Mt, Kt, Nt, device, threshold=0.99):
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    out_dev = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)
    kernel_fn(to_dram(a_torch, device), to_dram(b_torch, device), out_dev)
    result = ttnn.to_torch(out_dev).float()
    golden = (a_torch @ b_torch).float()
    assert_pcc(golden, result, threshold=threshold)


BIAS_FUSED_PARAMS = [
    # (Mt, Kt, Nt, bm, bn)
    (4, 3, 4, 2, 2),  # 128x96x128 (identical to examples/matmul_acc.py).
    (8, 8, 8, 2, 2),  # 256x256x256, 4x deeper K, 4x more output blocks.
    (8, 16, 4, 2, 2),  # 256x512x128, deep K accumulation.
]

SINGLE_K_PARAMS = [
    # (Mt, Kt, Nt, bm, bn)
    (4, 4, 4, 2, 2),  # 128x128x128, K=4 in one DFB fill.
    (8, 4, 8, 2, 2),  # 256x128x256, 4x more output blocks.
]

EXPLICIT_K_PARAMS = [
    # (Mt, Kt, Nt, bm, bk, bn)
    (4, 4, 4, 2, 2, 2),  # 128x128x128, bk=2 inner, 2 outer K blocks.
    (8, 8, 8, 2, 4, 2),  # 256x256x256, bk=4 inner, 2 outer K blocks.
    (4, 8, 8, 2, 2, 2),  # 128x256x256, bk=2, 4 outer K, 4 N blocks.
]


@pytest.mark.parametrize(
    "Mt,Kt,Nt,bm,bn",
    BIAS_FUSED_PARAMS,
    ids=[f"{m}x{k}x{n}_bm{bm}bn{bn}" for m, k, n, bm, bn in BIAS_FUSED_PARAMS],
)
@pytest.mark.requires_device
def test_matmul_bias_fused(Mt, Kt, Nt, bm, bn, device):
    """Strategy 1: Kt=1 inner, fused outer K accumulation with bias."""
    _run_bias(_make_matmul_bias_fused(bm, bn), Mt, Kt, Nt, device)


@pytest.mark.parametrize(
    "Mt,Kt,Nt,bm,bn",
    SINGLE_K_PARAMS,
    ids=[f"{m}x{k}x{n}_bm{bm}bn{bn}" for m, k, n, bm, bn in SINGLE_K_PARAMS],
)
@pytest.mark.requires_device
def test_matmul_single_k(Mt, Kt, Nt, bm, bn, device):
    """Strategy 2: Kt>1 inner, entire K in one DFB fill."""
    _run_matmul(_make_matmul_single_k(bm, bn), Mt, Kt, Nt, device)


@pytest.mark.parametrize(
    "Mt,Kt,Nt,bm,bk,bn",
    EXPLICIT_K_PARAMS,
    ids=[
        f"{m}x{k}x{n}_bm{bm}bk{bk}bn{bn}" for m, k, n, bm, bk, bn in EXPLICIT_K_PARAMS
    ],
)
@pytest.mark.requires_device
def test_matmul_bias_explicit_k(Mt, Kt, Nt, bm, bk, bn, device):
    """Strategy 3: Kt>1 inner, explicit outer K accumulation with bias."""
    _run_bias(_make_matmul_bias_explicit_k(bm, bk, bn), Mt, Kt, Nt, device)
