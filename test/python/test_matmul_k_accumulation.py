# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul K-accumulation precision: verify error stays bounded as K grows.

Two strategies tested:
  - Kt=1 streaming: explicit accumulation with separate partial and acc DFBs.
  - Kt>1 single fill: entire K in one DFB, compiler-generated K loop.

Accuracy requirements (PCC > 0.999, max/mean error scaling as sqrt(K)):
  - Each K step adds an independent bf16 rounding error. For random inputs
    these errors are uncorrelated, so the accumulated error grows as
    O(sqrt(K)) (random walk).
  - Kt>1 with matmul_full_fp32 accumulates in f32 DST without intermediate
    bf16 truncation, so error bounds are tighter than Kt=1 which truncates
    to bf16 at each CB round-trip.
  - Error that grows faster than sqrt(K) indicates a correctness bug
    (e.g., wrong tile indexing, missing subblocking, or fp32_dest_acc_en
    not propagated to all compute ops).
"""

import math

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32


def _make_matmul_k1(k_tiles, block_n):
    """Kt=1 streaming: explicit accumulation via partial + acc DFBs."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, w, out):
        Nt = w.shape[1] // TILE

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(1, block_n), block_count=2)
        mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, block_n), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, block_n), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, block_n), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(Nt // block_n):
                with a_dfb.wait() as av, w_dfb.wait() as wv, mm_dfb.reserve() as mm:
                    mm.store(av @ wv)
                with mm_dfb.wait() as mv, acc_dfb.reserve() as acc:
                    acc.store(mv)
                for _ in range(k_tiles - 1):
                    with a_dfb.wait() as av, w_dfb.wait() as wv, mm_dfb.reserve() as mm:
                        mm.store(av @ wv)
                    with (
                        mm_dfb.wait() as mv,
                        acc_dfb.wait() as old,
                        acc_dfb.reserve() as new,
                    ):
                        new.store(old + mv)
                with acc_dfb.wait() as final, out_dfb.reserve() as o:
                    o.store(final)

        @ttl.datamovement()
        def dm_read():
            for ni in range(Nt // block_n):
                n_off = ni * block_n
                for kt in range(k_tiles):
                    with a_dfb.reserve() as blk:
                        ttl.copy(a[0, kt], blk).wait()
                    with w_dfb.reserve() as blk:
                        ttl.copy(w[kt, n_off : n_off + block_n], blk).wait()

        @ttl.datamovement()
        def dm_write():
            for ni in range(Nt // block_n):
                n_off = ni * block_n
                with out_dfb.wait() as blk:
                    ttl.copy(blk, out[0, n_off : n_off + block_n]).wait()

    return kernel


def _make_matmul_kn(k_tiles, block_n):
    """Kt>1 single fill: entire K in one DFB, compiler K loop."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, w, out):
        Nt = w.shape[1] // TILE

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, k_tiles), block_count=2)
        w_dfb = ttl.make_dataflow_buffer_like(
            w, shape=(k_tiles, block_n), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, block_n), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(Nt // block_n):
                a_blk = a_dfb.wait()
                w_blk = w_dfb.wait()
                with out_dfb.reserve() as out_blk:
                    out_blk.store(a_blk @ w_blk)
                a_blk.pop()
                w_blk.pop()

        @ttl.datamovement()
        def dm_read():
            for ni in range(Nt // block_n):
                n_off = ni * block_n
                with a_dfb.reserve() as blk:
                    ttl.copy(a[0, 0:k_tiles], blk).wait()
                with w_dfb.reserve() as blk:
                    ttl.copy(w[0:k_tiles, n_off : n_off + block_n], blk).wait()

        @ttl.datamovement()
        def dm_write():
            for ni in range(Nt // block_n):
                n_off = ni * block_n
                with out_dfb.wait() as blk:
                    ttl.copy(blk, out[0, n_off : n_off + block_n]).wait()

    return kernel


def _run(make_fn, k_tiles, block_n, device, max_err_limit, mean_err_limit):
    k_dim = k_tiles * TILE
    n_dim = max(256, block_n * TILE)
    torch.manual_seed(42 + k_tiles)
    a = torch.randn(32, k_dim, dtype=torch.bfloat16)
    w = torch.randn(k_dim, n_dim, dtype=torch.bfloat16)
    golden = (a.float() @ w.float()).float()
    out = to_dram(torch.zeros(32, n_dim, dtype=torch.bfloat16), device)
    kernel = make_fn(k_tiles, block_n)
    kernel(to_dram(a, device), to_dram(w, device), out)
    result = ttnn.to_torch(out).float()
    assert_pcc(golden, result, threshold=0.999)
    max_err = (result - golden).abs().max().item()
    mean_err = (result - golden).abs().mean().item()
    assert (
        max_err < max_err_limit
    ), f"MaxErr {max_err:.4f} exceeds limit {max_err_limit} at K={k_tiles}"
    assert (
        mean_err < mean_err_limit
    ), f"MeanErr {mean_err:.4f} exceeds limit {mean_err_limit} at K={k_tiles}"


K_TILES = [2, 4, 8, 16, 32]
BLOCK_NS = [2, 4, 8]  # 2: fits f32 DST; 4: exact f32 DST; 8: requires subblocking.


@pytest.mark.parametrize("block_n", BLOCK_NS, ids=[f"N{n}" for n in BLOCK_NS])
@pytest.mark.parametrize("k_tiles", K_TILES, ids=[f"K{k}" for k in K_TILES])
@pytest.mark.requires_device
def test_matmul_k_accumulation_streaming(k_tiles, block_n, device):
    """Kt=1 streaming accumulation: error scales with sqrt(K)."""
    scale = math.sqrt(k_tiles)
    _run(
        _make_matmul_k1,
        k_tiles,
        block_n,
        device,
        max_err_limit=0.5 * scale,
        mean_err_limit=0.05 * scale,
    )


@pytest.mark.parametrize("block_n", BLOCK_NS, ids=[f"N{n}" for n in BLOCK_NS])
@pytest.mark.parametrize("k_tiles", K_TILES, ids=[f"K{k}" for k in K_TILES])
@pytest.mark.requires_device
def test_matmul_k_accumulation_single_fill(k_tiles, block_n, device):
    """Kt>1 single-fill accumulation: tighter bounds (f32 DST)."""
    scale = math.sqrt(k_tiles)
    _run(
        _make_matmul_kn,
        k_tiles,
        block_n,
        device,
        max_err_limit=0.1 * scale,
        mean_err_limit=0.01 * scale,
    )
