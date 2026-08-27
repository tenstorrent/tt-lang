# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul K-accumulation precision and multicore correctness coverage.

The parameterized precision tests compare two accumulation forms as K grows:
  - Kt=1 streaming: each K tile computes a partial result, and explicit DFB
    state accumulates those partials.
  - Kt>1 single fill: one input DFB contains the full K block, and
    compiler-generated matmul_block accumulation keeps intermediate results in
    DST when the output block fits DST capacity.

For the precision tests, PCC must exceed 0.999 and max/mean error must scale
as sqrt(K). Each K step adds an independent bf16 rounding error; for random
inputs these errors are uncorrelated. The Kt>1 DST-resident form accumulates
in f32 DST without intermediate bf16 truncation, so its bounds are tighter
than Kt=1 streaming, which stores through DFB state after each K step.

The multicore test uses the Kt>1 DST-resident form on a 2x2 grid to verify
that M/N block distribution preserves the same K-accumulation semantics across
cores.
"""

import math

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from ttl.utils.correctness import assert_pcc

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


def _make_matmul_kn_multicore(k_tiles, block_m, block_n, grid):
    """Kt>1 single fill with M/N output blocks distributed across cores."""

    @ttl.operation(grid=grid)
    def kernel(a, w, out):
        Mt = a.shape[0] // TILE
        Nt = w.shape[1] // TILE

        M_num = Mt // block_m
        N_num = Nt // block_n

        grid_n, grid_m = ttl.grid_size(dims=2)
        m_per = -(-M_num // grid_m)
        n_per = -(-N_num // grid_n)

        a_dfb = ttl.make_dataflow_buffer_like(
            a, shape=(block_m, k_tiles), block_count=2
        )
        w_dfb = ttl.make_dataflow_buffer_like(
            w, shape=(k_tiles, block_n), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(block_m, block_n), block_count=2
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_per):
                m_block = node_m * m_per + local_m
                if m_block < M_num:
                    for local_n in range(n_per):
                        n_block = node_n * n_per + local_n
                        if n_block < N_num:
                            with (
                                a_dfb.wait() as a_blk,
                                w_dfb.wait() as w_blk,
                                out_dfb.reserve() as out_blk,
                            ):
                                out_blk.store(a_blk @ w_blk)

        @ttl.datamovement()
        def dm_read():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_per):
                m_block = node_m * m_per + local_m
                if m_block < M_num:
                    m_offset = m_block * block_m
                    for local_n in range(n_per):
                        n_block = node_n * n_per + local_n
                        if n_block < N_num:
                            n_offset = n_block * block_n
                            with a_dfb.reserve() as a_blk:
                                ttl.copy(
                                    a[m_offset : m_offset + block_m, 0:k_tiles],
                                    a_blk,
                                ).wait()
                            with w_dfb.reserve() as w_blk:
                                ttl.copy(
                                    w[0:k_tiles, n_offset : n_offset + block_n],
                                    w_blk,
                                ).wait()

        @ttl.datamovement()
        def dm_write():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_per):
                m_block = node_m * m_per + local_m
                if m_block < M_num:
                    m_offset = m_block * block_m
                    for local_n in range(n_per):
                        n_block = node_n * n_per + local_n
                        if n_block < N_num:
                            n_offset = n_block * block_n
                            with out_dfb.wait() as out_blk:
                                ttl.copy(
                                    out_blk,
                                    out[
                                        m_offset : m_offset + block_m,
                                        n_offset : n_offset + block_n,
                                    ],
                                ).wait()

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
    """Kt>1 single-fill accumulation.

    When the output block fits in DST (block_n <= 4 for f32), matmul_block
    accumulates all K tiles in f32 DST with one bf16 truncation at the end
    (tighter bounds). When the output exceeds DST capacity (block_n > 4
    for f32), the compiler tiles K to 1 for L1 accumulation, producing one
    bf16 truncation per K step (same bounds as the streaming test).
    """
    scale = math.sqrt(k_tiles)
    # DST capacity with fp32_dest_acc_en=true is 4. Output block is
    # 1 x block_n. When block_n > 4, L1 acc activates with per-K-step
    # bf16 truncation, requiring relaxed error bounds.
    uses_l1_acc = block_n > 4
    if uses_l1_acc:
        max_err = 0.5 * scale
        mean_err = 0.05 * scale
    else:
        max_err = 0.1 * scale
        mean_err = 0.01 * scale
    _run(
        _make_matmul_kn,
        k_tiles,
        block_n,
        device,
        max_err_limit=max_err,
        mean_err_limit=mean_err,
    )


@pytest.mark.requires_device
def test_matmul_dst_resident_accumulation_multicore(device):
    """DST-resident K accumulation remains correct across a 2x2 grid."""
    k_tiles = 8
    block_m = 2
    block_n = 2
    m_blocks = 4
    n_blocks = 4

    m_dim = block_m * m_blocks * TILE
    k_dim = k_tiles * TILE
    n_dim = block_n * n_blocks * TILE

    torch.manual_seed(9001)
    a = torch.randn(m_dim, k_dim, dtype=torch.bfloat16)
    w = torch.randn(k_dim, n_dim, dtype=torch.bfloat16)
    golden = (a.float() @ w.float()).float()

    out = to_dram(torch.zeros(m_dim, n_dim, dtype=torch.bfloat16), device)
    kernel = _make_matmul_kn_multicore(k_tiles, block_m, block_n, grid=(2, 2))
    kernel(to_dram(a, device), to_dram(w, device), out)

    result = ttnn.to_torch(out).float()
    assert_pcc(golden, result, threshold=0.999)
