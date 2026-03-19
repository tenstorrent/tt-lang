# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone matmul tests with varying M and N block sizes (K=1).

TODO: K>1 requires a user-level K-loop with per-step DFBs and DST
accumulation (acc=True or explicit temp-CB pattern). The tt-mlir
experimental::matmul_block wrapper supports kt_dim>1 but only has 1x1
test coverage; the proven tt-metal pattern uses kt_dim=1 with an external
K-loop. K>1 tests are deferred until accumulation support lands.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram, to_l1

TILE = 32


@ttl.kernel(grid=(1, 1))
def matmul_kernel(a, b, out):
    Mt = a.shape[0] // TILE
    Kt = a.shape[1] // TILE
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(Mt, Kt), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(Kt, Nt), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        o = out_dfb.reserve()
        result = a_blk @ b_blk
        o.store(result)
        a_blk.pop()
        b_blk.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:Mt, 0:Kt], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:Kt, 0:Nt], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:Mt, 0:Nt])
            tx.wait()


@pytest.mark.parametrize(
    "Mt,Kt,Nt",
    [
        (1, 1, 1),  # Minimal: single tile.
        (2, 1, 2),  # Multi-tile output, K=1 outer product.
        (2, 1, 3),  # Non-square output, K=1.
        (1, 1, 4),  # Wide output, K=1.
        (4, 1, 1),  # Tall output, K=1.
        (2, 1, 4),  # 2x4 output = 8 tiles, max for bf16 DST.
        (4, 1, 2),  # 4x2 output = 8 tiles, tall rectangle at max DST.
        (1, 1, 8),  # 1x8 output = 8 tiles, wide at max DST.
    ],
    ids=["1x1x1", "2x1x2", "2x1x3", "1x1x4", "4x1x1", "2x1x4", "4x1x2", "1x1x8"],
)
@pytest.mark.requires_device
def test_matmul_block_sizes(Mt, Kt, Nt, device):
    """Standalone matmul with varying block dimensions."""
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    out_torch = torch.zeros(M, N, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(out_torch, device)

    matmul_kernel(a, b, out)

    result = ttnn.to_torch(out)
    golden = a_torch @ b_torch

    # bf16 matmul with K accumulation: PCC > 0.999 is the standard threshold.
    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.999, (
        f"PCC {pcc:.6f} < 0.999 for {Mt}x{Kt}x{Nt} matmul. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )


@pytest.mark.requires_device
def test_matmul_m_tiling_distinct(device):
    """M-tiling with distinct fixed values verifies tile indexing.

    A[2x1] @ B[1x1] = C[2x1]. Top tile = 1, bottom tile = 3, B = ones.
    C top = 32, C bottom = 96. A tile swap would produce 96 on top.
    """
    a_torch = torch.ones(2 * TILE, TILE, dtype=torch.bfloat16)
    a_torch[TILE:, :] = 3.0
    b_torch = torch.ones(TILE, TILE, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(2 * TILE, TILE, dtype=torch.bfloat16), device)

    matmul_kernel(a, b, out)

    result = ttnn.to_torch(out)
    golden = a_torch @ b_torch
    assert_allclose(result.float(), golden.float(), rtol=1e-2, atol=1e-1)


@pytest.mark.requires_device
def test_matmul_n_tiling_distinct(device):
    """N-tiling with distinct per-tile random values verifies tile indexing.

    A[1x1] @ B[1x2] = C[1x2]. B's column tiles have different value ranges
    so a tile-indexing bug would produce visibly wrong per-tile results.
    """
    a_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5
    b_torch = torch.zeros(TILE, 2 * TILE, dtype=torch.bfloat16)
    b_torch[:, :TILE] = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5 + 1.0
    b_torch[:, TILE:] = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5 - 1.0

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(TILE, 2 * TILE, dtype=torch.bfloat16), device)

    matmul_kernel(a, b, out)

    result = ttnn.to_torch(out)
    golden = a_torch @ b_torch
    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.999, (
        f"PCC {pcc:.6f} < 0.999 for N-tiling distinct matmul. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )


@pytest.mark.requires_device
def test_matmul_l1(device):
    """Matmul with tensors in L1 memory (instead of DRAM)."""
    a_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    b_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    out = to_l1(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    matmul_kernel(a, b, out)

    result = ttnn.to_torch(out)
    golden = a_torch @ b_torch
    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.999, f"PCC {pcc:.6f} < 0.999 for L1 matmul"
