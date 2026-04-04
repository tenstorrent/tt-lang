# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone matmul tests with varying M, N, and K block sizes.

When Kt > 1 in the DFB block shape, the compiler emits matmul_block with
kt_dim > 1 so the hardware accumulates across K tiles internally in DST,
avoiding CB round-trips per K step.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram, to_l1

TILE = 32


def _make_matmul_kernel(buffer_factor=2):
    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE

        a_dfb = ttl.make_dataflow_buffer_like(
            a, shape=(Mt, Kt), buffer_factor=buffer_factor
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b, shape=(Kt, Nt), buffer_factor=buffer_factor
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(Mt, Nt), buffer_factor=buffer_factor
        )

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

    return kernel


matmul_kernel = _make_matmul_kernel(buffer_factor=2)


BF16_SHAPES = [
    # K=1: varying M and N block sizes.
    (1, 1, 1),  # Minimal: single tile.
    (2, 1, 2),  # Multi-tile output, K=1 outer product.
    (2, 1, 3),  # Non-square output, K=1.
    (1, 1, 4),  # Wide output, K=1.
    (4, 1, 1),  # Tall output, K=1.
    (2, 1, 4),  # 2x4 output = 8 tiles, max for bf16 DST.
    (4, 1, 2),  # 4x2 output = 8 tiles, tall rectangle at max DST.
    (1, 1, 8),  # 1x8 output = 8 tiles, wide at max DST.
    # K > 1: hardware accumulates across K tiles in DST.
    (1, 2, 1),  # Minimal K > 1.
    (2, 4, 2),  # M*N=4, moderate K.
    (1, 8, 1),  # Large K, minimal output.
    (2, 8, 2),  # K > 1 with multi-tile M and N.
    # M*N > DST capacity with K > 1: auto-subblocked.
    (4, 4, 4),  # M*N=16 > f32 DST(4), heavy subblocking + K loop.
    (8, 2, 8),  # M*N=64, 16 subblocks at f32 DST(4), K=2.
    (1, 4, 8),  # Asymmetric: wide output, subblock ct != fullN.
    (4, 1, 4),  # K=1 with f32 subblocking regression check.
    (3, 5, 3),  # Odd dimensions, subblocking falls back to 1x1.
    # L1 pressure (moderate): 384 tiles * 2KB = 768 KB (~53%).
    (8, 8, 8),
    # L1 pressure (high): 672 tiles * 2KB = 1344 KB (~93%).
    (12, 8, 12),
]

# f32 tiles are 2x larger (4KB vs 2KB), so L1 limits are tighter.
# Selected shapes cover: K=1, K>1, subblocking, odd dims, and high L1.
F32_SHAPES = [
    (1, 1, 1),  # Minimal: single tile.
    (2, 1, 2),  # Multi-tile output, K=1.
    (1, 1, 4),  # Wide output, fills f32 DST(4) exactly.
    (2, 1, 4),  # M*N=8 > f32 DST(4), requires subblocking.
    (1, 2, 1),  # Minimal K > 1.
    (2, 4, 2),  # M*N=4, moderate K.
    (1, 8, 1),  # Large K, minimal output.
    (4, 4, 4),  # M*N=16, heavy subblocking + K loop.
    (1, 4, 8),  # Asymmetric: wide output, subblock ct != fullN.
    (3, 5, 3),  # Odd dimensions.
    # L1 pressure (high): 322 tiles * 4KB = 1288 KB (~89%).
    (7, 8, 7),
]

DTYPE_SHAPES = [(shape, torch.bfloat16) for shape in BF16_SHAPES] + [
    (shape, torch.float32) for shape in F32_SHAPES
]

DTYPE_IDS = [
    f"{m}x{k}x{n}_{'bf16' if dt == torch.bfloat16 else 'f32'}"
    for (m, k, n), dt in DTYPE_SHAPES
]


@pytest.mark.parametrize("shape,dtype", DTYPE_SHAPES, ids=DTYPE_IDS)
@pytest.mark.requires_device
def test_matmul_block_sizes(shape, dtype, device):
    """Standalone matmul with varying block dimensions and dtypes."""
    Mt, Kt, Nt = shape
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=dtype)
    b_torch = torch.randn(K, N, dtype=dtype)
    out_torch = torch.zeros(M, N, dtype=dtype)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(out_torch, device)

    matmul_kernel(a, b, out)

    result = ttnn.to_torch(out)
    golden = a_torch @ b_torch

    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    threshold = 0.9999 if dtype == torch.float32 else 0.999
    assert pcc > threshold, (
        f"PCC {pcc:.6f} < {threshold} for {Mt}x{Kt}x{Nt} "
        f"{'f32' if dtype == torch.float32 else 'bf16'} matmul. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )


# buffer_factor=1 (single-buffered) and buffer_factor=3 (triple-buffered)
# exercise different CB allocation sizes and synchronization patterns.
# Shapes chosen to cover K=1, K>1, and subblocking at each buffer factor.
BUFFER_FACTOR_PARAMS = [
    # (shape, dtype, buffer_factor)
    # buffer_factor=1: single-buffered, no overlap between DM and compute.
    ((1, 1, 1), torch.bfloat16, 1),
    ((2, 4, 2), torch.bfloat16, 1),
    ((4, 4, 4), torch.bfloat16, 1),
    ((2, 4, 2), torch.float32, 1),
    # buffer_factor=3: triple-buffered, extra L1 per CB.
    ((1, 1, 1), torch.bfloat16, 3),
    ((2, 4, 2), torch.bfloat16, 3),
    ((4, 4, 4), torch.bfloat16, 3),
    ((2, 4, 2), torch.float32, 3),
    ((4, 4, 4), torch.float32, 3),
]

BUFFER_FACTOR_IDS = [
    f"{m}x{k}x{n}_{'bf16' if dt == torch.bfloat16 else 'f32'}_bf{bf}"
    for (m, k, n), dt, bf in BUFFER_FACTOR_PARAMS
]


@pytest.mark.parametrize(
    "shape,dtype,buffer_factor", BUFFER_FACTOR_PARAMS, ids=BUFFER_FACTOR_IDS
)
@pytest.mark.requires_device
def test_matmul_buffer_factor(shape, dtype, buffer_factor, device):
    """Matmul with non-default buffer_factor (single- and triple-buffered)."""
    Mt, Kt, Nt = shape
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=dtype)
    b_torch = torch.randn(K, N, dtype=dtype)
    out_torch = torch.zeros(M, N, dtype=dtype)

    out = to_dram(out_torch, device)
    kernel = _make_matmul_kernel(buffer_factor=buffer_factor)
    kernel(to_dram(a_torch, device), to_dram(b_torch, device), out)

    result = ttnn.to_torch(out).float()
    golden = (a_torch @ b_torch).float()

    pcc = torch.corrcoef(torch.stack([result.flatten(), golden.flatten()]))[0, 1].item()
    threshold = 0.9999 if dtype == torch.float32 else 0.999
    assert pcc > threshold, (
        f"PCC {pcc:.6f} < {threshold} for {Mt}x{Kt}x{Nt} " f"bf={buffer_factor} matmul"
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
