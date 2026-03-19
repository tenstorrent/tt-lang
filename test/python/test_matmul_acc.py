# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul with explicit K-accumulation using the decomposed pattern.

Each K-step: matmul to intermediate CB, then add with accumulator.
Follows the same compute-local CB pattern as simple_add_loop.py.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram

TILE = 32


@ttl.kernel(grid=(1, 1))
def matmul_acc_kernel(a, b, out):
    Mt = a.shape[0] // TILE
    Kt = a.shape[1] // TILE
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(Mt, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, Nt), buffer_factor=2)
    partial_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)
    # Compute-local accumulator. DM writer does NOT touch this.
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)
    # Output DFB: only written once after accumulation completes.
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        # First K-step: matmul directly to accumulator.
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        with acc_dfb.reserve() as acc:
            acc.store(a_blk @ b_blk)
        a_blk.pop()
        b_blk.pop()

        # Remaining K-steps: matmul to partial, add with accumulator.
        for _ in range(Kt - 1):
            a_blk = a_dfb.wait()
            b_blk = b_dfb.wait()
            with partial_dfb.reserve() as p:
                p.store(a_blk @ b_blk)
            a_blk.pop()
            b_blk.pop()

            with partial_dfb.wait() as new, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + new)

        # Copy final accumulator to output (single push, DM writer sees this).
        with acc_dfb.wait() as final:
            with out_dfb.reserve() as o:
                o.store(final)

    @ttl.datamovement()
    def dm_read():
        for kt in range(Kt):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:Mt, kt : kt + 1], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[kt : kt + 1, 0:Nt], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:Mt, 0:Nt])
            tx.wait()


@pytest.mark.parametrize(
    "Mt,Kt,Nt",
    [
        (1, 1, 1),
        (1, 2, 1),
        (1, 4, 1),
        (2, 1, 2),
        (2, 2, 2),
        (2, 4, 2),
        (1, 3, 1),  # Odd K.
        (2, 2, 4),  # 2x4 output = 8 tiles at max DST, K=2.
        (1, 4, 4),  # Wide output with K accumulation.
        (1, 2, 2),  # Non-square: A[1,2] @ B[2,2] = C[1,2].
    ],
    ids=[
        "1x1x1",
        "1x2x1",
        "1x4x1",
        "2x1x2",
        "2x2x2",
        "2x4x2",
        "1x3x1",
        "2x2x4",
        "1x4x4",
        "1x2x2",
    ],
)
@pytest.mark.requires_device
def test_matmul_accumulate(Mt, Kt, Nt, device):
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    matmul_acc_kernel(a, b, out)

    result = ttnn.to_torch(out)
    golden = a_torch @ b_torch

    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.99, (
        f"PCC {pcc:.6f} < 0.99 for {Mt}x{Kt}x{Nt} matmul. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )


# =============================================================================
# Matmul + broadcast bias: Y = (A @ B) + broadcast_col(bias)
# =============================================================================


@ttl.kernel(grid=(1, 1))
def matmul_bcast_bias_kernel(a, b, bias, out):
    """Matmul with column-broadcast bias: Y[Mt,Nt] = (A @ B) + bcast(bias[1,Nt])."""
    Mt = a.shape[0] // TILE
    Kt = a.shape[1] // TILE
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(Mt, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, Nt), buffer_factor=2)
    partial_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, Nt), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        # K-accumulation (same as matmul_acc_kernel).
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        with acc_dfb.reserve() as acc:
            acc.store(a_blk @ b_blk)
        a_blk.pop()
        b_blk.pop()

        for _ in range(Kt - 1):
            a_blk = a_dfb.wait()
            b_blk = b_dfb.wait()
            with partial_dfb.reserve() as p:
                p.store(a_blk @ b_blk)
            a_blk.pop()
            b_blk.pop()
            with partial_dfb.wait() as new, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + new)

        # Add broadcast bias: bcast(bias[1,Nt]) -> (Mt,Nt), then add with acc.
        with bias_dfb.wait() as bias_blk, acc_dfb.wait() as acc_blk:
            with out_dfb.reserve() as o:
                bias_expanded = ttl.math.broadcast(bias_blk, o, dims=[0])
                o.store(bias_expanded + acc_blk)

    @ttl.datamovement()
    def dm_read():
        for kt in range(Kt):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:Mt, kt : kt + 1], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[kt : kt + 1, 0:Nt], blk)
                tx.wait()
        with bias_dfb.reserve() as blk:
            tx = ttl.copy(bias[0:1, 0:Nt], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:Mt, 0:Nt])
            tx.wait()


@pytest.mark.parametrize(
    "Mt,Kt,Nt",
    [
        (1, 1, 1),
        (2, 1, 2),
        (2, 2, 2),
    ],
    ids=["1x1x1", "2x1x2", "2x2x2"],
)
@pytest.mark.requires_device
def test_matmul_bcast_bias(Mt, Kt, Nt, device):
    """Matmul + column-broadcast bias addition."""
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    bias_torch = torch.randn(TILE, N, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    bias = to_dram(bias_torch, device)
    out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    matmul_bcast_bias_kernel(a, b, bias, out)

    result = ttnn.to_torch(out)
    # Broadcast bias: (1 tile row, Nt cols) -> (Mt tile rows, Nt cols).
    bias_expanded = bias_torch.repeat(Mt, 1)
    golden = a_torch @ b_torch + bias_expanded

    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    # Relaxed threshold: matmul + broadcast + add chain in bf16 accumulates
    # more rounding error than matmul alone.
    assert pcc > 0.95, (
        f"PCC {pcc:.6f} < 0.95 for {Mt}x{Kt}x{Nt} matmul+bcast_bias. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )


# =============================================================================
# Distinct per-tile values: verifies tile indexing in multi-tile matmul
# =============================================================================


@pytest.mark.requires_device
def test_matmul_distinct_tiles(device):
    """[2x2] @ [2x2] = [2x2] with distinct random values per tile.

    Each tile is filled with random values from a different range so that
    a tile-indexing bug would produce visibly wrong per-tile results.
    """
    a_torch = torch.zeros(2 * TILE, 2 * TILE, dtype=torch.bfloat16)
    a_torch[:TILE, :TILE] = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5 + 1.0
    a_torch[:TILE, TILE:] = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5 - 1.0
    a_torch[TILE:, :TILE] = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5 + 2.0
    a_torch[TILE:, TILE:] = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5 - 2.0
    b_torch = torch.randn(2 * TILE, 2 * TILE, dtype=torch.bfloat16) * 0.5

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(2 * TILE, 2 * TILE, dtype=torch.bfloat16), device)

    matmul_acc_kernel(a, b, out)

    result = ttnn.to_torch(out)
    golden = a_torch @ b_torch
    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.99, (
        f"PCC {pcc:.6f} < 0.99 for 2x2x2 distinct-tile matmul. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )
