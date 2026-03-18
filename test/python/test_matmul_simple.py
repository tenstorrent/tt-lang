# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone matmul tests: a_blk @ b_blk with various block sizes.

Parametrized over (M, K, N) block dimensions. Each test performs a single
matmul_block call through the full pipeline and validates against torch.matmul.

Block sizes tested:
  1x1x1:  Minimal, no loops in the compute.
  1x4x1:  K > 1, matmul_block handles K internally.
  2x2x2:  Multi-tile output and K.
  2x4x3:  Non-square, catches index computation bugs.
  4x4x4:  Larger square, exercises more DST registers.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram

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
        (1, 1, 1),  # Minimal: single tile matmul.
        (1, 4, 1),  # K > 1: matmul_block handles K-accumulation internally.
        (2, 2, 2),  # Multi-tile square output + K.
        (2, 4, 3),  # Non-square: catches per-operand index bugs.
        (4, 4, 4),  # Larger square.
    ],
    ids=["1x1x1", "1x4x1", "2x2x2", "2x4x3", "4x4x4"],
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
