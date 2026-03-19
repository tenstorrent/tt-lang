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

from ttlang_test_utils import to_dram

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
    ],
    ids=["1x1x1", "1x2x1", "1x4x1", "2x1x2", "2x2x2", "2x4x2"],
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
