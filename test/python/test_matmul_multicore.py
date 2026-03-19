# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multicore matmul: output rows partitioned across a 1D grid.
Each core computes C_strip = A_strip @ B where A_strip is the core's
row-block of A and B is read in full by every core.
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


@ttl.kernel(grid=(1, 2))
def matmul_multicore_2rows(a, b, out):
    """2-core matmul: each core computes one row-block of the output."""
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, Nt), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, Nt), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        with out_dfb.reserve() as o:
            o.store(a_blk @ b_blk)
        a_blk.pop()
        b_blk.pop()

    @ttl.datamovement()
    def dm_read():
        _, core_row = ttl.core(dims=2)
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[core_row, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:1, 0:Nt], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        _, core_row = ttl.core(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[core_row, 0:Nt])
            tx.wait()


@pytest.mark.parametrize("Nt", [1, 2, 4], ids=["Nt1", "Nt2", "Nt4"])
@pytest.mark.requires_device
def test_matmul_multicore_2rows(Nt, device):
    """2-core matmul partitioned by output rows."""
    M, K, N = 2 * TILE, TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    matmul_multicore_2rows(a, b, out)

    result = ttnn.to_torch(out)
    golden = a_torch @ b_torch

    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.999, (
        f"PCC {pcc:.6f} < 0.999 for 2-core matmul Nt={Nt}. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )
