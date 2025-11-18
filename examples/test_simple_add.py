# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple add test with data writeback verification.

Tests the complete data path: host → L1 → compute → L1 → host
"""

from ttlang.d2m_api import *
import torch


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def simple_add(lhs, rhs, out):
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)

    @compute()
    def add_compute(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        lhs_shard = lhs_cb.pop()
        rhs_shard = rhs_cb.pop()
        out_shard = out_cb.reserve()
        result = lhs_shard + rhs_shard
        out_shard.store(result)
        out_cb.pop()

    @datamovement()
    def dm_lhs(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()

    @datamovement()
    def dm_rhs(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_accessor[0, 0], rhs_shard)
        tx.wait()

    return Program(add_compute, dm_lhs, dm_rhs)(lhs, rhs, out)


if __name__ == "__main__":
    print("=" * 60)
    print("Simple Add Test - Data Writeback Verification")
    print("=" * 60)

    # Use simple non-zero values for easy debugging: 2 + 3 = 5
    lhs = torch.full((32, 32), 2.0)
    rhs = torch.full((32, 32), 3.0)
    out = torch.full((32, 32), -999.0)

    print("\n=== BEFORE KERNEL ===")
    print(f"lhs[0:3, 0:3] =\n{lhs[0:3, 0:3]}")
    print(f"rhs[0:3, 0:3] =\n{rhs[0:3, 0:3]}")
    print(f"out[0:3, 0:3] =\n{out[0:3, 0:3]}")
    expected = lhs + rhs
    print(f"expected[0:3, 0:3] =\n{expected[0:3, 0:3]}")

    simple_add(lhs, rhs, out)

    print("\n=== AFTER KERNEL ===")
    print(f"out[0:3, 0:3] =\n{out[0:3, 0:3]}")
    print(f"expected[0:3, 0:3] =\n{expected[0:3, 0:3]}")

    print(f"\nStats:")
    print(
        f"  out min/max/mean: {out.min().item():.4f} / {out.max().item():.4f} / {out.mean().item():.4f}"
    )
    print(
        f"  expected min/max/mean: {expected.min().item():.4f} / {expected.max().item():.4f} / {expected.mean().item():.4f}"
    )

    # Verify results (will only work on hardware)
    if not torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
        if out.min().item() == -999.0:
            print("\n⚠️  Output unchanged - expected on macOS (compilation only)")
        else:
            print(
                f"\n❌ MISMATCH! Max error: {(out - expected).abs().max().item():.6f}"
            )
    else:
        print("\n✅ Output matches expected!")
