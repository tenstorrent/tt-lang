# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple add test with profiling signposts for Tracy visualization.

Demonstrates the use of signpost() markers to create profiling regions
that will be visible when running the compiled flatbuffer with Tracy enabled.
"""

from ttlang.d2m_api import *
import torch


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def signpost_add(lhs, rhs, out):
    """Add two tensors with profiling signposts."""
    lhs_stream = Stream(lhs)
    rhs_stream = Stream(rhs)

    @compute()
    async def add_compute(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        signpost("compute_start")

        l = lhs_cb.pop()
        r = rhs_cb.pop()

        signpost("compute_add_operation")
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        out_cb.pop()

        signpost("compute_end")

    @datamovement()
    async def dm_lhs(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        signpost("dm_lhs_start")
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_stream[0, 0], lhs_shard)
        tx.wait()
        signpost("dm_lhs_complete")

    @datamovement()
    async def dm_rhs(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        signpost("dm_rhs_start")
        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_stream[0, 0], rhs_shard)
        tx.wait()
        signpost("dm_rhs_complete")

    return Program(add_compute, dm_lhs, dm_rhs)(lhs, rhs, out)


if __name__ == "__main__":
    print("=" * 60)
    print("Signpost Add Test - Profiling Markers for Tracy")
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

    print("\n=== COMPILING KERNEL WITH SIGNPOSTS ===")
    print("Signpost markers will be visible in Tracy profiler:")
    print("  - dm_lhs_start")
    print("  - dm_lhs_complete")
    print("  - dm_rhs_start")
    print("  - dm_rhs_complete")
    print("  - compute_start")
    print("  - compute_add_operation")
    print("  - compute_end")

    signpost_add(lhs, rhs, out)

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
            print("    To run with profiling, use: ttrt perf <flatbuffer>")
        else:
            print(
                f"\n❌ MISMATCH! Max error: {(out - expected).abs().max().item():.6f}"
            )
    else:
        print("\n✅ Output matches expected!")
        print("    Run with: ttrt perf <flatbuffer> to see Tracy profiling markers")
