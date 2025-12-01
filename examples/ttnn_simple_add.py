# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple add test using TTNN tensors.

Tests the TTNN interop path: ttnn.Tensor (on device) → compiled kernel → ttnn.Tensor (on device)
This allows tt-lang kernels to be used within existing TTNN programs.
"""

from ttlang.d2m_api import *
import torch

try:
    import ttnn
except ImportError:
    print("TTNN not available - this example requires ttnn to be installed")
    exit(0)


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
    print("TTNN Simple Add Test - TTNN Tensor Interop")
    print("=" * 60)

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        # Create torch tensors first
        lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        out_torch = torch.full((32, 32), -999.0, dtype=torch.bfloat16)
        expected = lhs_torch + rhs_torch

        # Convert to TTNN tensors on device (tilized, DRAM)
        lhs = ttnn.from_torch(
            lhs_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rhs = ttnn.from_torch(
            rhs_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Allocate output on device
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print("\n=== TTNN TENSORS ===")
        print(f"lhs shape: {lhs.shape}, dtype: {lhs.dtype}")
        print(f"rhs shape: {rhs.shape}, dtype: {rhs.dtype}")
        print(f"out shape: {out.shape}, dtype: {out.dtype}")

        # Run the tt-lang kernel with TTNN tensors
        print("\n=== RUNNING KERNEL ===")
        simple_add(lhs, rhs, out)

        # Copy result back to host for verification
        out_result = ttnn.to_torch(out)

        print("\n=== AFTER KERNEL ===")
        print(f"out[0:3, 0:3] =\n{out_result[0:3, 0:3]}")
        print(f"expected[0:3, 0:3] =\n{expected[0:3, 0:3]}")

        print(f"\nStats:")
        print(
            f"  out min/max/mean: {out_result.min().item():.4f} / {out_result.max().item():.4f} / {out_result.mean().item():.4f}"
        )
        print(
            f"  expected min/max/mean: {expected.min().item():.4f} / {expected.max().item():.4f} / {expected.mean().item():.4f}"
        )

        # Verify results
        if torch.allclose(out_result.float(), expected.float(), rtol=1e-2, atol=1e-2):
            print("\n✅ Output matches expected!")
        else:
            print(
                f"\n❌ MISMATCH! Max error: {(out_result.float() - expected.float()).abs().max().item():.6f}"
            )

    finally:
        ttnn.close_device(device)
