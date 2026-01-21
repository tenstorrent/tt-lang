# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: %python %s

"""
Precision tests for f32 computation.

Tests stricter f32 validation with tighter tolerances to verify true f32
precision is used (not bf16 approximation). Uses precision-sensitive operations
like exp() and reciprocal to amplify differences between f32 and bf16.
"""

import os
import sys

# Note: Not setting TTLANG_COMPILE_ONLY so kernel actually executes

import ttl

try:
    import ttnn
    import torch
except ImportError as e:
    print(f"TTNN or torch not available - exiting: {e}")
    exit(0)


@ttl.kernel(grid=(1, 1))
def exp_kernel_f32(lhs, out):
    """Compute exp(x) with f32 precision."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_exp():
        with lhs_cb.wait() as l, out_cb.reserve() as o:
            result = ttl.exp(l)
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with lhs_cb.reserve() as lhs_blk:
            tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
            tx_lhs.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()

    return ttl.Program(compute_exp, dm_read, dm_write)(lhs, out)


@ttl.kernel(grid=(1, 1))
def mul_kernel_f32(lhs, rhs, out):
    """Compute multiplication with f32 precision - tests DST capacity."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_mul():
        with lhs_cb.wait() as l, rhs_cb.wait() as r, out_cb.reserve() as o:
            result = l * r
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with lhs_cb.reserve() as lhs_blk:
            tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
            tx_lhs.wait()

        with rhs_cb.reserve() as rhs_blk:
            tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
            tx_rhs.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()

    return ttl.Program(compute_mul, dm_read, dm_write)(lhs, rhs, out)


def test_exp_f32_precision():
    """Test exp() with tighter tolerance to verify f32 precision."""
    print("\n=== Testing exp() with f32 precision ===")

    device = ttnn.open_device(device_id=0)

    try:
        # exp() is precision-sensitive: small input differences lead to larger output differences
        # This helps distinguish f32 from bf16 precision
        torch.manual_seed(42)

        # Use small values to keep exp() in reasonable range
        lhs_torch = torch.linspace(0.1, 2.0, 32 * 32, dtype=torch.float32).reshape(
            32, 32
        )
        out_torch = torch.zeros((32, 32), dtype=torch.float32)

        # Compute expected result with torch (full float32 precision)
        expected = torch.exp(lhs_torch)

        lhs = ttnn.from_torch(
            lhs_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        lhs = ttnn.to_memory_config(lhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling and executing float32 exp kernel...")
        exp_kernel_f32(lhs, out)

        # Validate result with tighter tolerance
        result = ttnn.to_torch(out)

        print(f"Input sample:    {lhs_torch[0, :5]}")
        print(f"Result sample:   {result[0, :5]}")
        print(f"Expected sample: {expected[0, :5]}")

        # Tighter tolerances for f32 (vs loose 1e-2 for bf16)
        # rtol=1e-4, atol=1e-5 should pass for f32 but fail for bf16
        if torch.allclose(result, expected, rtol=1e-4, atol=1e-5):
            print("✓ exp() results match expected f32 values (tight tolerance)")
            return True
        else:
            max_diff = torch.max(torch.abs(result - expected)).item()
            mean_diff = torch.mean(torch.abs(result - expected)).item()
            print(f"✗ exp() results don't match tight tolerance!")
            print(f"  Max difference:  {max_diff}")
            print(f"  Mean difference: {mean_diff}")
            # Try looser tolerance to see if it's just f32 vs bf16
            if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
                print(
                    f"  Note: Passes with loose tolerance (rtol=1e-2) - may indicate bf16 precision used"
                )
            return False

    finally:
        ttnn.close_device(device)


def test_mul_f32_dst_capacity():
    """Test multiplication to verify f32 DST capacity (4 tiles) doesn't overflow."""
    print("\n=== Testing mul with f32 DST capacity ===")

    device = ttnn.open_device(device_id=0)

    try:
        torch.manual_seed(42)
        lhs_torch = torch.rand((32, 32), dtype=torch.float32) * 2.0  # [0, 2)
        rhs_torch = torch.rand((32, 32), dtype=torch.float32) * 2.0
        out_torch = torch.zeros((32, 32), dtype=torch.float32)

        expected = lhs_torch * rhs_torch

        lhs = ttnn.from_torch(
            lhs_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rhs = ttnn.from_torch(
            rhs_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        lhs = ttnn.to_memory_config(lhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        rhs = ttnn.to_memory_config(rhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling and executing float32 mul kernel...")
        mul_kernel_f32(lhs, rhs, out)

        result = ttnn.to_torch(out)

        print(f"Input A sample:  {lhs_torch[0, :5]}")
        print(f"Input B sample:  {rhs_torch[0, :5]}")
        print(f"Result sample:   {result[0, :5]}")
        print(f"Expected sample: {expected[0, :5]}")

        # Tighter tolerance for mul
        if torch.allclose(result, expected, rtol=1e-5, atol=1e-5):
            print("✓ mul() results match expected f32 values (tight tolerance)")
            return True
        else:
            max_diff = torch.max(torch.abs(result - expected)).item()
            mean_diff = torch.mean(torch.abs(result - expected)).item()
            print(f"✗ mul() results don't match tight tolerance!")
            print(f"  Max difference:  {max_diff}")
            print(f"  Mean difference: {mean_diff}")
            if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
                print(f"  Note: Passes with loose tolerance (rtol=1e-2)")
            return False

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    try:
        exp_passed = test_exp_f32_precision()
        mul_passed = test_mul_f32_dst_capacity()

        print("\n" + "=" * 60)
        if exp_passed and mul_passed:
            print("=== All f32 precision tests PASSED ===")
            sys.exit(0)
        else:
            print("=== Some f32 precision tests FAILED ===")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
