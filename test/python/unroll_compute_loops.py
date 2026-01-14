# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_FINAL_MLIR=%t.final.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: %python %s

"""
Test loop unrolling for DST register utilization optimization.

Verifies that tile compute loops are unrolled to maximize DST register usage:
- 4 tiles with 2 inputs (footprint=3) and capacity=8 → unroll_factor=2
- DST indices are correctly updated per unrolled iteration
"""

import os

# Temporarily disable unrolling for debugging
# os.environ["TTLANG_DISABLE_UNROLL"] = "1"

import ttnn
from ttlang import ttl


@ttl.kernel(grid=(1, 1))
def add_unroll_kernel(lhs, rhs, out):
    """Add kernel with 1x4 tiles to test loop unrolling (unroll factor 2)."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 4), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 4), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 4), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_cb.reserve()
        tx_lhs = ttl.copy(lhs[0:1, 0:4], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[0:1, 0:4], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0:1, 0:4])
        tx.wait()
        out_cb.pop()

    return ttl.Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


# =============================================================================
# Final IR Checks - Verify unrolling is reflected in EmitC output
# =============================================================================

# CHECK-LABEL: func.func @add_compute

# Verify loop step constant is 2 (unrolled by factor 2)
# With 4 tiles, 2 inputs, footprint per iteration = 3 (2 inputs + 1 output)
# DST capacity = 8, so unroll_factor = floor(8/3) = 2
# CHECK: %[[STEP:.+]] = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t

# Verify loop bound is still 4
# CHECK: %[[BOUND:.+]] = "emitc.constant"() <{value = 4 : index}> : () -> !emitc.size_t

# Verify the emitc.for loop uses step 2
# CHECK: emitc.for %{{.+}} = %{{.+}} to %[[BOUND]] step %[[STEP]]

# =============================================================================
# C++ Kernel Checks - Verify unrolled loop generates correct iteration count
# =============================================================================

# CHECK-CPP: // add_compute
# CHECK-CPP: void kernel_main()

# With unroll_factor=2, loop constants should include step=2 and bound=4
# CHECK-CPP-DAG: size_t [[STEP:v[0-9]+]] = 2;
# CHECK-CPP-DAG: size_t [[BOUND:v[0-9]+]] = 4;

# Verify the unrolled loop uses these constants
# CHECK-CPP: for (size_t {{.+}} = {{.+}}; {{.+}} < [[BOUND]]; {{.+}} += [[STEP]]) {


if __name__ == "__main__":
    import torch
    from test_helpers import require_hardware

    print("=== Loop Unroll Compute Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        # 32x128 = 1x4 tiles (should trigger unroll factor 2)
        # Use iota pattern for easier verification
        lhs_torch = torch.arange(32 * 128, dtype=torch.bfloat16).reshape(32, 128)
        rhs_torch = torch.arange(32 * 128, dtype=torch.bfloat16).reshape(32, 128) * 2
        out_torch = torch.zeros((32, 128), dtype=torch.bfloat16)
        expected = lhs_torch + rhs_torch

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
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        lhs = ttnn.to_memory_config(lhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        rhs = ttnn.to_memory_config(rhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print(
            "Compiling add kernel with loop unrolling (1x4 tiles, unroll_factor=2)..."
        )
        add_unroll_kernel(lhs, rhs, out)

        # Verify results if not compile-only mode
        if "TTLANG_COMPILE_ONLY" not in os.environ:
            result = ttnn.to_torch(out)

            # Check for NaN or inf
            if torch.isnan(result).any() or torch.isinf(result).any():
                num_nan = torch.isnan(result).sum().item()
                num_inf = torch.isinf(result).sum().item()
                print(f"✗ Results contain {num_nan} NaN and {num_inf} inf values!")
                print(f"First 8 expected: {expected[0, :8]}")
                print(f"First 8 got: {result[0, :8]}")
                raise AssertionError("Results contain NaN or inf")

            # Compare with tolerance
            if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
                print("✓ Results match expected values (iota pattern verified)")
            else:
                diff = (result - expected).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                max_idx = diff.argmax().item()
                max_row, max_col = max_idx // 128, max_idx % 128
                print(f"✗ Results do not match!")
                print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
                print(f"  Max diff at [{max_row}, {max_col}]:")
                print(f"    Expected: {expected[max_row, max_col].item():.6f}")
                print(f"    Got: {result[max_row, max_col].item():.6f}")
                print(f"  First row sample:")
                print(f"    Expected: {expected[0, :8]}")
                print(f"    Got: {result[0, :8]}")
                raise AssertionError(f"Results mismatch: max_diff={max_diff:.6f}")

        print("=== Loop Unroll Compute Test Complete ===")

    finally:
        ttnn.close_device(device)
