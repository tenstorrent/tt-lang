# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Debug test for add kernel with 1x2 tile shape (32x64 tensor).
Based on test_binary_op[add-1x2tiles] which is failing.
"""

import torch
import ttnn
from ttlang import make_circular_buffer_like, ttl
from ttlang.operators import copy
from ttlang.ttl_api import Program

# Configuration for 1x2 tiles
TILE_ROWS = 1
TILE_COLS = 2
TENSOR_SHAPE = (32, 64)  # 1x2 tiles = 32 x 64
BUFFER_FACTOR = TILE_ROWS * TILE_COLS  # = 2


@ttl.kernel(grid=(1, 1))
def add_1x2_kernel(lhs, rhs, out):
    """Add kernel for 1x2 tiles (32x64 tensor)."""
    lhs_cb = make_circular_buffer_like(lhs, shape=(TILE_ROWS, TILE_COLS), buffer_factor=BUFFER_FACTOR)
    rhs_cb = make_circular_buffer_like(rhs, shape=(TILE_ROWS, TILE_COLS), buffer_factor=BUFFER_FACTOR)
    out_cb = make_circular_buffer_like(out, shape=(TILE_ROWS, TILE_COLS), buffer_factor=BUFFER_FACTOR)

    @ttl.compute()
    def compute_fn():
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
        lhs_cb.reserve()
        tx_lhs = copy(lhs[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(compute_fn, dm_read, dm_write)(lhs, rhs, out)


# Also test 1x1 for comparison
@ttl.kernel(grid=(1, 1))
def add_1x1_kernel(lhs, rhs, out):
    """Add kernel for 1x1 tiles (32x32 tensor)."""
    lhs_cb = make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=1)
    rhs_cb = make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=1)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=1)

    @ttl.compute()
    def compute_fn():
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
        lhs_cb.reserve()
        tx_lhs = copy(lhs[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(compute_fn, dm_read, dm_write)(lhs, rhs, out)


def run_test(name, kernel, shape):
    """Run a test and print detailed output."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Shape: {shape}")
    print(f"{'='*60}")

    device = ttnn.open_device(device_id=0)

    try:
        lhs_torch = torch.full(shape, 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full(shape, 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros(shape, dtype=torch.bfloat16)
        expected = lhs_torch + rhs_torch  # 5.0

        print(f"Input lhs: all 2.0")
        print(f"Input rhs: all 3.0")
        print(f"Expected output: all 5.0")

        # Create DRAM tensors
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

        print(f"\nRunning kernel...")
        kernel(lhs, rhs, out)

        result = ttnn.to_torch(out)

        print(f"\n--- Result Tensor ---")
        print(f"Result shape: {result.shape}")
        print(f"Result dtype: {result.dtype}")
        print(f"Result[0,0]: {result[0, 0].item()}")
        print(f"Result[0,-1]: {result[0, -1].item()}")
        print(f"Result[-1,0]: {result[-1, 0].item()}")
        print(f"Result[-1,-1]: {result[-1, -1].item()}")
        print(f"Result min: {result.min().item()}")
        print(f"Result max: {result.max().item()}")
        print(f"Result mean: {result.float().mean().item()}")

        # Check for all-zeros (common failure mode)
        if result.abs().max().item() == 0:
            print("\nWARNING: Result is all zeros!")

        # Check unique values
        unique_vals = torch.unique(result)
        print(f"Unique values in result: {unique_vals.tolist()}")

        # Full comparison
        if torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2):
            print(f"\nPASS: {name}")
        else:
            diff = (result.float() - expected.float()).abs()
            print(f"\nFAIL: {name}")
            print(f"Max difference: {diff.max().item()}")
            print(f"Mean difference: {diff.mean().item()}")
            # Print first few differing values
            nonzero_diff = (diff > 0.01).nonzero()
            if len(nonzero_diff) > 0:
                print(f"First differing indices: {nonzero_diff[:5].tolist()}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    print("=== Debug Add Kernel Shape Test ===")

    # Test 1x1 first (should work)
    run_test("add_1x1_kernel", add_1x1_kernel, (32, 32))

    # Test 1x2 (the failing case)
    run_test("add_1x2_kernel", add_1x2_kernel, (32, 64))

    print("\n=== Test Complete ===")
