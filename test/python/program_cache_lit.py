# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

"""
Test program caching behavior in tt-lang.

This test verifies that:
1. Same kernel + same tensor config = cache hit (no recompile)
2. Same kernel + different shapes = cache miss (recompile)
3. Same kernel + different memory space = cache miss (recompile)
4. Different kernels with same tensors = separate cache entries
"""

import torch
import ttnn
import ttl
from test_helpers import to_dram, to_l1


@ttl.kernel(grid=(1, 1))
def add_kernel(lhs, rhs, out):
    """Simple add kernel for cache testing."""
    lhs_cb = ttl.CircularBuffer(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.CircularBuffer(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.CircularBuffer(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        o.store(l + r)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_cb.reserve()
        tx = ttl.copy(lhs[0, 0], lhs_blk)
        tx.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx = ttl.copy(rhs[0, 0], rhs_blk)
        tx.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


@ttl.kernel(grid=(1, 1))
def mul_kernel(lhs, rhs, out):
    """Multiply kernel - separate from add_kernel for cache isolation test."""
    lhs_cb = ttl.CircularBuffer(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.CircularBuffer(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.CircularBuffer(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        o.store(l * r)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_cb.reserve()
        tx = ttl.copy(lhs[0, 0], lhs_blk)
        tx.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx = ttl.copy(rhs[0, 0], rhs_blk)
        tx.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


def count_compiles(output: str) -> int:
    """Count number of MLIR compilations from output."""
    return output.count("TTNN INTEROP: Compiling kernel")


print("=== Testing Program Cache ===")
# CHECK: Testing Program Cache

device = ttnn.open_device(device_id=0)

try:
    # =========================================================================
    # Test 1: Cache hit - same kernel, same tensor config
    # =========================================================================
    print("\n--- Test 1: Cache hit (same config) ---")
    # CHECK: Test 1: Cache hit

    # First call - should compile
    lhs1 = to_l1(torch.full((32, 32), 2.0, dtype=torch.bfloat16), device)
    rhs1 = to_l1(torch.full((32, 32), 3.0, dtype=torch.bfloat16), device)
    out1 = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    print("Call 1 (should compile):")
    # CHECK: Call 1 (should compile)
    # CHECK: TTNN INTEROP: Compiling kernel
    add_kernel(lhs1, rhs1, out1)
    result1 = ttnn.to_torch(out1)
    assert torch.allclose(result1.float(), torch.full((32, 32), 5.0), rtol=1e-2)
    print("Result 1 correct: 2 + 3 = 5")
    # CHECK: Result 1 correct

    # Second call with new tensors but same shapes - should NOT compile
    lhs2 = to_l1(torch.full((32, 32), 10.0, dtype=torch.bfloat16), device)
    rhs2 = to_l1(torch.full((32, 32), 7.0, dtype=torch.bfloat16), device)
    out2 = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    print("Call 2 (should use cache):")
    # CHECK: Call 2 (should use cache)
    # CHECK-NOT: TTNN INTEROP: Compiling kernel
    add_kernel(lhs2, rhs2, out2)
    result2 = ttnn.to_torch(out2)
    assert torch.allclose(result2.float(), torch.full((32, 32), 17.0), rtol=1e-2)
    print("Result 2 correct: 10 + 7 = 17")
    # CHECK: Result 2 correct

    print("PASS: Cache hit works")
    # CHECK: PASS: Cache hit works

    # =========================================================================
    # Test 2: Cache miss - different shapes
    # =========================================================================
    print("\n--- Test 2: Cache miss (different shapes) ---")
    # CHECK: Test 2: Cache miss (different shapes)

    # Different shape - should recompile
    lhs3 = to_l1(torch.full((64, 64), 1.0, dtype=torch.bfloat16), device)
    rhs3 = to_l1(torch.full((64, 64), 2.0, dtype=torch.bfloat16), device)
    out3 = to_l1(torch.zeros((64, 64), dtype=torch.bfloat16), device)

    print("Call 3 (different shape, should compile):")
    # CHECK: Call 3 (different shape, should compile)
    # CHECK: TTNN INTEROP: Compiling kernel
    add_kernel(lhs3, rhs3, out3)
    result3 = ttnn.to_torch(out3)
    # Kernel only processes first tile, so check first 32x32
    assert torch.allclose(
        result3[:32, :32].float(), torch.full((32, 32), 3.0), rtol=1e-2
    )
    print("Result 3 correct: 1 + 2 = 3 (first tile)")
    # CHECK: Result 3 correct

    print("PASS: Different shapes cause recompile")
    # CHECK: PASS: Different shapes cause recompile

    # =========================================================================
    # Test 3: Cache miss - different memory space
    # =========================================================================
    print("\n--- Test 3: Cache miss (different memory space) ---")
    # CHECK: Test 3: Cache miss (different memory space)

    # Same shape but DRAM instead of L1 - should recompile
    lhs4 = to_dram(torch.full((32, 32), 4.0, dtype=torch.bfloat16), device)
    rhs4 = to_dram(torch.full((32, 32), 5.0, dtype=torch.bfloat16), device)
    out4 = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    print("Call 4 (DRAM instead of L1, should compile):")
    # CHECK: Call 4 (DRAM instead of L1, should compile)
    # CHECK: TTNN INTEROP: Compiling kernel
    add_kernel(lhs4, rhs4, out4)
    result4 = ttnn.to_torch(out4)
    assert torch.allclose(result4.float(), torch.full((32, 32), 9.0), rtol=1e-2)
    print("Result 4 correct: 4 + 5 = 9")
    # CHECK: Result 4 correct

    print("PASS: Different memory space causes recompile")
    # CHECK: PASS: Different memory space causes recompile

    # =========================================================================
    # Test 4: Separate kernels have separate caches
    # =========================================================================
    print("\n--- Test 4: Separate kernel caches ---")
    # CHECK: Test 4: Separate kernel caches

    # mul_kernel should compile even though add_kernel with same config is cached
    lhs5 = to_l1(torch.full((32, 32), 3.0, dtype=torch.bfloat16), device)
    rhs5 = to_l1(torch.full((32, 32), 4.0, dtype=torch.bfloat16), device)
    out5 = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    print("Call 5 (mul_kernel, should compile):")
    # CHECK: Call 5 (mul_kernel, should compile)
    # CHECK: TTNN INTEROP: Compiling kernel
    mul_kernel(lhs5, rhs5, out5)
    result5 = ttnn.to_torch(out5)
    assert torch.allclose(result5.float(), torch.full((32, 32), 12.0), rtol=1e-2)
    print("Result 5 correct: 3 * 4 = 12")
    # CHECK: Result 5 correct

    # Second call to mul_kernel - should use cache
    lhs6 = to_l1(torch.full((32, 32), 2.0, dtype=torch.bfloat16), device)
    rhs6 = to_l1(torch.full((32, 32), 6.0, dtype=torch.bfloat16), device)
    out6 = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    print("Call 6 (mul_kernel again, should use cache):")
    # CHECK: Call 6 (mul_kernel again, should use cache)
    # CHECK-NOT: TTNN INTEROP: Compiling kernel
    mul_kernel(lhs6, rhs6, out6)
    result6 = ttnn.to_torch(out6)
    assert torch.allclose(result6.float(), torch.full((32, 32), 12.0), rtol=1e-2)
    print("Result 6 correct: 2 * 6 = 12")
    # CHECK: Result 6 correct

    print("PASS: Separate kernels have separate caches")
    # CHECK: PASS: Separate kernels have separate caches

    # =========================================================================
    # Test 5: Return to cached config still uses cache
    # =========================================================================
    print("\n--- Test 5: Return to cached config ---")
    # CHECK: Test 5: Return to cached config

    # Call add_kernel with original L1 32x32 config - should still be cached
    lhs7 = to_l1(torch.full((32, 32), 100.0, dtype=torch.bfloat16), device)
    rhs7 = to_l1(torch.full((32, 32), 23.0, dtype=torch.bfloat16), device)
    out7 = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    print("Call 7 (back to original config, should use cache):")
    # CHECK: Call 7 (back to original config, should use cache)
    # CHECK-NOT: TTNN INTEROP: Compiling kernel
    add_kernel(lhs7, rhs7, out7)
    result7 = ttnn.to_torch(out7)
    assert torch.allclose(result7.float(), torch.full((32, 32), 123.0), rtol=1e-2)
    print("Result 7 correct: 100 + 23 = 123")
    # CHECK: Result 7 correct

    print("PASS: Return to cached config uses cache")
    # CHECK: PASS: Return to cached config uses cache

finally:
    ttnn.close_device(device)

print("\n=== All Program Cache Tests Passed ===")
# CHECK: All Program Cache Tests Passed
