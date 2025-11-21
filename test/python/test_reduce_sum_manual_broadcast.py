# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: reduce_sum with broadcast=False, then manual broadcast_reduce_result call
# This separates the operations with CB synchronization to avoid compiler issues

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum, broadcast_reduce_result

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_reduce_manual_broadcast(A, ones, out):
    A_accessor = TensorAccessor(A)
    ones_accessor = TensorAccessor(ones)

    @compute()
    async def compute_reduce(A_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()
        ones_val = ones_cb.wait()
        o = out_cb.reserve()

        # Step 1: reduce_sum with broadcast=False (only fills column 0)
        reduced = reduce_sum(a, ones_val, dim=1, broadcast=False)
        o.store(reduced)
        out_cb.push()

        # CB synchronization between operations
        reduced_readback = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 2: Manual broadcast to all columns
        broadcasted = broadcast_reduce_result(reduced_readback, dim=1)

        o.store(broadcasted)
        A_cb.pop()
        ones_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(A_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = A_cb.reserve()
        tx = dma(A_accessor[0, 0], a_shard)
        tx.wait()
        A_cb.push()

        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

    return Program(compute_reduce, dm_loader)(A, ones, out)


# CHECK: func.func @test_reduce_manual_broadcast
# CHECK: "d2m.tile_reduce_sum"
# CHECK: linalg.generic
# CHECK: linalg.yield

# Test: reduce_sum(2.0 * 32 columns) = 64.0, then broadcast
A = torch.full((32, 32), 2.0)
ones = torch.ones((32, 32))
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"Input: 32x32 tensor, all values = 2.0")
print(f"Expected: reduce_sum(2.0 * 32) = 64.0 at ALL positions after broadcast")
print(f"Reduction dim: 1 (sum across columns)")

test_reduce_manual_broadcast(A, ones, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===

print(f"Output inspection:")
print(f"  out[0, 0] = {out[0, 0].item():.1f}")
# CHECK-OUTPUT: out[0, 0] = 64.0
print(f"  out[0, 15] = {out[0, 15].item():.1f}")
print(f"  out[0, 31] = {out[0, 31].item():.1f}")
print(f"  out[1, 0] = {out[1, 0].item():.1f}")

# Check if broadcasting worked
row_0_all_same = torch.allclose(out[0, :], torch.full((32,), 64.0), rtol=0.01)
all_rows_same = torch.allclose(out, torch.full((32, 32), 64.0), rtol=0.01)

print(f"\nBroadcast validation:")
print(f"  Row 0 all 64.0? {row_0_all_same}")
print(f"  All rows all 64.0? {all_rows_same}")
print(f"  Min value: {out.min().item():.1f}")
print(f"  Max value: {out.max().item():.1f}")

expected = 64.0
tolerance = 0.01

if all_rows_same and abs(out[0, 0].item() - expected) / expected < tolerance:
    print(f"\nPASS: Manual broadcast after reduce_sum produced correct result!")
    # CHECK-OUTPUT: PASS: Manual broadcast after reduce_sum produced correct result
else:
    print(f"\nFAIL: Expected all values to be {expected:.1f}")
    print(f"  Row 0: [{out[0, 0].item():.1f}, {out[0, 1].item():.1f}, ..., {out[0, 31].item():.1f}]")
