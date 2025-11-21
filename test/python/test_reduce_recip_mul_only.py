# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: reduce_sum → recip → mul (WITHOUT final matmul)
# This isolates whether the issue is in the mul or the final matmul

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum, recip

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_reduce_recip_mul(A, ones, out):
    A_accessor = TensorAccessor(A)
    ones_accessor = TensorAccessor(ones)

    @compute()
    async def compute_chain(A_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()
        ones_val = ones_cb.wait()
        o = out_cb.reserve()

        # Op 1: reduce_sum
        sum_result = reduce_sum(a, ones_val, dim=1)
        o.store(sum_result)
        out_cb.push()
        sum_result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 2: recip
        sum_recip = recip(sum_result)
        o.store(sum_recip)
        out_cb.push()
        sum_recip = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 3: mul (reuse 'a')
        result = a * sum_recip

        o.store(result)
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

    return Program(compute_chain, dm_loader)(A, ones, out)


# CHECK: func.func @test_reduce_recip_mul
# CHECK: "d2m.tile_reduce_sum"
# CHECK: "d2m.tile_recip"
# CHECK: "d2m.tile_mul"

# Use FA values
A = torch.full((32, 32), 1.058)  # exp_S
ones = torch.ones((32, 32))
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"A (exp_S): all {A[0, 0].item():.3f}")

# Manual calculation
sum_result = A[0, 0].item() * 32  # 33.856
sum_recip = 1.0 / sum_result      # 0.0295
normalized = A[0, 0].item() * sum_recip  # 0.03125 (should be 1/32)

print(f"Expected: A * recip(sum(A)) = {normalized:.6f}")

test_reduce_recip_mul(A, ones, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"Hardware: {out[0, 0].item():.6f}")
# CHECK-OUTPUT: Hardware:
print(f"Expected: {normalized:.6f}")

error = abs(out[0, 0].item() - normalized) / normalized
print(f"Error: {error*100:.1f}%")

if error < 0.2:
    print(f"PASS: reduce_sum → recip → mul works")
    # CHECK-OUTPUT: PASS: reduce_sum
else:
    print(f"FAIL: Expected {normalized:.6f}, got {out[0, 0].item():.6f}")
