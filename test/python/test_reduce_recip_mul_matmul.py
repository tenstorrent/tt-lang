# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: EXACT op sequence from Kernel 2
# reduce_sum → recip → mul (reuse input) → matmul

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum, recip

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_exact_kernel2_pattern(A, ones, C, out):
    """
    Mirrors Kernel 2 exactly:
    A = exp_S (reused)
    ones = ones (for reduce)
    C = V (for final matmul)
    """
    A_accessor = TensorAccessor(A)
    ones_accessor = TensorAccessor(ones)
    C_accessor = TensorAccessor(C)

    @compute()
    async def compute_chain(A_cb: CircularBuffer, ones_cb: CircularBuffer, C_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()        # exp_S in Kernel 2
        ones_val = ones_cb.wait()
        c = C_cb.wait()        # V in Kernel 2
        o = out_cb.reserve()

        # Op 1: reduce_sum (same as Kernel 2 step 4)
        sum_result = reduce_sum(a, ones_val, dim=1)
        o.store(sum_result)
        out_cb.push()
        sum_result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 2: recip (same as Kernel 2 step 5)
        sum_recip = recip(sum_result)
        o.store(sum_recip)
        out_cb.push()
        sum_recip = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 3: mul with REUSED input (same as Kernel 2 step 6)
        normalized = a * sum_recip
        o.store(normalized)
        out_cb.push()
        normalized = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 4: matmul (same as Kernel 2 step 7)
        result = normalized @ c

        o.store(result)
        A_cb.pop()
        ones_cb.pop()
        C_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(A_cb: CircularBuffer, ones_cb: CircularBuffer, C_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = A_cb.reserve()
        tx = dma(A_accessor[0, 0], a_shard)
        tx.wait()
        A_cb.push()

        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

        c_shard = C_cb.reserve()
        tx = dma(C_accessor[0, 0], c_shard)
        tx.wait()
        C_cb.push()

    return Program(compute_chain, dm_loader)(A, ones, C, out)


# CHECK: func.func @test_exact_kernel2_pattern
# CHECK: "d2m.tile_reduce_sum"
# CHECK: "d2m.tile_recip"
# CHECK: "d2m.tile_mul"
# CHECK: "d2m.tile_matmul"

# Use same values as FA to match Kernel 2 exactly
A = torch.full((32, 32), 1.058)  # exp_S
ones = torch.ones((32, 32))
C = torch.full((32, 32), 0.2)    # V
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"A (exp_S): all {A[0, 0].item():.3f}")
print(f"C (V): all {C[0, 0].item()}")

# Manual calculation matching Kernel 2
sum_result = A[0, 0].item() * 32  # reduce_sum
print(f"Step 1 - reduce_sum: {sum_result:.3f}")

sum_recip = 1.0 / sum_result  # recip
print(f"Step 2 - recip: {sum_recip:.6f}")

normalized = A[0, 0].item() * sum_recip  # mul (reuse A)
print(f"Step 3 - mul (normalize): {normalized:.6f}")

result_expected = normalized * C[0, 0].item() * 32  # matmul
print(f"Step 4 - matmul result: {result_expected:.6f}")

test_exact_kernel2_pattern(A, ones, C, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"Hardware: {out[0, 0].item():.6f}")
# CHECK-OUTPUT: Hardware:
print(f"Expected: {result_expected:.6f}")

error = abs(out[0, 0].item() - result_expected) / result_expected
print(f"Error: {error*100:.1f}%")

if error < 0.2:
    print(f"PASS: Exact Kernel 2 pattern works!")
    # CHECK-OUTPUT: PASS: Exact Kernel 2 pattern works
else:
    print(f"FAIL: Expected {result_expected:.6f}, got {out[0, 0].item():.6f}")
