# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: Three operations chained - recip → mul → matmul
# This mirrors the FA second half pattern exactly

import torch
from ttlang.d2m_api import *
from ttlang.operators import recip

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_recip_mul_matmul(A, B, C, out):
    A_accessor = TensorAccessor(A)
    B_accessor = TensorAccessor(B)
    C_accessor = TensorAccessor(C)

    @compute()
    async def compute_chain(A_cb: CircularBuffer, B_cb: CircularBuffer, C_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()
        b = B_cb.wait()
        c = C_cb.wait()
        o = out_cb.reserve()

        # Step 1: Reciprocal
        a_recip = recip(a)
        o.store(a_recip)
        out_cb.push()
        a_recip = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 2: Multiply (normalize)
        normalized = b * a_recip
        o.store(normalized)
        out_cb.push()
        normalized = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 3: Matmul (attention-weighted sum)
        result = normalized @ c

        o.store(result)
        A_cb.pop()
        B_cb.pop()
        C_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(A_cb: CircularBuffer, B_cb: CircularBuffer, C_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = A_cb.reserve()
        tx = dma(A_accessor[0, 0], a_shard)
        tx.wait()
        A_cb.push()

        b_shard = B_cb.reserve()
        tx = dma(B_accessor[0, 0], b_shard)
        tx.wait()
        B_cb.push()

        c_shard = C_cb.reserve()
        tx = dma(C_accessor[0, 0], c_shard)
        tx.wait()
        C_cb.push()

    return Program(compute_chain, dm_loader)(A, B, C, out)


# CHECK: func.func @test_recip_mul_matmul
# CHECK: "d2m.tile_recip"
# CHECK: "d2m.tile_mul"
# CHECK: "d2m.tile_matmul"

# Test: (8 / 4) @ 2 = 2 @ 2 = 2 * 2 * 32 = 128
A = torch.full((32, 32), 4.0)   # Denominator for division
B = torch.full((32, 32), 8.0)   # Numerator for division
C = torch.full((32, 32), 2.0)   # Matrix to multiply with
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"A: all 4.0, B: all 8.0, C: all 2.0")
print(f"Expected: (8 / 4) @ 2 = 2 @ 2 = {2.0 * 2.0 * 32}")

# Compute expected
A_recip = 1.0 / A  # All 0.25
normalized = B * A_recip  # 8 * 0.25 = 2
result_expected = normalized @ C  # 2 @ 2 = 2 * 2 * 32 = 128

test_recip_mul_matmul(A, B, C, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.1f}")
# CHECK-OUTPUT: out[0, 0] =
print(f"Expected: {result_expected[0, 0].item():.1f}")

if abs(out[0, 0].item() - result_expected[0, 0].item()) / result_expected[0, 0].item() < 0.1:
    print(f"PASS: recip → mul → matmul produced correct result")
    # CHECK-OUTPUT: PASS: recip
else:
    print(f"FAIL: Expected {result_expected[0, 0].item():.1f}, got {out[0, 0].item():.1f}")
