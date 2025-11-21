# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: Reciprocal followed by multiply (division via recip)

import torch
from ttlang.d2m_api import *
from ttlang.operators import recip

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_recip_mul(A, B, out):
    A_accessor = TensorAccessor(A)
    B_accessor = TensorAccessor(B)

    @compute()
    async def compute_chain(A_cb: CircularBuffer, B_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()
        b = B_cb.wait()
        o = out_cb.reserve()

        # Step 1: Reciprocal
        a_recip = recip(a)
        o.store(a_recip)
        out_cb.push()
        a_recip = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 2: Multiply (effectively dividing B by A)
        result = b * a_recip

        o.store(result)
        A_cb.pop()
        B_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(A_cb: CircularBuffer, B_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = A_cb.reserve()
        tx = dma(A_accessor[0, 0], a_shard)
        tx.wait()
        A_cb.push()

        b_shard = B_cb.reserve()
        tx = dma(B_accessor[0, 0], b_shard)
        tx.wait()
        B_cb.push()

    return Program(compute_chain, dm_loader)(A, B, out)


# CHECK: func.func @test_recip_mul
# CHECK: "d2m.tile_recip"
# CHECK: "d2m.tile_mul"

# Test: 8 / 4 = 8 * (1/4) = 8 * 0.25 = 2
A = torch.full((32, 32), 4.0)
B = torch.full((32, 32), 8.0)
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"A: all 4.0, B: all 8.0")
print(f"Expected: 8 / 4 = 2.0")

# Compute expected
result_expected = B / A  # All 2.0

test_recip_mul(A, B, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.6f}")
# CHECK-OUTPUT: out[0, 0] =
print(f"Expected: {result_expected[0, 0].item():.6f}")

if abs(out[0, 0].item() - result_expected[0, 0].item()) / result_expected[0, 0].item() < 0.1:
    print(f"PASS: recip → mul produced correct result")
    # CHECK-OUTPUT: PASS: recip
else:
    print(f"FAIL: Expected {result_expected[0, 0].item():.6f}, got {out[0, 0].item():.6f}")
