# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: Reuse an input CB value multiple times (like Kernel 2 does with exp_S)
# Pattern: a + b, then reuse 'a' in another operation: result * a

import torch
from ttlang.d2m_api import *

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_reuse_input(A, B, out):
    A_accessor = TensorAccessor(A)
    B_accessor = TensorAccessor(B)

    @compute()
    async def compute_reuse(A_cb: CircularBuffer, B_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()   # Load A once
        b = B_cb.wait()
        o = out_cb.reserve()

        # First use of 'a': a + b
        temp = a + b
        o.store(temp)
        out_cb.push()
        temp = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Second use of 'a': temp * a (reusing 'a' without popping!)
        result = temp * a

        o.store(result)
        A_cb.pop()  # Pop at the end
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

    return Program(compute_reuse, dm_loader)(A, B, out)


# CHECK: func.func @test_reuse_input
# CHECK: "d2m.tile_add"
# CHECK: "d2m.tile_mul"

# Test: (2 + 3) * 2 = 5 * 2 = 10
A = torch.full((32, 32), 2.0)
B = torch.full((32, 32), 3.0)
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"A: all 2.0, B: all 3.0")
print(f"Expected: (2 + 3) * 2 = 10.0")

test_reuse_input(A, B, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"Hardware: {out[0, 0].item():.1f}")
# CHECK-OUTPUT: Hardware:
print(f"Expected: 10.0")

expected = 10.0
if abs(out[0, 0].item() - expected) / expected < 0.1:
    print(f"PASS: Reusing input CB works")
    # CHECK-OUTPUT: PASS: Reusing input CB works
else:
    print(f"FAIL: Expected {expected:.1f}, got {out[0, 0].item():.1f}")
