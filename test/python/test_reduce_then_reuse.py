# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: Use input in reduce_sum, then reuse it in a simple add
# This isolates whether accumulator ops corrupt the input for reuse

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_reduce_then_reuse(A, ones, B, out):
    A_accessor = TensorAccessor(A)
    ones_accessor = TensorAccessor(ones)
    B_accessor = TensorAccessor(B)

    @compute()
    async def compute_chain(A_cb: CircularBuffer, ones_cb: CircularBuffer, B_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()        # Load A
        ones_val = ones_cb.wait()
        b = B_cb.wait()
        o = out_cb.reserve()

        # Op 1: Use 'a' in reduce_sum (accumulator op)
        sum_result = reduce_sum(a, ones_val, dim=1)
        o.store(sum_result)
        out_cb.push()
        sum_result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 2: Try to REUSE 'a' in a simple add
        result = a + b

        o.store(result)
        A_cb.pop()
        ones_cb.pop()
        B_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(A_cb: CircularBuffer, ones_cb: CircularBuffer, B_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = A_cb.reserve()
        tx = dma(A_accessor[0, 0], a_shard)
        tx.wait()
        A_cb.push()

        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

        b_shard = B_cb.reserve()
        tx = dma(B_accessor[0, 0], b_shard)
        tx.wait()
        B_cb.push()

    return Program(compute_chain, dm_loader)(A, ones, B, out)


# CHECK: func.func @test_reduce_then_reuse
# CHECK: "d2m.tile_reduce_sum"
# CHECK: "d2m.tile_add"

A = torch.full((32, 32), 5.0)
ones = torch.ones((32, 32))
B = torch.full((32, 32), 3.0)
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"A: all 5.0, B: all 3.0")
print(f"Op 1: reduce_sum(A, ones) = {5.0 * 32}")
print(f"Op 2: A + B = 5 + 3 = 8.0")
print(f"Expected final output: 8.0")

test_reduce_then_reuse(A, ones, B, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"Hardware: {out[0, 0].item():.1f}")
# CHECK-OUTPUT: Hardware:
print(f"Expected: 8.0")

if abs(out[0, 0].item() - 8.0) < 0.5:
    print(f"PASS: Can reuse input after reduce_sum")
    # CHECK-OUTPUT: PASS: Can reuse input after reduce_sum
else:
    print(f"FAIL: Expected 8.0, got {out[0, 0].item():.1f}")
    print(f"This suggests reduce_sum corrupts the input for reuse!")
