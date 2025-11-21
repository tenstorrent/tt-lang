# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: Four operations chained with CB cycling
# add → mul → add → mul (testing 4 CB cycles)

import torch
from ttlang.d2m_api import *

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_four_ops(A, B, C, out):
    A_accessor = TensorAccessor(A)
    B_accessor = TensorAccessor(B)
    C_accessor = TensorAccessor(C)

    @compute()
    async def compute_chain(A_cb: CircularBuffer, B_cb: CircularBuffer, C_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()
        b = B_cb.wait()
        c = C_cb.wait()
        o = out_cb.reserve()

        # Op 1: add
        temp1 = a + b
        o.store(temp1)
        out_cb.push()
        temp1 = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 2: mul
        temp2 = temp1 * c
        o.store(temp2)
        out_cb.push()
        temp2 = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 3: add (reuse 'a')
        temp3 = temp2 + a
        o.store(temp3)
        out_cb.push()
        temp3 = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 4: mul
        result = temp3 * b

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


# CHECK: func.func @test_four_ops
# CHECK: "d2m.tile_add"
# CHECK: "d2m.tile_mul"

# Test: ((2 + 3) * 4 + 2) * 3 = (5 * 4 + 2) * 3 = (20 + 2) * 3 = 22 * 3 = 66
A = torch.full((32, 32), 2.0)
B = torch.full((32, 32), 3.0)
C = torch.full((32, 32), 4.0)
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"A: all 2.0, B: all 3.0, C: all 4.0")
print(f"Expected: ((2 + 3) * 4 + 2) * 3 = 66.0")

# Manual calc
temp1 = 2.0 + 3.0  # 5
temp2 = temp1 * 4.0  # 20
temp3 = temp2 + 2.0  # 22
result = temp3 * 3.0  # 66

test_four_ops(A, B, C, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"Hardware: {out[0, 0].item():.1f}")
# CHECK-OUTPUT: Hardware:
print(f"Expected: {result:.1f}")

if abs(out[0, 0].item() - result) / result < 0.1:
    print(f"PASS: Four operations with CB cycling work")
    # CHECK-OUTPUT: PASS: Four operations
else:
    print(f"FAIL: Expected {result:.1f}, got {out[0, 0].item():.1f}")
