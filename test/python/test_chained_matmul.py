# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_chained_matmul(a, b, c, out):
    a_accessor = TensorAccessor(a)
    b_accessor = TensorAccessor(b)
    c_accessor = TensorAccessor(c)

    @compute()
    async def matmul_compute(a_cb: CircularBuffer, b_cb: CircularBuffer, c_cb: CircularBuffer, out_cb: CircularBuffer):
        a_val = a_cb.wait()
        b_val = b_cb.wait()
        c_val = c_cb.wait()
        o = out_cb.reserve()
        # (A @ B) @ C
        temp = a_val @ b_val
        result = temp @ c_val
        o.store(result)
        a_cb.pop()
        b_cb.pop()
        c_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_a(a_cb: CircularBuffer, b_cb: CircularBuffer, c_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = a_cb.reserve()
        tx = dma(a_accessor[0, 0], a_shard)
        tx.wait()

    @datamovement()
    async def dm_b(a_cb: CircularBuffer, b_cb: CircularBuffer, c_cb: CircularBuffer, out_cb: CircularBuffer):
        b_shard = b_cb.reserve()
        tx = dma(b_accessor[0, 0], b_shard)
        tx.wait()

    @datamovement()
    async def dm_c(a_cb: CircularBuffer, b_cb: CircularBuffer, c_cb: CircularBuffer, out_cb: CircularBuffer):
        c_shard = c_cb.reserve()
        tx = dma(c_accessor[0, 0], c_shard)
        tx.wait()

    return Program(matmul_compute, dm_a, dm_b, dm_c)(a, b, c, out)


# CHECK: func.func @test_chained_matmul
# CHECK: "d2m.tile_matmul"
# CHECK: "d2m.tile_matmul"

a = torch.eye(32)
b = torch.full((32, 32), 2.0)
c = torch.full((32, 32), 3.0)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"Testing chained matmul: (I @ 2s) @ 3s")
print(f"Each row of (I @ 2s) has one 2.0, rest zeros")
print(f"(I @ 2s) @ 3s gives each row sum of one column of 3s = 96")

test_chained_matmul(a, b, c, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===

# Expected: (I @ 2s) @ 3s
# I @ 2s gives a matrix where each row has a single 2.0 (diagonal)
# (I @ 2s) @ 3s: each element is dot product of a row with 2.0 in one position
# with a column of all 3s, giving 2*3 = 6 in diagonal positions,
# and 0 elsewhere since other positions multiply 0 with 3
expected = torch.eye(32) * 6.0
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print(f"PASS: Output matches expected (diagonal is 6, off-diagonal is 0)")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(f"FAIL: Expected 6I, got diagonal from {torch.diag(out).min().item():.1f} to {torch.diag(out).max().item():.1f}")
