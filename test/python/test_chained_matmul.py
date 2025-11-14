# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Tests chained matrix multiplications using separate kernels with intermediate CB
#   note to zoe: 7b59bec15a90c92b70d15d11e6a4784b41832757 has the change with MemRefProvenanceAnalysis to fix this.
# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

import torch
from ttlang.d2m_api import *


# First matmul: A @ B
@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def matmul_ab(a, b, temp_out):
    a_accessor = TensorAccessor(a)
    b_accessor = TensorAccessor(b)

    @compute()
    async def compute_ab(a_cb: CircularBuffer, b_cb: CircularBuffer, out_cb: CircularBuffer):
        a_val = a_cb.wait()
        b_val = b_cb.wait()
        o = out_cb.reserve()
        result = a_val @ b_val
        o.store(result)
        a_cb.pop()
        b_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_a(a_cb: CircularBuffer, b_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = a_cb.reserve()
        tx = dma(a_accessor[0, 0], a_shard)
        tx.wait()

    @datamovement()
    async def dm_b(a_cb: CircularBuffer, b_cb: CircularBuffer, out_cb: CircularBuffer):
        b_shard = b_cb.reserve()
        tx = dma(b_accessor[0, 0], b_shard)
        tx.wait()

    return Program(compute_ab, dm_a, dm_b)(a, b, temp_out)


# Second matmul: temp @ C
@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def matmul_temp_c(temp_in, c, out):
    temp_accessor = TensorAccessor(temp_in)
    c_accessor = TensorAccessor(c)

    @compute()
    async def compute_temp_c(temp_cb: CircularBuffer, c_cb: CircularBuffer, out_cb: CircularBuffer):
        temp_val = temp_cb.wait()
        c_val = c_cb.wait()
        o = out_cb.reserve()
        result = temp_val @ c_val
        o.store(result)
        temp_cb.pop()
        c_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_temp(temp_cb: CircularBuffer, c_cb: CircularBuffer, out_cb: CircularBuffer):
        temp_shard = temp_cb.reserve()
        tx = dma(temp_accessor[0, 0], temp_shard)
        tx.wait()

    @datamovement()
    async def dm_c(temp_cb: CircularBuffer, c_cb: CircularBuffer, out_cb: CircularBuffer):
        c_shard = c_cb.reserve()
        tx = dma(c_accessor[0, 0], c_shard)
        tx.wait()

    return Program(compute_temp_c, dm_temp, dm_c)(temp_in, c, out)


def test_chained_matmul(a, b, c, out):
    temp = torch.zeros(32, 32)
    matmul_ab(a, b, temp)
    matmul_temp_c(temp, c, out)


# CHECK: func.func @matmul_ab
# CHECK: "d2m.tile_matmul"
# CHECK: func.func @matmul_temp_c
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
