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


# Only the last kernel (matmul_temp_c) gets captured in MLIR dump
# CHECK: func.func @matmul
# CHECK: "d2m.tile_matmul"

a = torch.eye(32)
b = torch.full((32, 32), 2.0)
c = torch.full((32, 32), 3.0)
out = torch.zeros(32, 32)  # Pre-initialize with zeros (issue #31)

print("=== BEFORE KERNEL ===")
print(f"Testing chained matmul: (I @ 2s) @ 3s")
print(f"I @ 2s = all 2s (identity preserves uniform matrix)")
print(f"(all 2s) @ (all 3s) = each element sum(2*3) over 32 cols = 192")

test_chained_matmul(a, b, c, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===

# Expected: (I @ all 2s) @ all 3s = (all 2s) @ (all 3s) = all 192s
# Each element [i,j] = sum_k(temp[i,k] * c[k,j]) = sum_k(2 * 3) = 6 * 32 = 192
expected = torch.full((32, 32), 192.0)
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print(f"PASS: Output matches expected (all 192.0)")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(f"FAIL: Expected all 192.0, got values from {out.min().item():.1f} to {out.max().item():.1f}")
