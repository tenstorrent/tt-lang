# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum, exp

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)])
def test_chained_reduce(a, b, c, d, out):
    a_accessor = TensorAccessor(a)
    b_accessor = TensorAccessor(b)
    c_accessor = TensorAccessor(c)
    d_accessor = TensorAccessor(d)

    @compute()
    async def compute_chained(a_cb: CircularBuffer, b_cb: CircularBuffer, c_cb: CircularBuffer, d_cb: CircularBuffer, out_cb: CircularBuffer):
        a_val = a_cb.wait()
        b_val = b_cb.wait()
        c_val = c_cb.wait()
        d_val = d_cb.wait()
        o = out_cb.reserve()
        # Compute: exp(reduce_sum(a, b, c)) + d
        reduced = reduce_sum(a_val, b_val, c_val, dim=1)
        exp_result = exp(reduced)
        final_result = exp_result + d_val
        o.store(final_result)
        a_cb.pop()
        b_cb.pop()
        c_cb.pop()
        d_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_a(a_cb: CircularBuffer, b_cb: CircularBuffer, c_cb: CircularBuffer, d_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = a_cb.reserve()
        tx = dma(a_accessor[0, 0], a_shard)
        tx.wait()

    @datamovement()
    async def dm_b(a_cb: CircularBuffer, b_cb: CircularBuffer, c_cb: CircularBuffer, d_cb: CircularBuffer, out_cb: CircularBuffer):
        b_shard = b_cb.reserve()
        tx = dma(b_accessor[0, 0], b_shard)
        tx.wait()

    @datamovement()
    async def dm_c(a_cb: CircularBuffer, b_cb: CircularBuffer, c_cb: CircularBuffer, d_cb: CircularBuffer, out_cb: CircularBuffer):
        c_shard = c_cb.reserve()
        tx = dma(c_accessor[0, 0], c_shard)
        tx.wait()

    @datamovement()
    async def dm_d(a_cb: CircularBuffer, b_cb: CircularBuffer, c_cb: CircularBuffer, d_cb: CircularBuffer, out_cb: CircularBuffer):
        d_shard = d_cb.reserve()
        tx = dma(d_accessor[0, 0], d_shard)
        tx.wait()

    return Program(compute_chained, dm_a, dm_b, dm_c, dm_d)(a, b, c, d, out)


# CHECK: func.func @test_chained_reduce
# CHECK: "d2m.tile_reduce_sum"
# CHECK: "d2m.tile_exp"
# CHECK: "d2m.tile_add"

a = torch.ones((32, 32))
b = torch.ones((32, 32))
c = torch.zeros((32, 32))
d = torch.full((32, 32), 2.0)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"Testing chained: exp(reduce_sum(1*1, 0)) + 2 = exp(1) + 2 ≈ 4.718")

test_chained_reduce(a, b, c, d, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===

# Expected: exp(reduce_sum(1*1, 0)) + 2 = exp(1) + 2 ≈ 4.718
import math
expected = torch.full((32, 32), math.e + 2.0)
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print(f"PASS: Output matches expected (exp(1) + 2 = {expected[0,0].item():.3f})")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(f"FAIL: Expected {expected[0,0].item():.3f}, got values from {out.min().item():.3f} to {out.max().item():.3f}")
