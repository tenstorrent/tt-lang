# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_runtime_reduce_sum(a, b, c, out):
    a_accessor = TensorAccessor(a)
    b_accessor = TensorAccessor(b)
    c_accessor = TensorAccessor(c)

    @compute()
    async def compute_reduce(a_cb: CircularBuffer, b_cb: CircularBuffer, c_cb: CircularBuffer, out_cb: CircularBuffer):
        a_val = a_cb.wait()
        b_val = b_cb.wait()
        c_val = c_cb.wait()
        o = out_cb.reserve()
        # result = sum<dim>(a * b, c) - reduce over columns (dim=1)
        result = reduce_sum(a_val, b_val, c_val, dim=1)
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

    return Program(compute_reduce, dm_a, dm_b, dm_c)(a, b, c, out)


# CHECK: func.func @test_runtime_reduce_sum
# CHECK: "d2m.tile_reduce_sum"

a = torch.ones((32, 32))
b = torch.ones((32, 32))
c = torch.zeros((32, 32))
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"Testing reduce_sum compilation")

test_runtime_reduce_sum(a, b, c, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print("PASS: reduce_sum compiled successfully")
# CHECK-OUTPUT: PASS: reduce_sum compiled successfully
