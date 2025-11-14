# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_runtime_matmul(lhs, rhs, out):
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)

    @compute()
    async def compute_matmul(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l @ r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_lhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()

    @datamovement()
    async def dm_rhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_accessor[0, 0], rhs_shard)
        tx.wait()

    return Program(compute_matmul, dm_lhs, dm_rhs)(lhs, rhs, out)


# CHECK: func.func @test_runtime_matmul
# CHECK: "d2m.tile_matmul"

lhs = torch.randn((32, 32))
rhs = torch.randn((32, 32))
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"Testing matmul compilation")

test_runtime_matmul(lhs, rhs, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print("PASS: matmul compiled successfully")
# CHECK-OUTPUT: PASS: matmul compiled successfully
