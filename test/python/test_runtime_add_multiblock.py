# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: Simple add with multiple tiles per core
# Grid: 1x1, Tensor: 64x64 (2x2 tiles), Block factors: 2x2

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(2, 2), (2, 2), (2, 2)])
def test_add_multiblock(lhs, rhs, out):
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)

    @compute()
    async def add_compute(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()
        lhs_cb.push()

        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_accessor[0, 0], rhs_shard)
        tx.wait()
        rhs_cb.push()

    return Program(add_compute, dm_loader)(lhs, rhs, out)


# CHECK: func.func @test_add_multiblock
# CHECK: "d2m.tile_add"

# Test: 2.0 + 3.0 = 5.0
lhs = torch.full((64, 64), 2.0)
rhs = torch.full((64, 64), 3.0)
out = torch.zeros(64, 64)

print("=== BEFORE KERNEL ===")
print(f"Test: multiblock add (2x2 tiles on single core)")
print(f"lhs: all 2.0, rhs: all 3.0")
print(f"Expected: all 5.0")

test_add_multiblock(lhs, rhs, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===

expected = 5.0
print(f"Hardware: {out[0, 0].item():.1f}, Expected: {expected:.1f}")
# CHECK-OUTPUT: Hardware: {{[0-9]+\.[0-9]+}}, Expected: 5.0

print(f"Check corners:")
print(f"  out[0, 0] = {out[0, 0].item():.1f}")
print(f"  out[0, 63] = {out[0, 63].item():.1f}")
print(f"  out[63, 0] = {out[63, 0].item():.1f}")
print(f"  out[63, 63] = {out[63, 63].item():.1f}")

all_correct = torch.allclose(out, torch.full((64, 64), 5.0), rtol=0.01)

if all_correct:
    print(f"PASS: Multiblock add produced correct result!")
    # CHECK-OUTPUT: PASS: Multiblock add produced correct result
else:
    print(f"FAIL: Not all values are {expected:.1f}")
    print(f"  Min: {out.min().item():.1f}, Max: {out.max().item():.1f}")
