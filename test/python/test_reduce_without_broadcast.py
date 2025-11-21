# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: reduce_sum with broadcast=False to see what hardware actually produces
# Does tile_reduce_sum broadcast within the tile, or only fill certain positions?

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_reduce_no_broadcast(A, ones, out):
    A_accessor = TensorAccessor(A)
    ones_accessor = TensorAccessor(ones)

    @compute()
    async def compute_reduce(A_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()
        ones_val = ones_cb.wait()
        o = out_cb.reserve()

        # reduce_sum with experimental broadcast indexing maps
        result = reduce_sum(a, ones_val, dim=1)

        o.store(result)
        A_cb.pop()
        ones_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(A_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = A_cb.reserve()
        tx = dma(A_accessor[0, 0], a_shard)
        tx.wait()
        A_cb.push()

        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

    return Program(compute_reduce, dm_loader)(A, ones, out)


# CHECK: func.func @test_reduce_no_broadcast
# CHECK: "d2m.tile_reduce_sum"

# Test: What does hardware tile_reduce_sum actually produce?
A = torch.full((32, 32), 2.0)
ones = torch.ones((32, 32))
out = torch.zeros(32, 32)

print("=== TEST: Hardware tile_reduce_sum behavior ===")
print(f"Input: 32x32 tensor, all values = 2.0")
print(f"Expected sum: 2.0 * 32 = 64.0")
print(f"Question: Does tile produce 64.0 only at column 0, or broadcast to all columns?")

test_reduce_no_broadcast(A, ones, out)

print("\n=== AFTER KERNEL (experimental broadcast maps) ===")
# CHECK-OUTPUT: === AFTER KERNEL

print(f"\nColumn values in row 0:")
print(f"  out[0, 0] = {out[0, 0].item():.1f}")
print(f"  out[0, 1] = {out[0, 1].item():.1f}")
print(f"  out[0, 2] = {out[0, 2].item():.1f}")
print(f"  out[0, 15] = {out[0, 15].item():.1f}")
print(f"  out[0, 31] = {out[0, 31].item():.1f}")

row_0_all_same = torch.allclose(out[0, :], torch.full((32,), 64.0), rtol=0.01)
col_0_correct = abs(out[0, 0].item() - 64.0) < 1.0

print(f"\nAnalysis:")
if row_0_all_same:
    print(f"✅ Hardware DOES broadcast: all columns have 64.0!")
    print(f"   We can change the output map to (d0, d1)")
    # CHECK-OUTPUT: Hardware DOES broadcast
elif col_0_correct and not row_0_all_same:
    print(f"❌ Hardware does NOT broadcast: only column 0 has 64.0")
    print(f"   We NEED a separate broadcast operation")
    print(f"   (This is expected based on tt-mlir tests)")
    # CHECK-OUTPUT: Hardware does NOT broadcast
else:
    print(f"⚠️ Unexpected result - neither pattern matched")
    print(f"   Min: {out.min().item():.1f}, Max: {out.max().item():.1f}")
