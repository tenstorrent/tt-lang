# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: Load exp_S from host → reduce_sum
# This isolates the exact pattern from Kernel 2's first operation

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_loaded_reduce(exp_S, ones, out):
    exp_S_accessor = TensorAccessor(exp_S)
    ones_accessor = TensorAccessor(ones)

    @compute()
    async def compute_reduce(exp_S_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        exp_s = exp_S_cb.wait()
        ones_val = ones_cb.wait()
        o = out_cb.reserve()

        # Reduce sum (same as Kernel 2's first operation)
        sum_exp = reduce_sum(exp_s, ones_val, dim=1)

        o.store(sum_exp)
        exp_S_cb.pop()
        ones_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(exp_S_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        exp_s_shard = exp_S_cb.reserve()
        tx = dma(exp_S_accessor[0, 0], exp_s_shard)
        tx.wait()
        exp_S_cb.push()

        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

    return Program(compute_reduce, dm_loader)(exp_S, ones, out)


# CHECK: func.func @test_loaded_reduce
# CHECK: "d2m.tile_reduce_sum"

# Create exp_S as if it came from Kernel 1
exp_S = torch.full((32, 32), 1.058)
ones = torch.ones((32, 32))
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"exp_S: all {exp_S[0, 0].item():.6f}")
print(f"Expected sum: {exp_S[0, 0].item() * 32:.6f}")

test_loaded_reduce(exp_S, ones, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"Hardware: sum[0, 0] = {out[0, 0].item():.6f}")
# CHECK-OUTPUT: Hardware: sum[0, 0] =
print(f"Expected: {exp_S[0, 0].item() * 32:.6f}")

expected = exp_S[0, 0].item() * 32
error = abs(out[0, 0].item() - expected) / expected
print(f"Error: {error*100:.1f}%")

if error < 0.2:
    print(f"PASS: reduce_sum on loaded exp_S works")
    # CHECK-OUTPUT: PASS: reduce_sum on loaded exp_S works
else:
    print(f"FAIL: Expected {expected:.6f}, got {out[0, 0].item():.6f}")
