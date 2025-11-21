# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: Element-wise add followed by matmul

import torch
from ttlang.d2m_api import *

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_add_matmul(A, B, C, out):
    A_accessor = TensorAccessor(A)
    B_accessor = TensorAccessor(B)
    C_accessor = TensorAccessor(C)

    @compute()
    async def compute_chain(A_cb: CircularBuffer, B_cb: CircularBuffer, C_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()
        b = B_cb.wait()
        c = C_cb.wait()
        o = out_cb.reserve()

        # Step 1: Element-wise add
        ab = a + b
        o.store(ab)
        out_cb.push()
        ab = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 2: Matmul
        result = ab @ c

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


# CHECK: func.func @test_add_matmul
# CHECK: "d2m.tile_add"
# CHECK: "d2m.tile_matmul"

# Test: (2 + 3) @ 4 = 5 @ 4 = 5 * 4 * 32 = 640
A = torch.full((32, 32), 2.0)
B = torch.full((32, 32), 3.0)
C = torch.full((32, 32), 4.0)
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"A: all 2.0, B: all 3.0, C: all 4.0")
print(f"Expected: (2 + 3) @ 4 = 5 @ 4 = {5.0 * 4.0 * 32}")

# Compute expected
AB_expected = A + B  # All 5.0
result_expected = AB_expected @ C  # 5 * 4 * 32 = 640 per element

test_add_matmul(A, B, C, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.1f}")
# CHECK-OUTPUT: out[0, 0] =
print(f"Expected: {result_expected[0, 0].item():.1f}")

if abs(out[0, 0].item() - result_expected[0, 0].item()) / result_expected[0, 0].item() < 0.1:
    print(f"PASS: add → matmul produced correct result")
    # CHECK-OUTPUT: PASS: add
else:
    print(f"FAIL: Expected {result_expected[0, 0].item():.1f}, got {out[0, 0].item():.1f}")
