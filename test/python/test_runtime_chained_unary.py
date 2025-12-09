# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

import torch
from ttlang.d2m_api import *
from ttlang.operators import exp, sqrt


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def test_chained_unary(input_tensor, out):
    input_accessor = TensorAccessor(input_tensor)

    @compute()
    async def compute_chained(input_cb: CircularBuffer, out_cb: CircularBuffer):
        inp = input_cb.wait()
        o = out_cb.reserve()

        # Chain two unary ops with explicit CB synchronization
        # Input: 2.0 → exp(2.0) ≈ 7.389 → sqrt(7.389) ≈ 2.718

        # First op: exp(input)
        exp_result = exp(inp)
        o.store(exp_result)
        out_cb.push()  # Push intermediate result

        # Wait to read back intermediate
        intermediate = out_cb.wait()

        # Second op: sqrt(exp_result)
        sqrt_result = sqrt(intermediate)

        o.store(sqrt_result)
        input_cb.pop()
        out_cb.push()  # Push final result

    @datamovement()
    async def dm_loader(input_cb: CircularBuffer, out_cb: CircularBuffer):
        input_shard = input_cb.reserve()
        tx = dma(input_accessor[0, 0], input_shard)
        tx.wait()
        input_cb.push()

    return Program(compute_chained, dm_loader)(input_tensor, out)


# CHECK: func.func @test_chained_unary
# CHECK: "d2m.tile_exp"
# CHECK: "d2m.tile_sqrt"

# Test: sqrt(exp(2.0)) ≈ 2.718
input_tensor = torch.full((32, 32), 2.0)
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"Testing chained unary: sqrt(exp(2.0))")
print(f"exp(2.0) ≈ 7.389")
print(f"sqrt(7.389) ≈ 2.718")

test_chained_unary(input_tensor, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.3f}")
# CHECK-OUTPUT: out[0, 0] =

import math

expected_val = math.sqrt(math.exp(2.0))
print(f"Expected: {expected_val:.3f}")

# Check if we got the right answer
if abs(out[0, 0].item() - expected_val) < 0.1:
    print(f"PASS: Output matches expected (sqrt(exp(2.0)) ≈ 2.718)")
    # CHECK-OUTPUT: PASS: Output matches expected
elif abs(out[0, 0].item() - math.exp(2.0)) < 0.1:
    print(f"FAIL: Got exp(2.0) = {math.exp(2.0):.3f}, sqrt was skipped!")
elif abs(out[0, 0].item() - 2.0) < 0.1:
    print(f"FAIL: Got input value 2.0, both ops were skipped!")
else:
    print(f"FAIL: Expected {expected_val:.3f}, got {out[0, 0].item():.3f}")
