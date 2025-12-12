# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TODO: update to use pytest (see issue #91)
# UNSUPPORTED: true
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Tests chaining 8 operations to stress-test CB synchronization pattern
# Input: 2.0
# Chain: exp → sqrt → recip → exp → sqrt → recip → exp → sqrt
# Expected: sqrt(exp(recip(sqrt(exp(recip(sqrt(exp(2.0))))))))

import torch
from ttlang.d2m_api import *
from ttlang.operators import exp, sqrt, recip
import math


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def test_chained_8ops(input_tensor, out):
    input_accessor = TensorAccessor(input_tensor)

    @compute()
    async def compute_chain(input_cb: CircularBuffer, out_cb: CircularBuffer):
        inp = input_cb.wait()
        o = out_cb.reserve()

        # Op 1: exp(2.0) ≈ 7.389
        result = exp(inp)
        o.store(result)
        out_cb.push()
        result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 2: sqrt(7.389) ≈ 2.718
        result = sqrt(result)
        o.store(result)
        out_cb.push()
        result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 3: recip(2.718) ≈ 0.368
        result = recip(result)
        o.store(result)
        out_cb.push()
        result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 4: exp(0.368) ≈ 1.445
        result = exp(result)
        o.store(result)
        out_cb.push()
        result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 5: sqrt(1.445) ≈ 1.202
        result = sqrt(result)
        o.store(result)
        out_cb.push()
        result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 6: recip(1.202) ≈ 0.832
        result = recip(result)
        o.store(result)
        out_cb.push()
        result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 7: exp(0.832) ≈ 2.298
        result = exp(result)
        o.store(result)
        out_cb.push()
        result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 8: sqrt(2.298) ≈ 1.516
        result = sqrt(result)

        o.store(result)
        input_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(input_cb: CircularBuffer, out_cb: CircularBuffer):
        input_shard = input_cb.reserve()
        tx = dma(input_accessor[0, 0], input_shard)
        tx.wait()
        input_cb.push()

    return Program(compute_chain, dm_loader)(input_tensor, out)


# CHECK: func.func @test_chained_8ops
# CHECK: "d2m.tile_exp"
# CHECK: "d2m.tile_sqrt"
# CHECK: "d2m.tile_recip"

# Compute expected value
input_val = 2.0
v1 = math.exp(input_val)
v2 = math.sqrt(v1)
v3 = 1.0 / v2
v4 = math.exp(v3)
v5 = math.sqrt(v4)
v6 = 1.0 / v5
v7 = math.exp(v6)
v8 = math.sqrt(v7)
expected = v8

input_tensor = torch.full((32, 32), input_val)
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"Testing 8 chained ops on input: {input_val}")
print(f"exp → sqrt → recip → exp → sqrt → recip → exp → sqrt")
print(f"Expected final result: {expected:.6f}")

test_chained_8ops(input_tensor, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.6f}")
# CHECK-OUTPUT: out[0, 0] =
print(f"Expected: {expected:.6f}")

# Check if we got the right answer (within reasonable tolerance)
if abs(out[0, 0].item() - expected) / expected < 0.01:
    print(f"PASS: 8-op chain produced correct result")
    # CHECK-OUTPUT: PASS: 8-op chain produced correct result
else:
    print(f"FAIL: Expected {expected:.6f}, got {out[0, 0].item():.6f}")
