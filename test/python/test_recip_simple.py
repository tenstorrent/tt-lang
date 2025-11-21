# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test: Simple reciprocal operation

import torch
from ttlang.d2m_api import *
from ttlang.operators import recip

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def test_recip(input_tensor, out):
    input_accessor = TensorAccessor(input_tensor)

    @compute()
    async def compute_recip(input_cb: CircularBuffer, out_cb: CircularBuffer):
        inp = input_cb.wait()
        o = out_cb.reserve()

        result = recip(inp)

        o.store(result)
        input_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(input_cb: CircularBuffer, out_cb: CircularBuffer):
        input_shard = input_cb.reserve()
        tx = dma(input_accessor[0, 0], input_shard)
        tx.wait()
        input_cb.push()

    return Program(compute_recip, dm_loader)(input_tensor, out)


# CHECK: func.func @test_recip
# CHECK: "d2m.tile_recip"

# Test with simple value: 1/4 = 0.25
input_tensor = torch.full((32, 32), 4.0)
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"Input: all {input_tensor[0, 0].item()}")
print(f"Expected: 1/{input_tensor[0, 0].item()} = {1.0/input_tensor[0, 0].item()}")

test_recip(input_tensor, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.6f}")
# CHECK-OUTPUT: out[0, 0] =
print(f"Expected: 0.250000")

expected = 1.0 / input_tensor[0, 0].item()
if abs(out[0, 0].item() - expected) / expected < 0.1:
    print(f"PASS: recip produced correct result")
    # CHECK-OUTPUT: PASS: recip produced correct result
else:
    print(f"FAIL: Expected {expected:.6f}, got {out[0, 0].item():.6f}")
