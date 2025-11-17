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

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_chained_reduce(input_tensor, scaler, bias, out):
    input_accessor = TensorAccessor(input_tensor)
    scaler_accessor = TensorAccessor(scaler)
    bias_accessor = TensorAccessor(bias)

    @compute()
    async def compute_chained(input_cb: CircularBuffer, scaler_cb: CircularBuffer, bias_cb: CircularBuffer, out_cb: CircularBuffer):
        inp = input_cb.wait()
        scale = scaler_cb.wait()
        bias_val = bias_cb.wait()
        o = out_cb.reserve()
        # Compute: exp(reduce_sum(input, scaler)) + bias
        reduced = reduce_sum(inp, scale, dim=1)
        exp_result = exp(reduced)
        final_result = exp_result + bias_val
        o.store(final_result)
        input_cb.pop()
        scaler_cb.pop()
        bias_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_input(input_cb: CircularBuffer, scaler_cb: CircularBuffer, bias_cb: CircularBuffer, out_cb: CircularBuffer):
        input_shard = input_cb.reserve()
        tx = dma(input_accessor[0, 0], input_shard)
        tx.wait()
        input_cb.push()

    @datamovement()
    async def dm_scaler(input_cb: CircularBuffer, scaler_cb: CircularBuffer, bias_cb: CircularBuffer, out_cb: CircularBuffer):
        scaler_shard = scaler_cb.reserve()
        tx = dma(scaler_accessor[0, 0], scaler_shard)
        tx.wait()
        scaler_cb.push()

    @datamovement()
    async def dm_bias(input_cb: CircularBuffer, scaler_cb: CircularBuffer, bias_cb: CircularBuffer, out_cb: CircularBuffer):
        bias_shard = bias_cb.reserve()
        tx = dma(bias_accessor[0, 0], bias_shard)
        tx.wait()
        bias_cb.push()

    return Program(compute_chained, dm_input, dm_scaler, dm_bias)(input_tensor, scaler, bias, out)


# CHECK: func.func @test_chained_reduce
# CHECK: "d2m.tile_reduce_sum"
# CHECK: "d2m.tile_exp"
# CHECK: "d2m.tile_add"

# Test: exp(reduce_sum(1*1)) + 2 = exp(sum of 32 ones) + 2 = exp(32) + 2
input_tensor = torch.ones((32, 32))
scaler = torch.ones((32, 32))
bias = torch.full((32, 32), 2.0)
out = torch.zeros(32, 32)  # Pre-initialize with zeros (issue #31)

print("=== BEFORE KERNEL ===")
print(f"Testing chained: exp(reduce_sum(1*1)) + 2")
print(f"reduce_sum(1*1) over 32 cols = 32")
print(f"exp(32) ≈ 7.9e13")
print(f"exp(32) + 2 ≈ 7.9e13")

test_chained_reduce(input_tensor, scaler, bias, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.3e}")
# CHECK-OUTPUT: out[0, 0] =

# Expected: exp(32) + 2 ≈ 7.9e13
import math
expected_val = math.exp(32) + 2.0
print(f"Expected in first column: {expected_val:.3e}")
if out[0, 0].item() > 1e13:
    print(f"PASS: Output is in expected range (exp(32) + 2)")
    # CHECK-OUTPUT: PASS: Output is in expected range
else:
    print(f"FAIL: Expected ~7.9e13, got {out[0, 0].item():.3e}")
