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


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_runtime_reduce_sum(input_tensor, scaler, out):
    input_accessor = TensorAccessor(input_tensor)
    scaler_accessor = TensorAccessor(scaler)

    @compute()
    async def compute_reduce(input_cb: CircularBuffer, scaler_cb: CircularBuffer, out_cb: CircularBuffer):
        inp = input_cb.wait()
        scale = scaler_cb.wait()
        o = out_cb.reserve()
        # result = sum<dim>(input * scaler) - reduce over columns (dim=1)
        result = reduce_sum(inp, scale, dim=1)
        o.store(result)
        input_cb.pop()
        scaler_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_input(input_cb: CircularBuffer, scaler_cb: CircularBuffer, out_cb: CircularBuffer):
        input_shard = input_cb.reserve()
        tx = dma(input_accessor[0, 0], input_shard)
        tx.wait()
        input_cb.push()

    @datamovement()
    async def dm_scaler(input_cb: CircularBuffer, scaler_cb: CircularBuffer, out_cb: CircularBuffer):
        scaler_shard = scaler_cb.reserve()
        tx = dma(scaler_accessor[0, 0], scaler_shard)
        tx.wait()
        scaler_cb.push()

    return Program(compute_reduce, dm_input, dm_scaler)(input_tensor, scaler, out)


# CHECK: func.func @test_runtime_reduce_sum
# CHECK: "d2m.tile_reduce_sum"

# Test: sum(2 * 1) over 32 columns = 2 * 32 = 64
input_tensor = torch.full((32, 32), 2.0)
scaler = torch.ones((32, 32))
out = torch.zeros(32, 32)  # Pre-initialize with zeros (issue #31)

print("=== BEFORE KERNEL ===")
print(f"input: all 2.0, scaler: all 1.0")
print(f"Operation: reduce_sum(input, scaler, dim=1)")
print(f"Expected: sum(2*1) over 32 columns = 64 in first column")
print(out)

test_runtime_reduce_sum(input_tensor, scaler, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.3f}")
print(f"First column: min={out[:, 0].min().item():.3f}, max={out[:, 0].max().item():.3f}, mean={out[:, 0].mean().item():.3f}")
print(out[:, :2])

# Expected: sum(2*1) over 32 columns = 64 in first column (32x1 output)
expected_col = torch.full((32,), 64.0)
if torch.allclose(out[:, 0], expected_col, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches expected (reduce_sum dim=1, first column = 64.0)")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(f"FAIL: Expected first column all 64.0, got values from {out[:, 0].min().item():.3f} to {out[:, 0].max().item():.3f}")
