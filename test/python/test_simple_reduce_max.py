# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Simple reduce test matching TTIR pattern - just reduce one input, no fusion

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_max

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def simple_reduce_max(input_tensor, scaler, out):
    input_accessor = TensorAccessor(input_tensor)
    scaler_accessor = TensorAccessor(scaler)

    @compute()
    async def compute_reduce(input_cb: CircularBuffer, scaler_cb: CircularBuffer, out_cb: CircularBuffer):
        inp = input_cb.wait()
        scale = scaler_cb.wait()
        o = out_cb.reserve()
        # Just reduce with scaler=1.0, no c term
        result = reduce_max(inp, scale, o, dim=1)
        o.store(result)
        input_cb.pop()
        scaler_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_input(input_cb: CircularBuffer, scaler_cb: CircularBuffer, out_cb: CircularBuffer):
        input_shard = input_cb.reserve()
        tx = dma(input_accessor[0, 0], input_shard)
        tx.wait()

    @datamovement()
    async def dm_scaler(input_cb: CircularBuffer, scaler_cb: CircularBuffer, out_cb: CircularBuffer):
        scaler_shard = scaler_cb.reserve()
        tx = dma(scaler_accessor[0, 0], scaler_shard)
        tx.wait()

    return Program(compute_reduce, dm_input, dm_scaler)(input_tensor, scaler, out)

# Test: max(6, 6, 6, ...) over 32 columns = 6
input_tensor = torch.full((32, 32), 6.0)
scaler = torch.ones((32, 32))  # Scaler of 1.0
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"input: all 6.0, scaler: all 1.0")
print(f"Operation: reduce_max(input, scaler, out_cb, dim=1)")
print(f"Expected: max(6*1) over 32 columns = 6 in first column")
print(out)

simple_reduce_max(input_tensor, scaler, out)

print("\n=== AFTER KERNEL ===")
print(f"out[0, 0] = {out[0, 0].item():.3f}")
print(f"First column: min={out[:, 0].min().item():.3f}, max={out[:, 0].max().item():.3f}")
print(out[:, :2])

expected_col = torch.full((32,), 6.0)
if torch.allclose(out[:, 0], expected_col, rtol=1e-2, atol=1e-2):
    print("PASS: Simple reduce_max works correctly")
else:
    print(f"FAIL: Expected first column all 6.0, got {out[:, 0].unique()}")
