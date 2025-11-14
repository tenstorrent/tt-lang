# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Debug test for reduce_sum to identify the +1 error

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_reduce_sum(input_tensor, scaler, out):
    input_accessor = TensorAccessor(input_tensor)
    scaler_accessor = TensorAccessor(scaler)

    @compute()
    async def compute_reduce(input_cb: CircularBuffer, scaler_cb: CircularBuffer, out_cb: CircularBuffer):
        inp = input_cb.wait()
        scale = scaler_cb.wait()
        o = out_cb.reserve()
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

    @datamovement()
    async def dm_scaler(input_cb: CircularBuffer, scaler_cb: CircularBuffer, out_cb: CircularBuffer):
        scaler_shard = scaler_cb.reserve()
        tx = dma(scaler_accessor[0, 0], scaler_shard)
        tx.wait()

    return Program(compute_reduce, dm_input, dm_scaler)(input_tensor, scaler, out)


print("=" * 80)
print("TEST 1: input=1.0, scaler=1.0")
print("Expected: sum(1*1) over 32 columns = 32")
print("=" * 80)
input1 = torch.ones((32, 32))
scaler1 = torch.ones((32, 32))
out1 = torch.full((32, 32), -999.0)
test_reduce_sum(input1, scaler1, out1)
print(f"Got: {out1[0, 0].item():.1f} (error: {out1[0, 0].item() - 32:.1f})")
print(f"First column unique values: {out1[:, 0].unique()}")

print("\n" + "=" * 80)
print("TEST 2: input=4.0, scaler=1.0")
print("Expected: sum(4*1) over 32 columns = 128")
print("=" * 80)
input2 = torch.full((32, 32), 4.0)
scaler2 = torch.ones((32, 32))
out2 = torch.full((32, 32), -999.0)
test_reduce_sum(input2, scaler2, out2)
print(f"Got: {out2[0, 0].item():.1f} (error: {out2[0, 0].item() - 128:.1f})")
print(f"First column unique values: {out2[:, 0].unique()}")

print("\n" + "=" * 80)
print("TEST 3: input=2.0, scaler=2.0")
print("Expected: sum(2*2) over 32 columns = 128")
print("=" * 80)
input3 = torch.full((32, 32), 2.0)
scaler3 = torch.full((32, 32), 2.0)
out3 = torch.full((32, 32), -999.0)
test_reduce_sum(input3, scaler3, out3)
print(f"Got: {out3[0, 0].item():.1f} (error: {out3[0, 0].item() - 128:.1f})")
print(f"First column unique values: {out3[:, 0].unique()}")

print("\n" + "=" * 80)
print("TEST 4: input=10.0, scaler=1.0")
print("Expected: sum(10*1) over 32 columns = 320")
print("=" * 80)
input4 = torch.full((32, 32), 10.0)
scaler4 = torch.ones((32, 32))
out4 = torch.full((32, 32), -999.0)
test_reduce_sum(input4, scaler4, out4)
print(f"Got: {out4[0, 0].item():.1f} (error: {out4[0, 0].item() - 320:.1f})")
print(f"First column unique values: {out4[:, 0].unique()}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("If error is always +1, it's a fence-post bug")
print("If error scales with input, it's a calculation bug")
print("=" * 80)
