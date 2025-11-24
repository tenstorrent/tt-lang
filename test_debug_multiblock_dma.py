# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Debug test: Load 2x2 tiles and write back immediately (no compute)
# This tests if the DMA loading is working correctly for multiblock

import torch
from ttlang.d2m_api import *

@pykernel_gen(grid=(1, 1), block_factors=[(2, 2), (2, 2)])
def test_dma_passthrough(lhs, out):
    lhs_accessor = TensorAccessor(lhs)

    @compute()
    async def passthrough_compute(lhs_cb: CircularBuffer, out_cb: CircularBuffer):
        l = lhs_cb.wait()
        o = out_cb.reserve()
        o.store(l)  # Just copy input to output
        lhs_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(lhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()
        lhs_cb.push()

    return Program(passthrough_compute, dm_loader)(lhs, out)


# Test: Load all 2.0s and write back
lhs = torch.full((64, 64), 2.0)
out = torch.full((64, 64), -999.0)

print("=== BEFORE KERNEL ===")
print(f"Test: multiblock passthrough (2x2 tiles, just DMA + copy)")
print(f"lhs: all 2.0")
print(f"Expected: all 2.0 in output")

test_dma_passthrough(lhs, out)

print("\n=== AFTER KERNEL ===")
print(f"Hardware: out[0,0]={out[0, 0].item():.1f}, out[0,63]={out[0, 63].item():.1f}")
print(f"Hardware: out[63,0]={out[63, 0].item():.1f}, out[63,63]={out[63, 63].item():.1f}")

expected = 2.0
all_correct = torch.allclose(out, torch.full((64, 64), expected), rtol=0.01)

if all_correct:
    print(f"PASS: All values are {expected:.1f}")
else:
    print(f"FAIL: Not all values are {expected:.1f}")
    print(f"  Min: {out.min().item():.1f}, Max: {out.max().item():.1f}")
    # Show first few values
    print(f"  out[0:2, 0:2] = {out[0:2, 0:2]}")
