# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin

# Test: Just reduce_sum, output it directly (no CB cycling)
# Check if all elements in a row are the same

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_reduce_output(A, ones, out):
    A_accessor = TensorAccessor(A)
    ones_accessor = TensorAccessor(ones)

    @compute()
    async def compute_reduce(A_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()
        ones_val = ones_cb.wait()
        o = out_cb.reserve()

        # Just reduce_sum, output directly
        result = reduce_sum(a, ones_val, dim=1)

        o.store(result)
        A_cb.pop()
        ones_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(A_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = A_cb.reserve()
        tx = dma(A_accessor[0, 0], a_shard)
        tx.wait()
        A_cb.push()

        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

    return Program(compute_reduce, dm_loader)(A, ones, out)


A = torch.full((32, 32), 2.0)
ones = torch.ones((32, 32))
out = torch.zeros(32, 32)

print("Test: reduce_sum output inspection")
print(f"Input: all 2.0")
print(f"Expected: all rows have value 64.0 (2.0 * 32)")

test_reduce_output(A, ones, out)

print(f"\nHardware output:")
print(f"  out[0, 0] = {out[0, 0].item():.1f}")
print(f"  out[0, 31] = {out[0, 31].item():.1f} (should be same as [0,0])")
print(f"  out[1, 0] = {out[1, 0].item():.1f} (should be same as [0,0])")
print(f"  Row 0 all same? {torch.allclose(out[0, :], out[0, 0])}")
print(f"  All rows same? {torch.allclose(out, out[0, 0])}")
print(f"  Min/Max: {out.min().item():.1f} / {out.max().item():.1f}")
