# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin

# Test: reduce_sum → matmul (skip recip/mul, test if matmul works after reduce_sum)

import torch
from ttlang.d2m_api import *
from ttlang.operators import reduce_sum

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_reduce_matmul(A, ones, C, out):
    A_accessor = TensorAccessor(A)
    ones_accessor = TensorAccessor(ones)
    C_accessor = TensorAccessor(C)

    @compute()
    async def compute_chain(A_cb: CircularBuffer, ones_cb: CircularBuffer, C_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()
        ones_val = ones_cb.wait()
        c = C_cb.wait()
        o = out_cb.reserve()

        # Op 1: reduce_sum
        sum_result = reduce_sum(a, ones_val, dim=1)
        o.store(sum_result)
        out_cb.push()
        sum_result = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 2: matmul with the sum result
        result = sum_result @ c

        o.store(result)
        A_cb.pop()
        ones_cb.pop()
        C_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(A_cb: CircularBuffer, ones_cb: CircularBuffer, C_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = A_cb.reserve()
        tx = dma(A_accessor[0, 0], a_shard)
        tx.wait()
        A_cb.push()

        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

        c_shard = C_cb.reserve()
        tx = dma(C_accessor[0, 0], c_shard)
        tx.wait()
        C_cb.push()

    return Program(compute_chain, dm_loader)(A, ones, C, out)


# Test: sum(2.0 * 32) @ 3.0 = 64 @ 3 = 64 * 3 * 32 = 6144
A = torch.full((32, 32), 2.0)
ones = torch.ones((32, 32))
C = torch.full((32, 32), 3.0)
out = torch.zeros(32, 32)

print("Test: reduce_sum → matmul")
print(f"sum(2.0 * 32) @ 3.0 = 64 @ 3 = {64.0 * 3.0 * 32}")

test_reduce_matmul(A, ones, C, out)

print(f"Hardware: {out[0, 0].item():.1f}, Expected: {64.0 * 3.0 * 32:.1f}")
