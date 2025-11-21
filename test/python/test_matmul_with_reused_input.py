# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin

# Test: a @ b, then reuse 'a' in matmul again: result @ a
# Tests if reusing input in matmul (not reduce_sum) works

import torch
from ttlang.d2m_api import *

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_matmul_reuse(A, B, out):
    A_accessor = TensorAccessor(A)
    B_accessor = TensorAccessor(B)

    @compute()
    async def compute_chain(A_cb: CircularBuffer, B_cb: CircularBuffer, out_cb: CircularBuffer):
        a = A_cb.wait()
        b = B_cb.wait()
        o = out_cb.reserve()

        # Op 1: matmul
        temp = a @ b
        o.store(temp)
        out_cb.push()
        temp = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Op 2: matmul again, reusing 'a'
        result = temp @ a

        o.store(result)
        A_cb.pop()
        B_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(A_cb: CircularBuffer, B_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = A_cb.reserve()
        tx = dma(A_accessor[0, 0], a_shard)
        tx.wait()
        A_cb.push()

        b_shard = B_cb.reserve()
        tx = dma(B_accessor[0, 0], b_shard)
        tx.wait()
        B_cb.push()

    return Program(compute_chain, dm_loader)(A, B, out)


# Test: (2 @ 3) @ 2 = (2 * 3 * 32) @ 2 = 192 @ 2 = 192 * 2 * 32 = 12288
A = torch.full((32, 32), 2.0)
B = torch.full((32, 32), 3.0)
out = torch.zeros(32, 32)

temp_expected = (A @ B)[0, 0].item()
result_expected = (temp_expected * 2.0 * 32)

print(f"Test: (a @ b) @ a with a reused")
print(f"Expected: {result_expected:.1f}")

test_matmul_reuse(A, B, out)

print(f"Hardware: {out[0, 0].item():.1f}, Expected: {result_expected:.1f}")
