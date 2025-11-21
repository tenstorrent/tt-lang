# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin

# Test: Load pre-normalized P directly, then do P @ V
# This tests if the issue is with the matmul on normalized values

import torch
from ttlang.d2m_api import *

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_prenormalized_matmul(P, V, out):
    P_accessor = TensorAccessor(P)
    V_accessor = TensorAccessor(V)

    @compute()
    async def compute_matmul(P_cb: CircularBuffer, V_cb: CircularBuffer, out_cb: CircularBuffer):
        p = P_cb.wait()
        v = V_cb.wait()
        o = out_cb.reserve()

        # Just do the matmul
        result = p @ v

        o.store(result)
        P_cb.pop()
        V_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(P_cb: CircularBuffer, V_cb: CircularBuffer, out_cb: CircularBuffer):
        p_shard = P_cb.reserve()
        tx = dma(P_accessor[0, 0], p_shard)
        tx.wait()
        P_cb.push()

        v_shard = V_cb.reserve()
        tx = dma(V_accessor[0, 0], v_shard)
        tx.wait()
        V_cb.push()

    return Program(compute_matmul, dm_loader)(P, V, out)


# Use the expected P values from FA (1/32 per element)
P = torch.full((32, 32), 0.03125)
V = torch.full((32, 32), 0.2)
out = torch.zeros(32, 32)

result_expected = 0.03125 * 0.2 * 32

print(f"Test: P @ V with P=0.03125 (softmax values)")
print(f"Expected: {result_expected:.6f}")

test_prenormalized_matmul(P, V, out)

print(f"Hardware: {out[0, 0].item():.6f}, Expected: {result_expected:.6f}")
