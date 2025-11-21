# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Full Flash Attention with larger block factors (multiple tiles per core)
# Grid: 1x1, Tensor: 64x64 (2x2 tiles), Block factors: 2x2 (all tiles on one core)

import torch
from ttlang.d2m_api import *
from ttlang.operators import exp, reduce_sum, recip, bcast
import math

@pykernel_gen(grid=(1, 1), block_factors=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
def flash_attention_multiblock(Q, K, V, scale, ones, out):
    Q_accessor = TensorAccessor(Q)
    K_accessor = TensorAccessor(K)
    V_accessor = TensorAccessor(V)
    scale_accessor = TensorAccessor(scale)
    ones_accessor = TensorAccessor(ones)

    @compute()
    async def compute_attention(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        V_cb: CircularBuffer,
        scale_cb: CircularBuffer,
        ones_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        q = Q_cb.wait()
        k = K_cb.wait()
        v = V_cb.wait()
        scale_val = scale_cb.wait()
        ones_val = ones_cb.wait()
        o = out_cb.reserve()

        S = q @ k
        o.store(S)
        out_cb.push()
        S = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        S_scaled = S * scale_val
        o.store(S_scaled)
        out_cb.push()
        S_scaled = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        exp_S = exp(S_scaled)
        o.store(exp_S)
        out_cb.push()
        exp_S = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        sum_exp = reduce_sum(exp_S, ones_val, dim=1)
        o.store(sum_exp)
        out_cb.push()
        sum_exp = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        sum_exp_bcast = bcast(sum_exp, dim=1)
        o.store(sum_exp_bcast)
        out_cb.push()
        sum_exp_bcast = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        sum_recip = recip(sum_exp_bcast)
        o.store(sum_recip)
        out_cb.push()
        sum_recip = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        exp_S_for_mul = exp(S_scaled)
        P = exp_S_for_mul * sum_recip
        o.store(P)
        out_cb.push()
        P = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        O = P @ v

        o.store(O)
        Q_cb.pop()
        K_cb.pop()
        V_cb.pop()
        scale_cb.pop()
        ones_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        V_cb: CircularBuffer,
        scale_cb: CircularBuffer,
        ones_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        q_shard = Q_cb.reserve()
        tx = dma(Q_accessor[0, 0], q_shard)
        tx.wait()
        Q_cb.push()

        k_shard = K_cb.reserve()
        tx = dma(K_accessor[0, 0], k_shard)
        tx.wait()
        K_cb.push()

        v_shard = V_cb.reserve()
        tx = dma(V_accessor[0, 0], v_shard)
        tx.wait()
        V_cb.push()

        scale_shard = scale_cb.reserve()
        tx = dma(scale_accessor[0, 0], scale_shard)
        tx.wait()
        scale_cb.push()

        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

    return Program(compute_attention, dm_loader)(Q, K, V, scale, ones, out)


# CHECK: func.func @flash_attention_multiblock
# CHECK: "d2m.tile_matmul"
# CHECK: "d2m.tile_reduce_sum"
# CHECK: "d2m.tile_bcast"

# Test with 64x64 tensors, 2x2 tiles all on one core
Q = torch.ones((64, 64)) * 0.1
K = torch.ones((64, 64)) * 0.1
V = torch.ones((64, 64)) * 0.2
d_head = 64
scale = torch.full((64, 64), 1.0 / math.sqrt(d_head))
ones = torch.ones((64, 64))
out = torch.zeros(64, 64)

print("=== BEFORE KERNEL (Multi-block) ===")
print(f"Grid: 1x1 (single core)")
print(f"Tensor size: 64x64 (2x2 tiles)")
print(f"Block factors: 2x2 (all tiles on one core)")

S = Q @ K.T
S_scaled = S * scale[0, 0].item()
exp_S = torch.exp(S_scaled)
sum_exp = exp_S.sum(dim=1, keepdim=True)
P = exp_S / sum_exp
O_expected = P @ V

print(f"Expected O[0, 0]: {O_expected[0, 0].item():.6f}")

flash_attention_multiblock(Q, K, V, scale, ones, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===

print(f"Hardware: {out[0, 0].item():.6f}")
print(f"Expected: {O_expected[0, 0].item():.6f}")
# CHECK-OUTPUT: Hardware:

tolerance = 0.2
error = abs(out[0, 0].item() - O_expected[0, 0].item()) / O_expected[0, 0].item()

if error < tolerance:
    print(f"PASS: Multi-block FA produced correct result!")
    # CHECK-OUTPUT: PASS: Multi-block FA produced correct result
else:
    print(f"FAIL: Error {error*100:.1f}%")
