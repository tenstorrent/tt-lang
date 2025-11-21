# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Flash Attention split into two kernels with intermediate printing for debugging
# Kernel 1: Q @ K^T → scale → exp → reduce_sum
# Kernel 2: recip → mul → P @ V

import torch
from ttlang.d2m_api import *
from ttlang.operators import exp, reduce_sum, recip, bcast
import math

# Kernel 1: Q @ K^T → scale → exp (outputs exp_S)
@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def fa_first_half(Q, K, scale, out):
    Q_accessor = TensorAccessor(Q)
    K_accessor = TensorAccessor(K)
    scale_accessor = TensorAccessor(scale)

    @compute()
    async def compute_first(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        scale_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        q = Q_cb.wait()
        k = K_cb.wait()
        scale_val = scale_cb.wait()
        o = out_cb.reserve()

        # Step 1: Q @ K^T
        S = q @ k
        o.store(S)
        out_cb.push()
        S = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 2: Scale
        S_scaled = S * scale_val
        o.store(S_scaled)
        out_cb.push()
        S_scaled = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 3: exp (output this)
        exp_S = exp(S_scaled)

        o.store(exp_S)
        Q_cb.pop()
        K_cb.pop()
        scale_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_first(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        scale_cb: CircularBuffer,
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

        scale_shard = scale_cb.reserve()
        tx = dma(scale_accessor[0, 0], scale_shard)
        tx.wait()
        scale_cb.push()

    return Program(compute_first, dm_first)(Q, K, scale, out)


# Kernel 2: reduce_sum → recip → mul → P @ V (takes exp_S, uses same hardware values)
@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def fa_second_half(exp_S, ones, V, out):
    exp_S_accessor = TensorAccessor(exp_S)
    ones_accessor = TensorAccessor(ones)
    V_accessor = TensorAccessor(V)

    @compute()
    async def compute_second(
        exp_S_cb: CircularBuffer,
        ones_cb: CircularBuffer,
        V_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        exp_s = exp_S_cb.wait()
        ones_val = ones_cb.wait()
        v = V_cb.wait()
        o = out_cb.reserve()

        # Step 4: reduce_sum (recompute from hardware exp_S, only fills column 0)
        sum_exp = reduce_sum(exp_s, ones_val, dim=1)
        o.store(sum_exp)
        out_cb.push()
        sum_exp = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 4b: bcast to fill all columns
        sum_exp_bcast = bcast(sum_exp, dim=1)
        o.store(sum_exp_bcast)
        out_cb.push()
        sum_exp_bcast = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 5: Reciprocal (use broadcasted sum)
        sum_recip = recip(sum_exp_bcast)
        o.store(sum_recip)
        out_cb.push()
        sum_recip = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 6: Normalize (now exp_s and sum_exp are consistent!)
        P = exp_s * sum_recip
        o.store(P)
        out_cb.push()
        P = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 7: P @ V
        O = P @ v

        o.store(O)
        exp_S_cb.pop()
        ones_cb.pop()
        V_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_second(
        exp_S_cb: CircularBuffer,
        ones_cb: CircularBuffer,
        V_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        exp_s_shard = exp_S_cb.reserve()
        tx = dma(exp_S_accessor[0, 0], exp_s_shard)
        tx.wait()
        exp_S_cb.push()

        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

        v_shard = V_cb.reserve()
        tx = dma(V_accessor[0, 0], v_shard)
        tx.wait()
        V_cb.push()

    return Program(compute_second, dm_second)(exp_S, ones, V, out)


# CHECK: func.func @fa_first_half
# CHECK: func.func @fa_second_half

# Create inputs
Q = torch.ones((32, 32)) * 0.1
K = torch.ones((32, 32)) * 0.1
V = torch.ones((32, 32)) * 0.2
d_head = 32
scale = torch.full((32, 32), 1.0 / math.sqrt(d_head))
ones = torch.ones((32, 32))

print("=== INPUTS ===")
print(f"Q: all {Q[0, 0].item()}")
print(f"K: all {K[0, 0].item()}")
print(f"V: all {V[0, 0].item()}")
print(f"Scale: {scale[0, 0].item():.6f}")

# Manual calculation for reference
S_ref = Q @ K.T
print(f"\nManual: S (Q @ K^T) = {S_ref[0, 0].item():.6f}")

S_scaled_ref = S_ref * scale[0, 0].item()
print(f"Manual: S_scaled = {S_scaled_ref[0, 0].item():.6f}")

exp_S_ref = torch.exp(S_scaled_ref)
print(f"Manual: exp(S_scaled) = {exp_S_ref[0, 0].item():.6f}")

sum_exp_ref = exp_S_ref.sum(dim=1, keepdim=True)
print(f"Manual: sum(exp) = {sum_exp_ref[0, 0].item():.6f}")

P_ref = exp_S_ref / sum_exp_ref
print(f"Manual: P (softmax) = {P_ref[0, 0].item():.6f}")

O_ref = P_ref @ V
print(f"Manual: O (P @ V) = {O_ref[0, 0].item():.6f}")

# Run kernel 1 (outputs exp_S)
print("\n=== KERNEL 1: Q @ K^T → scale → exp ===")
exp_S_hw = torch.zeros(32, 32)
fa_first_half(Q, K, scale, exp_S_hw)

print(f"Hardware: exp_S[0, 0] = {exp_S_hw[0, 0].item():.6f}")
print(f"Manual:   exp_S[0, 0] = {exp_S_ref[0, 0].item():.6f}")
error_exp = abs(exp_S_hw[0, 0].item() - exp_S_ref[0, 0].item()) / exp_S_ref[0, 0].item()
print(f"Error: {error_exp*100:.1f}%")

# Run kernel 2 (takes hardware exp_S, recomputes sum internally)
print("\n=== KERNEL 2: reduce_sum → recip → mul → P @ V ===")
out = torch.zeros(32, 32)
fa_second_half(exp_S_hw, ones, V, out)

print(f"Hardware: O[0, 0] = {out[0, 0].item():.6f}")
print(f"Manual:   O[0, 0] = {O_ref[0, 0].item():.6f}")

error = abs(out[0, 0].item() - O_ref[0, 0].item()) / O_ref[0, 0].item()
print(f"\nError: {error*100:.1f}%")

tolerance = 0.2
if error < tolerance:
    print(f"PASS: Two-kernel FA produced correct result!")
    # CHECK-OUTPUT: PASS: Two-kernel FA produced correct result
else:
    print(f"FAIL: Expected {O_ref[0, 0].item():.6f}, got {out[0, 0].item():.6f}")
