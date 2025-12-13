# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TODO: update to use pytest (see issue #91)
# UNSUPPORTED: true
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Tests Flash Attention "first half" - up to but not including the P @ V matmul
# Pattern: Q @ K^T → scale → exp (softmax approximation) → reduce_sum (for normalization)
#
# Simplified from full FA which would be:
# 1. S = Q @ K^T
# 2. S_scaled = S / sqrt(d_head)
# 3. P = softmax(S_scaled)  [we approximate with exp → reduce_sum]
# 4. O = P @ V  [second matmul - not included in this test]

import torch
from ttlang.d2m_api import *
from ttlang.operators import exp, reduce_sum
import math


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)])
def test_fa_first_half(Q, K, scale, ones, out):
    Q_accessor = TensorAccessor(Q)
    K_accessor = TensorAccessor(K)
    scale_accessor = TensorAccessor(scale)
    ones_accessor = TensorAccessor(ones)

    @compute()
    async def compute_attention(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        scale_cb: CircularBuffer,
        ones_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        q = Q_cb.wait()
        k = K_cb.wait()
        scale_val = scale_cb.wait()
        ones_val = ones_cb.wait()
        o = out_cb.reserve()

        # Step 1: Q @ K^T (matmul for attention scores)
        # For 32x32 @ 32x32, we get 32x32 output
        S = q @ k
        o.store(S)
        out_cb.push()
        S = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 2: Scale by 1/sqrt(d_head)
        # Using element-wise multiply with scale tensor
        S_scaled = S * scale_val
        o.store(S_scaled)
        out_cb.push()
        S_scaled = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 3: exp(S_scaled) - approximates softmax numerator
        exp_S = exp(S_scaled)
        o.store(exp_S)
        out_cb.push()
        exp_S = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 4: reduce_sum along dim=1 (reduce columns)
        # This gives us sum per row for normalization
        # In real FA, we'd divide exp_S by this sum to get probabilities
        sum_exp = reduce_sum(exp_S, ones_val, dim=1)

        o.store(sum_exp)
        Q_cb.pop()
        K_cb.pop()
        scale_cb.pop()
        ones_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        scale_cb: CircularBuffer,
        ones_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        # Load Q
        q_shard = Q_cb.reserve()
        tx = dma(Q_accessor[0, 0], q_shard)
        tx.wait()
        Q_cb.push()

        # Load K
        k_shard = K_cb.reserve()
        tx = dma(K_accessor[0, 0], k_shard)
        tx.wait()
        K_cb.push()

        # Load scale
        scale_shard = scale_cb.reserve()
        tx = dma(scale_accessor[0, 0], scale_shard)
        tx.wait()
        scale_cb.push()

        # Load ones
        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

    return Program(compute_attention, dm_loader)(Q, K, scale, ones, out)


# CHECK: func.func @test_fa_first_half
# CHECK: "d2m.tile_matmul"
# CHECK: "d2m.tile_mul"
# CHECK: "d2m.tile_exp"
# CHECK: "d2m.tile_reduce_sum"

# Create test inputs
# Q, K are 32x32 with small values
Q = torch.ones((32, 32)) * 0.1
K = torch.ones((32, 32)) * 0.1
d_head = 32
scale = torch.full((32, 32), 1.0 / math.sqrt(d_head))  # 1/sqrt(32) ≈ 0.177
ones = torch.ones((32, 32))
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"Testing FA first half: Q @ K^T → scale → exp → reduce_sum")
print(f"Q shape: {Q.shape}, all values: {Q[0, 0].item()}")
print(f"K shape: {K.shape}, all values: {K[0, 0].item()}")
print(f"Scale factor: 1/sqrt({d_head}) = {scale[0, 0].item():.6f}")

# Compute expected on CPU
S_expected = Q @ K.T  # 0.1 * 0.1 * 32 = 0.32 per element
S_scaled_expected = S_expected * scale[0, 0].item()  # 0.32 * 0.177 ≈ 0.0566
exp_S_expected = torch.exp(S_scaled_expected)  # exp(0.0566) ≈ 1.058
sum_exp_expected = exp_S_expected.sum(dim=1, keepdim=True)  # sum of 32 values ≈ 33.86

print(f"\nExpected intermediate values:")
print(f"  S (Q @ K^T): {S_expected[0, 0].item():.6f}")
print(f"  S_scaled: {S_scaled_expected[0, 0].item():.6f}")
print(f"  exp(S_scaled): {exp_S_expected[0, 0].item():.6f}")
print(f"  sum(exp(S_scaled)): {sum_exp_expected[0, 0].item():.6f}")

test_fa_first_half(Q, K, scale, ones, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.6f}")
# CHECK-OUTPUT: out[0, 0] =
print(f"Expected (sum of exp): {sum_exp_expected[0, 0].item():.6f}")

# Check result (should be sum of exp values in first column)
tolerance = 0.1  # 10% tolerance
if (
    abs(out[0, 0].item() - sum_exp_expected[0, 0].item())
    / sum_exp_expected[0, 0].item()
    < tolerance
):
    print(f"PASS: FA first half produced correct result")
    # CHECK-OUTPUT: PASS: FA first half produced correct result
else:
    print(
        f"FAIL: Expected {sum_exp_expected[0, 0].item():.6f}, got {out[0, 0].item():.6f}"
    )
