# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Tests Flash Attention "second half" - the P @ V matmul after softmax
# Pattern: softmax(S) → P @ V → O
#
# Full FA pipeline:
# 1. S = Q @ K^T                [first half]
# 2. S_scaled = S / sqrt(d_head) [first half]
# 3. exp_S = exp(S_scaled)       [first half]
# 4. sum_exp = reduce_sum(exp_S) [first half]
# 5. P = exp_S / sum_exp         [second half - softmax normalization]
# 6. O = P @ V                   [second half - attention-weighted values]

import torch
from ttlang.d2m_api import *
from ttlang.operators import exp, recip
import math

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_fa_second_half(exp_S, sum_exp, V, out):
    exp_S_accessor = TensorAccessor(exp_S)
    sum_exp_accessor = TensorAccessor(sum_exp)
    V_accessor = TensorAccessor(V)

    @compute()
    async def compute_attention(
        exp_S_cb: CircularBuffer,
        sum_exp_cb: CircularBuffer,
        V_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        exp_s = exp_S_cb.wait()
        sum_val = sum_exp_cb.wait()
        v = V_cb.wait()
        o = out_cb.reserve()

        # Step 5: Normalize: P = exp_S / sum_exp
        # Using recip + multiply instead of division: P = exp_S * (1/sum_exp)
        sum_recip = recip(sum_val)
        o.store(sum_recip)
        out_cb.push()
        sum_recip = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        P = exp_s * sum_recip
        o.store(P)
        out_cb.push()
        P = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 6: P @ V (attention-weighted sum of values)
        O = P @ v

        o.store(O)
        exp_S_cb.pop()
        sum_exp_cb.pop()
        V_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(
        exp_S_cb: CircularBuffer,
        sum_exp_cb: CircularBuffer,
        V_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        # Load exp_S
        exp_s_shard = exp_S_cb.reserve()
        tx = dma(exp_S_accessor[0, 0], exp_s_shard)
        tx.wait()
        exp_S_cb.push()

        # Load sum_exp
        sum_shard = sum_exp_cb.reserve()
        tx = dma(sum_exp_accessor[0, 0], sum_shard)
        tx.wait()
        sum_exp_cb.push()

        # Load V
        v_shard = V_cb.reserve()
        tx = dma(V_accessor[0, 0], v_shard)
        tx.wait()
        V_cb.push()

    return Program(compute_attention, dm_loader)(exp_S, sum_exp, V, out)


# CHECK: func.func @test_fa_second_half
# CHECK: "d2m.tile_recip"
# CHECK: "d2m.tile_mul"
# CHECK: "d2m.tile_matmul"

# Create test inputs
# Simulate the output from first half
Q = torch.ones((32, 32)) * 0.1
K = torch.ones((32, 32)) * 0.1
d_head = 32
scale_factor = 1.0 / math.sqrt(d_head)

# Compute first half on CPU to get exp_S and sum_exp
S = Q @ K.T  # 0.1 * 0.1 * 32 = 0.32 per element
S_scaled = S * scale_factor  # 0.32 * 0.177 ≈ 0.0566
exp_S = torch.exp(S_scaled)  # exp(0.0566) ≈ 1.058 per element
sum_exp = exp_S.sum(dim=1, keepdim=True)  # sum of 32 values per row

# V matrix for attention-weighted sum
V = torch.ones((32, 32)) * 0.2  # Simple uniform values

print("=== BEFORE KERNEL ===")
print(f"Testing FA second half: softmax normalization → P @ V")
print(f"exp_S shape: {exp_S.shape}, sample value: {exp_S[0, 0].item():.6f}")
print(f"sum_exp shape: {sum_exp.shape}, sample value: {sum_exp[0, 0].item():.6f}")
print(f"V shape: {V.shape}, all values: {V[0, 0].item()}")

# Compute expected on CPU
P_expected = exp_S / sum_exp  # Softmax probabilities (should sum to 1 per row)
O_expected = P_expected @ V  # Attention-weighted values

print(f"\nExpected intermediate values:")
print(f"  P (softmax): {P_expected[0, 0].item():.6f} (should be ~1/32 = 0.03125)")
print(f"  P row sum: {P_expected[0, :].sum().item():.6f} (should be ~1.0)")
print(f"  O (P @ V): {O_expected[0, 0].item():.6f}")

out = torch.zeros(32, 32)
test_fa_second_half(exp_S, sum_exp, V, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.6f}")
# CHECK-OUTPUT: out[0, 0] =
print(f"Expected (P @ V): {O_expected[0, 0].item():.6f}")

# Check result
tolerance = 0.1  # 10% tolerance
if abs(out[0, 0].item() - O_expected[0, 0].item()) / O_expected[0, 0].item() < tolerance:
    print(f"PASS: FA second half produced correct result")
    # CHECK-OUTPUT: PASS: FA second half produced correct result
else:
    print(f"FAIL: Expected {O_expected[0, 0].item():.6f}, got {out[0, 0].item():.6f}")
