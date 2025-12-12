# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TODO: update to use pytest (see issue #91)
# UNSUPPORTED: true
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Full Flash Attention in a single kernel
# Pipeline: Q @ K^T → scale → exp → reduce_sum → normalize → P @ V

import torch
from ttlang.d2m_api import *
from ttlang.operators import exp, reduce_sum, recip, bcast
import math


@pykernel_gen(
    grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
)
def flash_attention(Q, K, V, scale, ones, out):
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

        # Step 1: Q @ K^T (attention scores)
        S = q @ k
        o.store(S)
        out_cb.push()
        S = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 2: Scale by 1/sqrt(d_head)
        S_scaled = S * scale_val
        o.store(S_scaled)
        out_cb.push()
        S_scaled = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 3: exp(S_scaled)
        exp_S = exp(S_scaled)
        o.store(exp_S)
        out_cb.push()
        exp_S = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 4: reduce_sum for normalization denominator (only fills column 0)
        sum_exp = reduce_sum(exp_S, ones_val, dim=1)
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

        # Step 5: Reciprocal for division (use broadcasted sum)
        sum_recip = recip(sum_exp_bcast)
        o.store(sum_recip)
        out_cb.push()
        sum_recip = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 6: Normalize (softmax): P = exp_S * sum_recip
        # Recompute exp_S (we can't reuse the earlier value due to SSA constraints)
        exp_S_for_mul = exp(S_scaled)
        P = exp_S_for_mul * sum_recip
        o.store(P)
        out_cb.push()
        P = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 7: P @ V (attention-weighted values)
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

        # Load V
        v_shard = V_cb.reserve()
        tx = dma(V_accessor[0, 0], v_shard)
        tx.wait()
        V_cb.push()

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

    return Program(compute_attention, dm_loader)(Q, K, V, scale, ones, out)


# CHECK: func.func @flash_attention
# CHECK: "d2m.tile_matmul"
# CHECK: "d2m.tile_mul"
# CHECK: "d2m.tile_exp"
# CHECK: "d2m.tile_reduce_sum"
# CHECK: "d2m.tile_bcast"
# CHECK: "d2m.tile_recip"
# CHECK: "d2m.tile_mul"
# CHECK: "d2m.tile_matmul"

# Create test inputs
Q = torch.ones((32, 32)) * 0.1
K = torch.ones((32, 32)) * 0.1
V = torch.ones((32, 32)) * 0.2
d_head = 32
scale = torch.full((32, 32), 1.0 / math.sqrt(d_head))
ones = torch.ones((32, 32))
out = torch.zeros(32, 32)

print("=== BEFORE KERNEL ===")
print(f"Testing Full Flash Attention")
print(f"Q shape: {Q.shape}, values: {Q[0, 0].item()}")
print(f"K shape: {K.shape}, values: {K[0, 0].item()}")
print(f"V shape: {V.shape}, values: {V[0, 0].item()}")
print(f"Scale: 1/sqrt({d_head}) = {scale[0, 0].item():.6f}")

# Compute expected - Manual calculation
S = Q @ K.T
S_scaled = S * scale[0, 0].item()
exp_S = torch.exp(S_scaled)
sum_exp = exp_S.sum(dim=1, keepdim=True)
P = exp_S / sum_exp
O_manual = P @ V

print(f"\nManual calculation (matching our ops):")
print(f"  S (Q @ K^T): {S[0, 0].item():.6f}")
print(f"  S_scaled: {S_scaled[0, 0].item():.6f}")
print(f"  exp(S_scaled): {exp_S[0, 0].item():.6f}")
print(f"  sum(exp): {sum_exp[0, 0].item():.6f}")
print(f"  P (softmax): {P[0, 0].item():.6f}")
print(f"  O (P @ V): {O_manual[0, 0].item():.6f}")

# Compute expected - PyTorch's scaled_dot_product_attention
# Need to add batch and head dimensions for torch API: (batch, heads, seq_len, dim)
Q_torch = Q.unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 32)
K_torch = K.unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 32)
V_torch = V.unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 32)

# PyTorch FA with same scale factor
with torch.no_grad():
    O_torch_fa = torch.nn.functional.scaled_dot_product_attention(
        Q_torch, K_torch, V_torch, dropout_p=0.0, scale=scale[0, 0].item()
    )
O_torch_fa = O_torch_fa.squeeze()  # Remove batch and head dims

print(f"\nPyTorch scaled_dot_product_attention:")
print(f"  O (reference): {O_torch_fa[0, 0].item():.6f}")

print(f"\nComparison:")
print(f"  Manual calc: {O_manual[0, 0].item():.6f}")
print(f"  PyTorch FA:  {O_torch_fa[0, 0].item():.6f}")
print(f"  Match: {torch.allclose(O_manual, O_torch_fa, rtol=1e-5, atol=1e-5)}")

# Use manual calculation as primary reference (matches our ops exactly)
O_expected = O_manual

flash_attention(Q, K, V, scale, ones, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"Hardware result:   {out[0, 0].item():.6f}")
# CHECK-OUTPUT: Hardware result:
print(f"Manual expected:   {O_manual[0, 0].item():.6f}")
print(f"PyTorch FA:        {O_torch_fa[0, 0].item():.6f}")

# Check against manual calculation (matches our ops)
tolerance = 0.2
error_manual = abs(out[0, 0].item() - O_manual[0, 0].item()) / O_manual[0, 0].item()
error_torch = abs(out[0, 0].item() - O_torch_fa[0, 0].item()) / O_torch_fa[0, 0].item()

print(f"\nError vs manual calc: {error_manual*100:.1f}%")
print(f"Error vs PyTorch FA:  {error_torch*100:.1f}%")

if error_manual < tolerance:
    print(f"PASS: Full Flash Attention produced correct result!")
    # CHECK-OUTPUT: PASS: Full Flash Attention produced correct result
else:
    ratio = out[0, 0].item() / O_manual[0, 0].item()
    print(
        f"FAIL: Expected {O_manual[0, 0].item():.6f}, got {out[0, 0].item():.6f} (ratio: {ratio:.3f})"
    )
