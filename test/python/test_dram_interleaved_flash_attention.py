# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: Flash Attention with DRAM interleaved tensors.
# DMAs read directly from DRAM into CBs without L1 intermediate.

import torch
from ttlang.ttl_api import *
from ttlang.operators import exp, reduce_sum, recip, bcast
import math

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== DRAM Interleaved Flash Attention Test Complete ===")
    exit(0)


@pykernel_gen(
    grid=(1, 1),
    block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
    ttnn_interop=True,
)
def flash_attention_dram(Q, K, V, scale, ones, out):
    """Full Flash Attention reading directly from DRAM interleaved tensors."""
    Q_accessor = TensorAccessor(Q)
    K_accessor = TensorAccessor(K)
    V_accessor = TensorAccessor(V)
    scale_accessor = TensorAccessor(scale)
    ones_accessor = TensorAccessor(ones)
    out_accessor = TensorAccessor(out)

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
        out_tile = out_cb.reserve()

        # Step 1: Q @ K^T (attention scores)
        S = q @ k
        out_tile.store(S)
        out_cb.push()
        S = out_cb.wait()
        out_cb.pop()
        out_tile = out_cb.reserve()

        # Step 2: Scale by 1/sqrt(d_head)
        S_scaled = S * scale_val
        out_tile.store(S_scaled)
        out_cb.push()
        S_scaled = out_cb.wait()
        out_cb.pop()
        out_tile = out_cb.reserve()

        # Step 3: exp(S_scaled)
        exp_S = exp(S_scaled)
        out_tile.store(exp_S)
        out_cb.push()
        exp_S = out_cb.wait()
        out_cb.pop()
        out_tile = out_cb.reserve()

        # Step 4: reduce_sum for normalization denominator (only fills column 0)
        sum_exp = reduce_sum(exp_S, ones_val, dim=1)
        out_tile.store(sum_exp)
        out_cb.push()
        sum_exp = out_cb.wait()
        out_cb.pop()
        out_tile = out_cb.reserve()

        # Step 4b: bcast to fill all columns
        sum_exp_bcast = bcast(sum_exp, dim=1)
        out_tile.store(sum_exp_bcast)
        out_cb.push()
        sum_exp_bcast = out_cb.wait()
        out_cb.pop()
        out_tile = out_cb.reserve()

        # Step 5: Reciprocal for division (use broadcasted sum)
        sum_recip = recip(sum_exp_bcast)
        out_tile.store(sum_recip)
        out_cb.push()
        sum_recip = out_cb.wait()
        out_cb.pop()
        out_tile = out_cb.reserve()

        # Step 6: Normalize (softmax): P = exp_S * sum_recip
        exp_S_for_mul = exp(S_scaled)
        P = exp_S_for_mul * sum_recip
        out_tile.store(P)
        out_cb.push()
        P = out_cb.wait()
        out_cb.pop()
        out_tile = out_cb.reserve()

        # Step 7: P @ V (attention-weighted values)
        result = P @ v

        out_tile.store(result)
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
        # Load Q from DRAM interleaved
        q_shard = Q_cb.reserve()
        tx = dma(Q_accessor[0, 0], q_shard)
        tx.wait()
        Q_cb.push()

        # Load K from DRAM interleaved
        k_shard = K_cb.reserve()
        tx = dma(K_accessor[0, 0], k_shard)
        tx.wait()
        K_cb.push()

        # Load V from DRAM interleaved
        v_shard = V_cb.reserve()
        tx = dma(V_accessor[0, 0], v_shard)
        tx.wait()
        V_cb.push()

        # Load scale from DRAM interleaved
        scale_shard = scale_cb.reserve()
        tx = dma(scale_accessor[0, 0], scale_shard)
        tx.wait()
        scale_cb.push()

        # Load ones from DRAM interleaved
        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

    @datamovement()
    async def dm_writer(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        V_cb: CircularBuffer,
        scale_cb: CircularBuffer,
        ones_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        out_shard = out_cb.wait()
        tx = dma(out_shard, out_accessor[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(compute_attention, dm_loader, dm_writer)(Q, K, V, scale, ones, out)


# CHECK: Testing DRAM Interleaved Flash Attention
print("=== Testing DRAM Interleaved Flash Attention ===")

device = ttnn.open_device(device_id=0)

try:
    # Create test inputs
    Q_torch = torch.ones((32, 32), dtype=torch.bfloat16) * 0.1
    K_torch = torch.ones((32, 32), dtype=torch.bfloat16) * 0.1
    V_torch = torch.ones((32, 32), dtype=torch.bfloat16) * 0.2
    d_head = 32
    scale_torch = torch.full((32, 32), 1.0 / math.sqrt(d_head), dtype=torch.bfloat16)
    ones_torch = torch.ones((32, 32), dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    # Create DRAM interleaved tensors - NO L1 intermediate!
    Q = ttnn.from_torch(
        Q_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    K = ttnn.from_torch(
        K_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    V = ttnn.from_torch(
        V_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scale = ttnn.from_torch(
        scale_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ones = ttnn.from_torch(
        ones_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"Q memory_config: {Q.memory_config()}")
    print(f"K memory_config: {K.memory_config()}")
    print(f"V memory_config: {V.memory_config()}")
    # CHECK: DRAM

    print(f"\nQ: {Q.shape}, K: {K.shape}, V: {V.shape}")
    print(f"Scale: 1/sqrt({d_head}) = {scale_torch[0, 0].item():.6f}")

    # Compute expected - Manual calculation
    S = Q_torch.float() @ K_torch.float().T
    S_scaled = S * scale_torch[0, 0].item()
    exp_S = torch.exp(S_scaled)
    sum_exp = exp_S.sum(dim=1, keepdim=True)
    P = exp_S / sum_exp
    O_expected = P @ V_torch.float()

    print(f"\nExpected output[0,0]: {O_expected[0, 0].item():.6f}")

    print("\n=== Running Flash Attention with DRAM tensors directly ===")
    flash_attention_dram(Q, K, V, scale, ones, out)

    result = ttnn.to_torch(out)

    print("\n=== Results ===")
    print(f"Hardware result:   {result[0, 0].item():.6f}")
    print(f"Expected:          {O_expected[0, 0].item():.6f}")

    tolerance = 0.2
    error = abs(result[0, 0].item() - O_expected[0, 0].item()) / O_expected[0, 0].item()
    print(f"Error: {error*100:.1f}%")

    if error < tolerance:
        print("\nPASS: DRAM interleaved Flash Attention works!")
        # CHECK: PASS: DRAM interleaved Flash Attention
    else:
        print(
            f"\nFAIL: Expected {O_expected[0, 0].item():.6f}, got {result[0, 0].item():.6f}"
        )

finally:
    ttnn.close_device(device)

print("\n=== DRAM Interleaved Flash Attention Test Complete ===")
# CHECK: Test Complete
