# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: Large Flash Attention with multiple heads from DRAM interleaved.
# Processes multiple independent attention heads, each reading directly from DRAM.

import torch
from ttlang.d2m_api import *
from ttlang.operators import exp, reduce_sum, recip, bcast
import math

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== DRAM Interleaved Large Flash Attention Test Complete ===")
    exit(0)


@pykernel_gen(
    grid=(1, 1),
    block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
    ttnn_interop=True,
)
def flash_attention_head_dram(Q, K, V, scale, ones, out):
    """Flash Attention for a single head, reading directly from DRAM interleaved."""
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

        # Step 3: exp
        exp_S = exp(S_scaled)
        o.store(exp_S)
        out_cb.push()
        exp_S = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 4: reduce_sum
        sum_exp = reduce_sum(exp_S, ones_val, dim=1)
        o.store(sum_exp)
        out_cb.push()
        sum_exp = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 4b: bcast
        sum_exp_bcast = bcast(sum_exp, dim=1)
        o.store(sum_exp_bcast)
        out_cb.push()
        sum_exp_bcast = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 5: recip
        sum_recip = recip(sum_exp_bcast)
        o.store(sum_recip)
        out_cb.push()
        sum_recip = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 6: softmax
        exp_S_for_mul = exp(S_scaled)
        P = exp_S_for_mul * sum_recip
        o.store(P)
        out_cb.push()
        P = out_cb.wait()
        out_cb.pop()
        o = out_cb.reserve()

        # Step 7: P @ V
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


def compute_expected_attention(Q, K, V, scale_val):
    """Compute expected flash attention output."""
    S = Q.float() @ K.float().T
    S_scaled = S * scale_val
    exp_S = torch.exp(S_scaled)
    sum_exp = exp_S.sum(dim=1, keepdim=True)
    P = exp_S / sum_exp
    return P @ V.float()


# CHECK: === DRAM Interleaved Large Flash Attention Test ===
print("=== DRAM Interleaved Large Flash Attention Test ===")

device = ttnn.open_device(device_id=0)

try:
    # Multi-head attention configuration
    NUM_HEADS = 16
    HEAD_DIM = 32
    SEQ_LEN = 32
    d_head = HEAD_DIM
    scale_val = 1.0 / math.sqrt(d_head)

    print(f"Configuration: {NUM_HEADS} attention heads")
    print(f"Each head: Q,K,V are {SEQ_LEN}x{HEAD_DIM}")
    print(f"Scale: 1/sqrt({d_head}) = {scale_val:.6f}")

    # Create Q, K, V tensors for all heads
    # Shape: (NUM_HEADS, SEQ_LEN, HEAD_DIM)
    Q_all = torch.randn((NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.bfloat16) * 0.1
    K_all = torch.randn((NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.bfloat16) * 0.1
    V_all = torch.randn((NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.bfloat16) * 0.2
    out_all = torch.zeros((NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.bfloat16)

    # Shared scale and ones tensors
    scale_torch = torch.full((SEQ_LEN, HEAD_DIM), scale_val, dtype=torch.bfloat16)
    ones_torch = torch.ones((SEQ_LEN, HEAD_DIM), dtype=torch.bfloat16)

    # Create DRAM interleaved tensors for each head
    print("\nCreating DRAM interleaved tensors for all heads...")

    Q_heads = []
    K_heads = []
    V_heads = []
    out_heads = []

    for h in range(NUM_HEADS):
        Q_heads.append(
            ttnn.from_torch(
                Q_all[h],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        K_heads.append(
            ttnn.from_torch(
                K_all[h],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        V_heads.append(
            ttnn.from_torch(
                V_all[h],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        out_heads.append(
            ttnn.from_torch(
                out_all[h],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    # Shared scale and ones (reused across heads)
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

    print(f"Created {NUM_HEADS} sets of Q,K,V,out tensors in DRAM")
    print(f"Q[0] memory_config: {Q_heads[0].memory_config()}")
    # CHECK: DRAM

    # Compile kernel once
    print("\nCompiling kernel (once)...")
    # CHECK: Compiling kernel
    kernel = flash_attention_head_dram.compile(
        Q_heads[0], K_heads[0], V_heads[0], scale, ones, out_heads[0]
    )
    print(f"Kernel compiled!")

    # Process all heads
    print(f"\nProcessing {NUM_HEADS} attention heads from DRAM...")
    # CHECK: Processing 16 attention heads

    for h in range(NUM_HEADS):
        kernel(Q_heads[h], K_heads[h], V_heads[h], scale, ones, out_heads[h])
        if h % 4 == 0:
            print(f"  Processed head {h}/{NUM_HEADS}")

    print(f"\nAll {NUM_HEADS} heads processed!")
    # CHECK: All 16 heads processed

    # Verify results
    print("\n=== Verification ===")

    num_correct = 0
    tolerance = 0.2

    for h in range(NUM_HEADS):
        result = ttnn.to_torch(out_heads[h])
        expected = compute_expected_attention(Q_all[h], K_all[h], V_all[h], scale_val)

        error = abs(result[0, 0].item() - expected[0, 0].item()) / (
            abs(expected[0, 0].item()) + 1e-6
        )

        if error < tolerance:
            num_correct += 1
        else:
            print(
                f"  Head {h}: FAIL - expected {expected[0,0].item():.6f}, got {result[0,0].item():.6f}"
            )

    print(f"\nResults: {num_correct}/{NUM_HEADS} heads correct")

    if num_correct == NUM_HEADS:
        print("\nPASS: Large DRAM interleaved Flash Attention works!")
        # CHECK: PASS: Large DRAM interleaved Flash Attention
    else:
        print(f"\nFAIL: {NUM_HEADS - num_correct} heads failed")

finally:
    ttnn.close_device(device)

print("\n=== DRAM Interleaved Large Flash Attention Test Complete ===")
# CHECK: Test Complete
