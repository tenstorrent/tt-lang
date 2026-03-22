# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Attention kernel for Qwen 2.5 0.5B.

Single-head attention: attn = softmax(Q @ K^T / sqrt(d)) @ V

For GQA (14 Q heads, 2 KV heads), the host orchestrates:
  - Call attention 14 times, selecting appropriate Q and KV head

Per-head shapes:
  Q: [seq, head_dim] = [512, 64] = [16, 2] tiles
  K^T: [head_dim, seq] = [64, 512] = [2, 16] tiles (pre-transposed on host)
  V: [seq, head_dim] = [512, 64] = [16, 2] tiles
  Scores: Q @ K^T = [16, 16] tiles
  Output: softmax(scores) @ V = [16, 2] tiles

Softmax is done on host (pulling scores back) since reduce_max/reduce_sum
are not available on the compiler.

Strategy:
  1. scores_kernel: compute Q @ K^T on device → [seq, seq]
  2. Host: apply scale, mask, softmax
  3. attn_out_kernel: compute softmax_weights @ V on device → [seq, head_dim]
"""

import torch
import ttl
import ttnn

TILE = 32


# Reuse linear_kernel for matmul operations
from kernels.linear import linear_kernel


def attention_single_head(q_device, k_t_device, v_device, causal_mask, scale_val, device):
    """Single-head attention with host-side softmax.

    Args:
        q_device: [seq, head_dim] on device
        k_t_device: [head_dim, seq] on device (pre-transposed)
        v_device: [seq, head_dim] on device
        causal_mask: [seq, seq] torch tensor (float, -inf upper triangle)
        scale_val: 1/sqrt(head_dim)
        device: TTNN device

    Returns:
        attn_out: [seq, head_dim] on device
    """
    q_shape = (q_device.shape[0], q_device.shape[1])  # padded shapes
    seq = q_shape[0]
    head_dim = q_shape[1]

    # Step 1: scores = Q @ K^T on device
    scores_device = ttnn.from_torch(
        torch.zeros(seq, seq, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    linear_kernel(q_device, k_t_device, scores_device)

    # Step 2: Host-side scale + mask + softmax
    scores = ttnn.to_torch(scores_device).float()
    scores = scores * scale_val
    scores = scores + causal_mask.float()
    attn_weights = torch.nn.functional.softmax(scores, dim=-1).bfloat16()

    attn_weights_device = ttnn.from_torch(
        attn_weights, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Step 3: attn_out = softmax_weights @ V on device
    attn_out_device = ttnn.from_torch(
        torch.zeros(seq, head_dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    linear_kernel(attn_weights_device, v_device, attn_out_device)

    return attn_out_device


def gqa_attention(q_device, k_device, v_device, causal_mask, device,
                  num_q_heads=14, num_kv_heads=2, head_dim=64, seq_len=512):
    """Grouped Query Attention.

    Args:
        q_device: [seq, num_q_heads * head_dim] = [512, 896] on device
        k_device: [seq, num_kv_heads * head_dim] = [512, 128] on device
        v_device: [seq, num_kv_heads * head_dim] = [512, 128] on device
        causal_mask: [seq, seq] torch tensor
        device: TTNN device

    Returns:
        attn_out: [seq, num_q_heads * head_dim] = [512, 896] on device
    """
    scale_val = 1.0 / (head_dim ** 0.5)
    heads_per_group = num_q_heads // num_kv_heads  # 7

    # Pull Q, K, V to host for reshaping
    q_torch = ttnn.to_torch(q_device).float()  # [seq, 896]
    k_torch = ttnn.to_torch(k_device).float()  # [seq, 128]
    v_torch = ttnn.to_torch(v_device).float()  # [seq, 128]

    # Reshape to per-head
    q_heads = q_torch.view(seq_len, num_q_heads, head_dim)   # [512, 14, 64]
    k_heads = k_torch.view(seq_len, num_kv_heads, head_dim)  # [512, 2, 64]
    v_heads = v_torch.view(seq_len, num_kv_heads, head_dim)  # [512, 2, 64]

    # Apply RoPE on host (will be passed in already applied in the full model)
    # For now, skip RoPE in this standalone test

    outputs = []
    for kv_idx in range(num_kv_heads):
        # Transpose K for this KV head: [seq, head_dim] -> [head_dim, seq]
        k_head = k_heads[:, kv_idx, :].contiguous()  # [512, 64]
        k_t = k_head.t().contiguous().bfloat16()       # [64, 512]
        v_head = v_heads[:, kv_idx, :].contiguous().bfloat16()  # [512, 64]

        k_t_device = ttnn.from_torch(
            k_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_device_head = ttnn.from_torch(
            v_head, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Process all Q heads in this group
        for q_local in range(heads_per_group):
            q_idx = kv_idx * heads_per_group + q_local
            q_head = q_heads[:, q_idx, :].contiguous().bfloat16()  # [512, 64]

            q_device_head = ttnn.from_torch(
                q_head, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            attn_out = attention_single_head(
                q_device_head, k_t_device, v_device_head,
                causal_mask, scale_val, device,
            )
            outputs.append(ttnn.to_torch(attn_out))  # [512, 64]

    # Concatenate all head outputs: [512, 14*64] = [512, 896]
    combined = torch.cat(outputs, dim=-1).bfloat16()

    return ttnn.from_torch(
        combined, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# =========================================================================
# Test
# =========================================================================
def test_attention_single_head(device):
    seq, head_dim = 512, 64
    print(f"  attention single head [{seq}x{head_dim}]...", end="", flush=True)

    Q_t = torch.randn(seq, head_dim, dtype=torch.bfloat16) * 0.1
    K_t = torch.randn(seq, head_dim, dtype=torch.bfloat16) * 0.1
    V_t = torch.randn(seq, head_dim, dtype=torch.bfloat16) * 0.1
    K_t_T = K_t.t().contiguous()  # [64, 512]

    causal_mask = torch.triu(
        torch.full((seq, seq), float("-inf")), diagonal=1
    )
    scale_val = 1.0 / (head_dim ** 0.5)

    Q = ttnn.from_torch(Q_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    K_T = ttnn.from_torch(K_t_T.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    V = ttnn.from_torch(V_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    result_device = attention_single_head(Q, K_T, V, causal_mask, scale_val, device)
    result = ttnn.to_torch(result_device)

    # PyTorch reference
    scores = Q_t.float() @ K_t.float().t() * scale_val
    scores = scores + causal_mask
    weights = torch.nn.functional.softmax(scores, dim=-1)
    expected = (weights @ V_t.float()).bfloat16()

    score = torch.corrcoef(
        torch.stack([result.float().flatten(), expected.float().flatten()])
    )[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.98, f" FAIL"
    print(" PASS")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "examples/qwen")

    device = ttnn.open_device(device_id=0)
    try:
        print("Attention tests:")
        test_attention_single_head(device)
        print("Attention test passed!")
    finally:
        ttnn.close_device(device)
