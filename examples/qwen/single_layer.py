# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3: Single Qwen 2.5 0.5B transformer layer.

Orchestrates component kernels:
  1. RMSNorm(x) → normed
  2. Q = linear_bias(normed, q_w, q_b)
  3. K = linear_bias(normed, k_w, k_b)
  4. V = linear_bias(normed, v_w, v_b)
  5. Apply RoPE on host
  6. GQA attention (14 Q heads, 2 KV heads)
  7. O projection: linear(attn_out, o_w)
  8. Residual: x + proj
  9. RMSNorm(post_attn) → normed2
  10. MLP: silu(gate_proj(normed2)) * up_proj(normed2), then down_proj
  11. Residual: post_attn + mlp_out

Usage:
    source build/env/activate
    cd examples/qwen
    python single_layer.py
"""

import math
import sys
import os

import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from kernels.linear import linear_kernel, linear_bias_kernel
from kernels.elementwise import add_kernel, silu_mul_kernel
from kernels.rmsnorm import rmsnorm

TILE = 32
HIDDEN = 896
NUM_Q_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
INTERMEDIATE = 4864
SEQ_LEN = 512
ROPE_THETA = 1_000_000.0
RMS_NORM_EPS = 1e-6


def to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def alloc_zeros(shape, device):
    return to_device(torch.zeros(shape, dtype=torch.bfloat16), device)


def apply_rope(q, k, cos, sin, seq_len, num_q_heads, num_kv_heads, head_dim):
    """Apply Rotary Position Embeddings on host.

    q: [seq, num_q_heads * head_dim] float tensor
    k: [seq, num_kv_heads * head_dim] float tensor
    cos, sin: [seq, head_dim] float tensors
    """
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(t, cos_t, sin_t, num_heads):
        t = t.view(seq_len, num_heads, head_dim)
        cos_t = cos_t.unsqueeze(1)  # [seq, 1, head_dim]
        sin_t = sin_t.unsqueeze(1)
        return (t * cos_t + rotate_half(t) * sin_t).view(seq_len, num_heads * head_dim)

    q_rot = apply_rotary(q, cos[:seq_len], sin[:seq_len], num_q_heads)
    k_rot = apply_rotary(k, cos[:seq_len], sin[:seq_len], num_kv_heads)
    return q_rot, k_rot


def transformer_layer(x_device, layer_weights, causal_mask, rope_cos, rope_sin, device):
    """Run one Qwen transformer layer.

    Args:
        x_device: [seq, hidden] on device
        layer_weights: dict with weight tensors on device
        causal_mask: [seq, seq] torch tensor
        rope_cos, rope_sin: [seq, head_dim] torch tensors
        device: TTNN device

    Returns:
        output: [seq, hidden] on device
    """
    # === Attention Block ===

    # 1. RMSNorm
    normed = rmsnorm(x_device, layer_weights["input_layernorm_weight"], device, RMS_NORM_EPS)

    # 2. Q/K/V projections
    q_out = alloc_zeros((SEQ_LEN, HIDDEN), device)
    linear_bias_kernel(normed, layer_weights["q_proj_weight"], layer_weights["q_proj_bias"], q_out)

    k_out = alloc_zeros((SEQ_LEN, NUM_KV_HEADS * HEAD_DIM), device)
    linear_bias_kernel(normed, layer_weights["k_proj_weight"], layer_weights["k_proj_bias"], k_out)

    v_out = alloc_zeros((SEQ_LEN, NUM_KV_HEADS * HEAD_DIM), device)
    linear_bias_kernel(normed, layer_weights["v_proj_weight"], layer_weights["v_proj_bias"], v_out)

    # 3. RoPE on host
    q_torch = ttnn.to_torch(q_out).float()
    k_torch = ttnn.to_torch(k_out).float()
    q_rot, k_rot = apply_rope(q_torch, k_torch, rope_cos, rope_sin,
                               SEQ_LEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)

    # 4. GQA Attention (host orchestrated)
    scale_val = 1.0 / math.sqrt(HEAD_DIM)
    heads_per_group = NUM_Q_HEADS // NUM_KV_HEADS

    q_heads = q_rot.view(SEQ_LEN, NUM_Q_HEADS, HEAD_DIM)
    k_heads = k_rot.view(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
    v_torch = ttnn.to_torch(v_out).float()
    v_heads = v_torch.view(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)

    attn_outputs = []
    for kv_idx in range(NUM_KV_HEADS):
        k_head = k_heads[:, kv_idx, :].contiguous()
        k_t = k_head.t().contiguous().bfloat16()
        v_head = v_heads[:, kv_idx, :].contiguous().bfloat16()

        k_t_device = to_device(k_t, device)
        v_device_head = to_device(v_head, device)

        for q_local in range(heads_per_group):
            q_idx = kv_idx * heads_per_group + q_local
            q_head = q_heads[:, q_idx, :].contiguous().bfloat16()
            q_device_head = to_device(q_head, device)

            # Scores = Q @ K^T
            scores_device = alloc_zeros((SEQ_LEN, SEQ_LEN), device)
            linear_kernel(q_device_head, k_t_device, scores_device)

            # Host softmax
            scores = ttnn.to_torch(scores_device).float()
            scores = scores * scale_val + causal_mask.float()
            weights = torch.nn.functional.softmax(scores, dim=-1).bfloat16()
            weights_device = to_device(weights, device)

            # Attn output = weights @ V
            head_out_device = alloc_zeros((SEQ_LEN, HEAD_DIM), device)
            linear_kernel(weights_device, v_device_head, head_out_device)
            attn_outputs.append(ttnn.to_torch(head_out_device))

    # Concatenate heads
    attn_combined = torch.cat(attn_outputs, dim=-1).bfloat16()  # [512, 896]
    attn_device = to_device(attn_combined, device)

    # 5. Output projection
    proj_out = alloc_zeros((SEQ_LEN, HIDDEN), device)
    linear_kernel(attn_device, layer_weights["o_proj_weight"], proj_out)

    # 6. Residual
    post_attn = alloc_zeros((SEQ_LEN, HIDDEN), device)
    add_kernel(x_device, proj_out, post_attn)

    # === MLP Block ===

    # 7. RMSNorm
    normed2 = rmsnorm(post_attn, layer_weights["post_attention_layernorm_weight"], device, RMS_NORM_EPS)

    # 8. Gate and Up projections
    gate_out = alloc_zeros((SEQ_LEN, INTERMEDIATE), device)
    linear_kernel(normed2, layer_weights["gate_proj_weight"], gate_out)

    up_out = alloc_zeros((SEQ_LEN, INTERMEDIATE), device)
    linear_kernel(normed2, layer_weights["up_proj_weight"], up_out)

    # 9. SiLU + elementwise mul
    hidden = alloc_zeros((SEQ_LEN, INTERMEDIATE), device)
    silu_mul_kernel(gate_out, up_out, hidden)

    # 10. Down projection
    mlp_out = alloc_zeros((SEQ_LEN, HIDDEN), device)
    linear_kernel(hidden, layer_weights["down_proj_weight"], mlp_out)

    # 11. Residual
    output = alloc_zeros((SEQ_LEN, HIDDEN), device)
    add_kernel(post_attn, mlp_out, output)

    return output


# =========================================================================
# Test with actual Qwen weights
# =========================================================================
def test_single_layer():
    from utils import load_checkpoint, pcc, assert_pcc

    print("Loading weights...")
    ckpt = load_checkpoint()
    layer_data = ckpt["layers"][0]

    device = ttnn.open_device(device_id=0)
    try:
        print("Uploading layer 0 weights to device...")
        layer_weights = {k: to_device(v, device) for k, v in layer_data.items()}

        # RoPE tables
        rope_cos = ckpt["rope_cos"].float()
        rope_sin = ckpt["rope_sin"].float()

        # Causal mask
        causal_mask = torch.triu(
            torch.full((SEQ_LEN, SEQ_LEN), float("-inf")), diagonal=1
        )

        # Random input
        x_torch = torch.randn(SEQ_LEN, HIDDEN, dtype=torch.bfloat16) * 0.1
        x_device = to_device(x_torch, device)

        print("Running transformer layer 0...")
        output_device = transformer_layer(
            x_device, layer_weights, causal_mask, rope_cos, rope_sin, device,
        )
        result = ttnn.to_torch(output_device)

        # PyTorch reference using HuggingFace model
        print("Computing PyTorch reference...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16
        )
        model.eval()

        with torch.no_grad():
            hidden_states = x_torch.float().unsqueeze(0).bfloat16()  # [1, seq, hidden]
            position_ids = torch.arange(SEQ_LEN).unsqueeze(0)

            layer_module = model.model.layers[0]

            # Compute position embeddings (RoPE cos/sin)
            rotary_emb = model.model.rotary_emb
            position_embeddings = rotary_emb(hidden_states, position_ids)

            causal_mask_4d = torch.triu(
                torch.full((1, 1, SEQ_LEN, SEQ_LEN), float("-inf")), diagonal=1
            ).bfloat16()

            layer_out = layer_module(
                hidden_states,
                attention_mask=causal_mask_4d,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            expected = layer_out[0].squeeze(0).bfloat16()  # [seq, hidden]

        score = torch.corrcoef(
            torch.stack([result.float().flatten(), expected.float().flatten()])
        )[0, 1].item()
        print(f"\nSingle layer PCC: {score:.6f}")
        if score > 0.90:
            print("PASS (PCC > 0.90)")
        else:
            print(f"FAIL (PCC = {score:.6f}, expected > 0.90)")
            print(f"Max diff: {(result.float() - expected.float()).abs().max().item():.6f}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_single_layer()
