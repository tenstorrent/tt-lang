# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Extract Qwen2.5-0.5B-Instruct weights for tt-lang execution.

Downloads the model from HuggingFace, extracts all weights as bf16 tensors,
pads to tile boundaries (multiples of 32), pre-transposes weight matrices
for matmul compatibility (tt-lang does x @ W, HF stores W as [out, in]),
and pre-computes RoPE cos/sin tables.

Output: a single .pt checkpoint file ready for device upload.

Usage:
    python examples/qwen/weight_extractor.py [--output weights/qwen2.5-0.5b.pt]
"""

import argparse
import math
import os
from pathlib import Path

import torch


TILE_SIZE = 32
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Qwen 2.5 0.5B architecture constants
HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_Q_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64  # 896 / 14
INTERMEDIATE_SIZE = 4864
VOCAB_SIZE = 151936
ROPE_THETA = 1_000_000.0
RMS_NORM_EPS = 1e-6
MAX_SEQ_LEN = 512


def pad_to_tile(tensor: torch.Tensor) -> torch.Tensor:
    """Pad tensor dimensions to multiples of TILE_SIZE (32)."""
    padded_shape = []
    pad_amounts = []
    for dim_size in reversed(tensor.shape):
        remainder = dim_size % TILE_SIZE
        pad_needed = (TILE_SIZE - remainder) % TILE_SIZE
        padded_shape.insert(0, dim_size + pad_needed)
        pad_amounts.extend([0, pad_needed])

    if all(p == 0 for p in pad_amounts):
        return tensor

    return torch.nn.functional.pad(tensor, pad_amounts, value=0.0)


def compute_rope_tables(max_seq_len: int, head_dim: int, theta: float) -> tuple:
    """Pre-compute RoPE cos/sin tables.

    Returns:
        cos_table: [max_seq_len, head_dim] padded to tile boundaries
        sin_table: [max_seq_len, head_dim] padded to tile boundaries
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )

    # Compute position * frequency
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [seq_len, head_dim/2]

    # Duplicate for full head_dim (interleaved pattern)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16)  # [seq, head_dim]
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16)  # [seq, head_dim]

    return pad_to_tile(cos_table), pad_to_tile(sin_table)


def extract_weights(output_path: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    state_dict = model.state_dict()

    print("Extracting and preparing weights...")
    checkpoint = {
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_size": INTERMEDIATE_SIZE,
            "vocab_size": VOCAB_SIZE,
            "rope_theta": ROPE_THETA,
            "rms_norm_eps": RMS_NORM_EPS,
            "max_seq_len": MAX_SEQ_LEN,
        },
        "layers": [],
    }

    # Embedding weights: [vocab_size, hidden_size]
    # For tt-lang matmul (x @ W), embedding lookup is done on host.
    # lm_head is tied to embedding, so we store once.
    embed_weight = state_dict["model.embed_tokens.weight"].bfloat16()
    print(f"  embed_tokens: {list(embed_weight.shape)}")
    checkpoint["embed_weight"] = pad_to_tile(embed_weight)

    # Final RMSNorm weight: [hidden_size] -> [32, hidden_size]
    final_norm = state_dict["model.norm.weight"].bfloat16()
    checkpoint["final_norm_weight"] = pad_to_tile(final_norm.unsqueeze(0).expand(TILE_SIZE, -1).contiguous())
    print(f"  final_norm: {list(final_norm.shape)}")

    # Per-layer weights
    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}"
        layer = {}

        # --- Attention weights ---
        # HuggingFace stores weights as [out_features, in_features].
        # tt-lang does x @ W, so we need W as [in_features, out_features].
        # Therefore we transpose all weight matrices.

        # Q projection: [896, 896] -> transposed to [896, 896]
        q_w = state_dict[f"{prefix}.self_attn.q_proj.weight"].bfloat16()
        layer["q_proj_weight"] = pad_to_tile(q_w.t().contiguous())

        # Q bias: [896] -> [32, 896] with same values in every row
        q_b = state_dict[f"{prefix}.self_attn.q_proj.bias"].bfloat16()
        layer["q_proj_bias"] = pad_to_tile(q_b.unsqueeze(0).expand(TILE_SIZE, -1).contiguous())

        # K projection: [128, 896] -> transposed to [896, 128]
        k_w = state_dict[f"{prefix}.self_attn.k_proj.weight"].bfloat16()
        layer["k_proj_weight"] = pad_to_tile(k_w.t().contiguous())

        # K bias: [128] -> [32, 128] with same values in every row
        k_b = state_dict[f"{prefix}.self_attn.k_proj.bias"].bfloat16()
        layer["k_proj_bias"] = pad_to_tile(k_b.unsqueeze(0).expand(TILE_SIZE, -1).contiguous())

        # V projection: [128, 896] -> transposed to [896, 128]
        v_w = state_dict[f"{prefix}.self_attn.v_proj.weight"].bfloat16()
        layer["v_proj_weight"] = pad_to_tile(v_w.t().contiguous())

        # V bias: [128] -> [32, 128] with same values in every row
        v_b = state_dict[f"{prefix}.self_attn.v_proj.bias"].bfloat16()
        layer["v_proj_bias"] = pad_to_tile(v_b.unsqueeze(0).expand(TILE_SIZE, -1).contiguous())

        # O projection: [896, 896] -> transposed to [896, 896] (no bias)
        o_w = state_dict[f"{prefix}.self_attn.o_proj.weight"].bfloat16()
        layer["o_proj_weight"] = pad_to_tile(o_w.t().contiguous())

        # --- MLP weights ---
        # gate_proj: [4864, 896] -> transposed to [896, 4864]
        gate_w = state_dict[f"{prefix}.mlp.gate_proj.weight"].bfloat16()
        layer["gate_proj_weight"] = pad_to_tile(gate_w.t().contiguous())

        # up_proj: [4864, 896] -> transposed to [896, 4864]
        up_w = state_dict[f"{prefix}.mlp.up_proj.weight"].bfloat16()
        layer["up_proj_weight"] = pad_to_tile(up_w.t().contiguous())

        # down_proj: [896, 4864] -> transposed to [4864, 896]
        down_w = state_dict[f"{prefix}.mlp.down_proj.weight"].bfloat16()
        layer["down_proj_weight"] = pad_to_tile(down_w.t().contiguous())

        # --- Norm weights ---
        # input_layernorm: [896] -> [32, 896] with same values in every row
        ln1 = state_dict[f"{prefix}.input_layernorm.weight"].bfloat16()
        layer["input_layernorm_weight"] = pad_to_tile(ln1.unsqueeze(0).expand(TILE_SIZE, -1).contiguous())

        # post_attention_layernorm: [896] -> [32, 896]
        ln2 = state_dict[f"{prefix}.post_attention_layernorm.weight"].bfloat16()
        layer["post_attention_layernorm_weight"] = pad_to_tile(ln2.unsqueeze(0).expand(TILE_SIZE, -1).contiguous())

        checkpoint["layers"].append(layer)

        if layer_idx == 0:
            print(f"  Layer 0 shapes (after transpose + pad):")
            for k, v in layer.items():
                print(f"    {k}: {list(v.shape)}")
        elif layer_idx == NUM_LAYERS - 1:
            print(f"  ... (layers 1-{NUM_LAYERS-2} same)")
            print(f"  Layer {NUM_LAYERS-1}: done")

    # RoPE tables
    print("Computing RoPE tables...")
    cos_table, sin_table = compute_rope_tables(MAX_SEQ_LEN, HEAD_DIM, ROPE_THETA)
    checkpoint["rope_cos"] = cos_table
    checkpoint["rope_sin"] = sin_table
    print(f"  rope_cos: {list(cos_table.shape)}")
    print(f"  rope_sin: {list(sin_table.shape)}")

    # Pre-compute causal mask [seq, seq] for prefill
    print("Computing causal mask...")
    causal_mask = torch.triu(
        torch.full((MAX_SEQ_LEN, MAX_SEQ_LEN), float("-inf")), diagonal=1
    ).bfloat16()
    checkpoint["causal_mask"] = pad_to_tile(causal_mask)
    print(f"  causal_mask: {list(checkpoint['causal_mask'].shape)}")

    # Pre-compute attention scale as a tile filled with 1/sqrt(head_dim)
    scale_val = 1.0 / math.sqrt(HEAD_DIM)
    scale_tile = torch.full((TILE_SIZE, TILE_SIZE), scale_val, dtype=torch.bfloat16)
    checkpoint["attn_scale"] = scale_tile

    # Pre-compute RMSNorm scaler: 1/hidden_size as a tile
    norm_scaler = torch.full(
        (TILE_SIZE, TILE_SIZE), 1.0 / HIDDEN_SIZE, dtype=torch.bfloat16
    )
    checkpoint["norm_scaler"] = norm_scaler

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"\nSaving checkpoint to {output_path}...")
    torch.save(checkpoint, output_path)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Checkpoint saved: {file_size_mb:.1f} MB")

    # Verify
    print("\nVerifying checkpoint...")
    loaded = torch.load(output_path, weights_only=True)
    assert len(loaded["layers"]) == NUM_LAYERS
    assert loaded["embed_weight"].shape[0] >= VOCAB_SIZE
    assert loaded["embed_weight"].shape[1] >= HIDDEN_SIZE
    for i, layer in enumerate(loaded["layers"]):
        assert "q_proj_weight" in layer, f"Layer {i} missing q_proj_weight"
        assert "gate_proj_weight" in layer, f"Layer {i} missing gate_proj_weight"
    print("Verification passed!")

    # Save tokenizer path for later use
    tokenizer_path = os.path.join(os.path.dirname(output_path), "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Qwen2.5-0.5B-Instruct weights")
    parser.add_argument(
        "--output",
        default="examples/qwen/weights/qwen2.5-0.5b.pt",
        help="Output checkpoint path",
    )
    args = parser.parse_args()
    extract_weights(args.output)
