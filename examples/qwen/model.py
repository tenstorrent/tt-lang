# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen 2.5 0.5B-Instruct model wrapper for tt-lang.

Manages weight loading, device upload, KV cache, and the full forward pass
for both prefill and decode phases.
"""

import contextlib
import io
import math
import os
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from kernels.linear import linear_kernel, linear_bias_kernel
from kernels.elementwise import add_kernel, silu_mul_kernel
from kernels.rope import rope_kernel
from kernels.rmsnorm import device_rmsnorm, rmsnorm_mul_kernel
from kernels.softmax import device_softmax
from utils import load_checkpoint

TILE = 32


class QwenModel:
    """Qwen 2.5 0.5B-Instruct running on Tenstorrent Blackhole via tt-lang."""

    def __init__(self, device, checkpoint_path=None):
        self.device = device

        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), "weights", "qwen2.5-0.5b.pt"
            )
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.ckpt = torch.load(checkpoint_path, weights_only=True)
        self.config = self.ckpt["config"]

        self.hidden_size = self.config["hidden_size"]       # 896
        self.num_layers = self.config["num_layers"]          # 24
        self.num_q_heads = self.config["num_q_heads"]        # 14
        self.num_kv_heads = self.config["num_kv_heads"]      # 2
        self.head_dim = self.config["head_dim"]              # 64
        self.intermediate_size = self.config["intermediate_size"]  # 4864
        self.vocab_size = self.config["vocab_size"]          # 151936
        self.rms_norm_eps = self.config["rms_norm_eps"]      # 1e-6
        self.max_seq_len = self.config["max_seq_len"]        # 512
        self.padded_max_seq = ((self.max_seq_len + TILE - 1) // TILE) * TILE  # 512

        # RoPE tables — host copies for reference + device copies for kernel
        self.rope_cos = self.ckpt["rope_cos"].float()
        self.rope_sin = self.ckpt["rope_sin"].float()
        self.rope_cos_device = self._to_device(self.ckpt["rope_cos"].bfloat16())
        self.rope_sin_device = self._to_device(self.ckpt["rope_sin"].bfloat16())

        # Embedding weights (host-side for lookup)
        self.embed_weight = self.ckpt["embed_weight"].float()

        # Upload layer weights to device + pre-split per-head weights
        print(f"Uploading {self.num_layers} layers to device...")
        t0 = time.time()
        self.layer_weights = []
        heads_per_group = self.num_q_heads // self.num_kv_heads  # 7

        for i in range(self.num_layers):
            layer_data = self.ckpt["layers"][i]
            layer_on_device = {
                k: self._to_device(v) for k, v in layer_data.items()
            }

            # Pre-split Q/K/V weights per head for decode (avoids host reshape)
            # Q: [896, 896] → 14 × [896, 64]
            q_w = layer_data["q_proj_weight"]  # [896, 896]
            q_b = layer_data["q_proj_bias"]    # [32, 896]
            # Pre-scale Q weights by 1/sqrt(head_dim) so decode never needs runtime scaling
            attn_scale = 1.0 / math.sqrt(self.head_dim)
            layer_on_device["q_head_weights"] = []
            layer_on_device["q_head_biases"] = []
            for h in range(self.num_q_heads):
                col_start = h * self.head_dim
                col_end = col_start + self.head_dim
                layer_on_device["q_head_weights"].append(
                    self._to_device((q_w[:, col_start:col_end] * attn_scale).contiguous()))
                layer_on_device["q_head_biases"].append(
                    self._to_device((q_b[:, col_start:col_end] * attn_scale).contiguous()))

            # K: [896, 128] → 2 × [896, 64]
            k_w = layer_data["k_proj_weight"]  # [896, 128]
            k_b = layer_data["k_proj_bias"]    # [32, 128]
            layer_on_device["k_head_weights"] = []
            layer_on_device["k_head_biases"] = []
            for h in range(self.num_kv_heads):
                col_start = h * self.head_dim
                col_end = col_start + self.head_dim
                layer_on_device["k_head_weights"].append(
                    self._to_device(k_w[:, col_start:col_end].contiguous()))
                layer_on_device["k_head_biases"].append(
                    self._to_device(k_b[:, col_start:col_end].contiguous()))

            # V: [896, 128] → 2 × [896, 64]
            v_w = layer_data["v_proj_weight"]  # [896, 128]
            v_b = layer_data["v_proj_bias"]    # [32, 128]
            layer_on_device["v_head_weights"] = []
            layer_on_device["v_head_biases"] = []
            for h in range(self.num_kv_heads):
                col_start = h * self.head_dim
                col_end = col_start + self.head_dim
                layer_on_device["v_head_weights"].append(
                    self._to_device(v_w[:, col_start:col_end].contiguous()))
                layer_on_device["v_head_biases"].append(
                    self._to_device(v_b[:, col_start:col_end].contiguous()))

            self.layer_weights.append(layer_on_device)
            if (i + 1) % 8 == 0:
                print(f"  {i + 1}/{self.num_layers} layers uploaded")
        print(f"  All layers uploaded in {time.time() - t0:.1f}s")

        # Final norm weight
        self.final_norm_weight = self._to_device(self.ckpt["final_norm_weight"])

        # Pre-computed scalers for device-side reduce ops
        self.mean_scaler_device = self._to_device(
            torch.full((TILE, TILE), 1.0 / self.hidden_size, dtype=torch.bfloat16)
        )
        self.ones_scaler_device = self._to_device(
            torch.ones(TILE, TILE, dtype=torch.bfloat16)
        )
        # Attention scale: 1/sqrt(head_dim), pre-applied to Q before matmul
        self.attn_scale_device = self._to_device(
            torch.full((TILE, TILE), 1.0 / math.sqrt(self.head_dim), dtype=torch.bfloat16)
        )

        # KV cache (initialized on first prefill)
        self.kv_cache = None
        self.cache_pos = 0  # number of positions filled

        # Quiet mode: suppress compilation output
        self.quiet = False
        self._compile_log = os.path.join(
            os.path.dirname(__file__), "compile.log"
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _to_device(self, tensor):
        return ttnn.from_torch(
            tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _alloc_zeros(self, shape):
        return self._to_device(torch.zeros(shape, dtype=torch.bfloat16))

    @contextlib.contextmanager
    def _suppress_output(self):
        """Redirect stdout/stderr to compile log when quiet mode is on.

        Uses fd-level redirection to also capture C++ output from tt-metal
        runtime, which writes directly to fd 1/2 bypassing Python's sys.stdout.
        """
        if not self.quiet:
            yield
            return

        # Flush Python buffers first
        sys.stdout.flush()
        sys.stderr.flush()

        # Save original file descriptors
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)

        try:
            log_fd = os.open(self._compile_log,
                             os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.dup2(log_fd, stdout_fd)
            os.dup2(log_fd, stderr_fd)
            os.close(log_fd)
            yield
        finally:
            # Flush before restoring
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)

    # -----------------------------------------------------------------
    # RMSNorm helpers
    # -----------------------------------------------------------------
    def _rmsnorm_host(self, x_device, gamma_device):
        """RMSNorm with host-side reduction. For prefill (all rows matter)."""
        x_torch = ttnn.to_torch(x_device).float()
        seq, hidden = x_torch.shape
        mean_sq = (x_torch ** 2).mean(dim=-1, keepdim=True)
        rsqrt_val = torch.rsqrt(mean_sq + self.rms_norm_eps)
        scale = rsqrt_val.expand(-1, TILE).bfloat16().contiguous()
        if seq % TILE != 0:
            scale = torch.nn.functional.pad(scale, (0, 0, 0, TILE - seq % TILE))
        scale_device = self._to_device(scale)
        y_device = self._alloc_zeros(x_torch.shape)
        rmsnorm_mul_kernel(x_device, scale_device, gamma_device, y_device)
        return y_device

    def _rmsnorm_device(self, x_device, gamma_device):
        """RMSNorm fully on device. For decode (only row 0 matters)."""
        return device_rmsnorm(x_device, gamma_device,
                              self.mean_scaler_device, self.device)

    # -----------------------------------------------------------------
    # RoPE
    # -----------------------------------------------------------------
    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope_prefill(self, q, k, seq_len):
        """Apply RoPE to full sequence (prefill)."""
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]

        def apply_rotary(t, num_heads):
            t = t.view(seq_len, num_heads, self.head_dim)
            cos_t = cos.unsqueeze(1)
            sin_t = sin.unsqueeze(1)
            return (t * cos_t + self._rotate_half(t) * sin_t).view(
                seq_len, num_heads * self.head_dim
            )

        return apply_rotary(q, self.num_q_heads), apply_rotary(k, self.num_kv_heads)

    def _apply_rope_single(self, q, k, pos):
        """Apply RoPE to a single position (decode)."""
        cos = self.rope_cos[pos:pos+1]  # [1, head_dim]
        sin = self.rope_sin[pos:pos+1]

        def apply_rotary(t, num_heads):
            # t: [1, num_heads * head_dim]
            t = t.view(1, num_heads, self.head_dim)
            cos_t = cos.unsqueeze(1)
            sin_t = sin.unsqueeze(1)
            return (t * cos_t + self._rotate_half(t) * sin_t).view(
                1, num_heads * self.head_dim
            )

        return apply_rotary(q, self.num_q_heads), apply_rotary(k, self.num_kv_heads)

    # -----------------------------------------------------------------
    # KV Cache
    # -----------------------------------------------------------------
    def init_kv_cache(self):
        """Initialize KV cache with both host mirrors and device-resident tensors.

        Host: kv_cache[layer][kv_head] = {"k": [max_seq, head_dim], "v": same}
        Device: kv_cache_dev[layer][kv_head] = {"k_t": [head_dim, max_seq], "v": [max_seq, head_dim]}

        Device tensors persist across decode steps — no re-upload needed.
        """
        self.kv_cache = []
        self.kv_cache_dev = []
        for _ in range(self.num_layers):
            layer_cache = []
            layer_cache_dev = []
            for _ in range(self.num_kv_heads):
                layer_cache.append({
                    "k": torch.zeros(self.max_seq_len, self.head_dim, dtype=torch.float32),
                    "v": torch.zeros(self.max_seq_len, self.head_dim, dtype=torch.float32),
                })
                layer_cache_dev.append({
                    "k_t": self._to_device(torch.zeros(self.head_dim, self.max_seq_len, dtype=torch.bfloat16)),
                    "v": self._to_device(torch.zeros(self.max_seq_len, self.head_dim, dtype=torch.bfloat16)),
                })
            self.kv_cache.append(layer_cache)
            self.kv_cache_dev.append(layer_cache_dev)
        self.cache_pos = 0

    def _update_kv_cache(self, layer_idx, k_rot, v_float, start_pos, length):
        """Update host KV cache mirrors for a range of positions.

        k_rot: [length, num_kv_heads * head_dim] float
        v_float: [length, num_kv_heads * head_dim] float
        """
        k_heads = k_rot.view(length, self.num_kv_heads, self.head_dim)
        v_heads = v_float.view(length, self.num_kv_heads, self.head_dim)

        for kv_idx in range(self.num_kv_heads):
            self.kv_cache[layer_idx][kv_idx]["k"][start_pos:start_pos+length] = k_heads[:, kv_idx]
            self.kv_cache[layer_idx][kv_idx]["v"][start_pos:start_pos+length] = v_heads[:, kv_idx]

    def _get_kv_device(self, layer_idx, kv_idx):
        """Upload KV cache to device DRAM, return K^T and V as device tensors.

        Always uses padded_max_seq (512) for consistent tensor shapes,
        avoiding kernel recompilation as the cache grows.

        Returns:
            k_t_dev: [head_dim, padded_max_seq] on device
            v_dev: [padded_max_seq, head_dim] on device
        """
        cache = self.kv_cache[layer_idx][kv_idx]

        # K^T: [head_dim, max_seq] — full cache, zeros beyond current pos
        k_t = cache["k"][:self.max_seq_len].bfloat16().t().contiguous()
        k_t_dev = self._to_device(k_t)

        # V: [max_seq, head_dim]
        v = cache["v"][:self.max_seq_len].bfloat16().contiguous()
        v_dev = self._to_device(v)

        return k_t_dev, v_dev

    # -----------------------------------------------------------------
    # Attention
    # -----------------------------------------------------------------
    def _attention_prefill(self, q_rot, k_rot, v_float, layer_idx, seq_len, causal_mask):
        """GQA attention for prefill: uses freshly computed K,V and stores to cache."""
        scale_val = 1.0 / math.sqrt(self.head_dim)
        heads_per_group = self.num_q_heads // self.num_kv_heads

        # Store K,V into cache
        self._update_kv_cache(layer_idx, k_rot, v_float, 0, seq_len)

        q_heads = q_rot.view(seq_len, self.num_q_heads, self.head_dim)

        outputs = []
        for kv_idx in range(self.num_kv_heads):
            # Get cached K^T and V on device (full max_seq_len shape)
            k_t_dev, v_dev = self._get_kv_device(layer_idx, kv_idx)

            for q_local in range(heads_per_group):
                q_idx = kv_idx * heads_per_group + q_local
                q_head = q_heads[:seq_len, q_idx, :].contiguous().bfloat16()
                # Pad Q to tile boundary
                padded_seq = ((seq_len + TILE - 1) // TILE) * TILE
                if padded_seq > seq_len:
                    q_head = torch.nn.functional.pad(q_head, (0, 0, 0, padded_seq - seq_len))
                q_dev = self._to_device(q_head)

                # Scores = Q @ K^T
                scores_dev = self._alloc_zeros((padded_seq, self.padded_max_seq))
                linear_kernel(q_dev, k_t_dev, scores_dev)

                # Host softmax for prefill (device softmax only handles single-row decode)
                scores = ttnn.to_torch(scores_dev).float()
                prefill_mask = torch.full((padded_seq, self.padded_max_seq), float("-inf"))
                prefill_mask[:padded_seq, :padded_seq] = causal_mask
                scores = scores * scale_val + prefill_mask
                weights = torch.nn.functional.softmax(scores, dim=-1).bfloat16()
                weights_dev = self._to_device(weights)

                # Attn output = weights[padded_seq, max_seq] @ V[max_seq, head_dim]
                head_out_dev = self._alloc_zeros((padded_seq, self.head_dim))
                linear_kernel(weights_dev, v_dev, head_out_dev)
                outputs.append(ttnn.to_torch(head_out_dev)[:seq_len])

        return torch.cat(outputs, dim=-1).bfloat16()

    def _attention_decode(self, q_rot, layer_idx, pos):
        """GQA attention for decode: Q is single position, K/V from full cache.

        Uses fixed max_seq_len (512) for all tensor shapes to avoid recompilation.
        The decode mask ensures only positions 0..pos are attended to.
        """
        scale_val = 1.0 / math.sqrt(self.head_dim)
        heads_per_group = self.num_q_heads // self.num_kv_heads
        attend_len = pos + 1
        cache_len = self.padded_max_seq  # always 512

        q_heads = q_rot.view(1, self.num_q_heads, self.head_dim)

        # Decode mask: [TILE, cache_len]
        # Row 0: attend to 0..pos, -inf for pos+1..cache_len-1
        # Rows 1..31: all -inf (padding rows)
        decode_mask = torch.full((TILE, cache_len), float("-inf"))
        decode_mask[0, :attend_len] = 0.0

        # Pre-compute mask on device (same for all heads)
        decode_mask_dev = self._to_device(decode_mask.bfloat16())

        outputs = []
        for kv_idx in range(self.num_kv_heads):
            k_t_dev, v_dev = self._get_kv_device(layer_idx, kv_idx)

            for q_local in range(heads_per_group):
                q_idx = kv_idx * heads_per_group + q_local
                q_head = q_heads[0, q_idx, :].contiguous().bfloat16()
                # Pre-scale Q by 1/sqrt(head_dim) on host (tiny: 1 tile)
                q_head = q_head * scale_val
                q_padded = torch.zeros(TILE, self.head_dim, dtype=torch.bfloat16)
                q_padded[0] = q_head
                q_dev = self._to_device(q_padded)

                # Scores = scaled_Q @ K^T (already scaled, no post-scale needed)
                scores_dev = self._alloc_zeros((TILE, cache_len))
                linear_kernel(q_dev, k_t_dev, scores_dev)

                # Device softmax — no host transfer!
                weights_dev = device_softmax(
                    scores_dev, decode_mask_dev, self.ones_scaler_device, self.device
                )

                # Attn output = weights @ V
                head_out_dev = self._alloc_zeros((TILE, self.head_dim))
                linear_kernel(weights_dev, v_dev, head_out_dev)
                outputs.append(ttnn.to_torch(head_out_dev)[:1])

        return torch.cat(outputs, dim=-1).bfloat16()

    # -----------------------------------------------------------------
    # Transformer layer
    # -----------------------------------------------------------------
    def _run_mlp(self, post_attn_device, layer_idx, seq_len, decode=False):
        """MLP block shared between prefill and decode."""
        w = self.layer_weights[layer_idx]

        if decode:
            normed2 = self._rmsnorm_device(post_attn_device, w["post_attention_layernorm_weight"])
        else:
            normed2 = self._rmsnorm_host(post_attn_device, w["post_attention_layernorm_weight"])

        gate_out = self._alloc_zeros((seq_len, self.intermediate_size))
        linear_kernel(normed2, w["gate_proj_weight"], gate_out)

        up_out = self._alloc_zeros((seq_len, self.intermediate_size))
        linear_kernel(normed2, w["up_proj_weight"], up_out)

        hidden = self._alloc_zeros((seq_len, self.intermediate_size))
        silu_mul_kernel(gate_out, up_out, hidden)

        mlp_out = self._alloc_zeros((seq_len, self.hidden_size))
        linear_kernel(hidden, w["down_proj_weight"], mlp_out)

        output = self._alloc_zeros((seq_len, self.hidden_size))
        add_kernel(post_attn_device, mlp_out, output)
        return output

    def _run_attn_projections(self, normed_device, layer_idx, seq_len):
        """Q/K/V projections shared between prefill and decode."""
        w = self.layer_weights[layer_idx]

        q_out = self._alloc_zeros((seq_len, self.hidden_size))
        linear_bias_kernel(normed_device, w["q_proj_weight"], w["q_proj_bias"], q_out)

        kv_dim = self.num_kv_heads * self.head_dim
        k_out = self._alloc_zeros((seq_len, kv_dim))
        linear_bias_kernel(normed_device, w["k_proj_weight"], w["k_proj_bias"], k_out)

        v_out = self._alloc_zeros((seq_len, kv_dim))
        linear_bias_kernel(normed_device, w["v_proj_weight"], w["v_proj_bias"], v_out)

        return (
            ttnn.to_torch(q_out).float(),
            ttnn.to_torch(k_out).float(),
            ttnn.to_torch(v_out).float(),
        )

    def transformer_layer_prefill(self, x_device, layer_idx, seq_len, causal_mask):
        """Prefill: full sequence attention, populate KV cache."""
        w = self.layer_weights[layer_idx]
        padded_seq = ((seq_len + TILE - 1) // TILE) * TILE

        normed = self._rmsnorm_host(x_device, w["input_layernorm_weight"])

        q_torch, k_torch, v_torch = self._run_attn_projections(normed, layer_idx, padded_seq)

        q_rot, k_rot = self._apply_rope_prefill(q_torch[:seq_len], k_torch[:seq_len], seq_len)
        attn_combined = self._attention_prefill(q_rot, k_rot, v_torch[:seq_len], layer_idx, seq_len, causal_mask)

        # Pad attention output back to tile boundary
        if padded_seq > seq_len:
            attn_combined = torch.nn.functional.pad(attn_combined, (0, 0, 0, padded_seq - seq_len))
        attn_device = self._to_device(attn_combined)

        proj_out = self._alloc_zeros((padded_seq, self.hidden_size))
        linear_kernel(attn_device, w["o_proj_weight"], proj_out)

        post_attn = self._alloc_zeros((padded_seq, self.hidden_size))
        add_kernel(x_device, proj_out, post_attn)

        return self._run_mlp(post_attn, layer_idx, padded_seq)

    def transformer_layer_decode(self, x_device, layer_idx, pos):
        """Decode: single token, per-head device projections + RoPE + softmax.

        Remaining host transfers: KV cache update (4KB pull + 128KB push per KV head),
        head output collection (4KB pull per head + 56KB concat push).
        """
        w = self.layer_weights[layer_idx]
        heads_per_group = self.num_q_heads // self.num_kv_heads
        cache_len = self.padded_max_seq
        attend_len = pos + 1

        normed = self._rmsnorm_device(x_device, w["input_layernorm_weight"])

        # Decode mask (reuse across heads)
        if not hasattr(self, '_decode_mask_cache') or self._decode_mask_pos != pos:
            decode_mask = torch.full((TILE, cache_len), float("-inf"), dtype=torch.bfloat16)
            decode_mask[0, :attend_len] = 0.0
            self._decode_mask_dev = self._to_device(decode_mask)
            self._decode_mask_pos = pos

        # Reset attention buffer
        self._decode_attn_buf.zero_()

        for kv_idx in range(self.num_kv_heads):
            # Per-head K projection + RoPE on device
            k_dev = self._alloc_zeros((TILE, self.head_dim))
            linear_bias_kernel(normed, w["k_head_weights"][kv_idx],
                               w["k_head_biases"][kv_idx], k_dev)
            k_rot_dev = self._alloc_zeros((TILE, self.head_dim))
            rope_kernel(k_dev, self._decode_cos_dev, self._decode_sin_dev, k_rot_dev)

            # Per-head V projection (no RoPE)
            v_dev = self._alloc_zeros((TILE, self.head_dim))
            linear_bias_kernel(normed, w["v_head_weights"][kv_idx],
                               w["v_head_biases"][kv_idx], v_dev)

            # KV cache update: pull new K/V (4KB each), update host+device cache
            k_rot_host = ttnn.to_torch(k_rot_dev)  # [TILE, 64] = 4KB
            v_host = ttnn.to_torch(v_dev)            # [TILE, 64] = 4KB
            self.kv_cache[layer_idx][kv_idx]["k"][pos] = k_rot_host[0].float()
            self.kv_cache[layer_idx][kv_idx]["v"][pos] = v_host[0].float()

            # Update device cache: re-upload only the tile-row containing pos
            tile_row = pos // TILE
            row_s = tile_row * TILE
            row_e = row_s + TILE
            # K^T: update tile-columns at pos → rebuild the 2 affected columns
            k_slice = self.kv_cache[layer_idx][kv_idx]["k"][row_s:row_e].bfloat16().t().contiguous()
            # k_slice is [64, 32] = 2 tile-columns. Upload and overwrite in full cache.
            # For now, rebuild full K^T and V from host (still a transfer but with persistent device tensors)
            k_t_full = self.kv_cache[layer_idx][kv_idx]["k"][:self.max_seq_len].bfloat16().t().contiguous()
            v_full = self.kv_cache[layer_idx][kv_idx]["v"][:self.max_seq_len].bfloat16().contiguous()
            self.kv_cache_dev[layer_idx][kv_idx]["k_t"] = self._to_device(k_t_full)
            self.kv_cache_dev[layer_idx][kv_idx]["v"] = self._to_device(v_full)

            # Read device-resident cache for attention
            k_t_dev = self.kv_cache_dev[layer_idx][kv_idx]["k_t"]
            v_dev_cache = self.kv_cache_dev[layer_idx][kv_idx]["v"]

            for q_local in range(heads_per_group):
                q_idx = kv_idx * heads_per_group + q_local

                # Per-head Q projection + RoPE (weights pre-scaled by 1/sqrt(d))
                q_dev = self._alloc_zeros((TILE, self.head_dim))
                linear_bias_kernel(normed, w["q_head_weights"][q_idx],
                                   w["q_head_biases"][q_idx], q_dev)
                q_rot_dev = self._alloc_zeros((TILE, self.head_dim))
                rope_kernel(q_dev, self._decode_cos_dev, self._decode_sin_dev, q_rot_dev)

                # Attention on device: scores → softmax → output
                scores_dev = self._alloc_zeros((TILE, cache_len))
                linear_kernel(q_rot_dev, k_t_dev, scores_dev)

                weights_dev = device_softmax(
                    scores_dev, self._decode_mask_dev,
                    self.ones_scaler_device, self.device
                )

                head_out_dev = self._alloc_zeros((TILE, self.head_dim))
                linear_kernel(weights_dev, v_dev_cache, head_out_dev)

                # Collect head output (4KB pull, accumulate on host)
                head_out_host = ttnn.to_torch(head_out_dev)
                col_start = q_idx * self.head_dim
                col_end = col_start + self.head_dim
                self._decode_attn_buf[:, col_start:col_end] = head_out_host

        # Upload concatenated attention (56KB, single transfer)
        attn_out_device = self._to_device(self._decode_attn_buf)

        proj_out = self._alloc_zeros((TILE, self.hidden_size))
        linear_kernel(attn_out_device, w["o_proj_weight"], proj_out)

        post_attn = self._alloc_zeros((TILE, self.hidden_size))
        add_kernel(x_device, proj_out, post_attn)

        return self._run_mlp(post_attn, layer_idx, TILE, decode=True)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def prefill(self, input_ids):
        """Run prefill: embedding → 24 layers → norm → lm_head. Populates KV cache.

        Args:
            input_ids: list of token IDs

        Returns:
            logits: [seq_len, vocab_size] torch tensor on host
        """
        seq_len = len(input_ids)
        assert seq_len <= self.max_seq_len

        padded_seq = ((seq_len + TILE - 1) // TILE) * TILE

        # Init KV cache
        self.init_kv_cache()

        # Embedding (host-side, no device calls)
        ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        x = self.embed_weight[ids_tensor]
        if padded_seq > seq_len:
            x = torch.nn.functional.pad(x, (0, 0, 0, padded_seq - seq_len))

        # Causal mask
        causal_mask = torch.triu(
            torch.full((padded_seq, padded_seq), float("-inf")), diagonal=1
        )

        # All device work in one suppress block
        with self._suppress_output():
            x_device = self._to_device(x.bfloat16())

            for layer_idx in range(self.num_layers):
                x_device = self.transformer_layer_prefill(
                    x_device, layer_idx, seq_len, causal_mask
                )

            self.cache_pos = seq_len

            x_device = self._rmsnorm_host(x_device, self.final_norm_weight)
            x_host = ttnn.to_torch(x_device).float()

        # lm_head on host
        embed_w = self.embed_weight[:self.vocab_size]
        logits = x_host[:seq_len] @ embed_w.t()

        return logits

    def decode_step(self, token_id, pos=None):
        """Run one decode step: single token through 24 layers.

        Args:
            token_id: int, the token to process
            pos: int, position in sequence (default: self.cache_pos)

        Returns:
            logits: [1, vocab_size] torch tensor on host
        """
        if pos is None:
            pos = self.cache_pos

        assert pos < self.max_seq_len, f"pos {pos} >= max_seq_len {self.max_seq_len}"

        # Embedding (host-side)
        x = self.embed_weight[token_id:token_id+1]
        x_padded = torch.zeros(TILE, self.hidden_size, dtype=torch.bfloat16)
        x_padded[0] = x[0].bfloat16()

        # Prepare position-specific RoPE cos/sin tiles (8KB upload, once per token)
        cos_pos = torch.ones(TILE, self.head_dim, dtype=torch.bfloat16)
        sin_pos = torch.zeros(TILE, self.head_dim, dtype=torch.bfloat16)
        cos_pos[0] = self.rope_cos[pos].bfloat16()
        sin_pos[0] = self.rope_sin[pos].bfloat16()

        # Attention buffer for head concatenation
        self._decode_attn_buf = torch.zeros(TILE, self.hidden_size, dtype=torch.bfloat16)

        # All device work in one suppress block
        with self._suppress_output():
            x_device = self._to_device(x_padded)
            self._decode_cos_dev = self._to_device(cos_pos)
            self._decode_sin_dev = self._to_device(sin_pos)

            for layer_idx in range(self.num_layers):
                x_device = self.transformer_layer_decode(x_device, layer_idx, pos)

            self.cache_pos = pos + 1

            x_device = self._rmsnorm_device(x_device, self.final_norm_weight)
            x_host = ttnn.to_torch(x_device).float()

        # lm_head on host
        embed_w = self.embed_weight[:self.vocab_size]
        logits = x_host[:1] @ embed_w.t()

        return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=0.0):
        """Prefill + decode loop. Yields generated token IDs.

        Args:
            input_ids: list of prompt token IDs
            max_new_tokens: max tokens to generate
            temperature: 0.0 for greedy, >0 for sampling
        """
        # Prefill
        logits = self.prefill(input_ids)

        # First generated token
        if temperature == 0.0:
            next_token = logits[-1].argmax().item()
        else:
            probs = torch.nn.functional.softmax(logits[-1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        yield next_token

        # Decode loop
        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(next_token)

            if temperature == 0.0:
                next_token = logits[0].argmax().item()
            else:
                probs = torch.nn.functional.softmax(logits[0] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            yield next_token
