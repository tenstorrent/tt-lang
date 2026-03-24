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
from kernels.linear import (
    linear_kernel, linear_bias_kernel,
    down_proj_partial_kernel, down_proj_reduce_kernel, DOWN_K_SPLITS,
)
from kernels.elementwise import add_kernel, silu_mul_kernel
from kernels.rope import batch_rope_kernel
from kernels.rmsnorm import fused_device_rmsnorm, fused_rmsnorm_kernel, rmsnorm_mul_kernel
from kernels.fused_attn import fused_attn_head_kernel
from kernels.group_attn import head_attn_kernels
from kernels.multicore_attn import (
    partial_attn_kernels, reduce_attn_kernels,
    parallel_partial_g0, parallel_partial_g1,
    parallel_reduce_g0, parallel_reduce_g1,
    TOTAL_PAR_CORES,
)
from kernels.kv_cache_update import get_kv_cache_update_kernel
from kernels.kv_cache_update_traced import kv_cache_update_traced, build_full_masks
from kernels.argmax import (
    DeviceArgmax, parallel_max_reduce_kernel, global_max_reduce_kernel,
    parallel_index_find_kernel, GRID_Y as ARGMAX_GRID_Y, GRID_X as ARGMAX_GRID_X,
)
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

        # Embedding weights (host-side for lookup + device-side for lm_head)
        self.embed_weight = self.ckpt["embed_weight"].float()
        # lm_head weight = embed_weight^T [hidden_size, vocab_size] on device
        self.lm_head_weight_device = self._to_device(
            self.embed_weight[:self.vocab_size].bfloat16().t().contiguous(),
        )

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

        # Device-side argmax for greedy decode (avoids 9.7MB logits readback)
        self.device_argmax = DeviceArgmax(device, vocab_size=self.vocab_size)

        # Quiet mode: suppress compilation output
        self.quiet = False
        self._compile_log = os.path.join(
            os.path.dirname(__file__), "compile.log"
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _to_device(self, tensor, dtype=None):
        return ttnn.from_torch(
            tensor, dtype=dtype or ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _alloc_zeros(self, shape, l1=False):
        t = torch.zeros(shape, dtype=torch.bfloat16)
        mem_cfg = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=mem_cfg,
        )

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

        # Pre-compute row/column masks for on-device KV cache updates
        self._row_masks = []
        self._inv_row_masks = []
        self._col_masks = []
        self._inv_col_masks = []
        for p in range(TILE):
            rm = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
            rm[p, :] = 1.0
            self._row_masks.append(self._to_device(rm))
            self._inv_row_masks.append(self._to_device(1.0 - rm))
            cm = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
            cm[:, p] = 1.0
            self._col_masks.append(self._to_device(cm))
            self._inv_col_masks.append(self._to_device(1.0 - cm))

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
            normed2 = fused_device_rmsnorm(post_attn_device, w["post_attention_layernorm_weight"],
                                            self.mean_scaler_device, self.device)
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
        """Decode: fully on-device. 29 kernel calls per layer, zero host transfers.

        Fused RMSNorm (1) + QKV projections (3) + batch RoPE (2) +
        KV cache update (1) + per-head attention (14, flash/online softmax,
        Q read at column offsets) + O proj (1) + add (1) + fused RMSNorm (1) +
        MLP (4) + add (1) = 29 calls. All on device.
        """
        w = self.layer_weights[layer_idx]
        heads_per_group = self.num_q_heads // self.num_kv_heads

        # 1. Fused RMSNorm (1 call)
        normed = fused_device_rmsnorm(x_device, w["input_layernorm_weight"],
                                       self.mean_scaler_device, self.device)

        # 2. Combined Q/K/V projections (3 calls, multi-core)
        q_out = self._alloc_zeros((TILE, self.hidden_size))
        linear_bias_kernel(normed, w["q_proj_weight"], w["q_proj_bias"], q_out)

        k_out = self._alloc_zeros((TILE, self.num_kv_heads * self.head_dim))
        linear_bias_kernel(normed, w["k_proj_weight"], w["k_proj_bias"], k_out)

        v_out = self._alloc_zeros((TILE, self.num_kv_heads * self.head_dim))
        linear_bias_kernel(normed, w["v_proj_weight"], w["v_proj_bias"], v_out)

        # 3. Batch RoPE (2 calls) — all heads at once
        q_rot = self._alloc_zeros((TILE, self.hidden_size))
        batch_rope_kernel(q_out, self._decode_cos_dev, self._decode_sin_dev, q_rot)

        k_rot = self._alloc_zeros((TILE, self.num_kv_heads * self.head_dim))
        batch_rope_kernel(k_out, self._decode_cos_dev, self._decode_sin_dev, k_rot)

        # 4. KV cache update — fully on device, zero host transfers
        tile_slot = pos // TILE
        sub_pos = pos % TILE
        update_kernel = get_kv_cache_update_kernel(tile_slot)
        update_kernel(
            k_rot, v_out,
            self.kv_cache_dev[layer_idx][0]["k_t"],
            self.kv_cache_dev[layer_idx][1]["k_t"],
            self.kv_cache_dev[layer_idx][0]["v"],
            self.kv_cache_dev[layer_idx][1]["v"],
            self._row_masks[sub_pos], self._inv_row_masks[sub_pos],
            self._col_masks[sub_pos], self._inv_col_masks[sub_pos],
        )

        # 5. Decode mask (created once per token, shared across all layers)
        decode_mask_dev = self._decode_mask_dev

        # 6. Parallel attention (4 calls: 2× partial-28-core + 2× reduce-7-core)
        attn_out_device = self._alloc_zeros((TILE, self.hidden_size))
        part_size = TOTAL_PAR_CORES * TILE  # 28 * 32 = 896
        part_m = self._alloc_zeros((TILE, part_size))
        part_d = self._alloc_zeros((TILE, part_size))
        part_o0 = self._alloc_zeros((TILE, part_size))
        part_o1 = self._alloc_zeros((TILE, part_size))

        for kv_idx, (p_kern, r_kern) in enumerate([
            (parallel_partial_g0, parallel_reduce_g0),
            (parallel_partial_g1, parallel_reduce_g1),
        ]):
            p_kern(q_rot, self.kv_cache_dev[layer_idx][kv_idx]["k_t"],
                   self.kv_cache_dev[layer_idx][kv_idx]["v"],
                   decode_mask_dev, self.ones_scaler_device,
                   self.attn_scale_device,
                   part_m, part_d, part_o0, part_o1)
            r_kern(part_m, part_d, part_o0, part_o1, attn_out_device)

        # 7. O projection + residual (2 calls)
        proj_out = self._alloc_zeros((TILE, self.hidden_size))
        linear_kernel(attn_out_device, w["o_proj_weight"], proj_out)

        post_attn = self._alloc_zeros((TILE, self.hidden_size))
        add_kernel(x_device, proj_out, post_attn)

        # 8. Fused RMSNorm + MLP (1 + 4 + 1 calls)
        return self._run_mlp(post_attn, layer_idx, TILE, decode=True)

    # -----------------------------------------------------------------
    # Traced decode (pre-allocated buffers + trace capture/replay)
    # -----------------------------------------------------------------

    def _init_trace_buffers(self):
        """Pre-allocate all intermediate tensors for traced decode."""
        H = self.hidden_size       # 896
        I = self.intermediate_size  # 4864
        KV = self.num_kv_heads * self.head_dim  # 128

        self._tb = {
            # All intermediates in L1 — avoids DRAM round-trips between kernels.
            # Total ~40KB/core, well within 1.5MB L1 budget.
            "x_a": self._alloc_zeros((TILE, H), l1=True),
            "x_b": self._alloc_zeros((TILE, H), l1=True),
            "normed": self._alloc_zeros((TILE, H), l1=True),
            "q_out": self._alloc_zeros((TILE, H), l1=True),
            "k_out": self._alloc_zeros((TILE, KV), l1=True),
            "v_out": self._alloc_zeros((TILE, KV), l1=True),
            "q_rot": self._alloc_zeros((TILE, H), l1=True),
            "k_rot": self._alloc_zeros((TILE, KV), l1=True),
            "attn_out": self._alloc_zeros((TILE, H), l1=True),
            "part_m": self._alloc_zeros((TILE, TOTAL_PAR_CORES * TILE), l1=True),
            "part_d": self._alloc_zeros((TILE, TOTAL_PAR_CORES * TILE), l1=True),
            "part_o0": self._alloc_zeros((TILE, TOTAL_PAR_CORES * TILE), l1=True),
            "part_o1": self._alloc_zeros((TILE, TOTAL_PAR_CORES * TILE), l1=True),
            "proj_out": self._alloc_zeros((TILE, H), l1=True),
            "post_attn": self._alloc_zeros((TILE, H), l1=True),
            "normed2": self._alloc_zeros((TILE, H), l1=True),
            "gate_out": self._alloc_zeros((TILE, I), l1=True),
            "up_out": self._alloc_zeros((TILE, I), l1=True),
            "mlp_hidden": self._alloc_zeros((TILE, I), l1=True),
            "mlp_out": self._alloc_zeros((TILE, H), l1=True),
            "final_out": self._alloc_zeros((TILE, H), l1=True),
            "cos_dev": self._alloc_zeros((TILE, self.head_dim)),
            "sin_dev": self._alloc_zeros((TILE, self.head_dim)),
            "mask_dev": self._alloc_zeros((TILE, self.padded_max_seq)),
            "logits": self._alloc_zeros((TILE, self.vocab_size)),
            # Argmax pipeline buffers (included in trace for zero dispatch overhead)
            # K-split down_proj partial buffer
            "down_proj_partial": self._alloc_zeros(
                (TILE, (self.hidden_size // TILE) * DOWN_K_SPLITS * TILE), l1=True),
            # Argmax pipeline buffers (included in trace for zero dispatch overhead)
            "argmax_scaler": self._to_device(
                torch.ones(TILE, TILE, dtype=torch.bfloat16)),
            "argmax_max_out": self._alloc_zeros(
                (TILE, ARGMAX_GRID_Y * ARGMAX_GRID_X * TILE)),
            "argmax_global_max": self._alloc_zeros((TILE, TILE)),
            "argmax_index_out": self._alloc_zeros(
                (TILE, ARGMAX_GRID_Y * ARGMAX_GRID_X * TILE)),
            # KV cache update masks (full-width, updated per token)
            "kv_row_masks": self._alloc_zeros((TILE, self.padded_max_seq)),
            "kv_irow_masks": self._to_device(
                torch.ones(TILE, self.padded_max_seq, dtype=torch.bfloat16)),
            "kv_col_masks": self._alloc_zeros((TILE, self.padded_max_seq)),
            "kv_icol_masks": self._to_device(
                torch.ones(TILE, self.padded_max_seq, dtype=torch.bfloat16)),
        }
        # Per-layer k_rot and v_out buffers (for KV cache update after trace)
        self._tb_k_rot = [self._alloc_zeros((TILE, KV)) for _ in range(self.num_layers)]
        self._tb_v_out = [self._alloc_zeros((TILE, KV)) for _ in range(self.num_layers)]
        self._trace_id = None
        self._trace_kv_ready = False  # Set True after first trace produces k_rot/v_out

    def _traced_decode_layers(self):
        """Run all 24 decode layers using pre-allocated buffers. No allocations.

        KV cache update is included in the trace — uses a SINGLE kernel variant
        that always operates on tile [0, col] of small staging tensors.
        The caller copies the right cache tile to/from staging before/after trace.

        Actually: KV cache is updated using per-layer k_rot/v_out buffers,
        so each layer writes its projections to a dedicated buffer.
        Cache update runs after trace replay using these saved buffers.
        """
        tb = self._tb
        heads_per_group = self.num_q_heads // self.num_kv_heads

        cur_in, cur_out = "x_a", "x_b"

        for layer_idx in range(self.num_layers):
            w = self.layer_weights[layer_idx]

            # 1. RMSNorm
            fused_rmsnorm_kernel(tb[cur_in], self.mean_scaler_device,
                                  w["input_layernorm_weight"], tb["normed"])

            # 2. QKV projections (V goes to per-layer buffer for KV cache update)
            linear_bias_kernel(tb["normed"], w["q_proj_weight"], w["q_proj_bias"], tb["q_out"])
            linear_bias_kernel(tb["normed"], w["k_proj_weight"], w["k_proj_bias"], tb["k_out"])
            linear_bias_kernel(tb["normed"], w["v_proj_weight"], w["v_proj_bias"],
                              self._tb_v_out[layer_idx])

            # 3. Batch RoPE (K goes to per-layer buffer for KV cache update)
            batch_rope_kernel(tb["q_out"], tb["cos_dev"], tb["sin_dev"], tb["q_rot"])
            batch_rope_kernel(tb["k_out"], tb["cos_dev"], tb["sin_dev"],
                              self._tb_k_rot[layer_idx])

            # 4. KV cache update (inside trace — masks encode target position)
            kv_cache_update_traced(
                self._tb_k_rot[layer_idx], self._tb_v_out[layer_idx],
                self.kv_cache_dev[layer_idx][0]["k_t"],
                self.kv_cache_dev[layer_idx][1]["k_t"],
                self.kv_cache_dev[layer_idx][0]["v"],
                self.kv_cache_dev[layer_idx][1]["v"],
                tb["kv_row_masks"], tb["kv_irow_masks"],
                tb["kv_col_masks"], tb["kv_icol_masks"],
            )

            # 5. Parallel attention (28-core partial + 7-core reduce, per KV group)
            for kv_idx, (p_kern, r_kern) in enumerate([
                (parallel_partial_g0, parallel_reduce_g0),
                (parallel_partial_g1, parallel_reduce_g1),
            ]):
                p_kern(tb["q_rot"], self.kv_cache_dev[layer_idx][kv_idx]["k_t"],
                       self.kv_cache_dev[layer_idx][kv_idx]["v"],
                       tb["mask_dev"], self.ones_scaler_device,
                       self.attn_scale_device,
                       tb["part_m"], tb["part_d"], tb["part_o0"], tb["part_o1"])
                r_kern(tb["part_m"], tb["part_d"], tb["part_o0"], tb["part_o1"],
                       tb["attn_out"])

            # 6. O projection + residual
            linear_kernel(tb["attn_out"], w["o_proj_weight"], tb["proj_out"])
            add_kernel(tb[cur_in], tb["proj_out"], tb["post_attn"])

            # 7. MLP
            fused_rmsnorm_kernel(tb["post_attn"], self.mean_scaler_device,
                                  w["post_attention_layernorm_weight"], tb["normed2"])
            linear_kernel(tb["normed2"], w["gate_proj_weight"], tb["gate_out"])
            linear_kernel(tb["normed2"], w["up_proj_weight"], tb["up_out"])
            silu_mul_kernel(tb["gate_out"], tb["up_out"], tb["mlp_hidden"])
            down_proj_partial_kernel(
                tb["mlp_hidden"], w["down_proj_weight"], tb["down_proj_partial"])
            down_proj_reduce_kernel(tb["down_proj_partial"], tb["mlp_out"])
            add_kernel(tb["post_attn"], tb["mlp_out"], tb[cur_out])

            cur_in, cur_out = cur_out, cur_in

        # Final norm + lm_head
        fused_rmsnorm_kernel(tb[cur_in], self.mean_scaler_device,
                              self.final_norm_weight, tb["final_out"])
        linear_kernel(tb["final_out"], self.lm_head_weight_device, tb["logits"])

        # Argmax pipeline (inside trace — zero dispatch overhead)
        parallel_max_reduce_kernel(
            tb["logits"], tb["argmax_scaler"], tb["argmax_max_out"])
        global_max_reduce_kernel(
            tb["argmax_max_out"], tb["argmax_scaler"], tb["argmax_global_max"])
        parallel_index_find_kernel(
            tb["logits"], tb["argmax_global_max"], tb["argmax_index_out"])

    def _read_traced_argmax(self):
        """Read the argmax result from the trace buffer (tiny readback).

        The argmax_index_out tensor has per-core results with tile_col at
        [0,1] and local_col at [0,0] for each core's tile.
        """
        num_cores = ARGMAX_GRID_Y * ARGMAX_GRID_X
        # Pre-allocate host buffer on first call
        if not hasattr(self, '_argmax_host_buf'):
            out_cols = num_cores * TILE
            self._argmax_host_buf = ttnn.from_torch(
                torch.zeros(TILE, out_cols, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            )
        ttnn.copy_device_to_host_tensor(
            self._tb["argmax_index_out"], self._argmax_host_buf)
        idx_bf16 = self._argmax_host_buf.to_torch().to(torch.bfloat16)
        idx_raw = idx_bf16.view(torch.int16).to(torch.int64)
        local_cols = idx_raw[0, ::TILE].numpy()[:num_cores] & 0xFFFF
        tile_cols = idx_raw[0, 1::TILE].numpy()[:num_cores] & 0xFFFF
        core_indices = tile_cols * TILE + local_cols
        valid_mask = (tile_cols > 0) | (local_cols > 0)
        if valid_mask.any():
            return int(core_indices[valid_mask].min())
        return 0

    def _capture_decode_trace(self):
        """Capture the full decode sequence as a trace."""
        print("  Capturing decode trace (warmup)...", end="", flush=True)
        self._traced_decode_layers()
        ttnn.synchronize_device(self.device)
        print(" compiling...", end="", flush=True)

        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._traced_decode_layers()
        ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)
        print(" done.")

    def decode_step_traced(self, token_id, pos=None, greedy=False):
        """Decode one token using trace replay. Much faster dispatch.

        Args:
            token_id: input token ID
            pos: position in sequence (default: self.cache_pos)
            greedy: if True, return token ID via device argmax (no logits readback)

        Returns:
            If greedy: int token ID
            Else: [1, vocab_size] logits tensor (host)
        """
        if pos is None:
            pos = self.cache_pos

        # Prepare inputs on host
        x = self.embed_weight[token_id:token_id+1]
        x_padded = torch.zeros(TILE, self.hidden_size, dtype=torch.bfloat16)
        x_padded[0] = x[0].bfloat16()

        cos_pos = torch.ones(TILE, self.head_dim, dtype=torch.bfloat16)
        sin_pos = torch.zeros(TILE, self.head_dim, dtype=torch.bfloat16)
        cos_pos[0] = self.rope_cos[pos].bfloat16()
        sin_pos[0] = self.rope_sin[pos].bfloat16()

        # Attention mask: attend to 0..pos (KV cache updated inside trace now)
        mask_t = torch.full((TILE, self.padded_max_seq), float("-inf"), dtype=torch.bfloat16)
        mask_t[0, :pos + 1] = 0.0

        # KV cache masks: encode target position for traced kernel
        kv_row_m, kv_irow_m, kv_col_m, kv_icol_m = build_full_masks(pos)

        with self._suppress_output():
            # Copy all inputs to pre-allocated device tensors
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(x_padded, layout=ttnn.TILE_LAYOUT), self._tb["x_a"])
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(cos_pos, layout=ttnn.TILE_LAYOUT), self._tb["cos_dev"])
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(sin_pos, layout=ttnn.TILE_LAYOUT), self._tb["sin_dev"])
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(mask_t, layout=ttnn.TILE_LAYOUT), self._tb["mask_dev"])
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(kv_row_m, layout=ttnn.TILE_LAYOUT), self._tb["kv_row_masks"])
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(kv_irow_m, layout=ttnn.TILE_LAYOUT), self._tb["kv_irow_masks"])
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(kv_col_m, layout=ttnn.TILE_LAYOUT), self._tb["kv_col_masks"])
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(kv_icol_m, layout=ttnn.TILE_LAYOUT), self._tb["kv_icol_masks"])

            # Replay the full decode (24 layers + lm_head + argmax pipeline)
            ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)
            self._trace_kv_ready = True

            if greedy:
                # Argmax result is already in the trace buffer — tiny readback
                result = self._read_traced_argmax()
            else:
                logits_host = ttnn.to_torch(self._tb["logits"]).float()
                result = logits_host[:1]

        self.cache_pos = pos + 1
        return result

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

            # Sync host KV cache to device-resident tensors for decode
            for li in range(self.num_layers):
                for kv_idx in range(self.num_kv_heads):
                    k_t = self.kv_cache[li][kv_idx]["k"][:self.max_seq_len].bfloat16().t().contiguous()
                    v = self.kv_cache[li][kv_idx]["v"][:self.max_seq_len].bfloat16().contiguous()
                    self.kv_cache_dev[li][kv_idx]["k_t"] = self._to_device(k_t)
                    self.kv_cache_dev[li][kv_idx]["v"] = self._to_device(v)

            x_device = self._rmsnorm_host(x_device, self.final_norm_weight)
            x_host = ttnn.to_torch(x_device).float()

        # lm_head on host
        embed_w = self.embed_weight[:self.vocab_size]
        logits = x_host[:seq_len] @ embed_w.t()

        return logits

    def decode_step(self, token_id, pos=None, greedy=False):
        """Run one decode step: single token through 24 layers.

        Args:
            token_id: int, the token to process
            pos: int, position in sequence (default: self.cache_pos)
            greedy: if True, return token ID via device argmax (no logits readback)

        Returns:
            If greedy: int token ID
            Else: [1, vocab_size] logits tensor (host)
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

            # Decode mask — create once, reuse across all 24 layers
            cache_len = self.padded_max_seq
            decode_mask = torch.full((TILE, cache_len), float("-inf"), dtype=torch.bfloat16)
            decode_mask[0, :pos + 1] = 0.0
            self._decode_mask_dev = self._to_device(decode_mask)

            for layer_idx in range(self.num_layers):
                x_device = self.transformer_layer_decode(x_device, layer_idx, pos)

            self.cache_pos = pos + 1

            x_device = fused_device_rmsnorm(x_device, self.final_norm_weight,
                                             self.mean_scaler_device, self.device)

            # lm_head on device
            logits_dev = self._alloc_zeros((TILE, self.vocab_size))
            linear_kernel(x_device, self.lm_head_weight_device, logits_dev)

            if greedy:
                result = self.device_argmax(logits_dev)
            else:
                logits_host = ttnn.to_torch(logits_dev).float()
                result = logits_host[:1]

        return result

    def generate(self, input_ids, max_new_tokens=50, temperature=0.0, use_trace=False):
        """Prefill + decode loop. Yields generated token IDs.

        Args:
            input_ids: list of prompt token IDs
            max_new_tokens: max tokens to generate
            temperature: 0.0 for greedy, >0 for sampling
            use_trace: if True, use trace capture/replay for faster decode
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

        # Set up trace if requested
        if use_trace and not getattr(self, '_trace_id', None):
            with self._suppress_output():
                self._init_trace_buffers()
                self._capture_decode_trace()

        # Decode loop
        greedy = (temperature == 0.0)
        decode_fn = self.decode_step_traced if use_trace else self.decode_step
        for _ in range(max_new_tokens - 1):
            result = decode_fn(next_token, greedy=greedy)

            if greedy:
                next_token = result  # device argmax returns int directly
            else:
                logits = result
                probs = torch.nn.functional.softmax(logits[0] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            yield next_token
