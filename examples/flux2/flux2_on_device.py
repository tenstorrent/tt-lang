# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2-klein-4B -- On-Device Transformer (zero host round-trips).

All transformer activations stay on Blackhole DRAM between operations.
Only the final output transfers back to host for VAE decode.

Performance target: 1ms/block warm -> 25ms per denoising step.
"""

import argparse
import time
import torch
import torch.nn.functional as F
import ttnn


def to_device(t, device):
    """Load a CPU tensor to device. TTNN handles tile padding internally."""
    t = t.to(torch.bfloat16).contiguous()
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class Flux2OnDevice:
    """FLUX.2 transformer running entirely on Blackhole -- no host round-trips."""

    def __init__(self, cpu_transformer, tt_device):
        self.device = tt_device
        self.config = cpu_transformer.config
        self.dtype = torch.bfloat16
        self.D = self.config.num_attention_heads * self.config.attention_head_dim  # 3072
        self.H = self.config.num_attention_heads   # 24
        self.HD = self.config.attention_head_dim   # 128
        self.n_double = self.config.num_layers      # 5
        self.n_single = self.config.num_single_layers  # 20
        self.in_channels = self.config.in_channels  # 128
        self.joint_dim = self.config.joint_attention_dim  # 7680
        self.rope_axes = self.config.axes_dims_rope
        self.rope_theta = self.config.rope_theta
        self.scale = 1.0 / (self.HD ** 0.5)

        # Forward diffusers attrs
        self.cache_context = cpu_transformer.cache_context

        # Load all weights to device
        sd = cpu_transformer.state_dict()
        self.w = {}
        print("  Loading weights to device DRAM...")
        t0 = time.time()
        for key, tensor in sd.items():
            if tensor.ndim >= 2:
                self.w[key] = to_device(tensor, tt_device)
            else:
                # 1D tensors (norm weights) -- keep on CPU for now, load as needed
                self.w[key] = tensor.to(torch.bfloat16)
        self._w_transposed = {}  # Cache for transposed weights
        n_dev = sum(1 for v in self.w.values() if not isinstance(v, torch.Tensor))
        # Also load 1D norm weights to device for QK-norm
        self._norm_cache = {}
        for key, tensor in self.w.items():
            if isinstance(tensor, torch.Tensor) and 'norm' in key:
                self._norm_cache[key] = to_device(tensor.unsqueeze(0), tt_device)
        print(f"  {n_dev} matrices + {len(self._norm_cache)} norm weights on device ({time.time()-t0:.1f}s)")

    def _matmul_w(self, x, key):
        """x @ W^T on device (weights stored as [out, in], transposed and cached)."""
        if key not in self._w_transposed:
            self._w_transposed[key] = ttnn.transpose(self.w[key], -2, -1)
        return ttnn.matmul(x, self._w_transposed[key])

    def _qk_norm(self, q, k, prefix):
        """Per-head RMSNorm on Q and K with learned scale weights (all on device)."""
        D, H, HD = self.D, self.H, self.HD
        S = q.shape[0]

        q_heads = ttnn.reshape(q, (S * H, HD))
        k_heads = ttnn.reshape(k, (S * H, HD))

        q_normed = ttnn.rms_norm(q_heads)
        k_normed = ttnn.rms_norm(k_heads)

        # Apply cached learned scale weights
        nq_key = f"{prefix}.norm_q.weight"
        nk_key = f"{prefix}.norm_k.weight"
        if nq_key in self._norm_cache:
            q_normed = ttnn.mul(q_normed, ttnn.repeat(self._norm_cache[nq_key], ttnn.Shape([S * H, 1])))
        if nk_key in self._norm_cache:
            k_normed = ttnn.mul(k_normed, ttnn.repeat(self._norm_cache[nk_key], ttnn.Shape([S * H, 1])))

        return ttnn.reshape(q_normed, (S, D)), ttnn.reshape(k_normed, (S, D))

    def _qk_norm_keys(self, q, k, q_weight_key, k_weight_key):
        """Per-head RMSNorm with explicit weight keys (all on device)."""
        D, H, HD = self.D, self.H, self.HD
        S = q.shape[0]

        q_heads = ttnn.reshape(q, (S * H, HD))
        k_heads = ttnn.reshape(k, (S * H, HD))

        q_normed = ttnn.rms_norm(q_heads)
        k_normed = ttnn.rms_norm(k_heads)

        if q_weight_key in self._norm_cache:
            q_normed = ttnn.mul(q_normed, ttnn.repeat(self._norm_cache[q_weight_key], ttnn.Shape([S * H, 1])))
        if k_weight_key in self._norm_cache:
            k_normed = ttnn.mul(k_normed, ttnn.repeat(self._norm_cache[k_weight_key], ttnn.Shape([S * H, 1])))

        return ttnn.reshape(q_normed, (S, D)), ttnn.reshape(k_normed, (S, D))

    def _apply_rope(self, q, k, rope_cos_half, rope_sin_half, H, HD):
        """Apply RoPE to Q and K entirely on device.

        q, k: (S, D) on device -- flat multi-head
        rope_cos_half, rope_sin_half: (S*H, HD/2) on device -- pre-expanded
        """
        if rope_cos_half is None or rope_sin_half is None:
            return q, k

        S = q.shape[0]
        D = H * HD
        SH = S * H

        # Reshape to per-head: (S, D) -> (S*H, HD)
        q_h = ttnn.reshape(q, (SH, HD))
        k_h = ttnn.reshape(k, (SH, HD))

        # Extract even/odd
        q_even, q_odd = q_h[:, 0::2], q_h[:, 1::2]
        k_even, k_odd = k_h[:, 0::2], k_h[:, 1::2]

        # Apply rotation
        qe = ttnn.sub(ttnn.mul(q_even, rope_cos_half), ttnn.mul(q_odd, rope_sin_half))
        qo = ttnn.add(ttnn.mul(q_even, rope_sin_half), ttnn.mul(q_odd, rope_cos_half))
        ke = ttnn.sub(ttnn.mul(k_even, rope_cos_half), ttnn.mul(k_odd, rope_sin_half))
        ko = ttnn.add(ttnn.mul(k_even, rope_sin_half), ttnn.mul(k_odd, rope_cos_half))

        # Interleave even/odd back
        q_rot = ttnn.reshape(ttnn.concat([ttnn.reshape(qe, (SH, HD//2, 1)),
                                           ttnn.reshape(qo, (SH, HD//2, 1))], dim=-1), (SH, HD))
        k_rot = ttnn.reshape(ttnn.concat([ttnn.reshape(ke, (SH, HD//2, 1)),
                                           ttnn.reshape(ko, (SH, HD//2, 1))], dim=-1), (SH, HD))

        return ttnn.reshape(q_rot, (S, D)), ttnn.reshape(k_rot, (S, D))

    def _single_block(self, x, mod, rope_cos, rope_sin, block_idx):
        """Single-stream block entirely on device."""
        p = f"single_transformer_blocks.{block_idx}"
        D, H, HD = self.D, self.H, self.HD

        shift = mod[:, :D]
        scale = mod[:, D:2*D]
        gate = mod[:, 2*D:3*D]

        S = x.shape[0]
        residual = x

        # AdaLN
        normed = ttnn.layer_norm(x)
        modulated = ttnn.add(ttnn.mul(ttnn.add(scale, 1.0), normed), shift)

        # Fused QKV + MLP projection
        proj = self._matmul_w(modulated, f"{p}.attn.to_qkv_mlp_proj.weight")
        qkv = proj[:, :3*D]
        mlp_hidden = proj[:, 3*D:]

        q = qkv[:, :D]
        k = qkv[:, D:2*D]
        v = qkv[:, 2*D:3*D]

        # QK-norm (per-head RMSNorm with learned scale)
        q, k = self._qk_norm(q, k, f"{p}.attn")

        # RoPE
        q, k = self._apply_rope(q, k, rope_cos, rope_sin, H, HD)

        # Multi-head attention using TTNN's optimized SDPA
        # Reshape: (S, D) -> (1, S, H, HD) -> (1, H, S, HD) for SDPA
        q_4d = ttnn.transpose(ttnn.reshape(q, (1, S, H, HD)), 1, 2)
        k_4d = ttnn.transpose(ttnn.reshape(k, (1, S, H, HD)), 1, 2)
        v_4d = ttnn.transpose(ttnn.reshape(v, (1, S, H, HD)), 1, 2)
        attn_4d = ttnn.transformer.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False)
        # Reshape back: (1, H, S, HD) -> (1, S, H, HD) -> (S, D)
        attn_2d = ttnn.transpose(attn_4d, 1, 2)
        attn_flat = ttnn.reshape(attn_2d, (S, D))

        # SwiGLU
        MLP_2x = mlp_hidden.shape[-1]
        MLP_D = MLP_2x // 2
        mlp_gate = mlp_hidden[:, :MLP_D]
        mlp_up = mlp_hidden[:, MLP_D:]
        swiglu = ttnn.mul(ttnn.silu(mlp_gate), mlp_up)

        # Fused output projection
        combined = ttnn.concat([attn_flat, swiglu], dim=-1)
        output = self._matmul_w(combined, f"{p}.attn.to_out.weight")

        return ttnn.add(residual, ttnn.mul(gate, output))

    def _double_block(self, img, txt, mod_img, mod_txt, rope_cos, rope_sin, block_idx):
        """Double-stream block entirely on device."""
        p = f"transformer_blocks.{block_idx}"
        D, H, HD = self.D, self.H, self.HD

        si_msa, sc_msa, g_msa = mod_img[:, :D], mod_img[:, D:2*D], mod_img[:, 2*D:3*D]
        si_mlp, sc_mlp, g_mlp = mod_img[:, 3*D:4*D], mod_img[:, 4*D:5*D], mod_img[:, 5*D:]
        ti_msa, tc_msa, tg_msa = mod_txt[:, :D], mod_txt[:, D:2*D], mod_txt[:, 2*D:3*D]
        ti_mlp, tc_mlp, tg_mlp = mod_txt[:, 3*D:4*D], mod_txt[:, 4*D:5*D], mod_txt[:, 5*D:]

        S_img = img.shape[0]
        S_txt = txt.shape[0]

        # Image AdaLN + QKV
        ni = ttnn.add(ttnn.mul(ttnn.add(sc_msa, 1.0), ttnn.layer_norm(img)), si_msa)
        qi = self._matmul_w(ni, f"{p}.attn.to_q.weight")
        ki = self._matmul_w(ni, f"{p}.attn.to_k.weight")
        vi = self._matmul_w(ni, f"{p}.attn.to_v.weight")

        # Text AdaLN + QKV
        nt = ttnn.add(ttnn.mul(ttnn.add(tc_msa, 1.0), ttnn.layer_norm(txt)), ti_msa)
        qt = self._matmul_w(nt, f"{p}.attn.add_q_proj.weight")
        kt = self._matmul_w(nt, f"{p}.attn.add_k_proj.weight")
        vt = self._matmul_w(nt, f"{p}.attn.add_v_proj.weight")

        # QK-norm (image uses norm_q/norm_k, text uses norm_added_q/norm_added_k)
        qi, ki = self._qk_norm(qi, ki, f"{p}.attn")
        # Text uses different weight keys
        qt, kt = self._qk_norm_keys(qt, kt,
                                      f"{p}.attn.norm_added_q.weight",
                                      f"{p}.attn.norm_added_k.weight")

        # Concat for joint attention
        q = ttnn.concat([qt, qi], dim=0)
        k = ttnn.concat([kt, ki], dim=0)
        v = ttnn.concat([vt, vi], dim=0)
        S_total = S_txt + S_img

        # RoPE (on concatenated sequence)
        q, k = self._apply_rope(q, k, rope_cos, rope_sin, H, HD)

        # Multi-head attention using TTNN SDPA
        q_4d = ttnn.transpose(ttnn.reshape(q, (1, S_total, H, HD)), 1, 2)
        k_4d = ttnn.transpose(ttnn.reshape(k, (1, S_total, H, HD)), 1, 2)
        v_4d = ttnn.transpose(ttnn.reshape(v, (1, S_total, H, HD)), 1, 2)
        attn_4d = ttnn.transformer.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False)
        attn_2d = ttnn.transpose(attn_4d, 1, 2)
        attn_flat = ttnn.reshape(attn_2d, (S_total, D))

        # Split back
        txt_attn = attn_flat[:S_txt, :]
        img_attn = attn_flat[S_txt:, :]

        # Output projections + gated residuals
        img_proj = self._matmul_w(img_attn, f"{p}.attn.to_out.0.weight")
        txt_proj = self._matmul_w(txt_attn, f"{p}.attn.to_add_out.weight")
        img = ttnn.add(img, ttnn.mul(g_msa, img_proj))
        txt = ttnn.add(txt, ttnn.mul(tg_msa, txt_proj))

        # Image MLP
        ni2 = ttnn.add(ttnn.mul(ttnn.add(sc_mlp, 1.0), ttnn.layer_norm(img)), si_mlp)
        mlp_h = self._matmul_w(ni2, f"{p}.ff.linear_in.weight")
        MLP_D = mlp_h.shape[-1] // 2
        swiglu_i = ttnn.mul(ttnn.silu(mlp_h[:, :MLP_D]), mlp_h[:, MLP_D:])
        mlp_out_i = self._matmul_w(swiglu_i, f"{p}.ff.linear_out.weight")
        img = ttnn.add(img, ttnn.mul(g_mlp, mlp_out_i))

        # Text MLP
        nt2 = ttnn.add(ttnn.mul(ttnn.add(tc_mlp, 1.0), ttnn.layer_norm(txt)), ti_mlp)
        mlp_h_t = self._matmul_w(nt2, f"{p}.ff_context.linear_in.weight")
        MLP_D_t = mlp_h_t.shape[-1] // 2
        swiglu_t = ttnn.mul(ttnn.silu(mlp_h_t[:, :MLP_D_t]), mlp_h_t[:, MLP_D_t:])
        mlp_out_t = self._matmul_w(swiglu_t, f"{p}.ff_context.linear_out.weight")
        txt = ttnn.add(txt, ttnn.mul(tg_mlp, mlp_out_t))

        return img, txt

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                txt_ids=None, img_ids=None, guidance=None,
                joint_attention_kwargs=None, return_dict=False, **kwargs):
        """Full forward pass on device."""
        t0 = time.time()
        D = self.D

        # Move inputs to device
        hs = to_device(hidden_states.reshape(-1, hidden_states.shape[-1]), self.device)
        es = to_device(encoder_hidden_states.reshape(-1, encoder_hidden_states.shape[-1]), self.device)

        S_img = hs.shape[0]
        S_txt = es.shape[0]

        # Timestep embedding (small, compute on CPU then transfer)
        def sinusoidal(t, dim):
            half = dim // 2
            freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half).float() / half)
            args = t.float() * freqs
            return torch.cat([torch.cos(args), torch.sin(args)], -1).to(torch.bfloat16)

        t_emb = sinusoidal(timestep * 1000, 256).unsqueeze(0)
        t_emb_tt = to_device(t_emb, self.device)
        temb = ttnn.silu(self._matmul_w(t_emb_tt, "time_guidance_embed.timestep_embedder.linear_1.weight"))
        temb = self._matmul_w(temb, "time_guidance_embed.timestep_embedder.linear_2.weight")

        # Modulation params (on device)
        mod_act = ttnn.silu(temb)
        d_mod_img = self._matmul_w(mod_act, "double_stream_modulation_img.linear.weight")
        d_mod_txt = self._matmul_w(mod_act, "double_stream_modulation_txt.linear.weight")
        s_mod = self._matmul_w(mod_act, "single_stream_modulation.linear.weight")

        # Input projections (on device)
        hs = self._matmul_w(hs, "x_embedder.weight")
        es = self._matmul_w(es, "context_embedder.weight")

        # Precompute RoPE tables on CPU, then load to device
        if txt_ids is not None and img_ids is not None:
            # Squeeze batch dim if present
            _txt_ids = txt_ids.squeeze(0) if txt_ids.ndim == 3 else txt_ids
            _img_ids = img_ids.squeeze(0) if img_ids.ndim == 3 else img_ids
            all_ids = torch.cat([_txt_ids, _img_ids], dim=0)  # (S_total, 4)

            all_cos, all_sin = [], []
            for axis_idx, dim in enumerate(self.rope_axes):
                half_dim = dim // 2
                freqs = 1.0 / (self.rope_theta ** (
                    torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
                angles = all_ids[:, axis_idx:axis_idx+1].float() * freqs.unsqueeze(0)
                all_cos.append(torch.cos(angles).repeat_interleave(2, dim=1))
                all_sin.append(torch.sin(angles).repeat_interleave(2, dim=1))
            rope_cos_cpu = torch.cat(all_cos, dim=-1).to(torch.bfloat16)
            rope_sin_cpu = torch.cat(all_sin, dim=-1).to(torch.bfloat16)
            rope_cos = to_device(rope_cos_cpu, self.device)
            rope_sin = to_device(rope_sin_cpu, self.device)
            # Pre-expand for RoPE: repeat_interleave + extract half dims ONCE
            cos_rep = ttnn.repeat_interleave(rope_cos, self.H, dim=0)
            sin_rep = ttnn.repeat_interleave(rope_sin, self.H, dim=0)
            rope_cos_half = cos_rep[:, 0::2]  # (S_total*H, HD/2)
            rope_sin_half = sin_rep[:, 0::2]
        else:
            rope_cos_half = rope_sin_half = None

        # Pre-broadcast modulation params to match sequence lengths.
        # TTNN TILE_LAYOUT doesn't support implicit row broadcasting in binary ops,
        # so we use ttnn.repeat to expand [1, X] -> [S, X].
        S_total = S_img + S_txt
        d_mod_img = ttnn.repeat(d_mod_img, ttnn.Shape([S_img, 1]))
        d_mod_txt = ttnn.repeat(d_mod_txt, ttnn.Shape([S_txt, 1]))
        s_mod = ttnn.repeat(s_mod, ttnn.Shape([S_total, 1]))

        # Transformer blocks
        for i in range(self.n_double):
            hs, es = self._double_block(hs, es, d_mod_img, d_mod_txt,
                                         rope_cos_half, rope_sin_half, i)
        hs = ttnn.concat([es, hs], dim=0)
        for i in range(self.n_single):
            hs = self._single_block(hs, s_mod, rope_cos_half, rope_sin_half, i)
        hs = hs[S_txt:S_txt + S_img, :]
        S_cur = hs.shape[0]

        # Output norm + projection
        out_emb = self._matmul_w(ttnn.silu(temb), "norm_out.linear.weight")
        out_emb = ttnn.repeat(out_emb, ttnn.Shape([S_cur, 1]))
        out_scale = out_emb[:, :D]
        out_shift = out_emb[:, D:]
        hs = ttnn.add(ttnn.mul(ttnn.add(out_scale, 1.0), ttnn.layer_norm(hs)), out_shift)
        hs = self._matmul_w(hs, "proj_out.weight")

        # Transfer back to CPU -- only transfer needed!
        result_cpu = ttnn.to_torch(hs)[:S_img, :self.in_channels]
        result_cpu = result_cpu.reshape(1, S_img, self.in_channels)

        dt = time.time() - t0
        print(f"    On-device transformer: {dt*1000:.0f}ms "
              f"({self.n_double}d+{self.n_single}s, S_img={S_img}, S_txt={S_txt})")

        if return_dict:
            return type('Out', (), {'sample': result_cpu})()
        return (result_cpu,)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="A cat sitting on a rainbow")
    parser.add_argument("--height", default=128, type=int)
    parser.add_argument("--width", default=128, type=int)
    parser.add_argument("--steps", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    print("=" * 60)
    print("FLUX.2-klein-4B -- On-Device (zero host round-trips)")
    print("=" * 60)
    print(f"  {args.prompt} | {args.width}x{args.height} | {args.steps} steps\n")

    tt_device = ttnn.open_device(device_id=0)

    # Accelerate text encoder matmuls via F.linear hook (same as flux2_blackhole.py)
    text_enc_weight_cache = {}
    original_linear = F.linear

    def accelerated_linear(input, weight, bias=None):
        """Route large matmuls to Blackhole with weight caching."""
        if input.numel() < 512 or weight.numel() < 512:
            return original_linear(input, weight, bias)
        wid = id(weight)
        if wid not in text_enc_weight_cache:
            w = weight.to(torch.bfloat16).contiguous()
            w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            text_enc_weight_cache[wid] = (ttnn.transpose(w_tt, -2, -1), weight.shape)
            ttnn.deallocate(w_tt)
        w_t, w_shape = text_enc_weight_cache[wid]
        orig_shape = input.shape
        x_2d = input.reshape(-1, input.shape[-1]).to(torch.bfloat16).contiguous()
        S = x_2d.shape[0]
        x_tt = ttnn.from_torch(x_2d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y_tt = ttnn.matmul(x_tt, w_t)
        y = ttnn.to_torch(y_tt)[:S, :w_shape[0]]
        ttnn.deallocate(x_tt)
        ttnn.deallocate(y_tt)
        y = y.reshape(*orig_shape[:-1], w_shape[0])
        if bias is not None:
            y = y + bias.to(torch.bfloat16)
        return y

    try:
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(
            "/proj_sw/ssokorac/work/flux2-klein-4b", torch_dtype=torch.bfloat16)
        pipe.vae = pipe.vae.to(torch.float32)
        orig_decode = pipe.vae._decode
        pipe.vae._decode = lambda z, **kw: orig_decode(z.float(), **kw)

        # Replace transformer with on-device version
        on_dev = Flux2OnDevice(pipe.transformer, tt_device)
        pipe.transformer = on_dev

        # Activate F.linear hook for text encoder acceleration
        F.linear = accelerated_linear

        # Warmup
        print("\nWarmup...")
        t0 = time.time()
        _ = pipe(prompt="warmup", height=args.height, width=args.width,
                 num_inference_steps=1, guidance_scale=1.0,
                 generator=torch.Generator("cpu").manual_seed(0))
        print(f"  Warmup: {time.time()-t0:.1f}s")

        # Generate
        print(f"\nGenerating...")
        t0 = time.time()
        result = pipe(prompt=args.prompt, height=args.height, width=args.width,
                      num_inference_steps=args.steps, guidance_scale=1.0,
                      generator=torch.Generator("cpu").manual_seed(args.seed))
        total = time.time() - t0
        print(f"\n  Total: {total:.1f}s ({total/args.steps:.1f}s/step)")

        result.images[0].save("/proj_sw/ssokorac/work/tt-lang/examples/flux2/flux2_blackhole_output.png")
        print("  Saved!")

    finally:
        F.linear = original_linear
        ttnn.close_device(tt_device)


if __name__ == "__main__":
    main()
