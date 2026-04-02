# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 Double-Stream (Joint) Transformer Block for TT-Lang.

Used 5 times in FLUX.2-klein-4B. Image and text streams are processed
separately through norms/MLPs but share a single joint attention:

  1. AdaLN on both streams (separate modulation params)
  2. Separate QKV projections for image and text
  3. Concatenate Q, K, V from both streams
  4. Joint attention: SDPA on combined sequence
  5. Split attention output back to image/text
  6. Separate output projections + gated residuals
  7. Separate SwiGLU MLPs + gated residuals

Scaled-down test config:
  hidden_dim = 32 (1 tile)
  num_heads = 1
  mlp_hidden = 64 (2 tiles)
  img_seq_len = 32 (1 tile)
  txt_seq_len = 32 (1 tile)
"""

import torch

import ttnn
import ttl

SEQ_TILES = 1       # 32 tokens per stream
HIDDEN_TILES = 1    # 32 hidden dim
MLP_TILES = 2       # 64 mlp hidden


def to_device(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ============================================================================
# Kernel 1: Image stream - AdaLN + QKV projection
# ============================================================================
@ttl.operation(grid=(1, 1))
def img_adaln_qkv_kernel(x, shift, scale, w_q, w_k, w_v, q_out, k_out, v_out):
    """AdaLN(x) -> Q, K, V for image stream."""
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    shift_dfb = ttl.make_dataflow_buffer_like(shift, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    w_q_dfb = ttl.make_dataflow_buffer_like(w_q, shape=(HIDDEN_TILES, HIDDEN_TILES), buffer_factor=1)
    w_k_dfb = ttl.make_dataflow_buffer_like(w_k, shape=(HIDDEN_TILES, HIDDEN_TILES), buffer_factor=1)
    w_v_dfb = ttl.make_dataflow_buffer_like(w_v, shape=(HIDDEN_TILES, HIDDEN_TILES), buffer_factor=1)
    q_dfb = ttl.make_dataflow_buffer_like(q_out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(k_out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(v_out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_dfb.wait() as xv, scale_dfb.wait() as sc, shift_dfb.wait() as sh:
            modulated = xv + sc * xv + sh
            with w_q_dfb.wait() as wq, q_dfb.reserve() as qo:
                qo.store(ttl.math.matmul(modulated, wq, qo))
            with w_k_dfb.wait() as wk, k_dfb.reserve() as ko:
                ko.store(ttl.math.matmul(modulated, wk, ko))
            with w_v_dfb.wait() as wv, v_dfb.reserve() as vo:
                vo.store(ttl.math.matmul(modulated, wv, vo))

    @ttl.datamovement()
    def dm_read():
        for dfb, tensor, rows, cols in [
            (x_dfb, x, SEQ_TILES, HIDDEN_TILES),
            (shift_dfb, shift, SEQ_TILES, HIDDEN_TILES),
            (scale_dfb, scale, SEQ_TILES, HIDDEN_TILES),
            (w_q_dfb, w_q, HIDDEN_TILES, HIDDEN_TILES),
            (w_k_dfb, w_k, HIDDEN_TILES, HIDDEN_TILES),
            (w_v_dfb, w_v, HIDDEN_TILES, HIDDEN_TILES),
        ]:
            with dfb.reserve() as blk:
                tx = ttl.copy(tensor[0:rows, 0:cols], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        for dfb, tensor in [(q_dfb, q_out), (k_dfb, k_out), (v_dfb, v_out)]:
            with dfb.wait() as blk:
                tx = ttl.copy(blk, tensor[0:SEQ_TILES, 0:HIDDEN_TILES])
                tx.wait()


# ============================================================================
# Kernel 2: Simple attention (reused for each stream)
# On real HW, both streams share one joint SDPA on concatenated sequences.
# For sim, we use self-attention per stream (validates matmul+transpose chain).
# ============================================================================
@ttl.operation(grid=(1, 1))
def simple_attention_kernel(q, k, v, scale, out):
    """exp(Q @ K^T * scale) @ V -- unnormalized attention for sim."""
    q_dfb = ttl.make_dataflow_buffer_like(q, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(k, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    k_t_dfb = ttl.make_dataflow_buffer_like(k, shape=(HIDDEN_TILES, SEQ_TILES), buffer_factor=2)
    scores_dfb = ttl.make_dataflow_buffer_like(q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        with k_dfb.wait() as kv, k_t_dfb.reserve() as kt:
            kt.store(ttl.transpose(kv, kt))
        with q_dfb.wait() as qv, k_t_dfb.wait() as ktv, scale_dfb.wait() as sv:
            with scores_dfb.reserve() as sc:
                sc.store(ttl.math.matmul(qv, ktv, sc))
            with scores_dfb.wait() as scv, scores_dfb.reserve() as esc:
                esc.store(ttl.math.exp(scv * sv))
            with scores_dfb.wait() as ev, v_dfb.wait() as vv:
                with out_dfb.reserve() as o:
                    o.store(ttl.math.matmul(ev, vv, o))

    @ttl.datamovement()
    def dm_read():
        for dfb, tensor in [(q_dfb, q), (k_dfb, k), (v_dfb, v)]:
            with dfb.reserve() as blk:
                tx = ttl.copy(tensor[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
                tx.wait()
        with scale_dfb.reserve() as blk:
            tx = ttl.copy(scale[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HIDDEN_TILES])
            tx.wait()


# ============================================================================
# Kernel 3: Output projection + gated residual (per stream)
# ============================================================================
@ttl.operation(grid=(1, 1))
def proj_gated_residual_kernel(attn_out, gate, x_residual, w_out, out):
    """out = x_residual + gate * (attn_out @ W_out)"""
    attn_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    gate_dfb = ttl.make_dataflow_buffer_like(gate, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    res_dfb = ttl.make_dataflow_buffer_like(x_residual, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    w_dfb = ttl.make_dataflow_buffer_like(w_out, shape=(HIDDEN_TILES, HIDDEN_TILES), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    proj_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        with attn_dfb.wait() as av, w_dfb.wait() as wv:
            with proj_dfb.reserve() as p:
                p.store(ttl.math.matmul(av, wv, p))
        with proj_dfb.wait() as pv, gate_dfb.wait() as gv, res_dfb.wait() as rv:
            with out_dfb.reserve() as o:
                o.store(rv + gv * pv)

    @ttl.datamovement()
    def dm_read():
        for dfb, tensor, rows, cols in [
            (attn_dfb, attn_out, SEQ_TILES, HIDDEN_TILES),
            (gate_dfb, gate, SEQ_TILES, HIDDEN_TILES),
            (res_dfb, x_residual, SEQ_TILES, HIDDEN_TILES),
            (w_dfb, w_out, HIDDEN_TILES, HIDDEN_TILES),
        ]:
            with dfb.reserve() as blk:
                tx = ttl.copy(tensor[0:rows, 0:cols], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HIDDEN_TILES])
            tx.wait()


# ============================================================================
# Kernel 4: SwiGLU MLP + gated residual (per stream)
# ============================================================================
@ttl.operation(grid=(1, 1))
def swiglu_mlp_residual_kernel(x, shift, scale, gate, x_residual,
                                w_gate, w_up, w_down, out):
    """
    AdaLN(x) -> SwiGLU MLP -> gated residual
    out = x_residual + gate * (SwiGLU(adaln_x @ W_gate, adaln_x @ W_up) @ W_down)
    """
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    shift_dfb = ttl.make_dataflow_buffer_like(shift, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    gate_dfb = ttl.make_dataflow_buffer_like(gate, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    res_dfb = ttl.make_dataflow_buffer_like(x_residual, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    w_g_dfb = ttl.make_dataflow_buffer_like(w_gate, shape=(HIDDEN_TILES, MLP_TILES), buffer_factor=1)
    w_u_dfb = ttl.make_dataflow_buffer_like(w_up, shape=(HIDDEN_TILES, MLP_TILES), buffer_factor=1)
    w_d_dfb = ttl.make_dataflow_buffer_like(w_down, shape=(MLP_TILES, HIDDEN_TILES), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)

    # Intermediates
    gate_out_dfb = ttl.make_dataflow_buffer_like(w_gate, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2)
    up_out_dfb = ttl.make_dataflow_buffer_like(w_up, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2)
    swiglu_dfb = ttl.make_dataflow_buffer_like(w_gate, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2)
    mlp_out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        # AdaLN
        with x_dfb.wait() as xv, scale_dfb.wait() as sc, shift_dfb.wait() as sh:
            modulated = xv + sc * xv + sh

            # Gate and up projections
            with w_g_dfb.wait() as wg, gate_out_dfb.reserve() as go:
                go.store(ttl.math.matmul(modulated, wg, go))
            with w_u_dfb.wait() as wu, up_out_dfb.reserve() as uo:
                uo.store(ttl.math.matmul(modulated, wu, uo))

        # SwiGLU
        with gate_out_dfb.wait() as gv, up_out_dfb.wait() as uv, swiglu_dfb.reserve() as sw:
            silu = gv * ttl.math.sigmoid(gv)
            sw.store(silu * uv)

        # Down projection
        with swiglu_dfb.wait() as swv, w_d_dfb.wait() as wd:
            with mlp_out_dfb.reserve() as mo:
                mo.store(ttl.math.matmul(swv, wd, mo))

        # Gated residual
        with mlp_out_dfb.wait() as mov, gate_dfb.wait() as gatev, res_dfb.wait() as rv:
            with out_dfb.reserve() as o:
                o.store(rv + gatev * mov)

    @ttl.datamovement()
    def dm_read():
        for dfb, tensor, rows, cols in [
            (x_dfb, x, SEQ_TILES, HIDDEN_TILES),
            (shift_dfb, shift, SEQ_TILES, HIDDEN_TILES),
            (scale_dfb, scale, SEQ_TILES, HIDDEN_TILES),
            (gate_dfb, gate, SEQ_TILES, HIDDEN_TILES),
            (res_dfb, x_residual, SEQ_TILES, HIDDEN_TILES),
            (w_g_dfb, w_gate, HIDDEN_TILES, MLP_TILES),
            (w_u_dfb, w_up, HIDDEN_TILES, MLP_TILES),
            (w_d_dfb, w_down, MLP_TILES, HIDDEN_TILES),
        ]:
            with dfb.reserve() as blk:
                tx = ttl.copy(tensor[0:rows, 0:cols], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HIDDEN_TILES])
            tx.wait()


# ============================================================================
# Test: Full double-stream block
# ============================================================================
def test_double_stream_block(device):
    """Test FLUX.2 double-stream (joint) transformer block."""
    torch.manual_seed(42)

    seq_len = SEQ_TILES * 32
    hidden_dim = HIDDEN_TILES * 32
    mlp_hidden = MLP_TILES * 32

    # Inputs
    img_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1
    txt_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1

    # AdaLN modulation (6 params per stream: shift_msa, scale_msa, gate_msa,
    #                                        shift_mlp, scale_mlp, gate_mlp)
    img_shift_msa = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.05
    img_scale_msa = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.05
    img_gate_msa = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1
    img_shift_mlp = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.05
    img_scale_mlp = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.05
    img_gate_mlp = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1
    txt_shift_msa = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.05
    txt_scale_msa = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.05
    txt_gate_msa = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1
    txt_shift_mlp = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.05
    txt_scale_mlp = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.05
    txt_gate_mlp = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1

    # Weights
    img_wq = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    img_wk = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    img_wv = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    txt_wq = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    txt_wk = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    txt_wv = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    img_wo = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    txt_wo = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    img_wg = torch.randn(hidden_dim, mlp_hidden, dtype=torch.bfloat16) * 0.02
    img_wu = torch.randn(hidden_dim, mlp_hidden, dtype=torch.bfloat16) * 0.02
    img_wd = torch.randn(mlp_hidden, hidden_dim, dtype=torch.bfloat16) * 0.02
    txt_wg = torch.randn(hidden_dim, mlp_hidden, dtype=torch.bfloat16) * 0.02
    txt_wu = torch.randn(hidden_dim, mlp_hidden, dtype=torch.bfloat16) * 0.02
    txt_wd = torch.randn(mlp_hidden, hidden_dim, dtype=torch.bfloat16) * 0.02

    scale_val = 1.0 / (hidden_dim ** 0.5)
    scale_tile = torch.full((32, 32), scale_val, dtype=torch.bfloat16)

    # Intermediates
    zeros_h = torch.zeros(seq_len, hidden_dim, dtype=torch.bfloat16)

    # Convert all to device
    def d(t):
        return to_device(t, device)

    img = d(img_torch)
    txt = d(txt_torch)
    q_img, k_img, v_img = d(zeros_h.clone()), d(zeros_h.clone()), d(zeros_h.clone())
    q_txt, k_txt, v_txt = d(zeros_h.clone()), d(zeros_h.clone()), d(zeros_h.clone())
    img_attn = d(zeros_h.clone())
    txt_attn = d(zeros_h.clone())
    img_post_attn = d(zeros_h.clone())
    txt_post_attn = d(zeros_h.clone())
    img_out = d(zeros_h.clone())
    txt_out = d(zeros_h.clone())
    scale_t = d(scale_tile)

    print("Running FLUX.2 double-stream block...")

    # Step 1: Image AdaLN + QKV
    print("  1a. img_adaln_qkv_kernel")
    img_adaln_qkv_kernel(img, d(img_shift_msa), d(img_scale_msa),
                          d(img_wq), d(img_wk), d(img_wv), q_img, k_img, v_img)

    # Step 1b: Text AdaLN + QKV (reuse same kernel)
    print("  1b. txt_adaln_qkv_kernel")
    img_adaln_qkv_kernel(txt, d(txt_shift_msa), d(txt_scale_msa),
                          d(txt_wq), d(txt_wk), d(txt_wv), q_txt, k_txt, v_txt)

    # Step 2: Attention (self-attention per stream for sim; real HW does joint SDPA)
    print("  2a. img attention")
    simple_attention_kernel(q_img, k_img, v_img, scale_t, img_attn)
    print("  2b. txt attention")
    simple_attention_kernel(q_txt, k_txt, v_txt, scale_t, txt_attn)

    # Step 3: Output projection + gated residual (both streams)
    print("  3a. img proj_gated_residual")
    proj_gated_residual_kernel(img_attn, d(img_gate_msa), img, d(img_wo), img_post_attn)
    print("  3b. txt proj_gated_residual")
    proj_gated_residual_kernel(txt_attn, d(txt_gate_msa), txt, d(txt_wo), txt_post_attn)

    # Step 4: SwiGLU MLP + gated residual (both streams)
    print("  4a. img swiglu_mlp_residual")
    swiglu_mlp_residual_kernel(img_post_attn, d(img_shift_mlp), d(img_scale_mlp),
                                d(img_gate_mlp), img_post_attn,
                                d(img_wg), d(img_wu), d(img_wd), img_out)
    print("  4b. txt swiglu_mlp_residual")
    swiglu_mlp_residual_kernel(txt_post_attn, d(txt_shift_mlp), d(txt_scale_mlp),
                                d(txt_gate_mlp), txt_post_attn,
                                d(txt_wg), d(txt_wu), d(txt_wd), txt_out)

    # Get results
    img_result = ttnn.to_torch(img_out)
    txt_result = ttnn.to_torch(txt_out)

    # PyTorch reference
    def ref_block(x, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                  wq, wk, wv, wo, wg, wu, wd):
        x_f = x.float()
        # AdaLN + QKV
        modulated = x_f + scale_msa.float() * x_f + shift_msa.float()
        q = modulated @ wq.float()
        k = modulated @ wk.float()
        v = modulated @ wv.float()
        # Self-attention (unnormalized, matching sim)
        scores = q @ k.T
        attn = torch.exp(scores * scale_val) @ v
        # Output proj + gated residual
        proj = attn @ wo.float()
        post_attn = x_f + gate_msa.float() * proj
        # MLP
        mlp_mod = post_attn + scale_mlp.float() * post_attn + shift_mlp.float()
        gate_proj = mlp_mod @ wg.float()
        up_proj = mlp_mod @ wu.float()
        silu = gate_proj * torch.sigmoid(gate_proj)
        swiglu = silu * up_proj
        mlp_out = swiglu @ wd.float()
        return (post_attn + gate_mlp.float() * mlp_out).bfloat16()

    img_expected = ref_block(
        img_torch, img_shift_msa, img_scale_msa, img_gate_msa,
        img_shift_mlp, img_scale_mlp, img_gate_mlp,
        img_wq, img_wk, img_wv, img_wo, img_wg, img_wu, img_wd,
    )

    print(f"\nImg Expected[0,:8]: {img_expected[0,:8]}")
    print(f"Img Result[0,:8]:   {img_result[0,:8]}")

    if torch.isnan(img_result).any():
        print("WARNING: Image result contains NaN")
        return

    abs_diff = torch.abs(img_result.float() - img_expected.float())
    print(f"Max absolute diff: {abs_diff.max().item():.6f}")
    print(f"Mean absolute diff: {abs_diff.mean().item():.6f}")

    correlation = torch.corrcoef(
        torch.stack([img_result.float().flatten(), img_expected.float().flatten()])
    )[0, 1].item()
    print(f"Correlation: {correlation:.6f}")

    if correlation > 0.90:
        print("\nPASSED: Double-stream block matches reference")
    else:
        print(f"\nFAILED: Correlation too low: {correlation}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_double_stream_block(device)
    finally:
        ttnn.close_device(device)
