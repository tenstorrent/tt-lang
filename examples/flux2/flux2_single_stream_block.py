# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 Single-Stream Transformer Block for TT-Lang.

This is the core block used 20 times in FLUX.2-klein-4B. It has a PARALLEL
design where attention and MLP run side-by-side:

  1. AdaLN: norm(x) * (1+scale) + shift  (using approximate norm for sim)
  2. Fused projection: norm_x @ W_qkv_mlp -> [Q, K, V, mlp_gate, mlp_up]
  3. Attention path: QK-norm -> (RoPE skipped in sim) -> SDPA -> attn_out
  4. MLP path: SwiGLU(mlp_gate, mlp_up) -> mlp_out
  5. Fused output: cat(attn_out, mlp_out) @ W_out
  6. Gated residual: x + gate * output

Scaled-down test config:
  hidden_dim = 32 (1 tile) -- single Tensix tile
  num_heads = 1 (head_dim = 32)
  mlp_hidden = 64 (2 tiles) -- mlp_ratio ~2 (real: 3.0)
  seq_len = 32 (1 tile)

The fused projection maps: hidden -> 3*hidden + 2*mlp_hidden
  = 32 -> 3*32 + 2*64 = 32 -> 224 (7 tiles)

The fused output maps: hidden + mlp_hidden -> hidden
  = 32 + 64 -> 32 = 96 -> 32 (3 tiles -> 1 tile)
"""

import torch
import torch.nn.functional as F

import ttnn
import ttl

# Scaled-down config
SEQ_TILES = 1       # 32 tokens
HIDDEN_TILES = 1    # 32 hidden dim (1 head, head_dim=32)
MLP_TILES = 2       # 64 mlp hidden dim
# Fused QKV+MLP projection: 3*HIDDEN + 2*MLP = 3 + 4 = 7 tiles output
QKV_TILES = 3       # Q, K, V each 1 tile = 3 tiles
FUSED_IN_TILES = QKV_TILES + 2 * MLP_TILES  # 3 + 4 = 7
# Fused output projection: HIDDEN + MLP = 1 + 2 = 3 tiles input
FUSED_OUT_IN_TILES = HIDDEN_TILES + MLP_TILES  # 3


def to_device(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ============================================================================
# Kernel 1: AdaLN + Fused QKV+MLP Projection
# ============================================================================
@ttl.operation(grid=(1, 1))
def adaln_fused_proj_kernel(x, shift, scale, w_q, w_k, w_v, w_mlp_gate, w_mlp_up,
                            q_out, k_out, v_out, mlp_gate_out, mlp_up_out):
    """
    AdaLN normalization (approximate) then separate projections for Q, K, V,
    mlp_gate, mlp_up.

    On hardware, the fused projection would be a single matmul to [7 tiles],
    but in sim we split it for simplicity since the sim doesn't support
    splitting a multi-tile output cleanly.
    """
    x_dfb = ttl.make_dataflow_buffer_like(
        x, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    shift_dfb = ttl.make_dataflow_buffer_like(
        shift, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1
    )
    scale_dfb = ttl.make_dataflow_buffer_like(
        scale, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1
    )
    # Weight DFBs (all bias-free linears)
    w_q_dfb = ttl.make_dataflow_buffer_like(
        w_q, shape=(HIDDEN_TILES, HIDDEN_TILES), buffer_factor=1
    )
    w_k_dfb = ttl.make_dataflow_buffer_like(
        w_k, shape=(HIDDEN_TILES, HIDDEN_TILES), buffer_factor=1
    )
    w_v_dfb = ttl.make_dataflow_buffer_like(
        w_v, shape=(HIDDEN_TILES, HIDDEN_TILES), buffer_factor=1
    )
    w_mg_dfb = ttl.make_dataflow_buffer_like(
        w_mlp_gate, shape=(HIDDEN_TILES, MLP_TILES), buffer_factor=1
    )
    w_mu_dfb = ttl.make_dataflow_buffer_like(
        w_mlp_up, shape=(HIDDEN_TILES, MLP_TILES), buffer_factor=1
    )
    # Output DFBs
    q_dfb = ttl.make_dataflow_buffer_like(
        q_out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    k_dfb = ttl.make_dataflow_buffer_like(
        k_out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    v_dfb = ttl.make_dataflow_buffer_like(
        v_out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    mg_dfb = ttl.make_dataflow_buffer_like(
        mlp_gate_out, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2
    )
    mu_dfb = ttl.make_dataflow_buffer_like(
        mlp_up_out, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        # AdaLN: approximate normalization + modulation
        # For sim, we just apply scale+shift without proper LN
        # (1 + scale) * x + shift
        with x_dfb.wait() as xv, scale_dfb.wait() as sc, shift_dfb.wait() as sh:
            # Modulated input (approximate AdaLN without proper norm)
            modulated = xv + sc * xv + sh

            # Q projection
            with w_q_dfb.wait() as wq, q_dfb.reserve() as qo:
                qo.store(ttl.math.matmul(modulated, wq, qo))

            # K projection
            with w_k_dfb.wait() as wk, k_dfb.reserve() as ko:
                ko.store(ttl.math.matmul(modulated, wk, ko))

            # V projection
            with w_v_dfb.wait() as wv, v_dfb.reserve() as vo:
                vo.store(ttl.math.matmul(modulated, wv, vo))

            # MLP gate projection
            with w_mg_dfb.wait() as wmg, mg_dfb.reserve() as mgo:
                mgo.store(ttl.math.matmul(modulated, wmg, mgo))

            # MLP up projection
            with w_mu_dfb.wait() as wmu, mu_dfb.reserve() as muo:
                muo.store(ttl.math.matmul(modulated, wmu, muo))

    @ttl.datamovement()
    def dm_read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with shift_dfb.reserve() as blk:
            tx = ttl.copy(shift[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with scale_dfb.reserve() as blk:
            tx = ttl.copy(scale[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with w_q_dfb.reserve() as blk:
            tx = ttl.copy(w_q[0:HIDDEN_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with w_k_dfb.reserve() as blk:
            tx = ttl.copy(w_k[0:HIDDEN_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with w_v_dfb.reserve() as blk:
            tx = ttl.copy(w_v[0:HIDDEN_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with w_mg_dfb.reserve() as blk:
            tx = ttl.copy(w_mlp_gate[0:HIDDEN_TILES, 0:MLP_TILES], blk)
            tx.wait()
        with w_mu_dfb.reserve() as blk:
            tx = ttl.copy(w_mlp_up[0:HIDDEN_TILES, 0:MLP_TILES], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with q_dfb.wait() as blk:
            tx = ttl.copy(blk, q_out[0:SEQ_TILES, 0:HIDDEN_TILES])
            tx.wait()
        with k_dfb.wait() as blk:
            tx = ttl.copy(blk, k_out[0:SEQ_TILES, 0:HIDDEN_TILES])
            tx.wait()
        with v_dfb.wait() as blk:
            tx = ttl.copy(blk, v_out[0:SEQ_TILES, 0:HIDDEN_TILES])
            tx.wait()
        with mg_dfb.wait() as blk:
            tx = ttl.copy(blk, mlp_gate_out[0:SEQ_TILES, 0:MLP_TILES])
            tx.wait()
        with mu_dfb.wait() as blk:
            tx = ttl.copy(blk, mlp_up_out[0:SEQ_TILES, 0:MLP_TILES])
            tx.wait()


# ============================================================================
# Kernel 2: Attention (SDPA without RoPE for sim testing)
# ============================================================================
@ttl.operation(grid=(1, 1))
def attention_kernel(q, k, v, scale, out):
    """
    Scaled dot-product attention: exp(Q @ K^T * scale) @ V
    No causal mask (FLUX.2 uses full attention, not causal).
    Softmax normalization omitted for sim (needs within-tile reduction).
    """
    q_dfb = ttl.make_dataflow_buffer_like(
        q, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    k_dfb = ttl.make_dataflow_buffer_like(
        k, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    v_dfb = ttl.make_dataflow_buffer_like(
        v, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )

    # Intermediates
    k_t_dfb = ttl.make_dataflow_buffer_like(
        k, shape=(HIDDEN_TILES, SEQ_TILES), buffer_factor=2
    )
    scores_dfb = ttl.make_dataflow_buffer_like(
        q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2
    )
    # For (1,1) tiles, softmax reduces within a tile which the sim can't do.
    # Use exp(scores * scale) / sum as an approximation: since all tiles are
    # the same shape, just do elementwise ops without broadcast.
    exp_dfb = ttl.make_dataflow_buffer_like(
        q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        # Transpose K
        with k_dfb.wait() as kv, k_t_dfb.reserve() as kt:
            kt.store(ttl.transpose(kv, kt))

        # Q @ K^T -> scores
        with q_dfb.wait() as qv, k_t_dfb.wait() as ktv:
            with scores_dfb.reserve() as sc:
                sc.store(ttl.math.matmul(qv, ktv, sc))

        # Scale and exp (approximate softmax: exp(scores * scale))
        # For (1,1) tiles, scale_dfb has the same 32x32 shape so elementwise works
        with scores_dfb.wait() as scv, scale_dfb.wait() as scalev:
            with exp_dfb.reserve() as ex:
                ex.store(ttl.math.exp(scv * scalev))

        # Attn output: exp_scores @ V (unnormalized attention for sim)
        with exp_dfb.wait() as exv, v_dfb.wait() as vv:
            with out_dfb.reserve() as o:
                o.store(ttl.math.matmul(exv, vv, o))

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(q[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(k[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with v_dfb.reserve() as blk:
            tx = ttl.copy(v[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
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
# Kernel 3: SwiGLU + Fused Output Projection + Gated Residual
# ============================================================================
@ttl.operation(grid=(1, 1))
def swiglu_output_residual_kernel(attn_out, mlp_gate, mlp_up, gate, x_residual,
                                   w_attn_out, w_mlp_out, out):
    """
    1. SwiGLU: silu(mlp_gate) * mlp_up
    2. Output proj (split): attn_out @ W_attn_out, swiglu_out @ W_mlp_out
    3. Combined: attn_proj + mlp_proj
    4. Gated residual: x + gate * combined

    On hardware, steps 2-3 would be a single fused matmul:
      cat(attn_out, swiglu_out) @ W_out
    But we split for sim compatibility.
    """
    attn_dfb = ttl.make_dataflow_buffer_like(
        attn_out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    mg_dfb = ttl.make_dataflow_buffer_like(
        mlp_gate, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2
    )
    mu_dfb = ttl.make_dataflow_buffer_like(
        mlp_up, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2
    )
    gate_dfb = ttl.make_dataflow_buffer_like(
        gate, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1
    )
    res_dfb = ttl.make_dataflow_buffer_like(
        x_residual, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1
    )
    w_ao_dfb = ttl.make_dataflow_buffer_like(
        w_attn_out, shape=(HIDDEN_TILES, HIDDEN_TILES), buffer_factor=1
    )
    w_mo_dfb = ttl.make_dataflow_buffer_like(
        w_mlp_out, shape=(MLP_TILES, HIDDEN_TILES), buffer_factor=1
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )

    # Intermediates
    swiglu_dfb = ttl.make_dataflow_buffer_like(
        mlp_gate, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2
    )
    attn_proj_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    mlp_proj_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        # 1. SwiGLU: silu(gate) * up
        with mg_dfb.wait() as gv, mu_dfb.wait() as uv, swiglu_dfb.reserve() as sw:
            silu_gate = gv * ttl.math.sigmoid(gv)
            sw.store(silu_gate * uv)

        # 2. Attention output projection
        with attn_dfb.wait() as av, w_ao_dfb.wait() as wao:
            with attn_proj_dfb.reserve() as ap:
                ap.store(ttl.math.matmul(av, wao, ap))

        # 3. MLP output projection
        with swiglu_dfb.wait() as swv, w_mo_dfb.wait() as wmo:
            with mlp_proj_dfb.reserve() as mp:
                mp.store(ttl.math.matmul(swv, wmo, mp))

        # 4. Combine + gated residual: x + gate * (attn_proj + mlp_proj)
        with (
            attn_proj_dfb.wait() as apv,
            mlp_proj_dfb.wait() as mpv,
            gate_dfb.wait() as gv,
            res_dfb.wait() as rv,
        ):
            with out_dfb.reserve() as o:
                combined = apv + mpv
                o.store(rv + gv * combined)

    @ttl.datamovement()
    def dm_read():
        with attn_dfb.reserve() as blk:
            tx = ttl.copy(attn_out[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with mg_dfb.reserve() as blk:
            tx = ttl.copy(mlp_gate[0:SEQ_TILES, 0:MLP_TILES], blk)
            tx.wait()
        with mu_dfb.reserve() as blk:
            tx = ttl.copy(mlp_up[0:SEQ_TILES, 0:MLP_TILES], blk)
            tx.wait()
        with gate_dfb.reserve() as blk:
            tx = ttl.copy(gate[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with res_dfb.reserve() as blk:
            tx = ttl.copy(x_residual[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with w_ao_dfb.reserve() as blk:
            tx = ttl.copy(w_attn_out[0:HIDDEN_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with w_mo_dfb.reserve() as blk:
            tx = ttl.copy(w_mlp_out[0:MLP_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HIDDEN_TILES])
            tx.wait()


# ============================================================================
# Test: Full single-stream block
# ============================================================================
def test_single_stream_block(device):
    """Test FLUX.2 single-stream transformer block."""
    torch.manual_seed(42)

    seq_len = SEQ_TILES * 32    # 32
    hidden_dim = HIDDEN_TILES * 32  # 32
    mlp_hidden = MLP_TILES * 32    # 64

    # Input
    x_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1

    # AdaLN modulation params (from timestep embedding)
    shift_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.05
    scale_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.05
    gate_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1

    # Weights (all bias-free, matching FLUX.2)
    w_q_torch = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    w_k_torch = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    w_v_torch = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    w_mg_torch = torch.randn(hidden_dim, mlp_hidden, dtype=torch.bfloat16) * 0.02
    w_mu_torch = torch.randn(hidden_dim, mlp_hidden, dtype=torch.bfloat16) * 0.02
    w_ao_torch = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
    w_mo_torch = torch.randn(mlp_hidden, hidden_dim, dtype=torch.bfloat16) * 0.02

    # Attention scalers
    scale_val = 1.0 / (hidden_dim ** 0.5)
    scale_tile = torch.full((32, 32), scale_val, dtype=torch.bfloat16)

    # Intermediate tensors
    q_torch = torch.zeros(seq_len, hidden_dim, dtype=torch.bfloat16)
    k_torch = torch.zeros(seq_len, hidden_dim, dtype=torch.bfloat16)
    v_torch = torch.zeros(seq_len, hidden_dim, dtype=torch.bfloat16)
    mg_out_torch = torch.zeros(seq_len, mlp_hidden, dtype=torch.bfloat16)
    mu_out_torch = torch.zeros(seq_len, mlp_hidden, dtype=torch.bfloat16)
    attn_out_torch = torch.zeros(seq_len, hidden_dim, dtype=torch.bfloat16)
    out_torch = torch.zeros(seq_len, hidden_dim, dtype=torch.bfloat16)

    # Convert to TTNN
    x = to_device(x_torch, device)
    shift = to_device(shift_torch, device)
    scale = to_device(scale_torch, device)
    gate = to_device(gate_torch, device)
    w_q = to_device(w_q_torch, device)
    w_k = to_device(w_k_torch, device)
    w_v = to_device(w_v_torch, device)
    w_mg = to_device(w_mg_torch, device)
    w_mu = to_device(w_mu_torch, device)
    w_ao = to_device(w_ao_torch, device)
    w_mo = to_device(w_mo_torch, device)
    scale_t = to_device(scale_tile, device)
    q = to_device(q_torch, device)
    k = to_device(k_torch, device)
    v = to_device(v_torch, device)
    mg_out = to_device(mg_out_torch, device)
    mu_out = to_device(mu_out_torch, device)
    attn_out = to_device(attn_out_torch, device)
    out = to_device(out_torch, device)

    print("Running FLUX.2 single-stream block...")

    # Kernel 1: AdaLN + projections
    print("  1. adaln_fused_proj_kernel")
    adaln_fused_proj_kernel(
        x, shift, scale, w_q, w_k, w_v, w_mg, w_mu,
        q, k, v, mg_out, mu_out
    )

    # Kernel 2: Attention
    print("  2. attention_kernel")
    attention_kernel(q, k, v, scale_t, attn_out)

    # Kernel 3: SwiGLU + output projection + gated residual
    print("  3. swiglu_output_residual_kernel")
    swiglu_output_residual_kernel(
        attn_out, mg_out, mu_out, gate, x, w_ao, w_mo, out
    )

    result = ttnn.to_torch(out)

    # PyTorch reference
    x_f = x_torch.float()
    shift_f = shift_torch.float()
    scale_f = scale_torch.float()
    gate_f = gate_torch.float()

    # AdaLN (approximate, no proper norm - matching kernel)
    modulated = x_f + scale_f * x_f + shift_f

    # Projections
    q_ref = modulated @ w_q_torch.float()
    k_ref = modulated @ w_k_torch.float()
    v_ref = modulated @ w_v_torch.float()
    mg_ref = modulated @ w_mg_torch.float()
    mu_ref = modulated @ w_mu_torch.float()

    # Attention (no RoPE, no causal mask, unnormalized for sim)
    scores = q_ref @ k_ref.T
    exp_scores = torch.exp(scores * scale_val)
    attn_ref = exp_scores @ v_ref

    # SwiGLU
    silu_gate = mg_ref * torch.sigmoid(mg_ref)
    swiglu_ref = silu_gate * mu_ref

    # Output projection (split: attn_proj + mlp_proj)
    attn_proj = attn_ref @ w_ao_torch.float()
    mlp_proj = swiglu_ref @ w_mo_torch.float()
    combined = attn_proj + mlp_proj

    # Gated residual
    expected = (x_f + gate_f * combined).bfloat16()

    print(f"\nExpected[0,:8]: {expected[0,:8]}")
    print(f"Result[0,:8]:   {result[0,:8]}")

    if torch.isnan(result).any():
        print("WARNING: Result contains NaN")
        return

    abs_diff = torch.abs(result.float() - expected.float())
    print(f"Max absolute diff: {abs_diff.max().item():.6f}")
    print(f"Mean absolute diff: {abs_diff.mean().item():.6f}")

    correlation = torch.corrcoef(
        torch.stack([result.float().flatten(), expected.float().flatten()])
    )[0, 1].item()
    print(f"Correlation: {correlation:.6f}")

    if correlation > 0.90:
        print("\nPASSED: Single-stream block matches reference")
    else:
        print(f"\nFAILED: Correlation too low: {correlation}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_single_stream_block(device)
    finally:
        ttnn.close_device(device)
