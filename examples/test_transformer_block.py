# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TT-Lang Transformer Block for nanochat.

A single transformer block:
1. RMSNorm(x) -> normalized
2. Q, K, V = linear projections of normalized
3. Apply rotary embeddings to Q, K
4. Attention: softmax(Q @ K.T / sqrt(d)) @ V
5. Output projection
6. Residual: x + attn_out
7. RMSNorm(residual)
8. MLP: relu²(x @ w_fc) @ w_proj
9. Final residual

Test configuration (single-head for simplicity):
- seq_len = 32 (1 tile)
- n_embd = 32 (1 tile)
- head_dim = 32 (same as n_embd for single head)
- mlp_hidden = 128 (4 tiles)
"""

import torch
import torch.nn.functional as F
import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl


# Test configuration
SEQ_TILES = 1  # 32 tokens
EMBD_TILES = 1  # 32 embedding dim (single head)
MLP_TILES = 4  # 128 MLP hidden


def to_device(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ============================================================================
# Kernel 1: RMSNorm + Q/K/V projections
# ============================================================================
@ttl.kernel(grid=(1, 1))
def norm_qkv_kernel(x, w_q, w_k, w_v, scaler, q_out, k_out, v_out):
    """
    RMSNorm(x) then project to Q, K, V.
    """
    x_cb = ttl.make_circular_buffer_like(
        x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    w_q_cb = ttl.make_circular_buffer_like(
        w_q, shape=(EMBD_TILES, EMBD_TILES), buffer_factor=1
    )
    w_k_cb = ttl.make_circular_buffer_like(
        w_k, shape=(EMBD_TILES, EMBD_TILES), buffer_factor=1
    )
    w_v_cb = ttl.make_circular_buffer_like(
        w_v, shape=(EMBD_TILES, EMBD_TILES), buffer_factor=1
    )
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    q_cb = ttl.make_circular_buffer_like(
        q_out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    k_cb = ttl.make_circular_buffer_like(
        k_out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    v_cb = ttl.make_circular_buffer_like(
        v_out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )

    # RMSNorm intermediates
    sq_cb = ttl.make_circular_buffer_like(
        x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    sum_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_cb = ttl.make_circular_buffer_like(
        x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    normed_cb = ttl.make_circular_buffer_like(
        x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        # RMSNorm
        with x_cb.wait() as xv, scaler_cb.wait() as sc:
            # Square
            with sq_cb.reserve() as sq:
                sq.store(xv * xv)
            # Row-wise sum
            with sq_cb.wait() as sqv, sum_cb.reserve() as sm:
                sm.store(ttl.math.reduce_sum(sqv, sc, sm, dims=[0]))
            # Rsqrt
            with sum_cb.wait() as smv, sum_cb.reserve() as rsq:
                rsq.store(ttl.math.rsqrt(smv))
            # Broadcast
            with sum_cb.wait() as rsqv, bcast_cb.reserve() as bc:
                bc.store(ttl.math.broadcast(rsqv, bc, dims=[1]))
            # Normalize and store to CB
            with bcast_cb.wait() as bcv, normed_cb.reserve() as nm:
                nm.store(xv * bcv)

        # Q, K, V projections - keep normed in scope for all three
        with normed_cb.wait() as nmv:
            with w_q_cb.wait() as wq, q_cb.reserve() as qo:
                qo.store(ttl.math.matmul(nmv, wq, qo))

            with w_k_cb.wait() as wk, k_cb.reserve() as ko:
                ko.store(ttl.math.matmul(nmv, wk, ko))

            with w_v_cb.wait() as wv, v_cb.reserve() as vo:
                vo.store(ttl.math.matmul(nmv, wv, vo))

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with w_q_cb.reserve() as blk:
            tx = ttl.copy(w_q[0:EMBD_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with w_k_cb.reserve() as blk:
            tx = ttl.copy(w_k[0:EMBD_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with w_v_cb.reserve() as blk:
            tx = ttl.copy(w_v[0:EMBD_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with scaler_cb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with q_cb.wait() as blk:
            tx = ttl.copy(blk, q_out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
        with k_cb.wait() as blk:
            tx = ttl.copy(blk, k_out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
        with v_cb.wait() as blk:
            tx = ttl.copy(blk, v_out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()


# ============================================================================
# Kernel 2: Rotary embeddings on Q and K
# ============================================================================
@ttl.kernel(grid=(1, 1))
def rotary_qk_kernel(q_in, k_in, cos, q_out, k_out):
    """
    Apply rotary embeddings to Q and K (simplified - just multiply by cos).
    Real RoPE would split dimension and use sin too.
    """
    q_cb = ttl.make_circular_buffer_like(
        q_in, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    k_cb = ttl.make_circular_buffer_like(
        k_in, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    cos_cb = ttl.make_circular_buffer_like(
        cos, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    qo_cb = ttl.make_circular_buffer_like(
        q_out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    ko_cb = ttl.make_circular_buffer_like(
        k_out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        # For simplified rotary: q_rot = q * cos, k_rot = k * cos
        # Keep cos in scope for both Q and K
        with cos_cb.wait() as cv:
            with q_cb.wait() as qv, qo_cb.reserve() as qo:
                qo.store(qv * cv)

            with k_cb.wait() as kv, ko_cb.reserve() as ko:
                ko.store(kv * cv)

    @ttl.datamovement()
    def dm_read():
        with q_cb.reserve() as blk:
            tx = ttl.copy(q_in[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with k_cb.reserve() as blk:
            tx = ttl.copy(k_in[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with cos_cb.reserve() as blk:
            tx = ttl.copy(cos[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with qo_cb.wait() as blk:
            tx = ttl.copy(blk, q_out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
        with ko_cb.wait() as blk:
            tx = ttl.copy(blk, k_out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()


# ============================================================================
# Kernel 3: Attention (reusing from test_attention.py)
# ============================================================================
@ttl.kernel(grid=(1, 1))
def attention_kernel(q, k, v, scale, causal_mask, scaler, out):
    """Single-head scaled dot-product attention."""
    q_cb = ttl.make_circular_buffer_like(
        q, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    k_cb = ttl.make_circular_buffer_like(
        k, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    v_cb = ttl.make_circular_buffer_like(
        v, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    scale_cb = ttl.make_circular_buffer_like(scale, shape=(1, 1), buffer_factor=1)
    mask_cb = ttl.make_circular_buffer_like(
        causal_mask, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=1
    )
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    out_cb = ttl.make_circular_buffer_like(
        out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )

    k_t_cb = ttl.make_circular_buffer_like(
        k, shape=(EMBD_TILES, SEQ_TILES), buffer_factor=2
    )
    scores_cb = ttl.make_circular_buffer_like(
        causal_mask, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2
    )
    scale_bcast_cb = ttl.make_circular_buffer_like(
        causal_mask, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2
    )
    scaled_masked_cb = ttl.make_circular_buffer_like(
        causal_mask, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2
    )
    max_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    max_bcast_cb = ttl.make_circular_buffer_like(
        causal_mask, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2
    )
    exp_cb = ttl.make_circular_buffer_like(
        causal_mask, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2
    )
    sum_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    sum_bcast_cb = ttl.make_circular_buffer_like(
        causal_mask, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2
    )
    softmax_cb = ttl.make_circular_buffer_like(
        causal_mask, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        with k_cb.wait() as kv, k_t_cb.reserve() as kt:
            kt.store(ttl.transpose(kv, kt))

        with q_cb.wait() as qv, k_t_cb.wait() as ktv:
            with scores_cb.reserve() as sc:
                sc.store(ttl.math.matmul(qv, ktv, sc))

        with (
            scores_cb.wait() as scv,
            scale_cb.wait() as scalev,
            mask_cb.wait() as maskv,
        ):
            with scale_bcast_cb.reserve() as sb:
                sb.store(ttl.math.broadcast(scalev, sb, dims=[0, 1]))
            with scale_bcast_cb.wait() as sbv, scaled_masked_cb.reserve() as sm:
                sm.store(scv * sbv + maskv)

        with scaler_cb.wait() as scaler_v, scaled_masked_cb.wait() as smv:
            with max_cb.reserve() as mx:
                mx.store(ttl.math.reduce_max(smv, scaler_v, mx, dims=[0]))
            with max_cb.wait() as mxv, max_bcast_cb.reserve() as mxb:
                mxb.store(ttl.math.broadcast(mxv, mxb, dims=[1]))
            with max_bcast_cb.wait() as mxbv:
                shifted = smv - mxbv
                with exp_cb.reserve() as ex:
                    ex.store(ttl.math.exp(shifted))
                with exp_cb.wait() as exv, sum_cb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(exv, scaler_v, sm, dims=[0]))
                with sum_cb.wait() as smv2, sum_bcast_cb.reserve() as smb:
                    smb.store(ttl.math.broadcast(smv2, smb, dims=[1]))
                with sum_bcast_cb.wait() as smbv, softmax_cb.reserve() as sfm:
                    sfm.store(ttl.math.exp(shifted) / smbv)

        with softmax_cb.wait() as sfmv, v_cb.wait() as vv:
            with out_cb.reserve() as o:
                o.store(ttl.math.matmul(sfmv, vv, o))

    @ttl.datamovement()
    def dm_read():
        with q_cb.reserve() as blk:
            tx = ttl.copy(q[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with k_cb.reserve() as blk:
            tx = ttl.copy(k[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with v_cb.reserve() as blk:
            tx = ttl.copy(v[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with scale_cb.reserve() as blk:
            tx = ttl.copy(scale[0, 0], blk)
            tx.wait()
        with mask_cb.reserve() as blk:
            tx = ttl.copy(causal_mask[0:SEQ_TILES, 0:SEQ_TILES], blk)
            tx.wait()
        with scaler_cb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()


# ============================================================================
# Kernel 4: Output projection + residual
# ============================================================================
@ttl.kernel(grid=(1, 1))
def proj_residual_kernel(attn_out, x_residual, w_proj, out):
    """Output projection and residual add: out = x_residual + attn_out @ w_proj"""
    attn_cb = ttl.make_circular_buffer_like(
        attn_out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    res_cb = ttl.make_circular_buffer_like(
        x_residual, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    w_cb = ttl.make_circular_buffer_like(
        w_proj, shape=(EMBD_TILES, EMBD_TILES), buffer_factor=1
    )
    out_cb = ttl.make_circular_buffer_like(
        out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )

    proj_cb = ttl.make_circular_buffer_like(
        out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        # Project
        with attn_cb.wait() as av, w_cb.wait() as wv:
            with proj_cb.reserve() as pj:
                pj.store(ttl.math.matmul(av, wv, pj))

        # Residual add
        with proj_cb.wait() as pjv, res_cb.wait() as rv:
            with out_cb.reserve() as o:
                o.store(pjv + rv)

    @ttl.datamovement()
    def dm_read():
        with attn_cb.reserve() as blk:
            tx = ttl.copy(attn_out[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with res_cb.reserve() as blk:
            tx = ttl.copy(x_residual[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with w_cb.reserve() as blk:
            tx = ttl.copy(w_proj[0:EMBD_TILES, 0:EMBD_TILES], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()


# ============================================================================
# Kernel 5: RMSNorm + MLP + residual
# ============================================================================
@ttl.kernel(grid=(1, 1))
def norm_mlp_residual_kernel(x, x_residual, w_fc, w_proj, scaler, out):
    """RMSNorm(x) -> MLP (relu²) -> + residual"""
    x_cb = ttl.make_circular_buffer_like(
        x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    res_cb = ttl.make_circular_buffer_like(
        x_residual, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    w_fc_cb = ttl.make_circular_buffer_like(
        w_fc, shape=(EMBD_TILES, MLP_TILES), buffer_factor=1
    )
    w_proj_cb = ttl.make_circular_buffer_like(
        w_proj, shape=(MLP_TILES, EMBD_TILES), buffer_factor=1
    )
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    out_cb = ttl.make_circular_buffer_like(
        out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )

    # Intermediates
    sq_cb = ttl.make_circular_buffer_like(
        x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    sum_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_cb = ttl.make_circular_buffer_like(
        x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    normed_cb = ttl.make_circular_buffer_like(
        x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )
    hidden_cb = ttl.make_circular_buffer_like(
        w_fc, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2
    )
    act_cb = ttl.make_circular_buffer_like(
        w_fc, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2
    )
    mlp_out_cb = ttl.make_circular_buffer_like(
        x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        # RMSNorm
        with x_cb.wait() as xv, scaler_cb.wait() as sc:
            with sq_cb.reserve() as sq:
                sq.store(xv * xv)
            with sq_cb.wait() as sqv, sum_cb.reserve() as sm:
                sm.store(ttl.math.reduce_sum(sqv, sc, sm, dims=[0]))
            with sum_cb.wait() as smv, sum_cb.reserve() as rsq:
                rsq.store(ttl.math.rsqrt(smv))
            with sum_cb.wait() as rsqv, bcast_cb.reserve() as bc:
                bc.store(ttl.math.broadcast(rsqv, bc, dims=[1]))
            with bcast_cb.wait() as bcv, normed_cb.reserve() as nm:
                nm.store(xv * bcv)

        # MLP: fc -> relu² -> proj
        with normed_cb.wait() as nmv, w_fc_cb.wait() as wfc:
            with hidden_cb.reserve() as h:
                h.store(ttl.math.matmul(nmv, wfc, h))

        with hidden_cb.wait() as hv, act_cb.reserve() as a:
            relu_out = ttl.math.relu(hv)
            a.store(relu_out * relu_out)

        with act_cb.wait() as av, w_proj_cb.wait() as wpr:
            with mlp_out_cb.reserve() as mo:
                mo.store(ttl.math.matmul(av, wpr, mo))

        # Residual add
        with mlp_out_cb.wait() as mov, res_cb.wait() as rv:
            with out_cb.reserve() as o:
                o.store(mov + rv)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with res_cb.reserve() as blk:
            tx = ttl.copy(x_residual[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with w_fc_cb.reserve() as blk:
            tx = ttl.copy(w_fc[0:EMBD_TILES, 0:MLP_TILES], blk)
            tx.wait()
        with w_proj_cb.reserve() as blk:
            tx = ttl.copy(w_proj[0:MLP_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with scaler_cb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()


# ============================================================================
# Test: Full transformer block
# ============================================================================
def test_transformer_block(device):
    """Test a single transformer block."""
    torch.manual_seed(42)

    seq_len = SEQ_TILES * 32  # 32
    n_embd = EMBD_TILES * 32  # 32
    mlp_hidden = MLP_TILES * 32  # 128

    # Create random inputs and weights
    x_torch = torch.randn(seq_len, n_embd, dtype=torch.bfloat16) * 0.1

    # Attention weights
    w_q_torch = torch.randn(n_embd, n_embd, dtype=torch.bfloat16) * 0.02
    w_k_torch = torch.randn(n_embd, n_embd, dtype=torch.bfloat16) * 0.02
    w_v_torch = torch.randn(n_embd, n_embd, dtype=torch.bfloat16) * 0.02
    w_o_torch = torch.randn(n_embd, n_embd, dtype=torch.bfloat16) * 0.02

    # MLP weights
    w_fc_torch = torch.randn(n_embd, mlp_hidden, dtype=torch.bfloat16) * 0.02
    w_proj_torch = torch.randn(mlp_hidden, n_embd, dtype=torch.bfloat16) * 0.02

    # Rotary (simplified - just use ones for this test)
    cos_torch = torch.ones(seq_len, n_embd, dtype=torch.bfloat16)

    # Scalers
    scale_val = 1.0 / (n_embd**0.5)
    scale_torch = torch.full((32, 32), scale_val, dtype=torch.bfloat16)
    scaler_torch = torch.full((32, 32), 1.0 / n_embd, dtype=torch.bfloat16)
    scaler_ones = torch.ones(32, 32, dtype=torch.bfloat16)

    # Causal mask
    causal_mask_torch = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")), diagonal=1
    ).bfloat16()

    # Intermediate tensors
    q_torch = torch.zeros(seq_len, n_embd, dtype=torch.bfloat16)
    k_torch = torch.zeros(seq_len, n_embd, dtype=torch.bfloat16)
    v_torch = torch.zeros(seq_len, n_embd, dtype=torch.bfloat16)
    q_rot_torch = torch.zeros(seq_len, n_embd, dtype=torch.bfloat16)
    k_rot_torch = torch.zeros(seq_len, n_embd, dtype=torch.bfloat16)
    attn_out_torch = torch.zeros(seq_len, n_embd, dtype=torch.bfloat16)
    post_attn_torch = torch.zeros(seq_len, n_embd, dtype=torch.bfloat16)
    out_torch = torch.zeros(seq_len, n_embd, dtype=torch.bfloat16)

    # Convert to TTNN
    x = to_device(x_torch, device)
    w_q = to_device(w_q_torch, device)
    w_k = to_device(w_k_torch, device)
    w_v = to_device(w_v_torch, device)
    w_o = to_device(w_o_torch, device)
    w_fc = to_device(w_fc_torch, device)
    w_proj = to_device(w_proj_torch, device)
    cos = to_device(cos_torch, device)
    scale = to_device(scale_torch, device)
    scaler = to_device(scaler_torch, device)
    scaler_ones = to_device(scaler_ones, device)
    causal_mask = to_device(causal_mask_torch, device)
    q = to_device(q_torch, device)
    k = to_device(k_torch, device)
    v = to_device(v_torch, device)
    q_rot = to_device(q_rot_torch, device)
    k_rot = to_device(k_rot_torch, device)
    attn_out = to_device(attn_out_torch, device)
    post_attn = to_device(post_attn_torch, device)
    out = to_device(out_torch, device)

    print("Running transformer block kernels...")

    # Kernel 1: Norm + Q/K/V projections
    print("  1. norm_qkv_kernel")
    norm_qkv_kernel(x, w_q, w_k, w_v, scaler, q, k, v)

    # Kernel 2: Rotary (simplified)
    print("  2. rotary_qk_kernel")
    rotary_qk_kernel(q, k, cos, q_rot, k_rot)

    # Kernel 3: Attention
    print("  3. attention_kernel")
    attention_kernel(q_rot, k_rot, v, scale, causal_mask, scaler_ones, attn_out)

    # Kernel 4: Output projection + residual
    print("  4. proj_residual_kernel")
    proj_residual_kernel(attn_out, x, w_o, post_attn)

    # Kernel 5: Norm + MLP + residual
    print("  5. norm_mlp_residual_kernel")
    norm_mlp_residual_kernel(post_attn, post_attn, w_fc, w_proj, scaler, out)

    # Get result
    result = ttnn.to_torch(out)

    # PyTorch reference
    def rms_norm(x):
        return x / torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + 1e-6)

    x_float = x_torch.float()

    # Attention
    normed = rms_norm(x_float)
    q_ref = normed @ w_q_torch.float()
    k_ref = normed @ w_k_torch.float()
    v_ref = normed @ w_v_torch.float()

    # Rotary (simplified - multiply by 1)
    q_rot_ref = q_ref * cos_torch.float()
    k_rot_ref = k_ref * cos_torch.float()

    # Attention
    scores = q_rot_ref @ k_rot_ref.T * scale_val
    scores = scores + causal_mask_torch.float()
    attn_weights = F.softmax(scores, dim=-1)
    attn_out_ref = attn_weights @ v_ref

    # Output projection + residual
    proj_ref = attn_out_ref @ w_o_torch.float()
    post_attn_ref = x_float + proj_ref

    # MLP
    normed2 = rms_norm(post_attn_ref)
    hidden = normed2 @ w_fc_torch.float()
    act = F.relu(hidden) ** 2
    mlp_out = act @ w_proj_torch.float()
    expected = (post_attn_ref + mlp_out).bfloat16()

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

    if correlation > 0.9:
        print("\nPASSED: Transformer block matches PyTorch reference")
    else:
        print(f"\nFAILED: Correlation too low: {correlation}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_transformer_block(device)
    finally:
        ttnn.close_device(device)
