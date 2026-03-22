# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fused single-head attention kernel using online (flash) softmax.

Processes one cache tile at a time, maintaining running max, sum, and output
accumulators with rescaling. No DRAM scratch needed — all intermediates
flow through bf=2 DFBs. Eliminates the DRAM race that caused PCC=0.19
with the scratch-based approach.

Algorithm (online softmax):
  For each cache tile kt:
    scores = Q @ K[:, kt]
    masked = scores + mask[kt]
    local_max = reduce_max(masked)
    m_new = max(m_old, local_max)
    alpha = exp(m_old - m_new)          # rescale factor
    p = exp(masked - broadcast(m_new))  # attention weights (unnormalized)
    d = d * alpha + reduce_sum(p)       # running denominator
    o = o * alpha + p @ V[kt]           # running output
  Final: output = o / d
"""

import torch
import ttl
import ttnn

TILE = 32
CACHE_TILES = 16  # max_seq / TILE = 512/32


@ttl.kernel(grid=(1, 1))
def fused_attn_head_kernel(Q_rot, K_T, V, mask, scaler, attn_out):
    """Fused attention for 1 Q head: score + online softmax + output.

    Q_rot:     [TILE, head_dim] = [32, 64] — single head Q, pre-rotated
    K_T:       [head_dim, max_seq] = [64, 512] = [2, 16] tiles
    V:         [max_seq, head_dim] = [512, 64] = [16, 2] tiles
    mask:      [TILE, max_seq] = [32, 512] = [1, 16] tiles
    scaler:    [TILE, TILE] ones
    attn_out:  [TILE, head_dim] = [32, 64] — output
    """
    Kt_q = 2   # head_dim / TILE
    Nt_s = K_T.shape[1] // TILE  # 16
    Nt_v = Kt_q  # 2

    # --- Reader → Compute DFBs ---
    q_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_T, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)

    # --- Compute-local DFBs (all bf=2) ---
    acc_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    # Dual masked: one for reduce_max, one for exp (computed twice, same result)
    masked_a_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    masked_b_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    p_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    # Running accumulators
    m_save_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    m_old_copy_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    m_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    alpha_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    alpha_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    d_save_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    d_scaled_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    o0_save_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)
    o1_save_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)
    o0_tmp_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)
    o1_tmp_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)

    # --- Compute → Writer DFBs ---
    out_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        # Scaler (once, at start)
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

        # For each cache tile: Q(×2), K(×2), mask(×1), V(×2)
        for kt in range(Nt_s):
            for ki in range(Kt_q):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(Q_rot[0, ki], blk)
                    tx.wait()
                with k_dfb.reserve() as blk:
                    tx = ttl.copy(K_T[ki, kt], blk)
                    tx.wait()
            with m_dfb.reserve() as blk:
                tx = ttl.copy(mask[0, kt], blk)
                tx.wait()
            for vi in range(Nt_v):
                with v_dfb.reserve() as blk:
                    tx = ttl.copy(V[kt, vi], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        with sc_dfb.wait() as sc_blk:

            # ============================================================
            # First cache tile (kt=0): initialize all accumulators
            # ============================================================

            # Score matmul: Q @ K[:, 0] (K-inner loop)
            with q_dfb.wait() as q0, k_dfb.wait() as k0:
                with acc_dfb.reserve() as acc:
                    acc.store(q0 @ k0)
            with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + q1 @ k1)

            # Mask (store twice: for reduce_max and for exp)
            with acc_dfb.wait() as score, m_dfb.wait() as msk:
                with masked_a_dfb.reserve() as ma:
                    ma.store(score + msk)
                with masked_b_dfb.reserve() as mb:
                    mb.store(score + msk)

            # reduce_max → initial max
            with masked_a_dfb.wait() as msk_a:
                with tmp_dfb.reserve() as tmp:
                    tmp.store(ttl.math.reduce_max(msk_a, sc_blk, tmp, dims=[1]))
            with tmp_dfb.wait() as local_m:
                with m_save_dfb.reserve() as ms:
                    ms.store(local_m)
                with m_bc_dfb.reserve() as mbc:
                    mbc.store(ttl.math.broadcast(local_m, mbc, dims=[0, 1]))

            # p = exp(masked - max_broadcast)
            with masked_b_dfb.wait() as msk_b, m_bc_dfb.wait() as m_bc:
                with p_dfb.reserve() as p:
                    p.store(ttl.math.exp(msk_b - m_bc))

            # Initialize d, o0, o1 from first p
            with p_dfb.wait() as p_blk:
                # d = reduce_sum(p)
                with tmp_dfb.reserve() as tmp:
                    tmp.store(ttl.math.reduce_sum(p_blk, sc_blk, tmp, dims=[1]))
                with tmp_dfb.wait() as local_d:
                    with d_save_dfb.reserve() as ds:
                        ds.store(local_d)

                # o0 = p @ V[0, 0]
                with v_dfb.wait() as v0:
                    with o0_save_dfb.reserve() as o0:
                        o0.store(p_blk @ v0)
                # o1 = p @ V[0, 1]
                with v_dfb.wait() as v1:
                    with o1_save_dfb.reserve() as o1:
                        o1.store(p_blk @ v1)

            # ============================================================
            # Remaining cache tiles (kt=1..15): accumulate with rescaling
            # ============================================================
            for _ in range(Nt_s - 1):
                # Score matmul: Q @ K[:, kt]
                with q_dfb.wait() as q0, k_dfb.wait() as k0:
                    with acc_dfb.reserve() as acc:
                        acc.store(q0 @ k0)
                with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + q1 @ k1)

                # Mask (twice)
                with acc_dfb.wait() as score, m_dfb.wait() as msk:
                    with masked_a_dfb.reserve() as ma:
                        ma.store(score + msk)
                    with masked_b_dfb.reserve() as mb:
                        mb.store(score + msk)

                # reduce_max → local_max
                with masked_a_dfb.wait() as msk_a:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(ttl.math.reduce_max(msk_a, sc_blk, tmp, dims=[1]))

                # m_new = max(m_old, local_max); save m_old for alpha
                with tmp_dfb.wait() as local_max, m_save_dfb.wait() as m_old:
                    with m_save_dfb.reserve() as m_new:
                        m_new.store(ttl.math.max(m_old, local_max))
                    with m_old_copy_dfb.reserve() as mo:
                        mo.store(m_old)

                # alpha = exp(m_old - m_new); broadcast m_new; save m_new
                with m_save_dfb.wait() as m_new_val, m_old_copy_dfb.wait() as m_old_val:
                    with alpha_dfb.reserve() as a:
                        a.store(ttl.math.exp(m_old_val - m_new_val))
                    with m_bc_dfb.reserve() as mbc:
                        mbc.store(ttl.math.broadcast(m_new_val, mbc, dims=[0, 1]))
                    with m_save_dfb.reserve() as ms:
                        ms.store(m_new_val)

                # p = exp(masked - m_new_broadcast)
                with masked_b_dfb.wait() as msk_b, m_bc_dfb.wait() as m_bc:
                    with p_dfb.reserve() as p:
                        p.store(ttl.math.exp(msk_b - m_bc))

                # Broadcast alpha
                with alpha_dfb.wait() as alpha_raw:
                    with alpha_bc_dfb.reserve() as abc:
                        abc.store(ttl.math.broadcast(alpha_raw, abc, dims=[0, 1]))

                # Update d, o0, o1 with rescaling (keep p and alpha_bc alive)
                with p_dfb.wait() as p_blk, alpha_bc_dfb.wait() as alpha_bc:
                    # reduce_sum(p) → local_sum
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(ttl.math.reduce_sum(p_blk, sc_blk, tmp, dims=[1]))

                    # d = d_old * alpha + local_sum
                    with d_save_dfb.wait() as d_old:
                        with d_scaled_dfb.reserve() as d_s:
                            d_s.store(d_old * alpha_bc)
                    with d_scaled_dfb.wait() as d_s, tmp_dfb.wait() as local_sum:
                        with d_save_dfb.reserve() as d_new:
                            d_new.store(d_s + local_sum)

                    # o0 = o0_old * alpha + p @ v0
                    with o0_save_dfb.wait() as o0_old:
                        with o0_tmp_dfb.reserve() as o0_s:
                            o0_s.store(o0_old * alpha_bc)
                    with v_dfb.wait() as v0, o0_tmp_dfb.wait() as o0_s:
                        with o0_save_dfb.reserve() as o0_new:
                            o0_new.store(o0_s + p_blk @ v0)

                    # o1 = o1_old * alpha + p @ v1
                    with o1_save_dfb.wait() as o1_old:
                        with o1_tmp_dfb.reserve() as o1_s:
                            o1_s.store(o1_old * alpha_bc)
                    with v_dfb.wait() as v1, o1_tmp_dfb.wait() as o1_s:
                        with o1_save_dfb.reserve() as o1_new:
                            o1_new.store(o1_s + p_blk @ v1)

            # ============================================================
            # Final normalization: output = o / d
            # ============================================================
            with d_save_dfb.wait() as d_final:
                with tmp_dfb.reserve() as d_bc_tmp:
                    d_bc_tmp.store(ttl.math.broadcast(d_final, d_bc_tmp, dims=[0, 1]))
            with tmp_dfb.wait() as d_bc:
                with o0_save_dfb.wait() as o0_final:
                    with out_dfb.reserve() as o:
                        o.store(o0_final * ttl.math.recip(d_bc))
                with o1_save_dfb.wait() as o1_final:
                    with out_dfb.reserve() as o:
                        o.store(o1_final * ttl.math.recip(d_bc))

    @ttl.datamovement()
    def write():
        for nt in range(Nt_v):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, attn_out[0, nt])
                tx.wait()


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_fused_attn(device, label, Q_t, K_T_t, V_t, mask_t, valid_n):
    """Test fused attention against PyTorch reference."""
    print(f"  {label}...", end="", flush=True)

    sc_t = torch.ones(TILE, TILE, dtype=torch.bfloat16)

    Q = _to_device(Q_t, device)
    K_T = _to_device(K_T_t, device)
    V = _to_device(V_t, device)
    mask = _to_device(mask_t, device)
    sc = _to_device(sc_t, device)
    out = _to_device(torch.zeros(TILE, 64, dtype=torch.bfloat16), device)

    fused_attn_head_kernel(Q, K_T, V, mask, sc, out)
    result = ttnn.to_torch(out)

    # Reference
    scores = Q_t[0:1, :64].float() @ K_T_t.float() + mask_t[0:1].float()
    weights = torch.nn.functional.softmax(scores, dim=-1)
    expected = (weights @ V_t.float()).bfloat16()

    pcc = torch.corrcoef(
        torch.stack([result[0, :64].float(), expected[0].float()])
    )[0, 1].item()
    status = 'PASS' if pcc > 0.98 else 'FAIL'
    print(f" PCC={pcc:.6f} scores=[{scores[0,:valid_n].min():.1f},{scores[0,:valid_n].max():.1f}] {status}")
    return pcc


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("Flash attention tests (no DRAM scratch):")

        # Test 1: Small random, 50 valid
        Q1 = torch.randn(TILE, 64, dtype=torch.bfloat16) * 0.01
        K1 = torch.randn(64, 512, dtype=torch.bfloat16) * 0.01
        V1 = torch.randn(512, 64, dtype=torch.bfloat16) * 0.01
        m1 = torch.full((TILE, 512), float("-inf"), dtype=torch.bfloat16)
        m1[0, :50] = 0.0
        pcc1 = test_fused_attn(device, "small random (50 valid)", Q1, K1, V1, m1, 50)

        # Test 2: Large scores, 50 valid
        Q2 = torch.randn(TILE, 64, dtype=torch.bfloat16) * 2.0
        K2 = torch.randn(64, 512, dtype=torch.bfloat16) * 2.0
        V2 = torch.randn(512, 64, dtype=torch.bfloat16) * 0.01
        m2 = torch.full((TILE, 512), float("-inf"), dtype=torch.bfloat16)
        m2[0, :50] = 0.0
        pcc2 = test_fused_attn(device, "large scores (50 valid)", Q2, K2, V2, m2, 50)

        # Test 3: Large scores, 6 valid (real model conditions)
        Q3 = torch.randn(TILE, 64, dtype=torch.bfloat16) * 2.0
        K3 = torch.randn(64, 512, dtype=torch.bfloat16) * 2.0
        V3 = torch.randn(512, 64, dtype=torch.bfloat16) * 0.01
        m3 = torch.full((TILE, 512), float("-inf"), dtype=torch.bfloat16)
        m3[0, :6] = 0.0
        pcc3 = test_fused_attn(device, "large scores (6 valid)", Q3, K3, V3, m3, 6)

        # Test 4: Moderate scores, 1 valid (edge case)
        Q4 = torch.randn(TILE, 64, dtype=torch.bfloat16) * 1.0
        K4 = torch.randn(64, 512, dtype=torch.bfloat16) * 1.0
        V4 = torch.randn(512, 64, dtype=torch.bfloat16) * 0.1
        m4 = torch.full((TILE, 512), float("-inf"), dtype=torch.bfloat16)
        m4[0, :1] = 0.0
        pcc4 = test_fused_attn(device, "moderate scores (1 valid)", Q4, K4, V4, m4, 1)

        print(f"\n=== Summary ===")
        print(f"  Small random (50):    PCC={pcc1:.6f} {'PASS' if pcc1 > 0.98 else 'FAIL'}")
        print(f"  Large scores (50):    PCC={pcc2:.6f} {'PASS' if pcc2 > 0.98 else 'FAIL'}")
        print(f"  Large scores (6):     PCC={pcc3:.6f} {'PASS' if pcc3 > 0.98 else 'FAIL'}")
        print(f"  Moderate scores (1):  PCC={pcc4:.6f} {'PASS' if pcc4 > 0.98 else 'FAIL'}")
    finally:
        ttnn.close_device(device)
