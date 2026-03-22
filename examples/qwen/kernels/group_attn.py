# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Per-head attention with on-device Q slicing and output placement.

Each kernel variant reads Q at a specific column offset from Q_rot [TILE, 896]
and writes output at a specific column offset to attn_out [TILE, 896].
Eliminates ALL host transfers for Q slicing and output concatenation.

14 variants (one per Q head), called 14 times per layer.
Same flash attention logic as fused_attn.py but with:
  - Score scaling by attn_scale (1/sqrt(head_dim))
  - Q read at Q_rot[0, q_col] instead of Q_rot[0, 0]
  - Output written to attn_out[0, out_col] instead of attn_out[0, 0]
"""

import torch
import ttl
import ttnn

TILE = 32


def _make_head_attn_kernel(q_col, out_col):
    """Create attention kernel for a specific Q head.

    q_col:   starting tile column in Q_rot for this head (0, 2, 4, ..., 26)
    out_col: starting tile column in attn_out for this head (same as q_col)
    """

    @ttl.kernel(grid=(1, 1))
    def head_attn_kernel(Q_rot, K_T, V, mask, scaler, attn_scale, attn_out):
        """Flash attention for 1 Q head, reading/writing at column offsets.

        Q_rot:      [TILE, 896]  — all heads combined [1, 28] tiles
        K_T:        [64, 512]    — K^T cache [2, 16] tiles
        V:          [512, 64]    — V cache [16, 2] tiles
        mask:       [TILE, 512]  — decode mask [1, 16] tiles
        scaler:     [TILE, TILE] — ones for reduce
        attn_scale: [TILE, TILE] — 1/sqrt(head_dim)
        attn_out:   [TILE, 896]  — output [1, 28] tiles (write at offset)
        """
        Kt_q = 2   # head_dim / TILE
        Nt_s = K_T.shape[1] // TILE  # 16
        Nt_v = Kt_q  # 2

        # --- Reader → Compute DFBs ---
        q_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(K_T, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        asc_dfb = ttl.make_dataflow_buffer_like(attn_scale, shape=(1, 1), buffer_factor=2)
        m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)

        # --- Compute-local DFBs ---
        acc_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
        scaled_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
        masked_a_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
        masked_b_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
        tmp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        p_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
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
        out_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)

        @ttl.datamovement()
        def read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk)
                tx.wait()
            with asc_dfb.reserve() as blk:
                tx = ttl.copy(attn_scale[0, 0], blk)
                tx.wait()
            for kt in range(Nt_s):
                for ki in range(Kt_q):
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(Q_rot[0, q_col + ki], blk)
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
            with sc_dfb.wait() as sc_blk, asc_dfb.wait() as attn_sc:

                # === First cache tile (kt=0) ===
                with q_dfb.wait() as q0, k_dfb.wait() as k0:
                    with acc_dfb.reserve() as acc:
                        acc.store(q0 @ k0)
                with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + q1 @ k1)
                with acc_dfb.wait() as score:
                    with scaled_dfb.reserve() as sc_score:
                        sc_score.store(score * attn_sc)

                with scaled_dfb.wait() as score, m_dfb.wait() as msk:
                    with masked_a_dfb.reserve() as ma:
                        ma.store(score + msk)
                    with masked_b_dfb.reserve() as mb:
                        mb.store(score + msk)

                with masked_a_dfb.wait() as msk_a:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(ttl.math.reduce_max(msk_a, sc_blk, tmp, dims=[1]))
                with tmp_dfb.wait() as local_m:
                    with m_save_dfb.reserve() as ms:
                        ms.store(local_m)
                    with m_bc_dfb.reserve() as mbc:
                        mbc.store(ttl.math.broadcast(local_m, mbc, dims=[0, 1]))

                with masked_b_dfb.wait() as msk_b, m_bc_dfb.wait() as m_bc:
                    with p_dfb.reserve() as p:
                        p.store(ttl.math.exp(msk_b - m_bc))

                with p_dfb.wait() as p_blk:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(ttl.math.reduce_sum(p_blk, sc_blk, tmp, dims=[1]))
                    with tmp_dfb.wait() as local_d:
                        with d_save_dfb.reserve() as ds:
                            ds.store(local_d)
                    with v_dfb.wait() as v0:
                        with o0_save_dfb.reserve() as o0:
                            o0.store(p_blk @ v0)
                    with v_dfb.wait() as v1:
                        with o1_save_dfb.reserve() as o1:
                            o1.store(p_blk @ v1)

                # === Remaining cache tiles (kt=1..15) ===
                for _ in range(Nt_s - 1):
                    with q_dfb.wait() as q0, k_dfb.wait() as k0:
                        with acc_dfb.reserve() as acc:
                            acc.store(q0 @ k0)
                    with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + q1 @ k1)
                    with acc_dfb.wait() as score:
                        with scaled_dfb.reserve() as sc_score:
                            sc_score.store(score * attn_sc)

                    with scaled_dfb.wait() as score, m_dfb.wait() as msk:
                        with masked_a_dfb.reserve() as ma:
                            ma.store(score + msk)
                        with masked_b_dfb.reserve() as mb:
                            mb.store(score + msk)

                    with masked_a_dfb.wait() as msk_a:
                        with tmp_dfb.reserve() as tmp:
                            tmp.store(ttl.math.reduce_max(msk_a, sc_blk, tmp, dims=[1]))

                    with tmp_dfb.wait() as local_max, m_save_dfb.wait() as m_old:
                        with m_save_dfb.reserve() as m_new:
                            m_new.store(ttl.math.max(m_old, local_max))
                        with m_old_copy_dfb.reserve() as mo:
                            mo.store(m_old)

                    with m_save_dfb.wait() as m_new_val, m_old_copy_dfb.wait() as m_old_val:
                        with alpha_dfb.reserve() as a:
                            a.store(ttl.math.exp(m_old_val - m_new_val))
                        with m_bc_dfb.reserve() as mbc:
                            mbc.store(ttl.math.broadcast(m_new_val, mbc, dims=[0, 1]))
                        with m_save_dfb.reserve() as ms:
                            ms.store(m_new_val)

                    with masked_b_dfb.wait() as msk_b, m_bc_dfb.wait() as m_bc:
                        with p_dfb.reserve() as p:
                            p.store(ttl.math.exp(msk_b - m_bc))

                    with alpha_dfb.wait() as alpha_raw:
                        with alpha_bc_dfb.reserve() as abc:
                            abc.store(ttl.math.broadcast(alpha_raw, abc, dims=[0, 1]))

                    with p_dfb.wait() as p_blk, alpha_bc_dfb.wait() as alpha_bc:
                        with tmp_dfb.reserve() as tmp:
                            tmp.store(ttl.math.reduce_sum(p_blk, sc_blk, tmp, dims=[1]))
                        with d_save_dfb.wait() as d_old:
                            with d_scaled_dfb.reserve() as d_s:
                                d_s.store(d_old * alpha_bc)
                        with d_scaled_dfb.wait() as d_s, tmp_dfb.wait() as local_sum:
                            with d_save_dfb.reserve() as d_new:
                                d_new.store(d_s + local_sum)
                        with o0_save_dfb.wait() as o0_old:
                            with o0_tmp_dfb.reserve() as o0_s:
                                o0_s.store(o0_old * alpha_bc)
                        with v_dfb.wait() as v0, o0_tmp_dfb.wait() as o0_s:
                            with o0_save_dfb.reserve() as o0_new:
                                o0_new.store(o0_s + p_blk @ v0)
                        with o1_save_dfb.wait() as o1_old:
                            with o1_tmp_dfb.reserve() as o1_s:
                                o1_s.store(o1_old * alpha_bc)
                        with v_dfb.wait() as v1, o1_tmp_dfb.wait() as o1_s:
                            with o1_save_dfb.reserve() as o1_new:
                                o1_new.store(o1_s + p_blk @ v1)

                # === Final normalization ===
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
            for vi in range(Nt_v):
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, attn_out[0, out_col + vi])
                    tx.wait()

    return head_attn_kernel


# Create 14 kernel variants (one per Q head)
head_attn_kernels = [_make_head_attn_kernel(q_col=h * 2, out_col=h * 2) for h in range(14)]


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_head_attn(device):
    """Compare head_attn_kernel vs fused_attn_head_kernel for heads 0 and 7."""
    import math
    from fused_attn import fused_attn_head_kernel

    print("Head attention test (column-offset Q/output):")

    head_dim = 64
    max_seq = 512
    hidden = 14 * head_dim

    Q_rot_t = torch.randn(TILE, hidden, dtype=torch.bfloat16) * 0.5
    K_T_t = torch.randn(head_dim, max_seq, dtype=torch.bfloat16) * 0.1
    V_t = torch.randn(max_seq, head_dim, dtype=torch.bfloat16) * 0.1
    mask_t = torch.full((TILE, max_seq), float("-inf"), dtype=torch.bfloat16)
    mask_t[0, :20] = 0.0
    sc_t = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    asc_t = torch.full((TILE, TILE), 1.0 / math.sqrt(head_dim), dtype=torch.bfloat16)
    scale_val = 1.0 / math.sqrt(head_dim)

    Q_rot = _to_device(Q_rot_t, device)
    K_T = _to_device(K_T_t, device)
    V = _to_device(V_t, device)
    mask = _to_device(mask_t, device)
    sc = _to_device(sc_t, device)
    asc = _to_device(asc_t, device)

    for h in [0, 3, 7, 13]:
        col_s = h * head_dim
        col_e = col_s + head_dim

        # Reference: fused_attn_head_kernel with pre-scaled Q
        q_head = Q_rot_t[:, col_s:col_e].clone()
        q_head[0] = q_head[0] * scale_val
        q_head_dev = _to_device(q_head, device)
        ref_out = _to_device(torch.zeros(TILE, head_dim, dtype=torch.bfloat16), device)
        fused_attn_head_kernel(q_head_dev, K_T, V, mask, sc, ref_out)
        ref_out_t = ttnn.to_torch(ref_out)

        # New: head_attn_kernel reading from Q_rot at offset
        full_out = _to_device(torch.zeros(TILE, hidden, dtype=torch.bfloat16), device)
        head_attn_kernels[h](Q_rot, K_T, V, mask, sc, asc, full_out)
        new_out_t = ttnn.to_torch(full_out)

        pcc = torch.corrcoef(torch.stack([
            new_out_t[0, col_s:col_e].float(),
            ref_out_t[0, :head_dim].float(),
        ]))[0, 1].item()
        print(f"  head {h:2d}: PCC={pcc:.6f} {'PASS' if pcc > 0.98 else 'FAIL'}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_head_attn(device)
    finally:
        ttnn.close_device(device)
