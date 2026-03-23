# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Multi-core attention: 4 cores compute partial attention, 1 core reduces.

Splits 16 cache tiles across 4 cores (4 tiles each). Each core runs
full online softmax on its 4 tiles. A reduce kernel merges the 4 partial
results using online softmax reduction.

Per head: 2 kernel calls (partial + reduce).
Partial: 4 cores × 4 tiles each ≈ 35μs (vs 139μs single-core)
Reduce: 4 partials merge ≈ 25μs
Total: ≈ 60μs (2.3x speedup over 139μs)
"""

import torch
import ttl
import ttnn

TILE = 32
NUM_CORES = 4
TILES_PER_CORE = 4  # 16 cache tiles / 4 cores


def _make_partial_attn_kernel(q_col):
    """4-core partial attention. Each core does 4 tiles with online softmax."""

    @ttl.kernel(grid=(1, NUM_CORES))
    def partial_attn_kernel(Q_rot, K_T, V, mask, scaler, attn_scale,
                            part_m, part_d, part_o0, part_o1):
        """Each core processes 4 cache tiles, writes partial m/d/o0/o1.

        part_m/d/o0/o1: [TILE, 4*TILE] = [32, 128] = [1, 4] tiles
        Core nid writes to position [0, nid].
        """
        Kt_q = 2
        Nt_per_core = TILES_PER_CORE  # 4

        # Reader DFBs
        q_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(K_T, shape=(1, 1), buffer_factor=2)
        m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        asc_dfb = ttl.make_dataflow_buffer_like(attn_scale, shape=(1, 1), buffer_factor=2)

        # Compute-local DFBs (same as fused_attn flash attention)
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
        o0_save_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        o1_save_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        o0_tmp_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        o1_tmp_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)

        # Writer DFBs
        out_m_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        out_d_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        out_o0_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        out_o1_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)

        y_size, x_size = ttl.grid_size(dims=2)

        @ttl.datamovement()
        def read():
            node_y, node_x = ttl.node(dims=2)
            nid = node_y * x_size + node_x
            kt_start = nid * Nt_per_core  # 0, 4, 8, 12

            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk)
                tx.wait()
            with asc_dfb.reserve() as blk:
                tx = ttl.copy(attn_scale[0, 0], blk)
                tx.wait()

            for kt_local in range(Nt_per_core):
                kt = kt_start + kt_local
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
                for vi in range(Kt_q):
                    with v_dfb.reserve() as blk:
                        tx = ttl.copy(V[kt, vi], blk)
                        tx.wait()

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc_blk, asc_dfb.wait() as attn_sc:
                # === First tile (local kt=0) ===
                with q_dfb.wait() as q0, k_dfb.wait() as k0:
                    with acc_dfb.reserve() as acc:
                        acc.store(q0 @ k0)
                with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + q1 @ k1)
                with acc_dfb.wait() as score:
                    with scaled_dfb.reserve() as s:
                        s.store(score * attn_sc)
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

                # === Remaining tiles (local kt=1..3) ===
                for _ in range(Nt_per_core - 1):
                    with q_dfb.wait() as q0, k_dfb.wait() as k0:
                        with acc_dfb.reserve() as acc:
                            acc.store(q0 @ k0)
                    with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + q1 @ k1)
                    with acc_dfb.wait() as score:
                        with scaled_dfb.reserve() as s:
                            s.store(score * attn_sc)
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

                # Write partial results (unnormalized — reduce kernel normalizes)
                with m_save_dfb.wait() as m_final:
                    with out_m_dfb.reserve() as om:
                        om.store(m_final)
                with d_save_dfb.wait() as d_final:
                    with out_d_dfb.reserve() as od:
                        od.store(d_final)
                with o0_save_dfb.wait() as o0_final:
                    with out_o0_dfb.reserve() as oo0:
                        oo0.store(o0_final)
                with o1_save_dfb.wait() as o1_final:
                    with out_o1_dfb.reserve() as oo1:
                        oo1.store(o1_final)

        @ttl.datamovement()
        def write():
            node_y, node_x = ttl.node(dims=2)
            nid = node_y * x_size + node_x
            with out_m_dfb.wait() as blk:
                tx = ttl.copy(blk, part_m[0, nid])
                tx.wait()
            with out_d_dfb.wait() as blk:
                tx = ttl.copy(blk, part_d[0, nid])
                tx.wait()
            with out_o0_dfb.wait() as blk:
                tx = ttl.copy(blk, part_o0[0, nid])
                tx.wait()
            with out_o1_dfb.wait() as blk:
                tx = ttl.copy(blk, part_o1[0, nid])
                tx.wait()

    return partial_attn_kernel


def _make_reduce_attn_kernel(out_col):
    """Merge 4 partial results using online softmax reduction."""

    @ttl.kernel(grid=(1, 1))
    def reduce_attn_kernel(part_m, part_d, part_o0, part_o1, attn_out):
        """Merge 4 partial attention results.

        part_m/d/o0/o1: [TILE, 4*TILE] = [1, 4] tiles
        attn_out: [TILE, 896] — output at column offset
        """
        Nt = NUM_CORES  # 4

        pm_dfb = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        pd_dfb = ttl.make_dataflow_buffer_like(part_d, shape=(1, 1), buffer_factor=2)
        po0_dfb = ttl.make_dataflow_buffer_like(part_o0, shape=(1, 1), buffer_factor=2)
        po1_dfb = ttl.make_dataflow_buffer_like(part_o1, shape=(1, 1), buffer_factor=2)
        m_save = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        m_old_copy = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        d_save = ttl.make_dataflow_buffer_like(part_d, shape=(1, 1), buffer_factor=2)
        d_tmp = ttl.make_dataflow_buffer_like(part_d, shape=(1, 1), buffer_factor=2)
        o0_save = ttl.make_dataflow_buffer_like(part_o0, shape=(1, 1), buffer_factor=2)
        o1_save = ttl.make_dataflow_buffer_like(part_o1, shape=(1, 1), buffer_factor=2)
        o0_tmp = ttl.make_dataflow_buffer_like(part_o0, shape=(1, 1), buffer_factor=2)
        o1_tmp = ttl.make_dataflow_buffer_like(part_o1, shape=(1, 1), buffer_factor=2)
        alpha_old_dfb = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        alpha_new_dfb = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        alpha_old_bc = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        alpha_new_bc = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        tmp_dfb = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)

        @ttl.datamovement()
        def read():
            for i in range(Nt):
                with pm_dfb.reserve() as blk:
                    tx = ttl.copy(part_m[0, i], blk)
                    tx.wait()
                with pd_dfb.reserve() as blk:
                    tx = ttl.copy(part_d[0, i], blk)
                    tx.wait()
                with po0_dfb.reserve() as blk:
                    tx = ttl.copy(part_o0[0, i], blk)
                    tx.wait()
                with po1_dfb.reserve() as blk:
                    tx = ttl.copy(part_o1[0, i], blk)
                    tx.wait()

        @ttl.compute()
        def compute():
            # Init from first partial
            with pm_dfb.wait() as m0:
                with m_save.reserve() as ms:
                    ms.store(m0)
            with pd_dfb.wait() as d0:
                with d_save.reserve() as ds:
                    ds.store(d0)
            with po0_dfb.wait() as o00:
                with o0_save.reserve() as os0:
                    os0.store(o00)
            with po1_dfb.wait() as o10:
                with o1_save.reserve() as os1:
                    os1.store(o10)

            # Merge remaining 3 partials
            for _ in range(Nt - 1):
                with pm_dfb.wait() as m_tile:
                    with m_save.wait() as m_old:
                        with m_save.reserve() as m_new:
                            m_new.store(ttl.math.max(m_old, m_tile))
                        with m_old_copy.reserve() as mo:
                            mo.store(m_old)
                    with m_save.wait() as m_new_val, m_old_copy.wait() as m_old_val:
                        with alpha_old_dfb.reserve() as ao:
                            ao.store(ttl.math.exp(m_old_val - m_new_val))
                        with alpha_new_dfb.reserve() as an:
                            an.store(ttl.math.exp(m_tile - m_new_val))
                        with m_save.reserve() as ms:
                            ms.store(m_new_val)
                with alpha_old_dfb.wait() as ao:
                    with alpha_old_bc.reserve() as aobc:
                        aobc.store(ttl.math.broadcast(ao, aobc, dims=[0, 1]))
                with alpha_new_dfb.wait() as an:
                    with alpha_new_bc.reserve() as anbc:
                        anbc.store(ttl.math.broadcast(an, anbc, dims=[0, 1]))
                with alpha_old_bc.wait() as ao_bc, alpha_new_bc.wait() as an_bc:
                    with pd_dfb.wait() as d_tile:
                        with d_save.wait() as d_old:
                            with d_tmp.reserve() as dt:
                                dt.store(d_old * ao_bc)
                        with d_tmp.wait() as dt:
                            with d_save.reserve() as d_new:
                                d_new.store(dt + d_tile * an_bc)
                    with po0_dfb.wait() as o0_tile:
                        with o0_save.wait() as o0_old:
                            with o0_tmp.reserve() as ot:
                                ot.store(o0_old * ao_bc)
                        with o0_tmp.wait() as ot:
                            with o0_save.reserve() as o0_new:
                                o0_new.store(ot + o0_tile * an_bc)
                    with po1_dfb.wait() as o1_tile:
                        with o1_save.wait() as o1_old:
                            with o1_tmp.reserve() as ot:
                                ot.store(o1_old * ao_bc)
                        with o1_tmp.wait() as ot:
                            with o1_save.reserve() as o1_new:
                                o1_new.store(ot + o1_tile * an_bc)

            # Final normalize
            with d_save.wait() as d_final:
                with tmp_dfb.reserve() as d_bc:
                    d_bc.store(ttl.math.broadcast(d_final, d_bc, dims=[0, 1]))
            with tmp_dfb.wait() as d_bc:
                with o0_save.wait() as o0_final:
                    with out_dfb.reserve() as o:
                        o.store(o0_final * ttl.math.recip(d_bc))
                with o1_save.wait() as o1_final:
                    with out_dfb.reserve() as o:
                        o.store(o1_final * ttl.math.recip(d_bc))

        @ttl.datamovement()
        def write():
            for vi in range(2):
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, attn_out[0, out_col + vi])
                    tx.wait()

    return reduce_attn_kernel


# Create 14 variants
partial_attn_kernels = [_make_partial_attn_kernel(q_col=h * 2) for h in range(14)]
reduce_attn_kernels = [_make_reduce_attn_kernel(out_col=h * 2) for h in range(14)]


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_multicore_attn(device):
    """Compare multi-core (4 cores) attention vs single-core."""
    import math
    from group_attn import head_attn_kernels

    print("Multi-core attention test (4 cores × 4 tiles):")
    HD, MAX_SEQ, H = 64, 512, 896
    PART_SIZE = NUM_CORES * TILE  # 128

    Q_rot_t = torch.randn(TILE, H, dtype=torch.bfloat16) * 0.5
    K_T_t = torch.randn(HD, MAX_SEQ, dtype=torch.bfloat16) * 0.1
    V_t = torch.randn(MAX_SEQ, HD, dtype=torch.bfloat16) * 0.1
    mask_t = torch.full((TILE, MAX_SEQ), float("-inf"), dtype=torch.bfloat16)
    mask_t[0, :20] = 0.0
    sc_t = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    asc_t = torch.full((TILE, TILE), 1.0 / math.sqrt(HD), dtype=torch.bfloat16)

    Q_rot = _to_device(Q_rot_t, device)
    K_T = _to_device(K_T_t, device)
    V = _to_device(V_t, device)
    mask = _to_device(mask_t, device)
    sc = _to_device(sc_t, device)
    asc = _to_device(asc_t, device)

    for h in [0, 7, 13]:
        col_s = h * HD
        col_e = col_s + HD

        # Reference
        ref_out = _to_device(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
        head_attn_kernels[h](Q_rot, K_T, V, mask, sc, asc, ref_out)
        ref_t = ttnn.to_torch(ref_out)

        # Multi-core
        pm = _to_device(torch.zeros(TILE, PART_SIZE, dtype=torch.bfloat16), device)
        pd = _to_device(torch.zeros(TILE, PART_SIZE, dtype=torch.bfloat16), device)
        po0 = _to_device(torch.zeros(TILE, PART_SIZE, dtype=torch.bfloat16), device)
        po1 = _to_device(torch.zeros(TILE, PART_SIZE, dtype=torch.bfloat16), device)
        mc_out = _to_device(torch.zeros(TILE, H, dtype=torch.bfloat16), device)

        partial_attn_kernels[h](Q_rot, K_T, V, mask, sc, asc, pm, pd, po0, po1)
        reduce_attn_kernels[h](pm, pd, po0, po1, mc_out)
        mc_t = ttnn.to_torch(mc_out)

        pcc = torch.corrcoef(torch.stack([
            mc_t[0, col_s:col_e].float(), ref_t[0, col_s:col_e].float()
        ]))[0, 1].item()
        print(f"  head {h:2d}: PCC={pcc:.6f} {'PASS' if pcc > 0.98 else 'FAIL'}")


# =========================================================================
# Parallel-group variants: all 7 heads on 28 cores (1 launch per KV group)
# =========================================================================
HEADS_PER_GROUP = 7
GRID_Y_PAR = 4  # 28 = 4 × 7
GRID_X_PAR = 7
TOTAL_PAR_CORES = HEADS_PER_GROUP * TILES_PER_CORE  # 28


def _make_parallel_partial_kernel(q_col_base):
    """28-core kernel: 7 heads × 4 tiles/head, all in parallel."""

    @ttl.kernel(grid=(GRID_Y_PAR, GRID_X_PAR))
    def parallel_partial_kernel(Q_rot, K_T, V, mask, scaler, attn_scale,
                                 part_m, part_d, part_o0, part_o1):
        Kt_q = 2
        Nt_per_core = TILES_PER_CORE

        q_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(K_T, shape=(1, 1), buffer_factor=2)
        m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        asc_dfb = ttl.make_dataflow_buffer_like(attn_scale, shape=(1, 1), buffer_factor=2)
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
        o0_save_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        o1_save_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        o0_tmp_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        o1_tmp_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        out_m_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        out_d_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        out_o0_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
        out_o1_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)

        y_size, x_size = ttl.grid_size(dims=2)

        @ttl.datamovement()
        def read():
            node_y, node_x = ttl.node(dims=2)
            nid = node_y * x_size + node_x
            head_local = nid // Nt_per_core
            tile_group = nid % Nt_per_core
            q_col = q_col_base + head_local * Kt_q
            kt_start = tile_group * Nt_per_core
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk)
                tx.wait()
            with asc_dfb.reserve() as blk:
                tx = ttl.copy(attn_scale[0, 0], blk)
                tx.wait()
            for kt_local in range(Nt_per_core):
                kt = kt_start + kt_local
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
                for vi in range(Kt_q):
                    with v_dfb.reserve() as blk:
                        tx = ttl.copy(V[kt, vi], blk)
                        tx.wait()

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc_blk, asc_dfb.wait() as attn_sc:
                # First tile
                with q_dfb.wait() as q0, k_dfb.wait() as k0:
                    with acc_dfb.reserve() as acc:
                        acc.store(q0 @ k0)
                with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + q1 @ k1)
                with acc_dfb.wait() as score:
                    with scaled_dfb.reserve() as s:
                        s.store(score * attn_sc)
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
                # Remaining tiles
                for _ in range(Nt_per_core - 1):
                    with q_dfb.wait() as q0, k_dfb.wait() as k0:
                        with acc_dfb.reserve() as acc:
                            acc.store(q0 @ k0)
                    with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + q1 @ k1)
                    with acc_dfb.wait() as score:
                        with scaled_dfb.reserve() as s:
                            s.store(score * attn_sc)
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
                with m_save_dfb.wait() as m_final:
                    with out_m_dfb.reserve() as om:
                        om.store(m_final)
                with d_save_dfb.wait() as d_final:
                    with out_d_dfb.reserve() as od:
                        od.store(d_final)
                with o0_save_dfb.wait() as o0_final:
                    with out_o0_dfb.reserve() as oo0:
                        oo0.store(o0_final)
                with o1_save_dfb.wait() as o1_final:
                    with out_o1_dfb.reserve() as oo1:
                        oo1.store(o1_final)

        @ttl.datamovement()
        def write():
            node_y, node_x = ttl.node(dims=2)
            nid = node_y * x_size + node_x
            with out_m_dfb.wait() as blk:
                tx = ttl.copy(blk, part_m[0, nid])
                tx.wait()
            with out_d_dfb.wait() as blk:
                tx = ttl.copy(blk, part_d[0, nid])
                tx.wait()
            with out_o0_dfb.wait() as blk:
                tx = ttl.copy(blk, part_o0[0, nid])
                tx.wait()
            with out_o1_dfb.wait() as blk:
                tx = ttl.copy(blk, part_o1[0, nid])
                tx.wait()

    return parallel_partial_kernel


def _make_parallel_reduce_kernel(out_col_base):
    """7-core reduce: each core reduces 4 partials for its head."""

    @ttl.kernel(grid=(1, HEADS_PER_GROUP))
    def parallel_reduce_kernel(part_m, part_d, part_o0, part_o1, attn_out):
        Nt = TILES_PER_CORE
        y_size, x_size = ttl.grid_size(dims=2)

        pm_dfb = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        pd_dfb = ttl.make_dataflow_buffer_like(part_d, shape=(1, 1), buffer_factor=2)
        po0_dfb = ttl.make_dataflow_buffer_like(part_o0, shape=(1, 1), buffer_factor=2)
        po1_dfb = ttl.make_dataflow_buffer_like(part_o1, shape=(1, 1), buffer_factor=2)
        m_save = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        m_old_copy = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        d_save = ttl.make_dataflow_buffer_like(part_d, shape=(1, 1), buffer_factor=2)
        d_tmp = ttl.make_dataflow_buffer_like(part_d, shape=(1, 1), buffer_factor=2)
        o0_save = ttl.make_dataflow_buffer_like(part_o0, shape=(1, 1), buffer_factor=2)
        o1_save = ttl.make_dataflow_buffer_like(part_o1, shape=(1, 1), buffer_factor=2)
        o0_tmp = ttl.make_dataflow_buffer_like(part_o0, shape=(1, 1), buffer_factor=2)
        o1_tmp = ttl.make_dataflow_buffer_like(part_o1, shape=(1, 1), buffer_factor=2)
        alpha_old_dfb = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        alpha_new_dfb = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        alpha_old_bc = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        alpha_new_bc = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        tmp_dfb = ttl.make_dataflow_buffer_like(part_m, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)

        @ttl.datamovement()
        def read():
            node_y, node_x = ttl.node(dims=2)
            head_local = node_y * x_size + node_x
            base = head_local * Nt
            for i in range(Nt):
                idx = base + i
                with pm_dfb.reserve() as blk:
                    tx = ttl.copy(part_m[0, idx], blk)
                    tx.wait()
                with pd_dfb.reserve() as blk:
                    tx = ttl.copy(part_d[0, idx], blk)
                    tx.wait()
                with po0_dfb.reserve() as blk:
                    tx = ttl.copy(part_o0[0, idx], blk)
                    tx.wait()
                with po1_dfb.reserve() as blk:
                    tx = ttl.copy(part_o1[0, idx], blk)
                    tx.wait()

        @ttl.compute()
        def compute():
            with pm_dfb.wait() as m0:
                with m_save.reserve() as ms:
                    ms.store(m0)
            with pd_dfb.wait() as d0:
                with d_save.reserve() as ds:
                    ds.store(d0)
            with po0_dfb.wait() as o00:
                with o0_save.reserve() as os0:
                    os0.store(o00)
            with po1_dfb.wait() as o10:
                with o1_save.reserve() as os1:
                    os1.store(o10)
            for _ in range(Nt - 1):
                with pm_dfb.wait() as m_tile:
                    with m_save.wait() as m_old:
                        with m_save.reserve() as m_new:
                            m_new.store(ttl.math.max(m_old, m_tile))
                        with m_old_copy.reserve() as mo:
                            mo.store(m_old)
                    with m_save.wait() as m_new_val, m_old_copy.wait() as m_old_val:
                        with alpha_old_dfb.reserve() as ao:
                            ao.store(ttl.math.exp(m_old_val - m_new_val))
                        with alpha_new_dfb.reserve() as an:
                            an.store(ttl.math.exp(m_tile - m_new_val))
                        with m_save.reserve() as ms:
                            ms.store(m_new_val)
                with alpha_old_dfb.wait() as ao:
                    with alpha_old_bc.reserve() as aobc:
                        aobc.store(ttl.math.broadcast(ao, aobc, dims=[0, 1]))
                with alpha_new_dfb.wait() as an:
                    with alpha_new_bc.reserve() as anbc:
                        anbc.store(ttl.math.broadcast(an, anbc, dims=[0, 1]))
                with alpha_old_bc.wait() as ao_bc, alpha_new_bc.wait() as an_bc:
                    with pd_dfb.wait() as d_tile:
                        with d_save.wait() as d_old:
                            with d_tmp.reserve() as dt:
                                dt.store(d_old * ao_bc)
                        with d_tmp.wait() as dt:
                            with d_save.reserve() as d_new:
                                d_new.store(dt + d_tile * an_bc)
                    with po0_dfb.wait() as o0_tile:
                        with o0_save.wait() as o0_old:
                            with o0_tmp.reserve() as ot:
                                ot.store(o0_old * ao_bc)
                        with o0_tmp.wait() as ot:
                            with o0_save.reserve() as o0_new:
                                o0_new.store(ot + o0_tile * an_bc)
                    with po1_dfb.wait() as o1_tile:
                        with o1_save.wait() as o1_old:
                            with o1_tmp.reserve() as ot:
                                ot.store(o1_old * ao_bc)
                        with o1_tmp.wait() as ot:
                            with o1_save.reserve() as o1_new:
                                o1_new.store(ot + o1_tile * an_bc)
            with d_save.wait() as d_final:
                with tmp_dfb.reserve() as d_bc:
                    d_bc.store(ttl.math.broadcast(d_final, d_bc, dims=[0, 1]))
            with tmp_dfb.wait() as d_bc:
                with o0_save.wait() as o0_final:
                    with out_dfb.reserve() as o:
                        o.store(o0_final * ttl.math.recip(d_bc))
                with o1_save.wait() as o1_final:
                    with out_dfb.reserve() as o:
                        o.store(o1_final * ttl.math.recip(d_bc))

        @ttl.datamovement()
        def write():
            node_y, node_x = ttl.node(dims=2)
            head_local = node_y * x_size + node_x
            out_col = out_col_base + head_local * 2
            for vi in range(2):
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, attn_out[0, out_col + vi])
                    tx.wait()

    return parallel_reduce_kernel


# Two group variants (one per KV group)
parallel_partial_g0 = _make_parallel_partial_kernel(q_col_base=0)
parallel_partial_g1 = _make_parallel_partial_kernel(q_col_base=14)
parallel_reduce_g0 = _make_parallel_reduce_kernel(out_col_base=0)
parallel_reduce_g1 = _make_parallel_reduce_kernel(out_col_base=14)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_multicore_attn(device)
    finally:
        ttnn.close_device(device)
