# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Masked flash decode shards for Gemma-style attention.

Two single-purpose variants of ``ttl.ops.flash_mla.make_flash_shard_core``
(python-bool branches do not trace inside atom bodies, so each variant is
its own body):

- ``make_flash_decode_core``: streams K and V plus a per-chunk additive
  mask row (0 / -inf from the host covers ring validity and partial fill).
  Sliding-window layers.
- ``make_flash_decode_kev_core``: V is the K stream (k_eq_v); global
  layers, halves the cache stream.

Both leave unnormalized ``(o, m, l)`` in the output DFBs like the parent
op, so the same tree-reduce / normalize cores fuse downstream.
"""

import torch

import ttl
from ttl.ops.mcast import mcast, mcast_rows


def make_flash_decode_core(B, PNHt, DHt, vDHt, Sk_chunk_t, N_CHUNKS, scale=1.0):
    St_per_core = Sk_chunk_t * N_CHUNKS

    @ttl.atom()
    def flash_decode_core(q_in: ttl.DFB, k, v, masks, o_out: ttl.DFB, m_out: ttl.DFB, l_out: ttl.DFB):
        col_c, row_c = ttl.node(dims=2)

        k_cb = ttl.make_dataflow_buffer_like(k, shape=(Sk_chunk_t, DHt), block_count=2)
        v_cb = ttl.make_dataflow_buffer_like(k, shape=(Sk_chunk_t, vDHt), block_count=2)
        mask_cb = ttl.make_dataflow_buffer_like(masks, shape=(PNHt, Sk_chunk_t), block_count=2)

        sv_cb    = ttl.make_dfb("bf16", shape=(PNHt, Sk_chunk_t), block_count=2)
        ex_cb    = ttl.make_dfb("bf16", shape=(PNHt, Sk_chunk_t), block_count=2)
        red_cb   = ttl.make_dfb("bf16", shape=(PNHt, 1),          block_count=1)
        mn_cb    = ttl.make_dfb("bf16", shape=(PNHt, 1),          block_count=2)
        alpha_cb = ttl.make_dfb("bf16", shape=(PNHt, 1),          block_count=2)
        pv_cb    = ttl.make_dfb("bf16", shape=(PNHt, vDHt),       block_count=1)

        k_base = col_c * St_per_core
        for c in range(N_CHUNKS):
            kc = k_base + c * Sk_chunk_t
            k_dst = k_cb.reserve()
            ttl.copy(k[kc:kc + Sk_chunk_t, 0:DHt], k_dst)
            v_dst = v_cb.reserve()
            ttl.copy(v[kc:kc + Sk_chunk_t, 0:vDHt], v_dst)
            m_dst = mask_cb.reserve()
            ttl.copy(masks[c * PNHt:(c + 1) * PNHt, 0:Sk_chunk_t], m_dst)

        m0 = m_out.reserve(); m0.store(ttl.block.fill(-1e30, shape=m0.shape))
        l0 = l_out.reserve(); l0.store(ttl.block.fill(0.0,   shape=l0.shape))
        o0 = o_out.reserve(); o0.store(ttl.block.fill(0.0,   shape=o0.shape))

        q_blk = q_in.wait()
        for _ in range(N_CHUNKS):
            k_blk = ttl.math.typecast(k_cb.wait(), torch.bfloat16)
            mask_blk = mask_cb.wait()
            sv_w = sv_cb.reserve()
            sv_w.store(ttl.add(ttl.mul(q_blk @ ttl.transpose(k_blk),
                               ttl.block.fill(scale, shape=sv_w.shape)), mask_blk))

            sv = sv_cb.wait()
            cm_w = red_cb.reserve(); cm_w.store(ttl.math.reduce_max(sv, dims=[1]))
            sv_re = sv_cb.reserve(); sv_re.store(sv)

            m_old = m_out.wait()
            cm = red_cb.wait()
            mn_w = mn_cb.reserve(); mn_w.store(ttl.math.max(m_old, cm))

            mn_for_alpha = mn_cb.wait()
            alpha_w = alpha_cb.reserve()
            alpha_w.store(ttl.exp(ttl.sub(m_old, mn_for_alpha)))
            mn_re = mn_cb.reserve(); mn_re.store(mn_for_alpha)

            mn_for_state = mn_cb.wait()
            sv2 = sv_cb.wait()
            ex_w = ex_cb.reserve()
            ex_w.store(ttl.exp(ttl.sub(sv2, ttl.block.broadcast(
                mn_for_state, dims=[1], shape=sv2.shape))))
            m_next = m_out.reserve(); m_next.store(mn_for_state)

            ex = ex_cb.wait()
            cs_w = red_cb.reserve(); cs_w.store(ttl.math.reduce_sum(ex, dims=[1]))
            ex_re = ex_cb.reserve(); ex_re.store(ex)

            alpha = alpha_cb.wait()
            l_old = l_out.wait()
            cs = red_cb.wait()
            l_next = l_out.reserve()
            l_next.store(ttl.add(ttl.mul(alpha, l_old), cs))
            alpha_re = alpha_cb.reserve(); alpha_re.store(alpha)

            ex2 = ex_cb.wait()
            v_blk = ttl.math.typecast(v_cb.wait(), torch.bfloat16)
            pv_w = pv_cb.reserve(); pv_w.store(ex2 @ v_blk)

            alpha2 = alpha_cb.wait()
            o_old = o_out.wait()
            pv_blk = pv_cb.wait()
            o_next = o_out.reserve()
            o_next.store(ttl.add(ttl.mul(ttl.block.broadcast(
                alpha2, dims=[1], shape=o_old.shape), o_old), pv_blk))

    return flash_decode_core


def make_flash_decode_kev_core(B, PNHt, DHt, Sk_chunk_t, N_CHUNKS, scale=1.0):
    St_per_core = Sk_chunk_t * N_CHUNKS

    @ttl.atom()
    def flash_decode_kev_core(q_in: ttl.DFB, k, masks, o_out: ttl.DFB, m_out: ttl.DFB, l_out: ttl.DFB):
        col_c, row_c = ttl.node(dims=2)

        k_cb = ttl.make_dataflow_buffer_like(k, shape=(Sk_chunk_t, DHt), block_count=2)
        mask_cb = ttl.make_dataflow_buffer_like(masks, shape=(PNHt, Sk_chunk_t), block_count=2)

        sv_cb    = ttl.make_dfb("bf16", shape=(PNHt, Sk_chunk_t), block_count=2)
        ex_cb    = ttl.make_dfb("bf16", shape=(PNHt, Sk_chunk_t), block_count=2)
        red_cb   = ttl.make_dfb("bf16", shape=(PNHt, 1),          block_count=1)
        mn_cb    = ttl.make_dfb("bf16", shape=(PNHt, 1),          block_count=2)
        alpha_cb = ttl.make_dfb("bf16", shape=(PNHt, 1),          block_count=2)
        pv_cb    = ttl.make_dfb("bf16", shape=(PNHt, DHt),        block_count=1)

        k_base = col_c * St_per_core
        for c in range(N_CHUNKS):
            kc = k_base + c * Sk_chunk_t
            k_dst = k_cb.reserve()
            ttl.copy(k[kc:kc + Sk_chunk_t, 0:DHt], k_dst)
            m_dst = mask_cb.reserve()
            ttl.copy(masks[c * PNHt:(c + 1) * PNHt, 0:Sk_chunk_t], m_dst)

        m0 = m_out.reserve(); m0.store(ttl.block.fill(-1e30, shape=m0.shape))
        l0 = l_out.reserve(); l0.store(ttl.block.fill(0.0,   shape=l0.shape))
        o0 = o_out.reserve(); o0.store(ttl.block.fill(0.0,   shape=o0.shape))

        q_blk = q_in.wait()
        for _ in range(N_CHUNKS):
            k_blk = ttl.math.typecast(k_cb.wait(), torch.bfloat16)
            mask_blk = mask_cb.wait()
            sv_w = sv_cb.reserve()
            sv_w.store(ttl.add(ttl.mul(q_blk @ ttl.transpose(k_blk),
                               ttl.block.fill(scale, shape=sv_w.shape)), mask_blk))

            sv = sv_cb.wait()
            cm_w = red_cb.reserve(); cm_w.store(ttl.math.reduce_max(sv, dims=[1]))
            sv_re = sv_cb.reserve(); sv_re.store(sv)

            m_old = m_out.wait()
            cm = red_cb.wait()
            mn_w = mn_cb.reserve(); mn_w.store(ttl.math.max(m_old, cm))

            mn_for_alpha = mn_cb.wait()
            alpha_w = alpha_cb.reserve()
            alpha_w.store(ttl.exp(ttl.sub(m_old, mn_for_alpha)))
            mn_re = mn_cb.reserve(); mn_re.store(mn_for_alpha)

            mn_for_state = mn_cb.wait()
            sv2 = sv_cb.wait()
            ex_w = ex_cb.reserve()
            ex_w.store(ttl.exp(ttl.sub(sv2, ttl.block.broadcast(
                mn_for_state, dims=[1], shape=sv2.shape))))
            m_next = m_out.reserve(); m_next.store(mn_for_state)

            ex = ex_cb.wait()
            cs_w = red_cb.reserve(); cs_w.store(ttl.math.reduce_sum(ex, dims=[1]))
            ex_re = ex_cb.reserve(); ex_re.store(ex)

            alpha = alpha_cb.wait()
            l_old = l_out.wait()
            cs = red_cb.wait()
            l_next = l_out.reserve()
            l_next.store(ttl.add(ttl.mul(alpha, l_old), cs))
            alpha_re = alpha_cb.reserve(); alpha_re.store(alpha)

            ex2 = ex_cb.wait()
            pv_w = pv_cb.reserve(); pv_w.store(ex2 @ k_blk)

            alpha2 = alpha_cb.wait()
            o_old = o_out.wait()
            pv_blk = pv_cb.wait()
            o_next = o_out.reserve()
            o_next.store(ttl.add(ttl.mul(ttl.block.broadcast(
                alpha2, dims=[1], shape=o_old.shape), o_old), pv_blk))

    return flash_decode_kev_core


def make_flash_decode(n_cols, B, PNHt, DHt, vDHt, Sk_chunk_t, N_CHUNKS, scale=1.0):
    """Standalone masked decode shard (K and V streams)."""
    core = make_flash_decode_core(B, PNHt, DHt, vDHt, Sk_chunk_t, N_CHUNKS, scale)

    @ttl.atom(grid=(n_cols, B))
    def flash_decode(q, k, v, masks, out_o, out_m, out_l):
        col_c, row_c = ttl.node(dims=2)

        q_net = ttl.PipeNet(mcast_rows(B, n_cols))
        q_stage = ttl.make_dataflow_buffer_like(q, shape=(PNHt, DHt), block_count=1)
        q_recv = ttl.make_dataflow_buffer_like(q, shape=(PNHt, DHt), block_count=1)
        o_b = ttl.make_dataflow_buffer_like(out_o, shape=(PNHt, vDHt), block_count=2)
        m_b = ttl.make_dataflow_buffer_like(out_m, shape=(PNHt, 1), block_count=2)
        l_b = ttl.make_dataflow_buffer_like(out_l, shape=(PNHt, 1), block_count=2)
        o_d = ttl.make_dataflow_buffer_like(out_o, shape=(PNHt, vDHt), block_count=2)
        m_d = ttl.make_dataflow_buffer_like(out_m, shape=(PNHt, 1), block_count=2)
        l_d = ttl.make_dataflow_buffer_like(out_l, shape=(PNHt, 1), block_count=2)

        mcast(q_net, q[0:PNHt, 0:DHt], q_stage, q_recv)
        core(q_recv, k, v, masks, o_b, m_b, l_b)

        m_dw = m_d.reserve(); m_dw.store(m_b.wait())
        l_dw = l_d.reserve(); l_dw.store(l_b.wait())
        o_dw = o_d.reserve(); o_dw.store(o_b.wait())

        base = (row_c * n_cols + col_c) * PNHt
        ttl.copy(m_d.wait(), out_m[base:base + PNHt, 0:1])
        ttl.copy(l_d.wait(), out_l[base:base + PNHt, 0:1])
        ttl.copy(o_d.wait(), out_o[base:base + PNHt, 0:vDHt])

    return flash_decode


def make_flash_decode_kev(n_cols, B, PNHt, DHt, Sk_chunk_t, N_CHUNKS, scale=1.0):
    """Standalone masked decode shard with V = K (global layers)."""
    core = make_flash_decode_kev_core(B, PNHt, DHt, Sk_chunk_t, N_CHUNKS, scale)

    @ttl.atom(grid=(n_cols, B))
    def flash_decode_kev(q, k, masks, out_o, out_m, out_l):
        col_c, row_c = ttl.node(dims=2)

        q_net = ttl.PipeNet(mcast_rows(B, n_cols))
        q_stage = ttl.make_dataflow_buffer_like(q, shape=(PNHt, DHt), block_count=1)
        q_recv = ttl.make_dataflow_buffer_like(q, shape=(PNHt, DHt), block_count=1)
        o_b = ttl.make_dataflow_buffer_like(out_o, shape=(PNHt, DHt), block_count=2)
        m_b = ttl.make_dataflow_buffer_like(out_m, shape=(PNHt, 1), block_count=2)
        l_b = ttl.make_dataflow_buffer_like(out_l, shape=(PNHt, 1), block_count=2)
        o_d = ttl.make_dataflow_buffer_like(out_o, shape=(PNHt, DHt), block_count=2)
        m_d = ttl.make_dataflow_buffer_like(out_m, shape=(PNHt, 1), block_count=2)
        l_d = ttl.make_dataflow_buffer_like(out_l, shape=(PNHt, 1), block_count=2)

        mcast(q_net, q[0:PNHt, 0:DHt], q_stage, q_recv)
        core(q_recv, k, masks, o_b, m_b, l_b)

        m_dw = m_d.reserve(); m_dw.store(m_b.wait())
        l_dw = l_d.reserve(); l_dw.store(l_b.wait())
        o_dw = o_d.reserve(); o_dw.store(o_b.wait())

        base = (row_c * n_cols + col_c) * PNHt
        ttl.copy(m_d.wait(), out_m[base:base + PNHt, 0:1])
        ttl.copy(l_d.wait(), out_l[base:base + PNHt, 0:1])
        ttl.copy(o_d.wait(), out_o[base:base + PNHt, 0:DHt])

    return flash_decode_kev
