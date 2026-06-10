# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fused per-card sliding attention atom (pre all-reduce), stage A.

One @ttl.atom on an (8, 2) grid; the only DRAM crossings inside are the
xn scratch row (block subviews are not available yet, #671) and the
KV cache. Column owns one head: cols 0-3 q, 4-5 k, 6-7 v.

  norm   (0,0) computes xn = x * invrms * (1+w) -> xn scratch
  qkv    [1,88t]@[88t,8t] per column, Kp=2 reduce -> head tile [1,8t]
  head   q/k: RMS-norm, then rotate-half RoPE as h*cos + (h@R)*sin with
         R the rotation permutation; v passthrough; heads -> DRAM rows
"""

import ttl
from ttl.ops.gemv import make_gemv_band_core
from ttl.ops.kv_append import make_kv_patch_core
from ttl.ops.mcast import mcast_block
from ttl.ops.pipe_util import pipe_send, pipe_recv
from ttl.ops.rmsnorm import make_rmsnorm_core
from ttl.ops.rope import make_rope_core
from ttl.ops.flash_decode import make_flash_window_core

TILE = 32


def make_attn_heads_atom(Ht, Dt, eps):
    """Stage A: norm -> QKV -> QK-norm/RoPE as ttl.ops core composition."""
    K_BAND = Ht // 2
    K_CH = K_BAND // 2

    norm_core = make_rmsnorm_core(Ht, K_CH, Ht * TILE, eps)
    band_core = make_gemv_band_core(K_CH, 2, Dt)
    head_norm_core = make_rmsnorm_core(Dt, Dt, Dt * TILE, eps)
    rope_core = make_rope_core(Dt)

    @ttl.atom(grid=(9, 2), fp32_dest_acc_en=True)
    def attn_heads(x, gamma, wqkv, cos, sin, qknorm, rot, heads):
        col_c, row_c = ttl.node(dims=2)

        nx_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        g_cb = ttl.make_dataflow_buffer_like(gamma, shape=(1, K_CH), block_count=2)
        xn_stage = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)

        xb_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        w_cb = ttl.make_dataflow_buffer_like(wqkv, shape=(K_CH, Dt), block_count=2)
        part_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        send_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        recv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        head_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=3)

        hg_cb = ttl.make_dataflow_buffer_like(qknorm, shape=(1, Dt), block_count=1)
        hn_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        c_cb = ttl.make_dataflow_buffer_like(cos, shape=(1, Dt), block_count=1)
        s_cb = ttl.make_dataflow_buffer_like(sin, shape=(1, Dt), block_count=1)
        r_cb = ttl.make_dataflow_buffer_like(rot, shape=(Dt, Dt), block_count=1)
        out_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)

        band0 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 0))])
        band1 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 1))])
        qkv_red = ttl.PipeNet([ttl.Pipe(src=(c, 1), dst=(c, 0)) for c in range(1, 9)])

        if col_c == 0 and row_c == 0:
            for c in range(4):
                xd = nx_cb.reserve(); ttl.copy(x[0:1, c * K_CH:(c + 1) * K_CH], xd)
            for c in range(4):
                xd = nx_cb.reserve(); ttl.copy(x[0:1, c * K_CH:(c + 1) * K_CH], xd)
                gd = g_cb.reserve(); ttl.copy(gamma[0:1, c * K_CH:(c + 1) * K_CH], gd)
            norm_core(nx_cb, g_cb, xn_stage)
        for ch in range(2):
            mcast_block(band0, xn_stage, xb_cb)
        for ch in range(2):
            mcast_block(band1, xn_stage, xb_cb)

        if col_c >= 1:
            kr = row_c * K_BAND
            nd = (col_c - 1) * Dt


            band_core(xb_cb, w_cb, part_cb)
            for ch in range(2):
                wd = w_cb.reserve()
                ttl.copy(wqkv[kr + ch * K_CH:kr + (ch + 1) * K_CH, nd:nd + Dt], wd)

            if row_c == 0:
                gq = hg_cb.reserve(); ttl.copy(qknorm[0:1, 0:Dt], gq)
                pipe_recv(qkv_red, recv_cb)
                hd = head_cb.reserve()
                hd.store(ttl.add(part_cb.wait(), recv_cb.wait()))
                h = head_cb.wait()
                h2 = head_cb.reserve(); h2.store(h)
                h3 = head_cb.reserve(); h3.store(h)
                head_norm_core(head_cb, hg_cb, hn_cb)
                if col_c < 7:
                    cd = c_cb.reserve(); ttl.copy(cos[0:1, 0:Dt], cd)
                    sd2 = s_cb.reserve(); ttl.copy(sin[0:1, 0:Dt], sd2)
                    rd = r_cb.reserve(); ttl.copy(rot[0:Dt, 0:Dt], rd)
                    rope_core(hn_cb, c_cb, s_cb, r_cb, out_cb)
                else:
                    ob = out_cb.reserve()
                    ob.store(hn_cb.wait())
                ttl.copy(out_cb.wait(), heads[col_c - 1:col_c, 0:Dt])
            else:
                sd = send_cb.reserve(); sd.store(part_cb.wait())
                pipe_send(qkv_red, send_cb)

    return attn_heads


def make_attn_patch_atom(Ht, Dt, St, eps):
    """Stage A (norm, QKV, QK-norm/RoPE) + KV cache patch on a (9, 2) grid.

    Cols 1-4 write rotated q heads to ``q_heads`` rows; cols 5-8 patch the
    ring caches at the runtime position. Flash runs as the next dispatch:
    the kv->q ready handshake inside one atom deadlocks today (any DM copy
    before a pipe_send hangs; see Gemma4ImplNotes), so dispatch order is
    the synchronization.
    """
    K_BAND = Ht // 2
    K_CH = K_BAND // 2

    norm_core = make_rmsnorm_core(Ht, K_CH, Ht * TILE, eps)
    band_core = make_gemv_band_core(K_CH, 2, Dt)
    head_norm_core = make_rmsnorm_core(Dt, Dt, Dt * TILE, eps)
    rope_core = make_rope_core(Dt)
    patch_core = make_kv_patch_core(Dt)

    @ttl.atom(grid=(9, 2), fp32_dest_acc_en=True)
    def attn_patch(x, gamma, wqkv, cos, sin, qknorm, rot, kc0, kc1, vc0, vc1,
                   pos_t, q_heads):
        col_c, row_c = ttl.node(dims=2)

        nx_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        g_cb = ttl.make_dataflow_buffer_like(gamma, shape=(1, K_CH), block_count=2)
        xn_stage = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)

        xb_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        w_cb = ttl.make_dataflow_buffer_like(wqkv, shape=(K_CH, Dt), block_count=1)
        part_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)
        send_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        recv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        head_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=3)

        hn_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)
        c_cb = ttl.make_dataflow_buffer_like(cos, shape=(1, Dt), block_count=2)
        hg_cb = ttl.make_dataflow_buffer_like(qknorm, shape=(1, Dt), block_count=1)
        s_cb = ttl.make_dataflow_buffer_like(cos, shape=(1, Dt), block_count=1)
        r_cb = ttl.make_dataflow_buffer_like(rot, shape=(Dt, Dt), block_count=1)
        out_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)

        pos_cb = ttl.make_dataflow_buffer_like(pos_t, shape=(1, 1), block_count=1)
        band_cb = ttl.make_dataflow_buffer_like(kc0, shape=(1, Dt), block_count=1)

        band0 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 0))])
        band1 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 1))])
        qkv_red = ttl.PipeNet([ttl.Pipe(src=(c, 1), dst=(c, 0)) for c in range(1, 9)])

        if col_c == 0 and row_c == 0:
            for c in range(4):
                xd = nx_cb.reserve(); ttl.copy(x[0:1, c * K_CH:(c + 1) * K_CH], xd)
            for c in range(4):
                xd = nx_cb.reserve(); ttl.copy(x[0:1, c * K_CH:(c + 1) * K_CH], xd)
                gd = g_cb.reserve(); ttl.copy(gamma[0:1, c * K_CH:(c + 1) * K_CH], gd)
            norm_core(nx_cb, g_cb, xn_stage)
        for ch in range(2):
            mcast_block(band0, xn_stage, xb_cb)
        for ch in range(2):
            mcast_block(band1, xn_stage, xb_cb)

        if col_c >= 1:
            kr = row_c * K_BAND
            nd = (col_c - 1) * Dt


            band_core(xb_cb, w_cb, part_cb)
            for ch in range(2):
                wd = w_cb.reserve()
                ttl.copy(wqkv[kr + ch * K_CH:kr + (ch + 1) * K_CH, nd:nd + Dt], wd)

            if row_c == 0:
                gq = hg_cb.reserve(); ttl.copy(qknorm[0:1, 0:Dt], gq)
                pipe_recv(qkv_red, recv_cb)
                hd = head_cb.reserve()
                hd.store(ttl.add(part_cb.wait(), recv_cb.wait()))
                h = head_cb.wait()
                h2 = head_cb.reserve(); h2.store(h)
                h3 = head_cb.reserve(); h3.store(h)
                head_norm_core(head_cb, hg_cb, hn_cb)
                if col_c < 7:
                    cd = c_cb.reserve(); ttl.copy(cos[0:1, 0:Dt], cd)
                    sd2 = s_cb.reserve(); ttl.copy(sin[0:1, 0:Dt], sd2)
                    rd = r_cb.reserve(); ttl.copy(rot[0:Dt, 0:Dt], rd)
                    rope_core(hn_cb, c_cb, s_cb, r_cb, out_cb)
                else:
                    ob = out_cb.reserve()
                    ob.store(hn_cb.wait())

                if col_c >= 5:
                    pd = pos_cb.reserve(); ttl.copy(pos_t[0, 0], pd)
                    pb = pos_cb.wait()
                    rr = ttl.read_index(pb, 0, 0)
                    intra = ttl.read_index(pb, 0, 1)
                    bd = band_cb.reserve()
                    if col_c == 5:
                        ttl.copy(kc0[rr:rr + 1, 0:Dt], bd)
                    elif col_c == 6:
                        ttl.copy(kc1[rr:rr + 1, 0:Dt], bd)
                    elif col_c == 7:
                        ttl.copy(vc0[rr:rr + 1, 0:Dt], bd)
                    else:
                        ttl.copy(vc1[rr:rr + 1, 0:Dt], bd)
                    bb = band_cb.wait()
                    patch_core(out_cb, bb, intra)
                    if col_c == 5:
                        ttl.copy(bb, kc0[rr:rr + 1, 0:Dt])
                    elif col_c == 6:
                        ttl.copy(bb, kc1[rr:rr + 1, 0:Dt])
                    elif col_c == 7:
                        ttl.copy(bb, vc0[rr:rr + 1, 0:Dt])
                    else:
                        ttl.copy(bb, vc1[rr:rr + 1, 0:Dt])
                else:
                    ttl.copy(out_cb.wait(), q_heads[col_c - 1:col_c, 0:Dt])

    return attn_patch


def make_flash_atom(Dt, St):
    """Flash decode over patched caches on a (4, 1) grid; one head per col.

    Reads the rotated q row from ``q_heads`` (written by the patch atom in
    the prior dispatch) and the full ring cache, writes normalized o rows.
    """
    n_chunks = 4
    chunk_t = St // n_chunks
    flash_core = make_flash_window_core(1, chunk_t, Dt, n_chunks)

    @ttl.atom(grid=(4, 1), fp32_dest_acc_en=True)
    def flash_atom(q_heads, kc0, kc1, vc0, vc1, masks, o_heads):
        col_c, row_c = ttl.node(dims=2)

        q_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)
        fkv_cb = ttl.make_dataflow_buffer_like(kc0, shape=(chunk_t, Dt), block_count=1)
        fmask_cb = ttl.make_dataflow_buffer_like(masks, shape=(1, chunk_t), block_count=1)
        fo_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=2)
        fm_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        fl_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        ostage_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)

        qw = q_cb.reserve()
        ttl.copy(q_heads[col_c:col_c + 1, 0:Dt], qw)
        for c in range(n_chunks):
            kd = fkv_cb.reserve()
            vd = fkv_cb.reserve()
            if col_c < 2:
                ttl.copy(kc0[c * chunk_t:(c + 1) * chunk_t, 0:Dt], kd)
                ttl.copy(vc0[c * chunk_t:(c + 1) * chunk_t, 0:Dt], vd)
            else:
                ttl.copy(kc1[c * chunk_t:(c + 1) * chunk_t, 0:Dt], kd)
                ttl.copy(vc1[c * chunk_t:(c + 1) * chunk_t, 0:Dt], vd)
            md = fmask_cb.reserve()
            ttl.copy(masks[c:c + 1, 0:chunk_t], md)
        flash_core(q_cb, fkv_cb, fkv_cb, fmask_cb, fo_cb, fm_cb, fl_cb)
        ofin = fo_cb.wait()
        mfin = fm_cb.wait()
        lfin = fl_cb.wait()
        os_w = ostage_cb.reserve()
        os_w.store(ttl.add(
            ttl.mul(ofin, ttl.block.broadcast(ttl.recip(lfin), dims=[1], shape=(1, Dt))),
            ttl.mul(ttl.block.broadcast(mfin, dims=[1], shape=(1, Dt)),
                    ttl.block.fill(0.0, shape=(1, Dt)))))
        ttl.copy(ostage_cb.wait(), o_heads[col_c:col_c + 1, 0:Dt])

    return flash_atom
