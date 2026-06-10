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
        ng_cb = ttl.make_dataflow_buffer_like(gamma, shape=(1, K_CH), block_count=2)
        nsq_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        nred_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        nacc_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        ninv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
        xn_stage = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)

        xb_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        w_cb = ttl.make_dataflow_buffer_like(wqkv, shape=(K_CH, Dt), block_count=2)
        part_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        send_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        recv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        head_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=2)

        hx_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=2)
        hg_cb = ttl.make_dataflow_buffer_like(qknorm, shape=(1, Dt), block_count=2)
        hsq_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        hred_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        hacc_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        hinv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
        hn_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        c_cb = ttl.make_dataflow_buffer_like(cos, shape=(1, Dt), block_count=1)
        s_cb = ttl.make_dataflow_buffer_like(sin, shape=(1, Dt), block_count=1)
        r_cb = ttl.make_dataflow_buffer_like(rot, shape=(Dt, Dt), block_count=1)
        rh_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        out_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)

        band0 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 0))])
        band1 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 1))])
        qkv_red = ttl.PipeNet([ttl.Pipe(src=(c, 1), dst=(c, 0)) for c in range(1, 9)])

        if col_c == 0 and row_c == 0:
            for c in range(4):
                xd = nx_cb.reserve(); ttl.copy(x[0:1, c * K_CH:(c + 1) * K_CH], xd)
            for c in range(4):
                xd = nx_cb.reserve(); ttl.copy(x[0:1, c * K_CH:(c + 1) * K_CH], xd)
                gd = ng_cb.reserve(); ttl.copy(gamma[0:1, c * K_CH:(c + 1) * K_CH], gd)
            norm_core(nx_cb, ng_cb, xn_stage, nsq_cb, nred_cb, nacc_cb, ninv_cb)
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
                pipe_recv(qkv_red, recv_cb)
                hd = head_cb.reserve()
                hd.store(ttl.add(part_cb.wait(), recv_cb.wait()))
                h = head_cb.wait()
                hx0 = hx_cb.reserve(); hx0.store(h)
                hre = head_cb.reserve(); hre.store(h)
                hx1 = hx_cb.reserve(); hx1.store(head_cb.wait())
                gq = hg_cb.reserve(); ttl.copy(qknorm[0:1, 0:Dt], gq)
                head_norm_core(hx_cb, hg_cb, hn_cb, hsq_cb, hred_cb, hacc_cb, hinv_cb)
                if col_c < 7:
                    cd = c_cb.reserve(); ttl.copy(cos[0:1, 0:Dt], cd)
                    sd2 = s_cb.reserve(); ttl.copy(sin[0:1, 0:Dt], sd2)
                    rd = r_cb.reserve(); ttl.copy(rot[0:Dt, 0:Dt], rd)
                    rope_core(hn_cb, c_cb, s_cb, r_cb, rh_cb, out_cb)
                else:
                    ob = out_cb.reserve()
                    ob.store(hn_cb.wait())
                ttl.copy(out_cb.wait(), heads[col_c - 1:col_c, 0:Dt])
            else:
                sd = send_cb.reserve(); sd.store(part_cb.wait())
                pipe_send(qkv_red, send_cb)

    return attn_heads


def make_attn_atom(Ht, Dt, St, eps):
    """Full pre-AR sliding attention as core composition on a (9, 2) grid.

    Stage A roles (norm col 0, QKV+head cols 1-8) plus: kv cores (5-8)
    patch the ring cache via kv_patch_core and signal q cores; q cores
    (1-4) run flash_decode_core over the cache, normalize, and mcast the
    o heads; every worker accumulates its O band via gemv_band_core with
    row-gated heads; row 0 drains o_part.
    """
    K_BAND = Ht // 2
    K_CH = K_BAND // 2
    n_chunks = 4
    chunk_t = St // n_chunks
    O_BAND = Ht // 8

    norm_core = make_rmsnorm_core(Ht, K_CH, Ht * TILE, eps)
    band_core = make_gemv_band_core(K_CH, 2, Dt)
    head_norm_core = make_rmsnorm_core(Dt, Dt, Dt * TILE, eps)
    rope_core = make_rope_core(Dt)
    patch_core = make_kv_patch_core(Dt)
    flash_core = make_flash_window_core(n_chunks)
    oband_core = make_gemv_band_core(Dt, 4, O_BAND)

    @ttl.atom(grid=(9, 2), fp32_dest_acc_en=True)
    def attn_atom(x, gamma, wqkv, cos, sin, qknorm, rot, kc0, kc1, vc0, vc1,
                  pos_t, masks, wo, o_part):
        col_c, row_c = ttl.node(dims=2)

        nx_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        ng_cb = ttl.make_dataflow_buffer_like(gamma, shape=(1, K_CH), block_count=2)
        nsq_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        nred_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        nacc_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        ninv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
        xn_stage = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)

        xb_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        w_cb = ttl.make_dataflow_buffer_like(wqkv, shape=(K_CH, Dt), block_count=2)
        part_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        send_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        recv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        head_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=2)

        hx_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=2)
        hg_cb = ttl.make_dataflow_buffer_like(qknorm, shape=(1, Dt), block_count=2)
        hsq_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        hred_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        hacc_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        hinv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
        hn_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        c_cb = ttl.make_dataflow_buffer_like(cos, shape=(1, Dt), block_count=1)
        s_cb = ttl.make_dataflow_buffer_like(sin, shape=(1, Dt), block_count=1)
        r_cb = ttl.make_dataflow_buffer_like(rot, shape=(Dt, Dt), block_count=1)
        rh_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        out_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)

        pos_cb = ttl.make_dataflow_buffer_like(pos_t, shape=(1, 1), block_count=1)
        band_cb = ttl.make_dataflow_buffer_like(kc0, shape=(1, Dt), block_count=1)
        tok_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        tok_stage = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)

        q_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)
        fo_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=2)
        fm_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        fl_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        fk_cb = ttl.make_dataflow_buffer_like(kc0, shape=(chunk_t, Dt), block_count=2)
        fv_cb = ttl.make_dataflow_buffer_like(vc0, shape=(chunk_t, Dt), block_count=2)
        fmask_cb = ttl.make_dataflow_buffer_like(masks, shape=(1, chunk_t), block_count=2)
        fsv_cb = ttl.make_dfb("bf16", shape=(1, chunk_t), block_count=2)
        fex_cb = ttl.make_dfb("bf16", shape=(1, chunk_t), block_count=2)
        fred_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=1)
        fmn_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        falpha_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        fpv_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)
        ostage_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)
        orecv_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=4)
        ox_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=2)
        wo_cb = ttl.make_dataflow_buffer_like(wo, shape=(Dt, O_BAND), block_count=2)
        op_cb = ttl.make_dataflow_buffer_like(x, shape=(1, O_BAND), block_count=1)
        osend_cb = ttl.make_dataflow_buffer_like(x, shape=(1, O_BAND), block_count=1)
        orcv2_cb = ttl.make_dataflow_buffer_like(x, shape=(1, O_BAND), block_count=1)
        osum_cb = ttl.make_dataflow_buffer_like(x, shape=(1, O_BAND), block_count=1)

        band0 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 0))])
        band1 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 1))])
        qkv_red = ttl.PipeNet([ttl.Pipe(src=(c, 1), dst=(c, 0)) for c in range(1, 9)])
        ready = ttl.PipeNet(
            [ttl.Pipe(src=(5, 0), dst=(slice(1, 3), 0)),
             ttl.Pipe(src=(6, 0), dst=(slice(3, 5), 0)),
             ttl.Pipe(src=(7, 0), dst=(slice(1, 3), 0)),
             ttl.Pipe(src=(8, 0), dst=(slice(3, 5), 0))])
        obc = ttl.PipeNet([ttl.Pipe(src=(c, 0), dst=(slice(1, 9), slice(0, 2)))
                           for c in range(1, 5)])
        o_red = ttl.PipeNet([ttl.Pipe(src=(c, 1), dst=(c, 0)) for c in range(1, 9)])

        if col_c == 0 and row_c == 0:
            for c in range(4):
                xd = nx_cb.reserve(); ttl.copy(x[0:1, c * K_CH:(c + 1) * K_CH], xd)
            for c in range(4):
                xd = nx_cb.reserve(); ttl.copy(x[0:1, c * K_CH:(c + 1) * K_CH], xd)
                gd = ng_cb.reserve(); ttl.copy(gamma[0:1, c * K_CH:(c + 1) * K_CH], gd)
            norm_core(nx_cb, ng_cb, xn_stage, nsq_cb, nred_cb, nacc_cb, ninv_cb)
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
                pipe_recv(qkv_red, recv_cb)
                hd = head_cb.reserve()
                hd.store(ttl.add(part_cb.wait(), recv_cb.wait()))
                h = head_cb.wait()
                hx0 = hx_cb.reserve(); hx0.store(h)
                hre = head_cb.reserve(); hre.store(h)
                hx1 = hx_cb.reserve(); hx1.store(head_cb.wait())
                gq = hg_cb.reserve(); ttl.copy(qknorm[0:1, 0:Dt], gq)
                head_norm_core(hx_cb, hg_cb, hn_cb, hsq_cb, hred_cb, hacc_cb, hinv_cb)
                if col_c < 7:
                    cd = c_cb.reserve(); ttl.copy(cos[0:1, 0:Dt], cd)
                    sd2 = s_cb.reserve(); ttl.copy(sin[0:1, 0:Dt], sd2)
                    rd = r_cb.reserve(); ttl.copy(rot[0:Dt, 0:Dt], rd)
                    rope_core(hn_cb, c_cb, s_cb, r_cb, rh_cb, out_cb)
                else:
                    ob = out_cb.reserve()
                    ob.store(hn_cb.wait())

                if col_c >= 5:
                    pd = pos_cb.reserve(); ttl.copy(pos_t[0, 0], pd)
                    pb = pos_cb.wait()
                    rr = ttl.read_index(pb, 0, 0)
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
                    patch_core(out_cb, band_cb, pos_cb)
                    if col_c == 5:
                        ttl.copy(bb, kc0[rr:rr + 1, 0:Dt])
                    elif col_c == 6:
                        ttl.copy(bb, kc1[rr:rr + 1, 0:Dt])
                    elif col_c == 7:
                        ttl.copy(bb, vc0[rr:rr + 1, 0:Dt])
                    else:
                        ttl.copy(bb, vc1[rr:rr + 1, 0:Dt])
                    ts = tok_stage.reserve(); ts.store(ttl.block.fill(1.0, shape=(1, 1)))
                    pipe_send(ready, tok_stage)
                else:
                    pipe_recv(ready, tok_cb)
                    pipe_recv(ready, tok_cb)
                    t1 = tok_cb.wait()
                    t2 = tok_cb.wait()
                    qw = q_cb.reserve()
                    qw.store(ttl.mul(out_cb.wait(),
                                     ttl.block.broadcast(ttl.mul(t1, t2), dims=[1], shape=(1, Dt))))
                    for c in range(n_chunks):
                        kd = fk_cb.reserve()
                        vd = fv_cb.reserve()
                        if col_c < 3:
                            ttl.copy(kc0[c * chunk_t:(c + 1) * chunk_t, 0:Dt], kd)
                            ttl.copy(vc0[c * chunk_t:(c + 1) * chunk_t, 0:Dt], vd)
                        else:
                            ttl.copy(kc1[c * chunk_t:(c + 1) * chunk_t, 0:Dt], kd)
                            ttl.copy(vc1[c * chunk_t:(c + 1) * chunk_t, 0:Dt], vd)
                        md = fmask_cb.reserve()
                        ttl.copy(masks[c:c + 1, 0:chunk_t], md)
                    flash_core(q_cb, fk_cb, fv_cb, fmask_cb, fo_cb, fm_cb, fl_cb,
                               fsv_cb, fex_cb, fred_cb, fmn_cb, falpha_cb, fpv_cb)
                    ofin = fo_cb.wait()
                    mfin = fm_cb.wait()
                    lfin = fl_cb.wait()
                    os_w = ostage_cb.reserve()
                    os_w.store(ttl.add(
                        ttl.mul(ofin, ttl.block.broadcast(ttl.recip(lfin), dims=[1], shape=(1, Dt))),
                        ttl.mul(ttl.block.broadcast(mfin, dims=[1], shape=(1, Dt)),
                                ttl.block.fill(0.0, shape=(1, Dt)))))
                mcast_block(obc, ostage_cb, orecv_cb)

            ob_off = (col_c - 1) * O_BAND
            for qh in range(4):
                oh = orecv_cb.wait()
                oxw = ox_cb.reserve()
                if row_c == qh % 2:
                    oxw.store(oh)
                else:
                    oxw.store(ttl.mul(oh, ttl.block.fill(0.0, shape=(1, Dt))))
                wod = wo_cb.reserve()
                ttl.copy(wo[qh * Dt:(qh + 1) * Dt, ob_off:ob_off + O_BAND], wod)
            oband_core(ox_cb, wo_cb, op_cb)
            if row_c == 0:
                pipe_recv(o_red, orcv2_cb)
                osm = osum_cb.reserve()
                osm.store(ttl.add(op_cb.wait(), orcv2_cb.wait()))
                ttl.copy(osum_cb.wait(), o_part[0:1, ob_off:ob_off + O_BAND])
            else:
                osd = osend_cb.reserve(); osd.store(op_cb.wait())
                pipe_send(o_red, osend_cb)

    return attn_atom
