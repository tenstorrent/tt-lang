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
from ttl.ops.mcast import mcast_block
from ttl.ops.pipe_util import pipe_send, pipe_recv

TILE = 32


def make_attn_heads_atom(Ht, Dt, eps):
    K_BAND = Ht // 2
    K_CH = K_BAND // 2
    WC = Ht // 8
    N_WC = 8
    inv_h = 1.0 / (Ht * TILE)
    inv_d = 1.0 / (Dt * TILE)

    @ttl.atom(grid=(9, 2), fp32_dest_acc_en=True)
    def attn_heads(x, gamma, wqkv, cos, sin, qknorm, rot, heads):
        col_c, row_c = ttl.node(dims=2)

        x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, WC), block_count=2)
        sq_cb = ttl.make_dataflow_buffer_like(x, shape=(1, WC), block_count=2)
        xband_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        g_cb = ttl.make_dataflow_buffer_like(gamma, shape=(1, K_CH), block_count=2)
        red_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        acc_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        inv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)

        xb_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        w_cb = ttl.make_dataflow_buffer_like(wqkv, shape=(K_CH, Dt), block_count=2)
        part_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        send_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        recv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        head_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=2)

        qk_g_cb = ttl.make_dataflow_buffer_like(qknorm, shape=(1, Dt), block_count=1)
        hsq_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        hred_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
        hn_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        c_cb = ttl.make_dataflow_buffer_like(cos, shape=(1, Dt), block_count=1)
        s_cb = ttl.make_dataflow_buffer_like(sin, shape=(1, Dt), block_count=1)
        r_cb = ttl.make_dataflow_buffer_like(rot, shape=(Dt, Dt), block_count=1)
        rh_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        out_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)

        xn_stage = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        band0 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 0))])
        band1 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 1))])
        qkv_red = ttl.PipeNet([ttl.Pipe(src=(c, 1), dst=(c, 0)) for c in range(1, 9)])

        if col_c == 0 and row_c == 0:
            a0 = acc_cb.reserve(); a0.store(ttl.block.fill(0.0, shape=(1, 1)))
            for c in range(N_WC):
                wc = c * WC
                xd = x_cb.reserve(); ttl.copy(x[0:1, wc:wc + WC], xd)
                xb = x_cb.wait()
                sq = sq_cb.reserve(); sq.store(ttl.mul(xb, xb))
                r = red_cb.reserve(); r.store(ttl.math.reduce_sum(sq_cb.wait(), dims=[1]))
                a_old = acc_cb.wait()
                a_new = acc_cb.reserve()
                a_new.store(ttl.add(a_old, red_cb.wait()))
            ivw = inv_cb.reserve()
            ivw.store(ttl.recip(ttl.sqrt(ttl.add(
                ttl.mul(acc_cb.wait(), ttl.block.fill(inv_h, shape=(1, 1))),
                ttl.block.fill(eps, shape=(1, 1))))))
            inv = inv_cb.wait()
            for ch in range(4):
                kb = ch * K_CH
                xd = xband_cb.reserve(); ttl.copy(x[0:1, kb:kb + K_CH], xd)
                gd = g_cb.reserve(); ttl.copy(gamma[0:1, kb:kb + K_CH], gd)
                xb = xband_cb.wait()
                xnw = xn_stage.reserve()
                xnw.store(ttl.mul(ttl.mul(xb, ttl.block.broadcast(inv, dims=[1], shape=xb.shape)),
                                  g_cb.wait()))
        for ch in range(2):
            mcast_block(band0, xn_stage, xb_cb)
        for ch in range(2):
            mcast_block(band1, xn_stage, xb_cb)

        if col_c >= 1:
            kr = row_c * K_BAND
            nd = (col_c - 1) * Dt

            p = part_cb.reserve()
            for ch in range(2):
                xn = xb_cb.wait()
                wb = w_cb.wait()
                p += xn @ wb
                wd = w_cb.reserve()
                ttl.copy(wqkv[kr + ch * K_CH:kr + (ch + 1) * K_CH, nd:nd + Dt], wd)

            if row_c == 0:
                pipe_recv(qkv_red, recv_cb)
                hd = head_cb.reserve()
                hd.store(ttl.add(part_cb.wait(), recv_cb.wait()))
                h = head_cb.wait()
                if col_c < 7:
                    gq = qk_g_cb.reserve(); ttl.copy(qknorm[0:1, 0:Dt], gq)
                    hsq = hsq_cb.reserve(); hsq.store(ttl.mul(h, h))
                    hr = hred_cb.reserve()
                    hr.store(ttl.math.reduce_sum(hsq_cb.wait(), dims=[1]))
                    hinv = ttl.recip(ttl.sqrt(ttl.add(
                        ttl.mul(hred_cb.wait(), ttl.block.fill(inv_d, shape=(1, 1))),
                        ttl.block.fill(eps, shape=(1, 1)))))
                    hn = hn_cb.reserve()
                    hn.store(ttl.mul(ttl.mul(h, ttl.block.broadcast(hinv, dims=[1], shape=h.shape)),
                                     qk_g_cb.wait()))
                    cd = c_cb.reserve(); ttl.copy(cos[0:1, 0:Dt], cd)
                    sd2 = s_cb.reserve(); ttl.copy(sin[0:1, 0:Dt], sd2)
                    rd = r_cb.reserve(); ttl.copy(rot[0:Dt, 0:Dt], rd)
                    hb = hn_cb.wait()
                    rh = rh_cb.reserve(); rh.store(hb @ r_cb.wait())
                    ow = out_cb.reserve()
                    ow.store(ttl.add(ttl.mul(hb, c_cb.wait()),
                                     ttl.mul(rh_cb.wait(), s_cb.wait())))
                else:
                    ow = out_cb.reserve()
                    ow.store(h)
                ttl.copy(out_cb.wait(), heads[col_c - 1:col_c, 0:Dt])
            else:
                sd = send_cb.reserve(); sd.store(part_cb.wait())
                pipe_send(qkv_red, send_cb)

    return attn_heads


def make_attn_atom(Ht, Dt, St, eps):
    """Full pre-AR sliding attention: heads + KV append + flash + O proj.

    Grid (9, 2). Col 0 row 0 norms x and mcasts bands. Cols 1-8 stream QKV
    (4q, 2k, 2v); k cores rope+norm K and patch the cache band; v cores
    norm V and patch theirs; q cores rope Q, run masked flash over the ring
    cache after the kv-ready token, then mcast o heads so every column adds
    its O-projection band partial; row 0 drains o_part to DRAM.
    """
    K_BAND = Ht // 2
    K_CH = K_BAND // 2
    WC = Ht // 8
    N_WC = 8
    inv_h = 1.0 / (Ht * TILE)
    inv_d = 1.0 / (Dt * TILE)
    n_chunks = 4
    chunk_t = St // n_chunks
    O_BAND = Ht // 8

    @ttl.atom(grid=(9, 2), fp32_dest_acc_en=True)
    def attn_atom(x, gamma, wqkv, cos, sin, qknorm, rot, kc0, kc1, vc0, vc1,
                  pos_t, masks, wo, o_part):
        col_c, row_c = ttl.node(dims=2)

        x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, WC), block_count=2)
        sq_cb = ttl.make_dataflow_buffer_like(x, shape=(1, WC), block_count=2)
        xband_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        g_cb = ttl.make_dataflow_buffer_like(gamma, shape=(1, K_CH), block_count=2)
        red_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        acc_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        inv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
        xn_stage = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        xb_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_CH), block_count=2)
        w_cb = ttl.make_dataflow_buffer_like(wqkv, shape=(K_CH, Dt), block_count=2)
        part_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        send_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        recv_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        head_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=2)
        qk_g_cb = ttl.make_dataflow_buffer_like(qknorm, shape=(1, Dt), block_count=1)
        hsq_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=1)
        hred_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
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

        kcb = ttl.make_dataflow_buffer_like(kc0, shape=(chunk_t, Dt), block_count=2)
        vcb = ttl.make_dataflow_buffer_like(vc0, shape=(chunk_t, Dt), block_count=2)
        mask_cb = ttl.make_dataflow_buffer_like(masks, shape=(1, chunk_t), block_count=2)
        sv_cb = ttl.make_dfb("bf16", shape=(1, chunk_t), block_count=2)
        ex_cb = ttl.make_dfb("bf16", shape=(1, chunk_t), block_count=2)
        fred_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=1)
        mn_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        al_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        pv_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)
        fo_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=2)
        fm_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        fl_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        onorm_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)
        ostage_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=1)
        orecv_cb = ttl.make_dfb("bf16", shape=(1, Dt), block_count=4)
        wo_cb = ttl.make_dataflow_buffer_like(wo, shape=(Dt, O_BAND), block_count=2)
        op_cb = ttl.make_dataflow_buffer_like(x, shape=(1, O_BAND), block_count=1)
        osend_cb = ttl.make_dataflow_buffer_like(x, shape=(1, O_BAND), block_count=1)
        orcv2_cb = ttl.make_dataflow_buffer_like(x, shape=(1, O_BAND), block_count=1)
        osum_cb = ttl.make_dataflow_buffer_like(x, shape=(1, O_BAND), block_count=1)

        band0 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 0))])
        band1 = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 9), 1))])
        qkv_red = ttl.PipeNet([ttl.Pipe(src=(c, 1), dst=(c, 0)) for c in range(1, 9)])
        # kv-ready tokens: k/v cores -> the two q cores of that kv head
        ready = ttl.PipeNet(
            [ttl.Pipe(src=(5, 0), dst=(slice(1, 3), 0)),
             ttl.Pipe(src=(6, 0), dst=(slice(3, 5), 0)),
             ttl.Pipe(src=(7, 0), dst=(slice(1, 3), 0)),
             ttl.Pipe(src=(8, 0), dst=(slice(3, 5), 0))])
        # o heads: q cores -> all 16 worker cores
        obc = ttl.PipeNet([ttl.Pipe(src=(c, 0), dst=(slice(1, 9), slice(0, 2)))
                           for c in range(1, 5)])
        o_red = ttl.PipeNet([ttl.Pipe(src=(c, 1), dst=(c, 0)) for c in range(1, 9)])

        if col_c == 0 and row_c == 0:
            a0 = acc_cb.reserve(); a0.store(ttl.block.fill(0.0, shape=(1, 1)))
            for c in range(N_WC):
                wc = c * WC
                xd = x_cb.reserve(); ttl.copy(x[0:1, wc:wc + WC], xd)
                xb = x_cb.wait()
                sq = sq_cb.reserve(); sq.store(ttl.mul(xb, xb))
                r = red_cb.reserve(); r.store(ttl.math.reduce_sum(sq_cb.wait(), dims=[1]))
                a_old = acc_cb.wait()
                a_new = acc_cb.reserve()
                a_new.store(ttl.add(a_old, red_cb.wait()))
            ivw = inv_cb.reserve()
            ivw.store(ttl.recip(ttl.sqrt(ttl.add(
                ttl.mul(acc_cb.wait(), ttl.block.fill(inv_h, shape=(1, 1))),
                ttl.block.fill(eps, shape=(1, 1))))))
            inv = inv_cb.wait()
            for ch in range(4):
                kb = ch * K_CH
                xd = xband_cb.reserve(); ttl.copy(x[0:1, kb:kb + K_CH], xd)
                gd = g_cb.reserve(); ttl.copy(gamma[0:1, kb:kb + K_CH], gd)
                xb = xband_cb.wait()
                xnw = xn_stage.reserve()
                xnw.store(ttl.mul(ttl.mul(xb, ttl.block.broadcast(inv, dims=[1], shape=xb.shape)),
                                  g_cb.wait()))
        for ch in range(2):
            mcast_block(band0, xn_stage, xb_cb)
        for ch in range(2):
            mcast_block(band1, xn_stage, xb_cb)

        if col_c >= 1:
            kr = row_c * K_BAND
            nd = (col_c - 1) * Dt

            p = part_cb.reserve()
            for ch in range(2):
                xn = xb_cb.wait()
                wb = w_cb.wait()
                p += xn @ wb
                wd = w_cb.reserve()
                ttl.copy(wqkv[kr + ch * K_CH:kr + (ch + 1) * K_CH, nd:nd + Dt], wd)

            if row_c == 0:
                pipe_recv(qkv_red, recv_cb)
                hd = head_cb.reserve()
                hd.store(ttl.add(part_cb.wait(), recv_cb.wait()))
                h = head_cb.wait()
                gq = qk_g_cb.reserve(); ttl.copy(qknorm[0:1, 0:Dt], gq)
                hsq = hsq_cb.reserve(); hsq.store(ttl.mul(h, h))
                hr = hred_cb.reserve()
                hr.store(ttl.math.reduce_sum(hsq_cb.wait(), dims=[1]))
                hinv = ttl.recip(ttl.sqrt(ttl.add(
                    ttl.mul(hred_cb.wait(), ttl.block.fill(inv_d, shape=(1, 1))),
                    ttl.block.fill(eps, shape=(1, 1)))))
                hnw = hn_cb.reserve()
                if col_c < 7:
                    hnw.store(ttl.mul(ttl.mul(h, ttl.block.broadcast(hinv, dims=[1], shape=h.shape)),
                                      qk_g_cb.wait()))
                else:
                    hnw.store(ttl.add(ttl.mul(h, ttl.block.broadcast(hinv, dims=[1], shape=h.shape)),
                                      ttl.mul(qk_g_cb.wait(), ttl.block.fill(0.0, shape=(1, Dt)))))
                ob = out_cb.reserve()
                if col_c < 7:
                    cd = c_cb.reserve(); ttl.copy(cos[0:1, 0:Dt], cd)
                    sd2 = s_cb.reserve(); ttl.copy(sin[0:1, 0:Dt], sd2)
                    rd = r_cb.reserve(); ttl.copy(rot[0:Dt, 0:Dt], rd)
                    hb = hn_cb.wait()
                    rh = rh_cb.reserve(); rh.store(hb @ r_cb.wait())
                    ob.store(ttl.add(ttl.mul(hb, c_cb.wait()),
                                     ttl.mul(rh_cb.wait(), s_cb.wait())))
                else:
                    ob.store(hn_cb.wait())

                if col_c >= 5:
                    # kv cores: patch own cache row, then signal readiness.
                    hv = out_cb.wait()
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
                    for cc in range(Dt * TILE):
                        vv = ttl.raw_element_read(hv, 0, cc)
                        ttl.raw_element_write(bb, intra, cc, vv)
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
                    # q cores: wait both ready tokens, then flash over ring.
                    pipe_recv(ready, tok_cb)
                    pipe_recv(ready, tok_cb)
                    t1 = tok_cb.wait()
                    t2 = tok_cb.wait()
                    qm = ttl.mul(out_cb.wait(),
                                 ttl.block.broadcast(ttl.mul(t1, t2), dims=[1], shape=(1, Dt)))

                    for fc in range(n_chunks):
                        kd = kcb.reserve()
                        vd = vcb.reserve()
                        md = mask_cb.reserve()
                        if col_c < 3:
                            ttl.copy(kc0[fc * chunk_t:(fc + 1) * chunk_t, 0:Dt], kd)
                            ttl.copy(vc0[fc * chunk_t:(fc + 1) * chunk_t, 0:Dt], vd)
                        else:
                            ttl.copy(kc1[fc * chunk_t:(fc + 1) * chunk_t, 0:Dt], kd)
                            ttl.copy(vc1[fc * chunk_t:(fc + 1) * chunk_t, 0:Dt], vd)
                        ttl.copy(masks[fc:fc + 1, 0:chunk_t], md)

                    m0 = fm_cb.reserve(); m0.store(ttl.block.fill(-1e30, shape=(1, 1)))
                    l0 = fl_cb.reserve(); l0.store(ttl.block.fill(0.0, shape=(1, 1)))
                    o0 = fo_cb.reserve(); o0.store(ttl.block.fill(0.0, shape=(1, Dt)))
                    for fc in range(n_chunks):
                        kb2 = kcb.wait()
                        mk = mask_cb.wait()
                        svw = sv_cb.reserve()
                        svw.store(ttl.add(qm @ ttl.transpose(kb2), mk))
                        sv = sv_cb.wait()
                        cmw = fred_cb.reserve(); cmw.store(ttl.math.reduce_max(sv, dims=[1]))
                        svr = sv_cb.reserve(); svr.store(sv)
                        m_old = fm_cb.wait()
                        cmx = fred_cb.wait()
                        mnw = mn_cb.reserve(); mnw.store(ttl.math.max(m_old, cmx))
                        mna = mn_cb.wait()
                        alw = al_cb.reserve(); alw.store(ttl.exp(ttl.sub(m_old, mna)))
                        mnr = mn_cb.reserve(); mnr.store(mna)
                        mns = mn_cb.wait()
                        sv2 = sv_cb.wait()
                        exw = ex_cb.reserve()
                        exw.store(ttl.exp(ttl.sub(sv2, ttl.block.broadcast(mns, dims=[1], shape=sv2.shape))))
                        mnx = fm_cb.reserve(); mnx.store(mns)
                        ex = ex_cb.wait()
                        csw = fred_cb.reserve(); csw.store(ttl.math.reduce_sum(ex, dims=[1]))
                        exr = ex_cb.reserve(); exr.store(ex)
                        alpha = al_cb.wait()
                        l_old = fl_cb.wait()
                        cs = fred_cb.wait()
                        lnw = fl_cb.reserve()
                        lnw.store(ttl.add(ttl.mul(alpha, l_old), cs))
                        alr = al_cb.reserve(); alr.store(alpha)
                        ex2 = ex_cb.wait()
                        vb2 = vcb.wait()
                        pvw = pv_cb.reserve(); pvw.store(ex2 @ vb2)
                        al2 = al_cb.wait()
                        o_old = fo_cb.wait()
                        pvb = pv_cb.wait()
                        onw = fo_cb.reserve()
                        onw.store(ttl.add(ttl.mul(ttl.block.broadcast(al2, dims=[1], shape=(1, Dt)), o_old), pvb))

                    ofin = fo_cb.wait()
                    lfin = fl_cb.wait()
                    onm = onorm_cb.reserve()
                    onm.store(ttl.mul(ofin, ttl.block.broadcast(ttl.recip(lfin), dims=[1], shape=(1, Dt))))
                    os = ostage_cb.reserve(); os.store(onorm_cb.wait())
                mcast_block(obc, ostage_cb, orecv_cb)

            # O proj: all worker cores; K = 4 q heads, Kp via rows.
            ob_off = (col_c - 1) * O_BAND
            opw = op_cb.reserve()
            for qh in range(4):
                ohead = orecv_cb.wait()
                wod = wo_cb.reserve()
                ttl.copy(wo[qh * Dt:(qh + 1) * Dt, ob_off:ob_off + O_BAND], wod)
                if row_c == qh % 2:
                    opw += ohead @ wo_cb.wait()
                else:
                    opw += ttl.mul(ohead, ttl.block.fill(0.0, shape=(1, Dt))) @ wo_cb.wait()
            if row_c == 0:
                pipe_recv(o_red, orcv2_cb)
                osm = osum_cb.reserve()
                osm.store(ttl.add(op_cb.wait(), orcv2_cb.wait()))
                ttl.copy(osum_cb.wait(), o_part[0:1, ob_off:ob_off + O_BAND])
            else:
                osd = osend_cb.reserve(); osd.store(op_cb.wait())
                pipe_send(o_red, osend_cb)

    return attn_atom
