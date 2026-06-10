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
