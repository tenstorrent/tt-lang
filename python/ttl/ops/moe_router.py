# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MoE router tail (after the logits GEMV): softmax, then top-k weight
renorm with the per-expert scale gathered at the routed ids.

Combined with the existing topk op these put the entire router on device;
the host only ever stages step-invariant tables.
"""

import ttl

TILE = 32


def make_softmax_row(Wt):
    """Numerically-stable row softmax over a ``[1, Wt]`` tile row."""

    @ttl.atom(grid=(1, 1))
    def softmax_row(x, out):
        x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Wt), block_count=2)
        m_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        e_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Wt), block_count=2)
        s_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, Wt), block_count=2)

        xd = x_cb.reserve()
        ttl.copy(x[0:1, 0:Wt], xd)
        xb = x_cb.wait()
        mw = m_cb.reserve()
        mw.store(ttl.math.reduce_max(xb, dims=[1]))
        ew = e_cb.reserve()
        ew.store(ttl.exp(ttl.sub(
            xb, ttl.block.broadcast(m_cb.wait(), dims=[1], shape=xb.shape))))
        eb = e_cb.wait()
        sw = s_cb.reserve()
        sw.store(ttl.math.reduce_sum(eb, dims=[1]))
        ow = out_cb.reserve()
        ow.store(ttl.mul(eb, ttl.block.broadcast(
            ttl.recip(s_cb.wait()), dims=[1], shape=eb.shape)))
        ttl.copy(out_cb.wait(), out[0:1, 0:Wt])

    return softmax_row


def make_moe_weights(K, Et):
    """Top-k gate weights: ``w[t] = vals[t] / sum(vals) * pe[idx[t]]``.

    ``vals``/``idx`` use the topk layout (result t at element column t*32);
    ``pe`` is one ``[1, Et]`` row; ``zero`` is a zeroed ``[1, Kt]`` row that
    seeds the gather tiles (topk leaves garbage off the winner columns).
    Output keeps the strided layout consumed by make_row_scale's ``s_col``.

    TP sharding needs no mask here: stage ``pe`` zeroed outside the card's
    expert range and off-card weights vanish; renorm uses the global sum.
    """
    Kt = K  # one tile column per result

    @ttl.atom(grid=(1, 1))
    def moe_weights(vals, idx, pe, zero, out):
        v_cb = ttl.make_dataflow_buffer_like(vals, shape=(1, Kt), block_count=1)
        r_cb = ttl.make_dataflow_buffer_like(vals, shape=(1, Kt), block_count=1)
        i_cb = ttl.make_dataflow_buffer_like(idx, shape=(1, Kt), block_count=1)
        pe_cb = ttl.make_dataflow_buffer_like(pe, shape=(1, Et), block_count=1)
        g_cb = ttl.make_dataflow_buffer_like(vals, shape=(1, Kt), block_count=1)
        s_cb = ttl.make_dataflow_buffer_like(vals, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, Kt), block_count=2)

        rd = r_cb.reserve()
        ttl.copy(vals[0:1, 0:Kt], rd)
        vd = v_cb.reserve()
        idd = i_cb.reserve()
        ttl.copy(idx[0:1, 0:Kt], idd)
        ped = pe_cb.reserve()
        ttl.copy(pe[0:1, 0:Et], ped)

        ib = i_cb.wait()
        peb = pe_cb.wait()
        rb = r_cb.wait()
        gd = g_cb.reserve()
        # topk leaves garbage off the winner columns: rebuild both rows on
        # zeros so the renorm reduce only sees the K winners.
        ttl.copy(zero[0:1, 0:Kt], gd)
        ttl.copy(zero[0:1, 0:Kt], vd)
        for t in range(K):
            e = ttl.read_index(ib, 0, t * TILE)
            ttl.raw_element_write(gd, 0, t * TILE, ttl.raw_element_read(peb, 0, e))
            ttl.raw_element_write(vd, 0, t * TILE,
                                  ttl.raw_element_read(rb, 0, t * TILE))

        vb = v_cb.wait()
        sw = s_cb.reserve()
        sw.store(ttl.math.reduce_sum(vb, dims=[1]))
        ow = out_cb.reserve()
        ow.store(ttl.mul(ttl.mul(vb, g_cb.wait()), ttl.block.broadcast(
            ttl.recip(s_cb.wait()), dims=[1], shape=vb.shape)))
        ttl.copy(out_cb.wait(), out[0:1, 0:Kt])

    return moe_weights


def make_moe_scale(Dt, t, recip=False):
    """Scale one tile row by a scalar: ``out[t] = s[t] * x[t]`` (``recip``
    scales by ``1/s``, the flash finalize). ``s`` uses the topk strided
    layout (scalar t at tile column t); zero gate weights kill off-card
    rows before the down GEMVs see them.

    One single-block program per row: multi-row chunk loops and the
    row_scale chunked-s variant corrupt the next dispatched program (heads
    atom returns zeros after ~24 calls).
    """

    @ttl.atom(grid=(1, 1))
    def moe_scale(x, s, out):
        x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=2)
        s_cb = ttl.make_dataflow_buffer_like(s, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, Dt), block_count=2)

        xd = x_cb.reserve()
        ttl.copy(x[t:t + 1, 0:Dt], xd)
        sd = s_cb.reserve()
        ttl.copy(s[0:1, t:t + 1], sd)
        sb = s_cb.wait()
        if recip:
            sb = ttl.recip(sb)
        xb = x_cb.wait()
        ow = out_cb.reserve()
        ow.store(ttl.mul(xb, ttl.block.broadcast(sb, dims=[1], shape=xb.shape)))
        ttl.copy(out_cb.wait(), out[t:t + 1, 0:Dt])

    return moe_scale


def make_idx_gather(K, Et):
    """Translate topk ids through a per-card LUT: ``out[t] = lut[idx[t]]``.

    TP expert sharding stages ``lut[e] = clamp(e - base, 0, local - 1)`` per
    card so indexed GEMVs read in-range rows; off-card weights are already
    zeroed by the pe staging in moe_weights. Layout matches topk (result t
    at element column t*32).
    """
    Kt = K

    @ttl.atom(grid=(1, 1))
    def idx_gather(idx, lut, out):
        i_cb = ttl.make_dataflow_buffer_like(idx, shape=(1, Kt), block_count=2)
        l_cb = ttl.make_dataflow_buffer_like(lut, shape=(1, Et), block_count=1)
        o_cb = ttl.make_dataflow_buffer_like(out, shape=(1, Kt), block_count=1)

        idd = i_cb.reserve()
        ttl.copy(idx[0:1, 0:Kt], idd)
        ld = l_cb.reserve()
        ttl.copy(lut[0:1, 0:Et], ld)

        ib = i_cb.wait()
        lb = l_cb.wait()
        od = o_cb.reserve()
        ttl.copy(idx[0:1, 0:Kt], od)
        for t in range(K):
            e = ttl.read_index(ib, 0, t * TILE)
            ttl.raw_element_write(od, 0, t * TILE, ttl.raw_element_read(lb, 0, e))
        ttl.copy(o_cb.wait(), out[0:1, 0:Kt])

    return idx_gather
