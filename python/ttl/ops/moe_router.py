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
    ``pe`` is one ``[1, Et]`` row. Output keeps the strided layout consumed
    by make_row_scale's ``s_col``.
    """
    Kt = K  # one tile column per result

    @ttl.atom(grid=(1, 1))
    def moe_weights(vals, idx, pe, out):
        v_cb = ttl.make_dataflow_buffer_like(vals, shape=(1, Kt), block_count=2)
        i_cb = ttl.make_dataflow_buffer_like(idx, shape=(1, Kt), block_count=1)
        pe_cb = ttl.make_dataflow_buffer_like(pe, shape=(1, Et), block_count=1)
        g_cb = ttl.make_dataflow_buffer_like(vals, shape=(1, Kt), block_count=1)
        s_cb = ttl.make_dataflow_buffer_like(vals, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, Kt), block_count=2)

        vd = v_cb.reserve()
        ttl.copy(vals[0:1, 0:Kt], vd)
        idd = i_cb.reserve()
        ttl.copy(idx[0:1, 0:Kt], idd)
        ped = pe_cb.reserve()
        ttl.copy(pe[0:1, 0:Et], ped)

        ib = i_cb.wait()
        peb = pe_cb.wait()
        gd = g_cb.reserve()
        # Pre-fill with vals (zero off-stride) so the elementwise mul below
        # never touches uninitialized L1.
        ttl.copy(vals[0:1, 0:Kt], gd)
        for t in range(K):
            e = ttl.read_index(ib, 0, t * TILE)
            s = ttl.raw_element_read(peb, 0, e)
            ttl.raw_element_write(gd, 0, t * TILE, s)

        vb = v_cb.wait()
        sw = s_cb.reserve()
        sw.store(ttl.math.reduce_sum(vb, dims=[1]))
        ow = out_cb.reserve()
        ow.store(ttl.mul(ttl.mul(vb, g_cb.wait()), ttl.block.broadcast(
            ttl.recip(s_cb.wait()), dims=[1], shape=vb.shape)))
        ttl.copy(out_cb.wait(), out[0:1, 0:Kt])

    return moe_weights
