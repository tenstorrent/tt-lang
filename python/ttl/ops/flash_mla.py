# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Flash MLA decode ops: a per-core online-softmax shard, an 8-core tree
reduce that merges the per-core partials, and a normalize tail.

Each op is a standalone @ttl.atom factory parametrized by tile shapes. A
shard core runs flash attention over its slice of the K/V sequence and
emits an unnormalized (o, m, l) partial; the tree reduce merges the
``n_cols`` partials of each batch row with the flash online-softmax
rescale; normalize divides by the running sum.
"""

import torch

import ttl


def make_flash_shard(n_cols, B, PNHt, DHt, vDHt, Sk_chunk_t, N_CHUNKS, scale=1.0):
    """Per-core flash attention over a K-dim split.

    The grid is ``(n_cols, B)``: column ``col_c`` owns the K/V slice
    starting at ``col_c * Sk_chunk_t * N_CHUNKS`` tiles, row ``row_c`` is
    the batch. Q is read from DRAM on every core; K/V are read per-core.
    Each core writes an unnormalized ``(o, m, l)`` partial to its row
    block of ``out_o`` / ``out_m`` / ``out_l``.

    K/V are typecast to bf16 before the matmuls so a lower-precision cache
    (e.g. bfp8) feeds the bf16 compute chain; the cast is a no-op when K/V
    are already bf16.

    ``scale`` is the SDPA scale; it is folded into ``qk`` per chunk.
    """
    St_per_core = Sk_chunk_t * N_CHUNKS

    @ttl.atom(grid=(n_cols, B))
    def flash_shard(q, k, v, out_o, out_m, out_l):
        col_c, row_c = ttl.node(dims=2)

        q_cb = ttl.make_dataflow_buffer_like(q, shape=(PNHt, DHt), block_count=2)
        k_cb = ttl.make_dataflow_buffer_like(k, shape=(Sk_chunk_t, DHt), block_count=2)
        v_cb = ttl.make_dataflow_buffer_like(k, shape=(Sk_chunk_t, vDHt), block_count=2)

        sv_cb        = ttl.make_dataflow_buffer_like(q, shape=(PNHt, Sk_chunk_t), block_count=2)
        ex_cb        = ttl.make_dataflow_buffer_like(q, shape=(PNHt, Sk_chunk_t), block_count=2)
        chunk_max_cb = ttl.make_dataflow_buffer_like(q, shape=(PNHt, 1),          block_count=2)
        chunk_sum_cb = ttl.make_dataflow_buffer_like(q, shape=(PNHt, 1),          block_count=2)
        alpha_cb     = ttl.make_dataflow_buffer_like(q, shape=(PNHt, 1),          block_count=2)
        m_new_cb     = ttl.make_dataflow_buffer_like(q, shape=(PNHt, 1),          block_count=2)
        o_corr_cb    = ttl.make_dataflow_buffer_like(q, shape=(PNHt, vDHt),       block_count=2)
        pv_cb        = ttl.make_dataflow_buffer_like(q, shape=(PNHt, vDHt),       block_count=2)

        m_state_cb   = ttl.make_dataflow_buffer_like(q, shape=(PNHt, 1),    block_count=2)
        l_state_cb   = ttl.make_dataflow_buffer_like(q, shape=(PNHt, 1),    block_count=2)
        o_state_cb   = ttl.make_dataflow_buffer_like(q, shape=(PNHt, vDHt), block_count=2)

        # Dedicated single-push output CBs: the compute thread waits the
        # state CBs and stores the finals here; the datamovement thread waits
        # these to copy to DRAM. We could copy each state CB straight to DRAM,
        # but then the state CB would be waited on the datamovement thread too
        # and the SPSC verifier rejects that cross-thread second consumer.
        out_o_cb = ttl.make_dataflow_buffer_like(out_o, shape=(PNHt, vDHt), block_count=2)
        out_m_cb = ttl.make_dataflow_buffer_like(out_m, shape=(PNHt, 1),    block_count=2)
        out_l_cb = ttl.make_dataflow_buffer_like(out_l, shape=(PNHt, 1),    block_count=2)

        qd = q_cb.reserve()
        ttl.copy(q[0:PNHt, 0:DHt], qd)

        k_base = col_c * St_per_core
        for c in range(N_CHUNKS):
            kc = k_base + c * Sk_chunk_t
            k_dst = k_cb.reserve()
            ttl.copy(k[kc:kc + Sk_chunk_t, 0:DHt], k_dst)
            v_dst = v_cb.reserve()
            ttl.copy(v[kc:kc + Sk_chunk_t, 0:vDHt], v_dst)

        m0 = m_state_cb.reserve(); m0.store(ttl.block.fill(-1e30, shape=m0.shape))
        l0 = l_state_cb.reserve(); l0.store(ttl.block.fill(0.0,   shape=l0.shape))
        o0 = o_state_cb.reserve(); o0.store(ttl.block.fill(0.0,   shape=o0.shape))

        q_blk = q_cb.wait()
        for _ in range(N_CHUNKS):
            k_blk = ttl.math.typecast(k_cb.wait(), torch.bfloat16)
            sv_w = sv_cb.reserve()
            sv_w.store(ttl.mul(q_blk @ ttl.transpose(k_blk),
                               ttl.block.fill(scale, shape=sv_w.shape)))

            sv = sv_cb.wait()
            cm_w = chunk_max_cb.reserve()
            cm_w.store(ttl.math.reduce_max(sv, dims=[1]))
            sv_re = sv_cb.reserve(); sv_re.store(sv)

            m_old = m_state_cb.wait()
            cm = chunk_max_cb.wait()
            mn_w = m_new_cb.reserve(); mn_w.store(ttl.math.max(m_old, cm))

            mn_for_alpha = m_new_cb.wait()
            alpha_w = alpha_cb.reserve()
            alpha_w.store(ttl.exp(ttl.sub(m_old, mn_for_alpha)))
            mn_re = m_new_cb.reserve(); mn_re.store(mn_for_alpha)

            mn_for_state = m_new_cb.wait()
            sv2 = sv_cb.wait()
            ex_w = ex_cb.reserve()
            ex_w.store(ttl.exp(ttl.sub(sv2, ttl.block.broadcast(
                mn_for_state, dims=[1], shape=sv2.shape))))
            m_next = m_state_cb.reserve(); m_next.store(mn_for_state)

            ex = ex_cb.wait()
            cs_w = chunk_sum_cb.reserve()
            cs_w.store(ttl.math.reduce_sum(ex, dims=[1]))
            ex_re = ex_cb.reserve(); ex_re.store(ex)

            alpha = alpha_cb.wait()
            l_old = l_state_cb.wait()
            cs = chunk_sum_cb.wait()
            l_next = l_state_cb.reserve()
            l_next.store(ttl.add(ttl.mul(alpha, l_old), cs))
            alpha_re = alpha_cb.reserve(); alpha_re.store(alpha)

            alpha2 = alpha_cb.wait()
            o_old = o_state_cb.wait()
            o_corr_w = o_corr_cb.reserve()
            o_corr_w.store(ttl.mul(ttl.block.broadcast(
                alpha2, dims=[1], shape=o_old.shape), o_old))

            ex2 = ex_cb.wait()
            v_blk = ttl.math.typecast(v_cb.wait(), torch.bfloat16)
            pv_w = pv_cb.reserve(); pv_w.store(ex2 @ v_blk)

            o_corr_blk = o_corr_cb.wait()
            pv_blk = pv_cb.wait()
            o_next = o_state_cb.reserve()
            o_next.store(ttl.add(o_corr_blk, pv_blk))

        base = (row_c * n_cols + col_c) * PNHt
        m_final = m_state_cb.wait()
        mo = out_m_cb.reserve(); mo.store(m_final)
        ttl.copy(out_m_cb.wait(), out_m[base:base + PNHt, 0:1])
        l_final = l_state_cb.wait()
        lo = out_l_cb.reserve(); lo.store(l_final)
        ttl.copy(out_l_cb.wait(), out_l[base:base + PNHt, 0:1])
        o_final = o_state_cb.wait()
        oo = out_o_cb.reserve(); oo.store(o_final)
        ttl.copy(out_o_cb.wait(), out_o[base:base + PNHt, 0:vDHt])

    return flash_shard


def make_flash_normalize(grid, PNHt, vDHt):
    """Finalize flash output: ``o_norm = o_unnorm / l``.

    ``l`` is broadcast from ``(PNHt, 1)`` to ``(PNHt, vDHt)`` and applied
    via reciprocal + multiply.
    """

    @ttl.atom(grid=grid)
    def flash_normalize(o_in, l_in, o_out):
        col_c, row_c = ttl.node(dims=2)
        base = row_c * PNHt

        o_cb = ttl.make_dataflow_buffer_like(o_in, shape=(PNHt, vDHt), block_count=2)
        l_cb = ttl.make_dataflow_buffer_like(l_in, shape=(PNHt, 1),    block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(o_out, shape=(PNHt, vDHt), block_count=2)

        od = o_cb.reserve(); ttl.copy(o_in[base:base + PNHt, 0:vDHt], od)
        ld = l_cb.reserve(); ttl.copy(l_in[base:base + PNHt, 0:1], ld)

        o = o_cb.wait()
        l = l_cb.wait()
        l_recip_bc = ttl.block.broadcast(
            ttl.math.recip(l), dims=[1], shape=o.shape,
        )
        ow = out_cb.reserve()
        ow.store(ttl.mul(o, l_recip_bc))
        o_blk = out_cb.wait()
        ttl.copy(o_blk, o_out[base:base + PNHt, 0:vDHt])

    return flash_normalize


def make_flash_tree_reduce(PNHt, vDHt, B=1):
    """8-core 3-step intra-row tree reduce, replicated across ``B`` rows.

    Grid is ``(8, B)``: ``col_c`` is the partial column, ``row_c`` the
    batch row. Each row independently merges its 8 ``(m, l, o)`` partials
    with the flash online-softmax rescale. On ``col_c == 0`` the merged
    unnormalized ``(o, m, l)`` is written to the row block of the outputs.
    """

    @ttl.atom()
    def tree_step_recv(
        m_in: ttl.DFB, l_in: ttl.DFB, o_in: ttl.DFB,
        m_out: ttl.DFB, l_out: ttl.DFB, o_out: ttl.DFB,
        m_peer: ttl.DFB, l_peer: ttl.DFB, o_peer: ttl.DFB,
        m_net: ttl.PipeNet, l_net: ttl.PipeNet, o_net: ttl.PipeNet,
    ):
        m_a = m_in.wait(); l_a = l_in.wait(); o_a = o_in.wait()

        def recv_m(pipe):
            p = m_peer.reserve(); ttl.copy(pipe, p)
        m_net.if_dst(recv_m)

        def recv_l(pipe):
            p = l_peer.reserve(); ttl.copy(pipe, p)
        l_net.if_dst(recv_l)

        def recv_o(pipe):
            p = o_peer.reserve(); ttl.copy(pipe, p)
        o_net.if_dst(recv_o)

        m_b = m_peer.wait(); l_b = l_peer.wait(); o_b = o_peer.wait()

        m_new = ttl.math.max(m_a, m_b)
        aa = ttl.exp(ttl.sub(m_a, m_new))
        ab = ttl.exp(ttl.sub(m_b, m_new))
        l_new = ttl.add(ttl.mul(aa, l_a), ttl.mul(ab, l_b))
        aa_bc = ttl.block.broadcast(aa, dims=[1], shape=o_a.shape)
        ab_bc = ttl.block.broadcast(ab, dims=[1], shape=o_b.shape)
        o_new = ttl.add(ttl.mul(aa_bc, o_a), ttl.mul(ab_bc, o_b))

        mw = m_out.reserve(); mw.store(m_new)
        lw = l_out.reserve(); lw.store(l_new)
        ow = o_out.reserve(); ow.store(o_new)

    @ttl.atom()
    def tree_step_send(
        m_state: ttl.DFB, l_state: ttl.DFB, o_state: ttl.DFB,
        m_net: ttl.PipeNet, l_net: ttl.PipeNet, o_net: ttl.PipeNet,
    ):
        def send_m(pipe):
            mb = m_state.wait(); ttl.copy(mb, pipe)
        m_net.if_src(send_m)

        def send_l(pipe):
            lb = l_state.wait(); ttl.copy(lb, pipe)
        l_net.if_src(send_l)

        def send_o(pipe):
            ob = o_state.wait(); ttl.copy(ob, pipe)
        o_net.if_src(send_o)

    @ttl.atom(grid=(8, B))
    def flash_tree_reduce(in_o, in_m, in_l, out_o, out_m, out_l):
        col_c, row_c = ttl.node(dims=2)

        s0_m = ttl.PipeNet([ttl.Pipe(src=(2 * i + 1, b), dst=(2 * i, b)) for b in range(B) for i in range(4)])
        s0_l = ttl.PipeNet([ttl.Pipe(src=(2 * i + 1, b), dst=(2 * i, b)) for b in range(B) for i in range(4)])
        s0_o = ttl.PipeNet([ttl.Pipe(src=(2 * i + 1, b), dst=(2 * i, b)) for b in range(B) for i in range(4)])
        s1_m = ttl.PipeNet([ttl.Pipe(src=(4 * i + 2, b), dst=(4 * i, b)) for b in range(B) for i in range(2)])
        s1_l = ttl.PipeNet([ttl.Pipe(src=(4 * i + 2, b), dst=(4 * i, b)) for b in range(B) for i in range(2)])
        s1_o = ttl.PipeNet([ttl.Pipe(src=(4 * i + 2, b), dst=(4 * i, b)) for b in range(B) for i in range(2)])
        s2_m = ttl.PipeNet([ttl.Pipe(src=(4, b), dst=(0, b)) for b in range(B)])
        s2_l = ttl.PipeNet([ttl.Pipe(src=(4, b), dst=(0, b)) for b in range(B)])
        s2_o = ttl.PipeNet([ttl.Pipe(src=(4, b), dst=(0, b)) for b in range(B)])

        # A single (m, l, o) DFB per stage would suffice: every wait is
        # sequential and a core only ever plays one role. But the SPSC
        # verifier counts a DFB's waits across all threads of the kernel, and
        # the same stage buffer is consumed by the recv path on the compute
        # thread and by the send path on the datamovement thread. We split
        # each stage buffer into a recv-side (`_rx`, compute consumer) and a
        # send-side (`_tx`, datamovement consumer) copy so each has a single
        # consumer thread; each recv routes its result into the copy whose
        # thread matches the next stage's consumer.
        #
        # TODO: these could be substantially reduced (to just 3 dfbs) if the
        # verifier was smarter about what actually might cause a race.
        m_rx0 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=2)
        l_rx0 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=2)
        o_rx0 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, vDHt), block_count=2)
        m_tx0 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=2)
        l_tx0 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=2)
        o_tx0 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, vDHt), block_count=2)
        m_rx1 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=2)
        l_rx1 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=2)
        o_rx1 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, vDHt), block_count=2)
        m_rx2 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=2)
        l_rx2 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=2)
        o_rx2 = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, vDHt), block_count=2)
        # Send-side buffer for stages 1 and 2: both are compute-produced and
        # datamovement-consumed, and no single core sends in both stages, so
        # they share one DFB (keeps us under the 32 hardware DFB limit).
        m_txc = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=2)
        l_txc = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=2)
        o_txc = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, vDHt), block_count=2)

        # Peer DFBs receive the partner's partial and are reused across all
        # 3 steps; block_count=3 = one slot per pipe that targets them.
        m_peer = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=3)
        l_peer = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, 1),    block_count=3)
        o_peer = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, vDHt), block_count=3)

        # Dedicated single-push output CBs: the compute thread stores the
        # merged result here and the datamovement thread copies it to DRAM.
        # The merge could write to DRAM directly, but then the state CB would
        # be waited on the datamovement thread too; the SPSC verifier rejects
        # that cross-thread second consumer.
        out_o_cb = ttl.make_dataflow_buffer_like(out_o, shape=(PNHt, vDHt), block_count=2)
        out_m_cb = ttl.make_dataflow_buffer_like(out_m, shape=(PNHt, 1),    block_count=2)
        out_l_cb = ttl.make_dataflow_buffer_like(out_l, shape=(PNHt, 1),    block_count=2)

        # Load this core's partial into the rx/tx copy matching its stage-0
        # role: receivers feed the compute thread, senders the datamovement.
        base = (row_c * 8 + col_c) * PNHt
        if s0_o.is_dst():
            od = o_rx0.reserve(); ttl.copy(in_o[base:base + PNHt, 0:vDHt], od)
            md = m_rx0.reserve(); ttl.copy(in_m[base:base + PNHt, 0:1], md)
            ld = l_rx0.reserve(); ttl.copy(in_l[base:base + PNHt, 0:1], ld)
        elif s0_o.is_src():
            od = o_tx0.reserve(); ttl.copy(in_o[base:base + PNHt, 0:vDHt], od)
            md = m_tx0.reserve(); ttl.copy(in_m[base:base + PNHt, 0:1], md)
            ld = l_tx0.reserve(); ttl.copy(in_l[base:base + PNHt, 0:1], ld)

        # Stage 0: receivers (cols 0,2,4,6) merge with their s0 peer and route
        # the result to rx1 if they receive again next, or tx1 if they send.
        if s1_o.is_dst():
            tree_step_recv(m_rx0, l_rx0, o_rx0, m_rx1, l_rx1, o_rx1,
                           m_peer, l_peer, o_peer, s0_m, s0_l, s0_o)
        elif s1_o.is_src():
            tree_step_recv(m_rx0, l_rx0, o_rx0, m_txc, l_txc, o_txc,
                           m_peer, l_peer, o_peer, s0_m, s0_l, s0_o)
        elif s0_o.is_src():
            tree_step_send(m_tx0, l_tx0, o_tx0, s0_m, s0_l, s0_o)

        # Stage 1: receivers (cols 0,4) merge with their s1 peer and route to
        # rx2 (col 0, receives again) or tx2 (col 4, sends to col 0).
        if s2_o.is_dst():
            tree_step_recv(m_rx1, l_rx1, o_rx1, m_rx2, l_rx2, o_rx2,
                           m_peer, l_peer, o_peer, s1_m, s1_l, s1_o)
        elif s2_o.is_src():
            tree_step_recv(m_rx1, l_rx1, o_rx1, m_txc, l_txc, o_txc,
                           m_peer, l_peer, o_peer, s1_m, s1_l, s1_o)
        elif s1_o.is_src():
            tree_step_send(m_txc, l_txc, o_txc, s1_m, s1_l, s1_o)

        if s2_o.is_dst():
            m_a = m_rx2.wait(); l_a = l_rx2.wait(); o_a = o_rx2.wait()

            def s2_recv_m(pipe):
                p = m_peer.reserve(); ttl.copy(pipe, p)
            s2_m.if_dst(s2_recv_m)

            def s2_recv_l(pipe):
                p = l_peer.reserve(); ttl.copy(pipe, p)
            s2_l.if_dst(s2_recv_l)

            def s2_recv_o(pipe):
                p = o_peer.reserve(); ttl.copy(pipe, p)
            s2_o.if_dst(s2_recv_o)

            m_b = m_peer.wait(); l_b = l_peer.wait(); o_b = o_peer.wait()

            m_fin = ttl.math.max(m_a, m_b)
            aa = ttl.exp(ttl.sub(m_a, m_fin))
            ab = ttl.exp(ttl.sub(m_b, m_fin))
            l_fin = ttl.add(ttl.mul(aa, l_a), ttl.mul(ab, l_b))
            aa_bc = ttl.block.broadcast(aa, dims=[1], shape=o_a.shape)
            ab_bc = ttl.block.broadcast(ab, dims=[1], shape=o_b.shape)
            o_unnorm = ttl.add(ttl.mul(aa_bc, o_a), ttl.mul(ab_bc, o_b))

            obase = row_c * PNHt
            mw = out_m_cb.reserve(); mw.store(m_fin)
            ttl.copy(out_m_cb.wait(), out_m[obase:obase + PNHt, 0:1])
            lw = out_l_cb.reserve(); lw.store(l_fin)
            ttl.copy(out_l_cb.wait(), out_l[obase:obase + PNHt, 0:1])
            ow = out_o_cb.reserve(); ow.store(o_unnorm)
            ttl.copy(out_o_cb.wait(), out_o[obase:obase + PNHt, 0:vDHt])
        elif s2_o.is_src():
            tree_step_send(m_txc, l_txc, o_txc, s2_m, s2_l, s2_o)

    return flash_tree_reduce
