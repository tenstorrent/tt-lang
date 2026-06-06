# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Flash MLA decode ops: a per-core online-softmax shard, an 8-core tree
reduce that merges the per-core partials, and a normalize tail.

Each phase is split into an inlinable **core** atom whose boundary buffers are
``ttl.DFB`` params (pure compute, no DRAM) and a thin **wrapper** atom that
stages DRAM <-> DFB and inlines the core. The wrappers are the standalone ops;
``make_flash_mla`` fuses all three cores into one kernel, moving every
inter-phase value through DFBs (q is multicast, partials and merged stats never
touch DRAM) instead of round-tripping DRAM between separate launches.
"""

import torch

import ttl
from ttl.ops.mcast import mcast, mcast_rows
from ttl.ops.pipe_util import pipe_send, pipe_recv


def make_flash_shard_core(B, PNHt, DHt, vDHt, Sk_chunk_t, N_CHUNKS, scale=1.0):
    """Per-core flash attention over a K-dim split, as an inlinable core.

    ``q`` arrives in the ``q_in`` DFB (multicast or staged by the wrapper);
    K/V are read per-core from DRAM and typecast to bf16 before the matmuls so
    a bfp8 cache feeds the bf16 chain. The running ``(o, m, l)`` accumulators
    live in the ``o_out`` / ``m_out`` / ``l_out`` DFBs directly, so the final
    chunk leaves the unnormalized partial there for the caller to drain or
    consume. ``scale`` is folded into ``qk`` per chunk.
    """
    St_per_core = Sk_chunk_t * N_CHUNKS

    @ttl.atom()
    def flash_shard_core(q_in: ttl.DFB, k, v, o_out: ttl.DFB, m_out: ttl.DFB, l_out: ttl.DFB):
        col_c, row_c = ttl.node(dims=2)

        k_cb = ttl.make_dataflow_buffer_like(k, shape=(Sk_chunk_t, DHt), block_count=2)
        v_cb = ttl.make_dataflow_buffer_like(k, shape=(Sk_chunk_t, vDHt), block_count=2)

        # These four stay explicit because their consumers are not DFB-input
        # ops, so the compiler cannot back them with implicit intermediates:
        # sv feeds reduce_max then a later exp, alpha feeds the l rescale then a
        # later o rescale (both second uses elementwise), and red_cb's reduce
        # results feed max/add binary combines. pv backs the ex @ v result that
        # the elementwise o update consumes. red_cb carries chunk_max then
        # chunk_sum (disjoint lifetimes). The exp(sv - max) result is left as
        # plain SSA: it feeds reduce_sum and the ex @ v matmul, both DFB-input
        # ops, so the compiler materializes it into one shared intermediate DFB.
        sv_cb        = ttl.make_dfb("bf16", shape=(PNHt, Sk_chunk_t), block_count=2)
        ex_cb        = ttl.make_dfb("bf16", shape=(PNHt, Sk_chunk_t), block_count=2)
        red_cb       = ttl.make_dfb("bf16", shape=(PNHt, 1),          block_count=2)
        mn_cb        = ttl.make_dfb("bf16", shape=(PNHt, 1),          block_count=2)
        alpha_cb     = ttl.make_dfb("bf16", shape=(PNHt, 1),          block_count=2)
        pv_cb        = ttl.make_dfb("bf16", shape=(PNHt, vDHt),       block_count=2)

        # The running m / l / o accumulators live in the output DFBs directly:
        # each chunk waits the prior value and reserves the next, and the final
        # chunk leaves exactly one block pushed for the consumer to drain.
        k_base = col_c * St_per_core
        for c in range(N_CHUNKS):
            kc = k_base + c * Sk_chunk_t
            k_dst = k_cb.reserve()
            ttl.copy(k[kc:kc + Sk_chunk_t, 0:DHt], k_dst)
            v_dst = v_cb.reserve()
            ttl.copy(v[kc:kc + Sk_chunk_t, 0:vDHt], v_dst)

        m0 = m_out.reserve(); m0.store(ttl.block.fill(-1e30, shape=m0.shape))
        l0 = l_out.reserve(); l0.store(ttl.block.fill(0.0,   shape=l0.shape))
        o0 = o_out.reserve(); o0.store(ttl.block.fill(0.0,   shape=o0.shape))

        q_blk = q_in.wait()
        for _ in range(N_CHUNKS):
            k_blk = ttl.math.typecast(k_cb.wait(), torch.bfloat16)
            sv_w = sv_cb.reserve()
            sv_w.store(ttl.mul(q_blk @ ttl.transpose(k_blk),
                               ttl.block.fill(scale, shape=sv_w.shape)))

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

    return flash_shard_core


def make_flash_shard(n_cols, B, PNHt, DHt, vDHt, Sk_chunk_t, N_CHUNKS, scale=1.0):
    """Standalone shard: q is multicast (col 0 reads DRAM, broadcasts to all
    columns), the core computes the per-core partial, and the wrapper drains it
    to ``out_o`` / ``out_m`` / ``out_l`` DRAM."""
    core = make_flash_shard_core(B, PNHt, DHt, vDHt, Sk_chunk_t, N_CHUNKS, scale)

    @ttl.atom(grid=(n_cols, B))
    def flash_shard(q, k, v, out_o, out_m, out_l):
        col_c, row_c = ttl.node(dims=2)

        q_net = ttl.PipeNet(mcast_rows(B, n_cols))
        q_stage = ttl.make_dataflow_buffer_like(q, shape=(PNHt, DHt), block_count=2)
        q_recv = ttl.make_dataflow_buffer_like(q, shape=(PNHt, DHt), block_count=2)
        o_b = ttl.make_dataflow_buffer_like(out_o, shape=(PNHt, vDHt), block_count=2)
        m_b = ttl.make_dataflow_buffer_like(out_m, shape=(PNHt, 1), block_count=2)
        l_b = ttl.make_dataflow_buffer_like(out_l, shape=(PNHt, 1), block_count=2)
        # The core accumulates the partial into o_b/m_b/l_b across chunks and
        # consumes them on compute, so the NCRISC drain needs its own buffer:
        # re-store the final partial on compute, then copy that to DRAM.
        o_d = ttl.make_dataflow_buffer_like(out_o, shape=(PNHt, vDHt), block_count=2)
        m_d = ttl.make_dataflow_buffer_like(out_m, shape=(PNHt, 1), block_count=2)
        l_d = ttl.make_dataflow_buffer_like(out_l, shape=(PNHt, 1), block_count=2)

        mcast(q_net, q[0:PNHt, 0:DHt], q_stage, q_recv)
        core(q_recv, k, v, o_b, m_b, l_b)

        m_dw = m_d.reserve(); m_dw.store(m_b.wait())
        l_dw = l_d.reserve(); l_dw.store(l_b.wait())
        o_dw = o_d.reserve(); o_dw.store(o_b.wait())

        base = (row_c * n_cols + col_c) * PNHt
        ttl.copy(m_d.wait(), out_m[base:base + PNHt, 0:1])
        ttl.copy(l_d.wait(), out_l[base:base + PNHt, 0:1])
        ttl.copy(o_d.wait(), out_o[base:base + PNHt, 0:vDHt])

    return flash_shard


def make_flash_normalize_core(PNHt, vDHt):
    """Finalize flash output ``o_norm = o_unnorm / l`` on column 0, as a core.

    ``o_in`` / ``l_in`` carry the merged unnormalized output and running sum;
    only column 0 holds them (the reduce roots there), so the work is guarded.
    """

    @ttl.atom()
    def flash_normalize_core(o_in: ttl.DFB, l_in: ttl.DFB, o_out: ttl.DFB):
        col_c, row_c = ttl.node(dims=2)
        if col_c == 0:
            o = o_in.wait()
            l = l_in.wait()
            l_recip_bc = ttl.block.broadcast(
                ttl.math.recip(l), dims=[1], shape=o.shape,
            )
            ow = o_out.reserve()
            ow.store(ttl.mul(o, l_recip_bc))

    return flash_normalize_core


def make_flash_normalize(grid, PNHt, vDHt):
    """Standalone normalize: stage ``o`` / ``l`` from DRAM, run the core, drain
    the normalized output to DRAM."""
    core = make_flash_normalize_core(PNHt, vDHt)

    @ttl.atom(grid=grid)
    def flash_normalize(o_in, l_in, o_out):
        col_c, row_c = ttl.node(dims=2)
        base = row_c * PNHt

        o_b = ttl.make_dataflow_buffer_like(o_in, shape=(PNHt, vDHt), block_count=2)
        l_b = ttl.make_dataflow_buffer_like(l_in, shape=(PNHt, 1), block_count=2)
        out_b = ttl.make_dataflow_buffer_like(o_out, shape=(PNHt, vDHt), block_count=2)

        od = o_b.reserve(); ttl.copy(o_in[base:base + PNHt, 0:vDHt], od)
        ld = l_b.reserve(); ttl.copy(l_in[base:base + PNHt, 0:1], ld)

        core(o_b, l_b, out_b)

        ttl.copy(out_b.wait(), o_out[base:base + PNHt, 0:vDHt])

    return flash_normalize


"""
                           ┌──────────────────────────────┐
                           │  q  (read ONCE from DRAM,    │
                           │      column 0)               │
                           └───────────────┬──────────────┘
                                           │   M U L T I C A S T  (1 → 8)
          ┌──────┬──────┬──────┬──────┬────┴─┬──────┬──────┬──────┐
          ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
        ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
        │ c0 │ │ c1 │ │ c2 │ │ c3 │ │ c4 │ │ c5 │ │ c6 │ │ c7 │   each column owns a
        └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘   slice of the K/V
          │      │      │      │      │      │      │      │       sequence and computes
          │      │      │      │      │      │      │      │       a partial (m,l,o) =
          ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼       flash(q, Kᵢ, Vᵢ)

  Then the partials collapse to column 0 with unicast pipes — odd→even, then quarter, then half:

     c0     c1     c2     c3     c4     c5     c6     c7
      │      │      │      │      │      │      │      │
      │◄─────┘      │◄─────┘      │◄─────┘      │◄─────┘     step 0 (unicast):
      │             │             │             │             1→0  3→2  5→4  7→6
      │             │             │             │
      │◄────────────┘             │◄────────────┘            step 1 (unicast):
      │                           │                           2→0       6→4
      │                           │
      │◄──────────────────────────┘                          step 2 (unicast):
      │                                                        4→0
      ▼
    (m,l,o) fully merged on c0  ──►  normalize (o / l)  ──►  output
"""

def make_flash_tree_reduce_core(PNHt, vDHt, B=1):
    """8-core 3-step intra-row tree reduce as a core: each core's ``(o, m, l)``
    partial arrives in the ``*_in`` DFBs, peers are exchanged over PipeNets, and
    column 0 pushes the merged unnormalized ``(o, l)`` to ``o_out`` / ``l_out``.

    The running max is consumed internally for the rescale but not emitted (the
    normalized output needs only ``o`` and ``l``).
    """

    @ttl.atom()
    def tree_step_recv(
        m_in: ttl.DFB, l_in: ttl.DFB, o_in: ttl.DFB,
        m_out: ttl.DFB, l_out: ttl.DFB, o_out: ttl.DFB,
        m_peer: ttl.DFB, l_peer: ttl.DFB, o_peer: ttl.DFB,
        m_net: ttl.PipeNet, l_net: ttl.PipeNet, o_net: ttl.PipeNet,
    ):
        m_a = m_in.wait(); l_a = l_in.wait(); o_a = o_in.wait()

        pipe_recv(m_net, m_peer)
        pipe_recv(l_net, l_peer)
        pipe_recv(o_net, o_peer)

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
        pipe_send(m_net, m_state)
        pipe_send(l_net, l_state)
        pipe_send(o_net, o_state)

    @ttl.atom()
    def flash_tree_reduce_core(
        o_in: ttl.DFB, m_in: ttl.DFB, l_in: ttl.DFB,
        o_out: ttl.DFB, l_out: ttl.DFB,
    ):
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

        # With the SPSC verifier relaxed (--ttl-relax-dfb-spsc), each stage's
        # working buffer can feed either the next compute-recv or the
        # datamovement-send -- the role guards make those mutually exclusive per
        # core -- so we keep one (m,l,o) buffer per stage instead of the
        # recv-side/send-side split that the verifier would otherwise require.
        m_s0 = ttl.make_dfb("bf16", shape=(PNHt, 1),    block_count=2)
        l_s0 = ttl.make_dfb("bf16", shape=(PNHt, 1),    block_count=2)
        o_s0 = ttl.make_dfb("bf16", shape=(PNHt, vDHt), block_count=2)
        m_s1 = ttl.make_dfb("bf16", shape=(PNHt, 1),    block_count=2)
        l_s1 = ttl.make_dfb("bf16", shape=(PNHt, 1),    block_count=2)
        o_s1 = ttl.make_dfb("bf16", shape=(PNHt, vDHt), block_count=2)

        m_peer = ttl.make_dfb("bf16", shape=(PNHt, 1),    block_count=3)
        l_peer = ttl.make_dfb("bf16", shape=(PNHt, 1),    block_count=3)
        o_peer = ttl.make_dfb("bf16", shape=(PNHt, vDHt), block_count=3)

        # Stage 0: receivers (cols 0,2,4,6) merge their own partial (*_in) with
        # the s0 peer into the stage-0 buffer; senders (cols 1,3,5,7) stage
        # their partial there and ship it.
        if s0_o.is_dst():
            tree_step_recv(m_in, l_in, o_in, m_s0, l_s0, o_s0,
                           m_peer, l_peer, o_peer, s0_m, s0_l, s0_o)
        elif s0_o.is_src():
            md = m_s0.reserve(); md.store(m_in.wait())
            ld = l_s0.reserve(); ld.store(l_in.wait())
            od = o_s0.reserve(); od.store(o_in.wait())
            tree_step_send(m_s0, l_s0, o_s0, s0_m, s0_l, s0_o)

        # Stage 1: receivers (cols 0,4) merge s0 with the s1 peer into s1;
        # senders (cols 2,6) ship their s0 result.
        if s1_o.is_dst():
            tree_step_recv(m_s0, l_s0, o_s0, m_s1, l_s1, o_s1,
                           m_peer, l_peer, o_peer, s1_m, s1_l, s1_o)
        elif s1_o.is_src():
            tree_step_send(m_s0, l_s0, o_s0, s1_m, s1_l, s1_o)

        # Stage 2: col 0 merges s1 with the final peer and writes the output;
        # col 4 ships its s1 result.
        if s2_o.is_dst():
            m_a = m_s1.wait(); l_a = l_s1.wait(); o_a = o_s1.wait()

            pipe_recv(s2_m, m_peer)
            pipe_recv(s2_l, l_peer)
            pipe_recv(s2_o, o_peer)

            m_b = m_peer.wait(); l_b = l_peer.wait(); o_b = o_peer.wait()

            m_fin = ttl.math.max(m_a, m_b)
            aa = ttl.exp(ttl.sub(m_a, m_fin))
            ab = ttl.exp(ttl.sub(m_b, m_fin))
            l_fin = ttl.add(ttl.mul(aa, l_a), ttl.mul(ab, l_b))
            aa_bc = ttl.block.broadcast(aa, dims=[1], shape=o_a.shape)
            ab_bc = ttl.block.broadcast(ab, dims=[1], shape=o_b.shape)
            o_unnorm = ttl.add(ttl.mul(aa_bc, o_a), ttl.mul(ab_bc, o_b))

            lw = l_out.reserve(); lw.store(l_fin)
            ow = o_out.reserve(); ow.store(o_unnorm)
        elif s2_o.is_src():
            tree_step_send(m_s1, l_s1, o_s1, s2_m, s2_l, s2_o)

    return flash_tree_reduce_core


def make_flash_tree_reduce(PNHt, vDHt, B=1):
    """Standalone tree reduce: stage each core's partial from DRAM, run the
    core, drain the merged ``(o, l)`` to DRAM on column 0."""
    core = make_flash_tree_reduce_core(PNHt, vDHt, B)

    @ttl.atom(grid=(8, B), options="--ttl-relax-dfb-spsc")
    def flash_tree_reduce(in_o, in_m, in_l, out_o, out_l):
        col_c, row_c = ttl.node(dims=2)
        base = (row_c * 8 + col_c) * PNHt

        o_b = ttl.make_dataflow_buffer_like(in_o, shape=(PNHt, vDHt), block_count=2)
        m_b = ttl.make_dataflow_buffer_like(in_m, shape=(PNHt, 1), block_count=2)
        l_b = ttl.make_dataflow_buffer_like(in_l, shape=(PNHt, 1), block_count=2)
        o_t = ttl.make_dataflow_buffer_like(out_o, shape=(PNHt, vDHt), block_count=2)
        l_t = ttl.make_dataflow_buffer_like(out_l, shape=(PNHt, 1), block_count=2)

        od = o_b.reserve(); ttl.copy(in_o[base:base + PNHt, 0:vDHt], od)
        md = m_b.reserve(); ttl.copy(in_m[base:base + PNHt, 0:1], md)
        ld = l_b.reserve(); ttl.copy(in_l[base:base + PNHt, 0:1], ld)

        core(o_b, m_b, l_b, o_t, l_t)

        obase = row_c * PNHt
        if col_c == 0:
            ttl.copy(l_t.wait(), out_l[obase:obase + PNHt, 0:1])
            ttl.copy(o_t.wait(), out_o[obase:obase + PNHt, 0:vDHt])

    return flash_tree_reduce


def make_flash_mla(n_cols, B, PNHt, DHt, vDHt, Sk_chunk_t, N_CHUNKS, scale=1.0):
    """Fully fused flash MLA decode: one kernel on grid ``(n_cols, B)``.

    q is multicast (column 0 reads DRAM, fans out to all columns); the shard,
    tree reduce, and normalize cores are inlined back to back with their
    boundary values carried through DFB bridges, so no partial or merged stat
    round-trips DRAM. Column 0 drains the normalized output. Requires
    ``n_cols == 8`` (the tree reduce topology).
    """
    shard = make_flash_shard_core(B, PNHt, DHt, vDHt, Sk_chunk_t, N_CHUNKS, scale)
    reduce = make_flash_tree_reduce_core(PNHt, vDHt, B)
    normalize = make_flash_normalize_core(PNHt, vDHt)

    # The fused kernel is two inline sites bridged by (so, sm, sl): the first
    # multicasts q and computes this core's shard partial, the second tree-
    # reduces the partials and normalizes. q_stage/q_recv live only in the first
    # site and the merged-stat scratch only in the second, so their L1 collapses
    # across the two sites; only the bridges stay at the fused top level.
    @ttl.atom()
    def q_shard(q, k, v, o_out: ttl.DFB, m_out: ttl.DFB, l_out: ttl.DFB):
        q_net = ttl.PipeNet(mcast_rows(B, n_cols))
        q_stage = ttl.make_dataflow_buffer_like(q, shape=(PNHt, DHt), block_count=2)
        q_recv = ttl.make_dataflow_buffer_like(q, shape=(PNHt, DHt), block_count=2)
        mcast(q_net, q[0:PNHt, 0:DHt], q_stage, q_recv)
        shard(q_recv, k, v, o_out, m_out, l_out)

    @ttl.atom()
    def reduce_norm(o_in: ttl.DFB, m_in: ttl.DFB, l_in: ttl.DFB, norm_out):
        col_c, row_c = ttl.node(dims=2)
        to = ttl.make_dfb("bf16", shape=(PNHt, vDHt), block_count=2)
        tl = ttl.make_dfb("bf16", shape=(PNHt, 1), block_count=2)
        # nout gets its own DFB: the datamovement drain below waits it on
        # NCRISC, and aliasing it onto the unnormalized partial lets NCRISC read
        # the wrong tile under the relaxed SPSC guard.
        nout = ttl.make_dataflow_buffer_like(norm_out, shape=(PNHt, vDHt), block_count=2)
        reduce(o_in, m_in, l_in, to, tl)
        normalize(to, tl, nout)
        if col_c == 0:
            ttl.copy(nout.wait(), norm_out[row_c * PNHt:row_c * PNHt + PNHt, 0:vDHt])

    @ttl.atom(grid=(n_cols, B), options="--ttl-relax-dfb-spsc")
    def flash_mla(q, k, v, norm_out):
        # shard -> tree bridges (this core's own partial)
        so = ttl.make_dataflow_buffer_like(q, shape=(PNHt, vDHt), block_count=2)
        sm = ttl.make_dfb("bf16", shape=(PNHt, 1), block_count=2)
        sl = ttl.make_dfb("bf16", shape=(PNHt, 1), block_count=2)

        q_shard(q, k, v, so, sm, sl)
        reduce_norm(so, sm, sl, norm_out)

    return flash_mla
