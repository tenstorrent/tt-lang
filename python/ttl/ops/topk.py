# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Top-K over the last dimension, e.g. MoE expert routing (top-k of N
experts per token).

Each core owns whole rows, so the per-row top-k is core-local (no cross-core
collective) and rows are spread across the full device grid. The k results
are produced by iterative argmax: each round takes the row max, records its
value and column index, then masks that column out before the next round.

Everything is full-tile arithmetic. The column index is recovered from an
``index`` ramp input (column j holds value j) via a masked reduce; equality
masks are built as ``1 - sign(abs(a - b))`` (exact since ``sign(0) == 0``).
Indices are carried in bf16, which is exact for ramps up to 256 (so N must
be <= 256). Each round writes its value/index straight to output tile-column
``r`` (a runtime slice), so the outputs are K tiles wide and no sub-tile
element writes are needed; the host reads column ``r*32`` of result ``r``.
"""

import ttl

# Knock-out magnitude: subtracted from the selected column so it cannot win
# again. Far larger than any routing logit, still well inside bf16 range.
_KNOCKOUT = 1e30


def make_topk(Rt, PNt, Wt, K, N):
    """Top-K over an ``N``-wide row for ``Rt`` row-tiles on the full grid.

    Rows are grouped into ``PNt``-row-tile blocks (``Rt`` divisible by
    ``PNt``) and spread across the device cores with a tail guard, so any
    row count works on any grid. ``N = Wt*32`` is the row width (number of
    candidates, must be <= 256 for exact bf16 indices). ``index`` is a ramp
    input whose column ``j`` holds value ``j`` (every row identical). The K
    descending results are written to tile-columns ``0..K-1`` of ``out_vals``
    / ``out_idxs`` (each K tiles wide); result ``r`` lands at element column
    ``r*32``.
    """
    if Rt % PNt != 0:
        raise ValueError(f"Rt ({Rt}) must be divisible by PNt ({PNt})")
    if N > 256:
        raise ValueError(f"N ({N}) must be <= 256 for exact bf16 indices")
    n_blocks = Rt // PNt
    Nm1 = N - 1

    @ttl.atom(grid="full")
    def topk(x, index, out_vals, out_idxs):
        col_c, row_c = ttl.node(dims=2)
        gx, gy = ttl.grid_size(dims=2)
        bpc = (n_blocks + gx * gy - 1) // (gx * gy)
        cid = col_c * gy + row_c

        xin_cb     = ttl.make_dataflow_buffer_like(x,        shape=(PNt, Wt), block_count=2)
        xs_cb      = ttl.make_dataflow_buffer_like(x,        shape=(PNt, Wt), block_count=2)
        idx_cb     = ttl.make_dataflow_buffer_like(index,    shape=(1, Wt), block_count=2)
        irev_cb    = ttl.make_dataflow_buffer_like(x,        shape=(1, Wt), block_count=2)
        m_cb       = ttl.make_dataflow_buffer_like(x,        shape=(PNt, 1), block_count=2)
        vd_cb      = ttl.make_dataflow_buffer_like(x,        shape=(PNt, Wt), block_count=2)
        vmask_cb   = ttl.make_dataflow_buffer_like(x,        shape=(PNt, Wt), block_count=2)
        contrib_cb = ttl.make_dataflow_buffer_like(x,        shape=(PNt, Wt), block_count=2)
        ridx_cb    = ttl.make_dataflow_buffer_like(x,        shape=(PNt, 1), block_count=2)
        fidx_cb    = ttl.make_dataflow_buffer_like(x,        shape=(PNt, 1), block_count=2)
        id_cb      = ttl.make_dataflow_buffer_like(x,        shape=(PNt, Wt), block_count=2)
        imask_cb   = ttl.make_dataflow_buffer_like(x,        shape=(PNt, Wt), block_count=2)
        ov_cb      = ttl.make_dataflow_buffer_like(out_vals, shape=(PNt, 1), block_count=2)
        oi_cb      = ttl.make_dataflow_buffer_like(out_idxs, shape=(PNt, 1), block_count=2)

        sW = (PNt, Wt)
        s1 = (PNt, 1)

        idx0 = idx_cb.reserve()
        ttl.copy(index[0:1, 0:Wt], idx0)
        idx_w = idx_cb.wait()

        # iota_rev: column j holds (N-1 - j), for first-index tie-breaking.
        irev_w = irev_cb.reserve()
        irev_w.store(ttl.sub(ttl.block.fill(Nm1, shape=(1, Wt)), idx_w))
        irev = irev_cb.wait()

        for blk in range(bpc):
            b = cid * bpc + blk
            if b < n_blocks:
                base = b * PNt

                # Load on the DM thread into xin, then initialize the compute
                # working set on the compute thread so xs has a single producer.
                xin0 = xin_cb.reserve()
                ttl.copy(x[base:base + PNt, 0:Wt], xin0)
                xs0 = xs_cb.reserve()
                xs0.store(xin_cb.wait())

                for r in range(K):
                    xs = xs_cb.wait()

                    m_w = m_cb.reserve()
                    m_w.store(ttl.math.reduce_max(xs, dims=[1]))
                    m = m_cb.wait()

                    # vmask: 1 at the row max column(s). m_bc - xs >= 0 so a
                    # plain sign (no abs) is exact: 0 at the max, 1 elsewhere.
                    # Materialized in two steps: a single op spanning many
                    # width tiles overflows the dst register file when nested
                    # deeper, leaving high tiles uninitialized.
                    vd_w = vd_cb.reserve()
                    vd_w.store(ttl.sub(
                        ttl.block.broadcast(m, dims=[1], shape=sW), xs))
                    vmask_w = vmask_cb.reserve()
                    vmask_w.store(ttl.sub(
                        ttl.block.fill(1.0, shape=sW), ttl.sign(vd_cb.wait())))
                    vmask = vmask_cb.wait()

                    # first column index of the max = (N-1) - max(vmask*iota_rev).
                    contrib_w = contrib_cb.reserve()
                    contrib_w.store(ttl.mul(
                        vmask, ttl.block.broadcast(irev, dims=[0], shape=sW)))
                    ridx_w = ridx_cb.reserve()
                    ridx_w.store(ttl.math.reduce_max(contrib_cb.wait(), dims=[1]))
                    fidx_w = fidx_cb.reserve()
                    fidx_w.store(ttl.sub(ttl.block.fill(Nm1, shape=s1), ridx_cb.wait()))
                    fidx = fidx_cb.wait()

                    ov_w = ov_cb.reserve(); ov_w.store(m)
                    ttl.copy(ov_cb.wait(), out_vals[base:base + PNt, r:r + 1])
                    oi_w = oi_cb.reserve(); oi_w.store(fidx)
                    ttl.copy(oi_cb.wait(), out_idxs[base:base + PNt, r:r + 1])

                    # knock out exactly the selected column (handles value
                    # ties). Materialized in two steps for the same dst-register
                    # reason as vmask above.
                    id_w = id_cb.reserve()
                    id_w.store(ttl.abs(ttl.sub(
                        ttl.block.broadcast(idx_w, dims=[0], shape=sW),
                        ttl.block.broadcast(fidx, dims=[1], shape=sW))))
                    imask_w = imask_cb.reserve()
                    imask_w.store(ttl.sub(
                        ttl.block.fill(1.0, shape=sW), ttl.sign(id_cb.wait())))
                    imask = imask_cb.wait()

                    xs_new = xs_cb.reserve()
                    xs_new.store(ttl.sub(xs, ttl.mul(
                        imask, ttl.block.fill(_KNOCKOUT, shape=sW))))

    return topk
