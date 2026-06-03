# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm op: ``y = x / sqrt(mean(x^2) + eps) * weight``.

The reduction is over the feature dimension and is local to a core (each
core owns whole rows), so this is a standalone @ttl.atom with no cross-core
collective. The row-tiles are laid out across whatever 2D core grid the atom
is launched on: each core walks a contiguous run of ``bpc`` row blocks, so
the op handles any number of rows independent of the core count. The feature
dimension is streamed in width-tile chunks so an arbitrarily wide row (e.g.
7168) fits L1: pass one accumulates the sum of squares over the chunks to
form the per-row inverse-RMS scale, pass two re-reads each chunk and writes
``x * inv_rms * weight``.
"""

import ttl


def make_rmsnorm(Rt, PNt, Dt, WCt, D, eps):
    """RMSNorm over ``Rt`` row-tiles spread across the full device grid.

    Rows are grouped into blocks of ``PNt`` row-tiles (``Rt`` must divide by
    ``PNt``); the ``Rt // PNt`` blocks are spread contiguously across the
    device cores, ``ceil(blocks / cores)`` blocks per core, with a tail guard
    so any block count works on any grid (idle cores when blocks < cores).
    ``Dt`` is the row width in tiles, streamed ``WCt`` tiles at a time (``Dt``
    must divide by ``WCt``). ``D`` is the true feature width for the mean;
    ``weight`` is one ``(1, Dt)`` row broadcast over rows.
    """
    if Dt % WCt != 0:
        raise ValueError(f"Dt ({Dt}) must be divisible by WCt ({WCt})")
    if Rt % PNt != 0:
        raise ValueError(f"Rt ({Rt}) must be divisible by PNt ({PNt})")
    n_blocks = Rt // PNt
    n_wc = Dt // WCt
    inv_d = 1.0 / D

    @ttl.atom(grid="full")
    def rmsnorm(x, weight, out):
        col_c, row_c = ttl.node(dims=2)
        gx, gy = ttl.grid_size(dims=2)
        cores = gx * gy
        bpc = (n_blocks + cores - 1) // cores
        cid = col_c * gy + row_c

        xq_cb   = ttl.make_dataflow_buffer_like(x,      shape=(PNt, WCt), block_count=2)
        xr_cb   = ttl.make_dataflow_buffer_like(x,      shape=(PNt, WCt), block_count=2)
        w_cb    = ttl.make_dataflow_buffer_like(weight, shape=(1, WCt),   block_count=2)
        out_cb  = ttl.make_dataflow_buffer_like(out,    shape=(PNt, WCt), block_count=2)
        sq_cb   = ttl.make_dataflow_buffer_like(x,      shape=(PNt, WCt), block_count=2)
        part_cb = ttl.make_dataflow_buffer_like(x,      shape=(PNt, 1),   block_count=2)
        acc_cb  = ttl.make_dataflow_buffer_like(x,      shape=(PNt, 1),   block_count=2)
        inv_cb  = ttl.make_dataflow_buffer_like(x,      shape=(PNt, 1),   block_count=2)

        for blk in range(bpc):
            b = cid * bpc + blk
            if b < n_blocks:
                base = b * PNt

                acc0 = acc_cb.reserve()
                acc0.store(ttl.block.fill(0.0, shape=acc0.shape))
                for c in range(n_wc):
                    wc = c * WCt
                    xqd = xq_cb.reserve()
                    ttl.copy(x[base:base + PNt, wc:wc + WCt], xqd)
                    xq = xq_cb.wait()
                    sqd = sq_cb.reserve()
                    sqd.store(ttl.mul(xq, xq))
                    part = part_cb.reserve()
                    part.store(ttl.math.reduce_sum(sq_cb.wait(), dims=[1]))
                    acc_old = acc_cb.wait()
                    acc_new = acc_cb.reserve()
                    acc_new.store(ttl.add(acc_old, part_cb.wait()))

                acc_final = acc_cb.wait()
                inv_w = inv_cb.reserve()
                inv_w.store(ttl.recip(ttl.sqrt(ttl.add(
                    ttl.mul(acc_final, ttl.block.fill(inv_d, shape=inv_w.shape)),
                    ttl.block.fill(eps, shape=inv_w.shape)))))

                inv = inv_cb.wait()
                for c in range(n_wc):
                    wc = c * WCt
                    xrd = xr_cb.reserve()
                    ttl.copy(x[base:base + PNt, wc:wc + WCt], xrd)
                    wrd = w_cb.reserve()
                    ttl.copy(weight[0:1, wc:wc + WCt], wrd)
                    xr = xr_cb.wait()
                    w = w_cb.wait()
                    ow = out_cb.reserve()
                    ow.store(ttl.mul(
                        ttl.mul(xr, ttl.block.broadcast(inv, dims=[1], shape=xr.shape)),
                        ttl.block.broadcast(w, dims=[0], shape=xr.shape)))
                    ttl.copy(out_cb.wait(), out[base:base + PNt, wc:wc + WCt])

    return rmsnorm
