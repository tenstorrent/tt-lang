# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Elementwise binary ops over tiled rows.

Rows split across the grid in ``PNt`` row-tile blocks, the feature
dimension streamed in ``WCt`` width chunks so any width fits L1. The
combine function is a block expression (e.g. add, gelu*mul), so swiglu
and residual add share this one streaming skeleton.
"""

import ttl


def make_binary(kind, Rt, PNt, Dt, WCt, a_off=(0, 0), b_off=(0, 0), out_off=(0, 0),
                scalar=1.0):
    """``out = <kind>(a, b)`` over ``Rt`` row-tiles by ``Dt`` width-tiles.

    ``kind``: add | swiglu (gelu(a)*b) | scaled_add ((a+b)*scalar). A
    trace-time switch: the tracer can't call closures inside atom bodies.
    ``a_off``/``b_off``/``out_off`` are (row, col) tile offsets into the
    operands, so inputs can be slices of a wider tensor (e.g. the g/u
    halves of a fused gate_up projection row). Scale-by-gate-weight runs
    as a separate row_scale dispatch.
    """
    if kind not in ("add", "swiglu", "scaled_add"):
        raise ValueError(f"unknown binary kind {kind}")
    if Dt % WCt != 0:
        raise ValueError(f"Dt ({Dt}) must be divisible by WCt ({WCt})")
    if Rt % PNt != 0:
        raise ValueError(f"Rt ({Rt}) must be divisible by PNt ({PNt})")
    n_blocks = Rt // PNt
    n_wc = Dt // WCt
    (ar, ac), (br, bc), (orr, oc) = a_off, b_off, out_off
    KIND_ADD = kind == "add"
    KIND_SWIGLU = kind == "swiglu"
    KIND_SCALED_ADD = kind == "scaled_add"

    @ttl.atom(grid="full")
    def binary(a, b, out):
        col_c, row_c = ttl.node(dims=2)
        gx, gy = ttl.grid_size(dims=2)
        cores = gx * gy
        bpc = (n_blocks + cores - 1) // cores
        cid = col_c * gy + row_c

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(PNt, WCt), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(b, shape=(PNt, WCt), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(PNt, WCt), block_count=2)

        for blk in range(bpc):
            blk_i = cid * bpc + blk
            if blk_i < n_blocks:
                base = blk_i * PNt
                for c in range(n_wc):
                    wc = c * WCt
                    ad = a_cb.reserve()
                    ttl.copy(a[ar + base:ar + base + PNt, ac + wc:ac + wc + WCt], ad)
                    bd = b_cb.reserve()
                    ttl.copy(b[br + base:br + base + PNt, bc + wc:bc + wc + WCt], bd)
                    ab = a_cb.wait()
                    bb = b_cb.wait()
                    ow = out_cb.reserve()
                    if KIND_ADD:
                        ow.store(ttl.add(ab, bb))
                    if KIND_SWIGLU:
                        ow.store(ttl.mul(ttl.gelu(ab), bb))
                    if KIND_SCALED_ADD:
                        ow.store(ttl.mul(ttl.add(ab, bb),
                                         ttl.block.fill(scalar, shape=(PNt, WCt))))
                    ttl.copy(out_cb.wait(), out[orr + base:orr + base + PNt, oc + wc:oc + wc + WCt])

    return binary


def make_add(Rt, PNt, Dt, WCt, **offsets):
    """Residual add: ``out = a + b``."""
    return make_binary("add", Rt, PNt, Dt, WCt, **offsets)


def make_copy(Rt, PNt, Dt, WCt, a_off=(0, 0), out_off=(0, 0)):
    """Strided tile copy (e.g. head extraction from a packed projection)."""
    if Dt % WCt or Rt % PNt:
        raise ValueError(f"Dt/Rt must divide: Rt={Rt} PNt={PNt} Dt={Dt} WCt={WCt}")
    n_blocks = Rt // PNt
    n_wc = Dt // WCt
    (ar, ac), (orr, oc) = a_off, out_off

    @ttl.atom(grid=(1, 1))
    def copy(a, out):
        a_cb = ttl.make_dataflow_buffer_like(a, shape=(PNt, WCt), block_count=2)
        for b in range(n_blocks):
            base = b * PNt
            for c in range(n_wc):
                wc = c * WCt
                ad = a_cb.reserve()
                ttl.copy(a[ar + base:ar + base + PNt, ac + wc:ac + wc + WCt], ad)
                ttl.copy(a_cb.wait(), out[orr + base:orr + base + PNt, oc + wc:oc + wc + WCt])

    return copy


def make_row_scale(Dt, WCt, recip=False, a_row=0, s_col=0, out_row=0):
    """``out[out_row, :] = s[0, 0] * a[a_row, :]`` with the scalar
    column-broadcast; ``recip`` scales by 1/s (flash finalize o * 1/l). MoE
    gate weighting reads expert t's scalar from tile column ``s_col``."""
    if Dt % WCt:
        raise ValueError(f"Dt ({Dt}) must be divisible by WCt ({WCt})")
    n_wc = Dt // WCt

    @ttl.atom(grid=(1, 1))
    def row_scale(a, s, out):
        a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, WCt), block_count=2)
        s_cb = ttl.make_dataflow_buffer_like(s, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, WCt), block_count=2)

        # Stage the scalar per chunk: a wait hoisted out of the loop is popped
        # after the first iteration, leaving later chunks reading stale L1.
        for c in range(n_wc):
            wc = c * WCt
            sd = s_cb.reserve()
            ttl.copy(s[0:1, s_col:s_col + 1], sd)
            ad = a_cb.reserve()
            ttl.copy(a[a_row:a_row + 1, wc:wc + WCt], ad)
            sb = s_cb.wait()
            ab = a_cb.wait()
            ow = out_cb.reserve()
            if recip:
                ow.store(ttl.mul(ab, ttl.block.broadcast(
                    ttl.recip(sb), dims=[1], shape=ab.shape)))
            else:
                ow.store(ttl.mul(ab, ttl.block.broadcast(
                    sb, dims=[1], shape=ab.shape)))
            ttl.copy(out_cb.wait(), out[out_row:out_row + 1, wc:wc + WCt])

    return row_scale
