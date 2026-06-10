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


def make_binary(fn, Rt, PNt, Dt, WCt):
    """``out = fn(a, b)`` over ``Rt`` row-tiles by ``Dt`` width-tiles."""
    if Dt % WCt != 0:
        raise ValueError(f"Dt ({Dt}) must be divisible by WCt ({WCt})")
    if Rt % PNt != 0:
        raise ValueError(f"Rt ({Rt}) must be divisible by PNt ({PNt})")
    n_blocks = Rt // PNt
    n_wc = Dt // WCt

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
                    ttl.copy(a[base:base + PNt, wc:wc + WCt], ad)
                    bd = b_cb.reserve()
                    ttl.copy(b[base:base + PNt, wc:wc + WCt], bd)
                    ow = out_cb.reserve()
                    ow.store(fn(a_cb.wait(), b_cb.wait()))
                    ttl.copy(out_cb.wait(), out[base:base + PNt, wc:wc + WCt])

    return binary


def make_add(Rt, PNt, Dt, WCt):
    """Residual add: ``out = a + b``."""
    return make_binary(ttl.add, Rt, PNt, Dt, WCt)
