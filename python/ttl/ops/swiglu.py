# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU activation: ``y = gelu(g) * u`` (gelu_pytorch_tanh).

Pure elementwise: rows split across the grid like rmsnorm, the feature
dimension streamed in width chunks so any width fits L1. Used between
gate/up and down projections; the fused decode atom inlines the body so
g and u arrive in DFBs instead of DRAM.
"""

import ttl


def make_swiglu(Rt, PNt, Dt, WCt):
    """gelu(g) * u over ``Rt`` row-tiles by ``Dt`` width-tiles."""
    if Dt % WCt != 0:
        raise ValueError(f"Dt ({Dt}) must be divisible by WCt ({WCt})")
    if Rt % PNt != 0:
        raise ValueError(f"Rt ({Rt}) must be divisible by PNt ({PNt})")
    n_blocks = Rt // PNt
    n_wc = Dt // WCt

    @ttl.atom(grid="full")
    def swiglu(g, u, out):
        col_c, row_c = ttl.node(dims=2)
        gx, gy = ttl.grid_size(dims=2)
        cores = gx * gy
        bpc = (n_blocks + cores - 1) // cores
        cid = col_c * gy + row_c

        g_cb = ttl.make_dataflow_buffer_like(g, shape=(PNt, WCt), block_count=2)
        u_cb = ttl.make_dataflow_buffer_like(u, shape=(PNt, WCt), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(PNt, WCt), block_count=2)

        for blk in range(bpc):
            b = cid * bpc + blk
            if b < n_blocks:
                base = b * PNt
                for c in range(n_wc):
                    wc = c * WCt
                    gd = g_cb.reserve()
                    ttl.copy(g[base:base + PNt, wc:wc + WCt], gd)
                    ud = u_cb.reserve()
                    ttl.copy(u[base:base + PNt, wc:wc + WCt], ud)
                    ow = out_cb.reserve()
                    ow.store(ttl.mul(ttl.gelu(g_cb.wait()), u_cb.wait()))
                    ttl.copy(out_cb.wait(), out[base:base + PNt, wc:wc + WCt])

    return swiglu
