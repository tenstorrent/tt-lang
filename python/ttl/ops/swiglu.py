# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU activation: ``y = gelu(g) * u`` (gelu_pytorch_tanh).

Pure elementwise: rows split across the grid like rmsnorm, the feature
dimension streamed in width chunks so any width fits L1. Used between
gate/up and down projections; the fused decode atom inlines the body so
g and u arrive in DFBs instead of DRAM.
"""

from ttl.ops.elementwise import make_binary


def make_swiglu(Rt, PNt, Dt, WCt, **kwargs):
    """gelu(g) * u over ``Rt`` row-tiles by ``Dt`` width-tiles."""
    return make_binary("swiglu", Rt, PNt, Dt, WCt, **kwargs)
