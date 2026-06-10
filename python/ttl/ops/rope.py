# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Rotary position embedding for decode (one position, heads packed on rows).

HF rotate-half form on the first ``rot_t`` width-tiles of each head row:
``y1 = x1*cos1 - x2*sin1``, ``y2 = x2*cos2 + x1*sin2`` where x1/x2 are the
two halves of the rotary span; tiles past ``rot_t`` pass through unchanged
(partial rotary, Gemma global layers rotate only the first quarter).

``cos``/``sin`` are ``[1, rot_t]`` tiles for the decode position with the
row replicated across all 32 tile rows (host slices per-layer-type tables
and broadcasts), so the multiply needs no in-kernel row broadcast.
"""

import ttl


def make_rope(Ht, rot_t):
    """RoPE over ``[1, Ht]`` row-tiles of packed heads, rotating ``rot_t``."""
    if rot_t % 2 or rot_t > Ht:
        raise ValueError(f"rot_t must be even and <= Ht: rot_t={rot_t} Ht={Ht}")
    half = rot_t // 2
    rest = Ht - rot_t

    @ttl.atom(grid=(1, 1))
    def rope(x, cos, sin, out):
        x1_cb = ttl.make_dataflow_buffer_like(x, shape=(1, half), block_count=2)
        x2_cb = ttl.make_dataflow_buffer_like(x, shape=(1, half), block_count=2)
        c1_cb = ttl.make_dataflow_buffer_like(cos, shape=(1, half), block_count=2)
        c2_cb = ttl.make_dataflow_buffer_like(cos, shape=(1, half), block_count=2)
        s1_cb = ttl.make_dataflow_buffer_like(sin, shape=(1, half), block_count=2)
        s2_cb = ttl.make_dataflow_buffer_like(sin, shape=(1, half), block_count=2)
        y1_cb = ttl.make_dataflow_buffer_like(out, shape=(1, half), block_count=2)
        y2_cb = ttl.make_dataflow_buffer_like(out, shape=(1, half), block_count=2)
        # Declared unconditionally: DFB decls must be top-level statements.
        r_cb = ttl.make_dataflow_buffer_like(x, shape=(1, max(rest, 1)), block_count=2)

        x1d = x1_cb.reserve(); ttl.copy(x[0:1, 0:half], x1d)
        x2d = x2_cb.reserve(); ttl.copy(x[0:1, half:rot_t], x2d)
        c1d = c1_cb.reserve(); ttl.copy(cos[0:1, 0:half], c1d)
        c2d = c2_cb.reserve(); ttl.copy(cos[0:1, half:rot_t], c2d)
        s1d = s1_cb.reserve(); ttl.copy(sin[0:1, 0:half], s1d)
        s2d = s2_cb.reserve(); ttl.copy(sin[0:1, half:rot_t], s2d)

        x1 = x1_cb.wait(); x2 = x2_cb.wait()
        c1 = c1_cb.wait(); c2 = c2_cb.wait()
        s1 = s1_cb.wait(); s2 = s2_cb.wait()

        y1 = y1_cb.reserve()
        y1.store(ttl.sub(ttl.mul(x1, c1), ttl.mul(x2, s1)))
        y2 = y2_cb.reserve()
        y2.store(ttl.add(ttl.mul(x2, c2), ttl.mul(x1, s2)))

        ttl.copy(y1_cb.wait(), out[0:1, 0:half])
        ttl.copy(y2_cb.wait(), out[0:1, half:rot_t])
        if rest:
            rd = r_cb.reserve(); ttl.copy(x[0:1, rot_t:Ht], rd)
            ttl.copy(r_cb.wait(), out[0:1, rot_t:Ht])

    return rope


def make_rope_core(Dt):
    """Inlinable rotate-half RoPE: ``y = h*cos + (h@R)*sin`` with ``R`` the
    rotation permutation streamed in ``r_in`` ([Dt, Dt]). cos/sin arrive as
    row-broadcast tiles. Partial rotary folds into R/cos/sin contents."""

    @ttl.atom()
    def rope_core(h_in: ttl.DFB, c_in: ttl.DFB, s_in: ttl.DFB,
                  r_in: ttl.DFB, rh: ttl.DFB, y_out: ttl.DFB):
        hb = h_in.wait()
        rw = rh.reserve(); rw.store(hb @ r_in.wait())
        yw = y_out.reserve()
        yw.store(ttl.add(ttl.mul(hb, c_in.wait()), ttl.mul(rh.wait(), s_in.wait())))

    return rope_core
