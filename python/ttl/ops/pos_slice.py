# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Position-dependent table slicing and the decode position counter.

A step-invariant table (cos/sin rows per position, causal mask rows, pos
LUT) lives in DRAM; per step a tiny single-core atom gathers the element
row at the runtime position into row 0 of a DRAM staging tensor consumed
unchanged by the downstream atoms (B=1: only row 0 of those tiles carries
data). Replaces every per-step host to_dev with one dispatch per table.

``pos`` tile layout (f32 for exactness past 256): (0,0)=q, (0,1)=m,
(0,2)=abs, (0,3)=ring q, (0,4)=ring m where q,m split abs (or ring) into
tile row / intra row for kv_append-style runtime slices.
"""

import ttl

TILE = 32


def make_pos_slice(Dt, col_off=0, pos_col=0, out_row=0):
    """Extract element row at runtime pos from a tile band -> one out row.

    Reads tile-row index q at ``pos[0, pos_col]`` and intra row m at
    ``pos[0, pos_col + 1]``; gathers row m of band q over ``Dt`` width
    tiles starting at tile col ``col_off``, writing row 0 of out tile-row
    ``out_row`` (chunked mask staging passes c as out_row and col_off).
    """

    @ttl.atom(grid=(1, 1))
    def pos_slice(table, pos, out):
        p_cb = ttl.make_dataflow_buffer_like(pos, shape=(1, 1), block_count=1)
        b_cb = ttl.make_dataflow_buffer_like(table, shape=(1, Dt), block_count=1)
        o_cb = ttl.make_dataflow_buffer_like(out, shape=(1, Dt), block_count=1)

        pd = p_cb.reserve()
        ttl.copy(pos[0, 0], pd)
        pb = p_cb.wait()
        q = ttl.read_index(pb, 0, pos_col)
        m = ttl.read_index(pb, 0, pos_col + 1)

        bd = b_cb.reserve()
        ttl.copy(table[q:q + 1, col_off:col_off + Dt], bd)
        band = b_cb.wait()
        od = o_cb.reserve()
        for c in range(Dt * TILE):
            ttl.raw_element_write(od, 0, c, ttl.raw_element_read(band, m, c))
        ttl.copy(o_cb.wait(), out[out_row:out_row + 1, 0:Dt])

    return pos_slice


def make_pos_step():
    """Advance the position tile: row p of the LUT holds p+1's tile.

    LUT (f32, host-staged once) row p columns: 0=q, 1=m, 2=abs, 3=ring q,
    4=ring m, all evaluated at p+1, so a single gather at the current
    (q, m) yields the next step's pos tile.
    """

    @ttl.atom(grid=(1, 1))
    def pos_step(lut, pos, out):
        p_cb = ttl.make_dataflow_buffer_like(pos, shape=(1, 1), block_count=1)
        l_cb = ttl.make_dataflow_buffer_like(lut, shape=(1, 1), block_count=1)
        o_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

        pd = p_cb.reserve()
        ttl.copy(pos[0, 0], pd)
        pb = p_cb.wait()
        # LUT row p holds the tile for p+1, so current (q, m) address it.
        q = ttl.read_index(pb, 0, 0)
        m = ttl.read_index(pb, 0, 1)
        ld = l_cb.reserve()
        ttl.copy(lut[q:q + 1, 0:1], ld)
        lb = l_cb.wait()
        od = o_cb.reserve()
        for k in range(5):
            ttl.raw_element_write(od, 0, k, ttl.raw_element_read(lb, m, k))
        ttl.copy(o_cb.wait(), out[0:1, 0:1])

    return pos_step
