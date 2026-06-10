# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Greedy argmax over a wide logits row, on device.

Three stages over 256-wide chunks: (1) the logits row is restacked one
chunk per tile row, and per-row top-1 (topk machinery, bf16-exact local
ids 0..255) leaves a value/id column; (2) a column collapse copies the
``n_chunks`` winners into a row (one dispatch for values, one for ids);
(3) top-1 over the chunk-winner row plus
a gather of the chunk-local id yields ``token = chunk * 256 + local``,
written split (tile row, intra row) so embed gather / kv_append consume
it directly. Quotient/remainder of the local id come from step-invariant
LUT rows (host stages a 256-wide //32 and %32 ramp once).
"""

import ttl

TILE = 32
CHUNK = 256
Wt = CHUNK // TILE


def make_restack(n_chunks):
    """logits[0, c*256:(c+1)*256] -> tall[c, :] for topk row processing."""

    @ttl.atom(grid=(1, 1))
    def restack(x, out):
        x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, Wt), block_count=2)
        for c in range(n_chunks):
            xd = x_cb.reserve()
            ttl.copy(x[0:1, c * Wt:(c + 1) * Wt], xd)
            ttl.copy(x_cb.wait(), out[c:c + 1, 0:Wt])

    return restack


def make_collapse(n_chunks, out_row=0):
    """col[c, 0] (element column 0) -> out[out_row, c].

    Single source so each DFB keeps one consumer thread; the caller
    dispatches it once for values and once for ids (the latter into the
    token-select stage row).
    """
    nt = (n_chunks + TILE - 1) // TILE

    @ttl.atom(grid=(1, 1))
    def collapse(col, out):
        c_cb = ttl.make_dataflow_buffer_like(col, shape=(1, 1), block_count=2)
        o_cb = ttl.make_dataflow_buffer_like(out, shape=(1, nt), block_count=1)

        od = o_cb.reserve()
        # topk wrote chunk c's winner at tile row c, element (0, 0).
        for c in range(n_chunks):
            cd = c_cb.reserve()
            ttl.copy(col[c:c + 1, 0:1], cd)
            cb = c_cb.wait()
            ttl.raw_element_write(od, 0, c, ttl.raw_element_read(cb, 0, 0))
        ttl.copy(o_cb.wait(), out[out_row:out_row + 1, 0:nt])

    return collapse


def make_token_select(n_chunks):
    """local-id row + winner chunk id -> token (row, intra) tile.

    ``stage`` packs everything the gather reads, so one DFB wait serves
    all of it (one consumer thread, like collapse). Tile rows: 0 = LUT
    (element rows 0 = local // 32, 1 = local % 32 over 256, 2 = chunk *
    8 over n_chunks), 1 = chunk-local id row, 2 = winner chunk id at
    (0, 0). token = win*256 + local, emitted as (token // 32, token %
    32) = (win*8 + local//32, local%32): the two pieces land in
    separate zeroed tensors via DM-only raw gathers; the caller
    combines them with a tile add.
    """
    nt = (n_chunks + TILE - 1) // TILE
    if nt > Wt:
        raise ValueError(f"n_chunks {n_chunks} exceeds stage row width {CHUNK}")

    @ttl.atom(grid=(1, 1))
    def token_select(stage, zero, out_a, out_b):
        s_cb = ttl.make_dataflow_buffer_like(stage, shape=(3, Wt), block_count=1)
        a_cb = ttl.make_dataflow_buffer_like(out_a, shape=(1, 1), block_count=1)
        b_cb = ttl.make_dataflow_buffer_like(out_b, shape=(1, 1), block_count=1)

        sd = s_cb.reserve()
        ttl.copy(stage[0:3, 0:Wt], sd)
        sb = s_cb.wait()
        j = ttl.read_index(sb, 2 * TILE, 0)
        local = ttl.read_index(sb, TILE, j)

        ad = a_cb.reserve()
        ttl.copy(zero[0:1, 0:1], ad)
        ttl.raw_element_write(ad, 0, 0, ttl.raw_element_read(sb, 2, j))
        ttl.copy(a_cb.wait(), out_a[0:1, 0:1])
        bd = b_cb.reserve()
        ttl.copy(zero[0:1, 0:1], bd)
        ttl.raw_element_write(bd, 0, 0, ttl.raw_element_read(sb, 0, local))
        ttl.raw_element_write(bd, 0, 1, ttl.raw_element_read(sb, 1, local))
        ttl.copy(b_cb.wait(), out_b[0:1, 0:1])

    return token_select


def make_elem_copy(out_row=0, out_col=0, width=1):
    """Copy row-0 elements by value (dtype-converting; tile copies are raw).

    Single source, single consumer thread; tile-granular RMW of the target.
    """

    @ttl.atom(grid=(1, 1))
    def elem_copy(x, out):
        x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
        o_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

        xd = x_cb.reserve()
        ttl.copy(x[0:1, 0:1], xd)
        xb = x_cb.wait()
        ot = out_row // TILE
        oc = out_col // TILE
        od = o_cb.reserve()
        ttl.copy(out[ot:ot + 1, oc:oc + 1], od)
        for j in range(width):
            ttl.raw_element_write(od, out_row % TILE, out_col % TILE + j,
                                  ttl.raw_element_read(xb, 0, j))
        ttl.copy(o_cb.wait(), out[ot:ot + 1, oc:oc + 1])

    return elem_copy


def make_pick_token(n):
    """Vocab-sharded winner merge after all_gather.

    ``g`` holds one [value | token | scratch] tile triple per card as tile
    rows (gather dim 0) with the winner card id written at (0, 64); one
    source so a single consumer thread reads everything.
    """

    @ttl.atom(grid=(1, 1))
    def pick_token(g, zero, out):
        g_cb = ttl.make_dataflow_buffer_like(g, shape=(n, 3), block_count=1)
        o_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

        gd = g_cb.reserve()
        ttl.copy(g[0:n, 0:3], gd)
        gb = g_cb.wait()
        c = ttl.read_index(gb, 0, 2 * TILE)
        od = o_cb.reserve()
        ttl.copy(zero[0:1, 0:1], od)
        ttl.raw_element_write(od, 0, 0, ttl.raw_element_read(gb, c * TILE, TILE))
        ttl.raw_element_write(od, 0, 1, ttl.raw_element_read(gb, c * TILE, TILE + 1))
        ttl.copy(o_cb.wait(), out[0:1, 0:1])

    return pick_token
