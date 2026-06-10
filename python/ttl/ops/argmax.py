# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Greedy argmax over a wide logits row, on device.

Three stages over 256-wide chunks: (1) the logits row is restacked one
chunk per tile row, and per-row top-1 (topk machinery, bf16-exact local
ids 0..255) leaves a value/id column; (2) a column collapse copies the
``n_chunks`` winners into a row; (3) top-1 over the chunk-winner row plus
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


def make_collapse(n_chunks):
    """vals[c, 0], ids[c, 0] (element columns 0/32) -> rows 0/1 of out."""
    nt = (n_chunks + TILE - 1) // TILE

    @ttl.atom(grid=(1, 1))
    def collapse(vals, ids, out):
        v_cb = ttl.make_dataflow_buffer_like(vals, shape=(nt, 1), block_count=1)
        i_cb = ttl.make_dataflow_buffer_like(ids, shape=(nt, 1), block_count=1)
        o_cb = ttl.make_dataflow_buffer_like(out, shape=(1, nt), block_count=1)

        vd = v_cb.reserve()
        ttl.copy(vals[0:nt, 0:1], vd)
        idd = i_cb.reserve()
        ttl.copy(ids[0:nt, 0:1], idd)
        vb, ib = v_cb.wait(), i_cb.wait()
        od = o_cb.reserve()
        # topk wrote chunk c's winner at tile row c -> element row c*32.
        for c in range(n_chunks):
            ttl.raw_element_write(od, 0, c, ttl.raw_element_read(vb, c * TILE, 0))
            ttl.raw_element_write(od, 1, c, ttl.raw_element_read(ib, c * TILE, 0))
        ttl.copy(o_cb.wait(), out[0:1, 0:nt])

    return collapse


def make_token_select(n_chunks):
    """rows (vals, local ids) + winner chunk id -> token (row, intra) tile.

    ``win`` holds the winning chunk id at element (0, 0) (topk output).
    LUT rows (step-invariant, host-staged once): 0 = local // 32,
    1 = local % 32 over 256, 2 = chunk * 8 over n_chunks. token =
    win*256 + local, emitted as (token // 32, token % 32) = (win*8 +
    local//32, local%32): the two pieces land in separate zeroed tiles
    via raw gathers and a tile add combines them (no scalar arithmetic
    on the DM thread).
    """
    nt = (n_chunks + TILE - 1) // TILE

    @ttl.atom(grid=(1, 1))
    def token_select(cw, win, lut, zero, out):
        c_cb = ttl.make_dataflow_buffer_like(cw, shape=(1, nt), block_count=1)
        w_cb = ttl.make_dataflow_buffer_like(win, shape=(1, 1), block_count=1)
        l_cb = ttl.make_dataflow_buffer_like(lut, shape=(1, Wt), block_count=1)
        a_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
        b_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
        o_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        cd = c_cb.reserve()
        ttl.copy(cw[0:1, 0:nt], cd)
        wd = w_cb.reserve()
        ttl.copy(win[0:1, 0:1], wd)
        ld = l_cb.reserve()
        ttl.copy(lut[0:1, 0:Wt], ld)

        cb, wb, lb = c_cb.wait(), w_cb.wait(), l_cb.wait()
        j = ttl.read_index(wb, 0, 0)
        local = ttl.read_index(cb, 1, j)

        ad = a_cb.reserve()
        ttl.copy(zero[0:1, 0:1], ad)
        ttl.raw_element_write(ad, 0, 0, ttl.raw_element_read(lb, 2, j))
        bd = b_cb.reserve()
        ttl.copy(zero[0:1, 0:1], bd)
        ttl.raw_element_write(bd, 0, 0, ttl.raw_element_read(lb, 0, local))
        ttl.raw_element_write(bd, 0, 1, ttl.raw_element_read(lb, 1, local))

        ow = o_cb.reserve()
        ow.store(ttl.add(a_cb.wait(), b_cb.wait()))
        ttl.copy(o_cb.wait(), out[0:1, 0:1])

    return token_select
