# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding gather: copy one table row at a runtime token id, scaled.

Reads the tile band ``table[id // 32]``, copies row ``id % 32`` into row 0
of the output tile and scales by ``scale`` (Gemma: sqrt(H)). ``tok`` holds
``id // 32`` at (0,0) and ``id % 32`` at (0,1) (host-split like kv_append).
"""

import ttl

TILE = 32


def make_embed_gather(Dt, scale):
    @ttl.atom(grid=(1, 1))
    def embed_gather(table, tok, out):
        tok_cb = ttl.make_dataflow_buffer_like(tok, shape=(1, 1), block_count=1)
        band_cb = ttl.make_dataflow_buffer_like(table, shape=(1, Dt), block_count=1)
        row_cb = ttl.make_dataflow_buffer_like(out, shape=(1, Dt), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, Dt), block_count=2)

        td = tok_cb.reserve()
        ttl.copy(tok[0, 0], td)
        tok_blk = tok_cb.wait()
        r = ttl.read_index(tok_blk, 0, 0)
        intra = ttl.read_index(tok_blk, 0, 1)

        bd = band_cb.reserve()
        ttl.copy(table[r:r + 1, 0:Dt], bd)
        band = band_cb.wait()

        rowd = row_cb.reserve()
        for c in range(Dt * TILE):
            v = ttl.raw_element_read(band, intra, c)
            ttl.raw_element_write(rowd, 0, c, v)

        row = row_cb.wait()
        ow = out_cb.reserve()
        ow.store(ttl.mul(row, ttl.block.fill(scale, shape=row.shape)))
        ttl.copy(out_cb.wait(), out[0:1, 0:Dt])

    return embed_gather
