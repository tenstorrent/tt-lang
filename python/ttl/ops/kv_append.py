# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""KV cache append: write one K row at a runtime position.

Reads the tile band ``cache[pos // 32]``, patches row ``pos % 32`` with
the new K row (read from ``k`` row 0), and writes the band back. ``pos``
arrives as a tensor holding ``pos // 32`` at (0,0) and ``pos % 32`` at
(0,1) (floats; exact integers) since index division is host-side; the
same atom works at every decode step. Sliding caches pass ring positions.
"""

import ttl

TILE = 32


def make_kv_append(St, Dt):
    """Patch ``cache[(pos//32) tile row]`` row ``pos%32`` with ``k[0, :]``."""

    @ttl.atom(grid=(1, 1))
    def kv_append(cache, k, pos, out):
        pos_cb = ttl.make_dataflow_buffer_like(pos, shape=(1, 1), block_count=1)
        k_cb = ttl.make_dataflow_buffer_like(k, shape=(1, Dt), block_count=1)
        band_cb = ttl.make_dataflow_buffer_like(cache, shape=(1, Dt), block_count=1)

        pd = pos_cb.reserve()
        ttl.copy(pos[0, 0], pd)
        pos_blk = pos_cb.wait()
        r = ttl.read_index(pos_blk, 0, 0)
        intra = ttl.read_index(pos_blk, 0, 1)

        kd = k_cb.reserve()
        ttl.copy(k[0:1, 0:Dt], kd)
        k_blk = k_cb.wait()

        bd = band_cb.reserve()
        ttl.copy(cache[r:r + 1, 0:Dt], bd)
        band = band_cb.wait()
        for c in range(Dt * TILE):
            v = ttl.raw_element_read(k_blk, 0, c)
            ttl.raw_element_write(band, intra, c, v)
        ttl.copy(band, out[r:r + 1, 0:Dt])

    return kv_append


def make_kv_patch_core(Dt):
    """Inlinable cache-row patch: write all elements of ``k_in`` row 0 into
    ``band`` at row ``intra``. The caller copies the cache band in and back
    out around it (read_index supplies the offsets)."""

    @ttl.atom()
    def kv_patch_core(k_in: ttl.DFB, band: ttl.DFB, pos_blk: ttl.DFB):
        kb = k_in.wait()
        bb = band.wait()
        pb = pos_blk.wait()
        intra = ttl.read_index(pb, 0, 1)
        for c in range(Dt * TILE):
            v = ttl.raw_element_read(kb, 0, c)
            ttl.raw_element_write(bb, intra, c, v)

    return kv_patch_core
