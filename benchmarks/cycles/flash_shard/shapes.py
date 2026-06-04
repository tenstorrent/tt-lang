# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared geometry for the flash-shard cycle A/B.

A per-core MLA decode slice: PNHt=1 query head-tile, DHt=18 (kvpe head dim,
576), vDHt=16 (kv_lora_rank, 512), one tile per K chunk. Both the tt-lang shard
(``ttl.py``) and the metal ``compute_sdpa_chunk`` baseline (``metal.py``) read
these so the two are measured on the same problem. ``N_CHUNKS`` is overridable
with ``CYCLES_N_CHUNKS``; the metal baseline keeps the whole K slice resident in
L1 (no DRAM streaming, to isolate compute), so it is capped to what fits.
"""

import math
import os

TILE = 32  # ttnn.TILE_SIZE

PNHt = 1
DHt = 18
vDHt = 16
Sk_chunk_t = 1

# 16 single-tile chunks = 512 K positions. Caps the L1-resident metal baseline
# (the whole K slice stays in L1 to isolate compute); the ttl shard streams from
# DRAM and can go higher (Phase A ran 128).
N_CHUNKS = int(os.environ.get("CYCLES_N_CHUNKS", "16"))

# The metal compute_sdpa_chunk caller groups K into `chunk_size` tile-rows per
# call (the deepseek test uses 4/8, never 1). We pick chunk_size and num_chunks
# so the total K-tile-rows (num_chunks * chunk_size) equals the ttl shard's
# (N_CHUNKS * Sk_chunk_t), keeping the two sides on the same total work.
METAL_CHUNK_SIZE = int(os.environ.get("CYCLES_METAL_CHUNK_SIZE", "8"))
METAL_NUM_CHUNKS = max(1, (N_CHUNKS * Sk_chunk_t) // METAL_CHUNK_SIZE)

WORKER_L1 = 1100000

SCALE = 1.0 / math.sqrt(DHt * TILE)


def print_result(name, d, n_chunks):
    """Print one variant's device kernel duration (cycles, us, us/chunk)."""
    print(f"{name}  N_CHUNKS={n_chunks} DHt={DHt} vDHt={vDHt}", flush=True)
    print(
        f"  device kernel: {d['cycles']} cyc  {d['us']:.1f} us  "
        f"({d['us'] / n_chunks:.2f} us/chunk)  pcc={d.get('pcc', float('nan')):.4f}",
        flush=True,
    )
    print(f"  per-risc cyc:  {d['per_risc']}", flush=True)
