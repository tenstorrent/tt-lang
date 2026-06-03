# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-core cycle benchmark: the ttl.ops.flash_mla shard.

Runs the online-softmax flash-decode shard on one core over its K/V slice and
reports the device kernel duration from the Tracy device profiler. This is the
tt-lang side of the flash-shard vs ``compute_sdpa_chunk`` A/B; the metal
baseline lives in ``benchmarks/cycles/metal_sdpa.py``.

Shapes mirror a per-core MLA decode slice: PNHt=1 query head-tile, DHt=18
(kvpe head dim), vDHt=16 (kv_lora_rank), one tile per K chunk over N_CHUNKS
chunks. Run on hardware with the profiler enabled, e.g.

    TT_METAL_DEVICE_PROFILER=1 python3 -m benchmarks.cycles.flash_shard
"""

import math

import torch

import ttnn
from ttl.ops.flash_mla import make_flash_shard

from benchmarks.common import (
    clear_profile_log,
    parse_kernel_duration,
    read_device_profiler,
)

TILE = ttnn.TILE_SIZE

# Per-core MLA decode slice; N_CHUNKS=128 single-tile chunks = 4096 K positions
# (one eighth of a 32k context), matching the per-core work of an 8-way K split.
PNHt = 1
DHt = 18
vDHt = 16
Sk_chunk_t = 1
N_CHUNKS = 128

WORKER_L1 = 1100000


def _dram(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    torch.manual_seed(7)
    PN, D, vD = PNHt * TILE, DHt * TILE, vDHt * TILE
    S = Sk_chunk_t * N_CHUNKS * TILE
    scale = 1.0 / math.sqrt(D)

    device = ttnn.open_device(device_id=0, worker_l1_size=WORKER_L1)
    try:
        q_t = torch.randn(PN, D, dtype=torch.bfloat16) * 0.1
        k_t = torch.randn(S, D, dtype=torch.bfloat16) * 0.1
        v_t = torch.randn(S, vD, dtype=torch.bfloat16) * 0.1

        # K/V are a bfp8 cache (production MLA-decode dtype); the shard
        # typecasts them back to bf16 for the matmuls.
        q_d = _dram(q_t, device)
        k_d = _dram(k_t, device, dtype=ttnn.bfloat8_b)
        v_d = _dram(v_t, device, dtype=ttnn.bfloat8_b)
        o_d = _dram(torch.zeros(PN, vD, dtype=torch.bfloat16), device)
        m_d = _dram(torch.zeros(PN, TILE, dtype=torch.bfloat16), device)
        l_d = _dram(torch.zeros(PN, TILE, dtype=torch.bfloat16), device)

        shard = make_flash_shard(
            n_cols=1, B=1,
            PNHt=PNHt, DHt=DHt, vDHt=vDHt,
            Sk_chunk_t=Sk_chunk_t, N_CHUNKS=N_CHUNKS, scale=scale,
        )

        clear_profile_log()
        shard(q_d, k_d, v_d, o_d, m_d, l_d)
        ttnn.synchronize_device(device)
        read_device_profiler(device)

        o_unnorm = ttnn.to_torch(o_d).reshape(PN, vD).float()
        l = ttnn.to_torch(l_d).reshape(PN, TILE).float()[:, 0:1]
        got = o_unnorm / l
    finally:
        # close_device flushes the device profiler CSV to disk.
        ttnn.close_device(device)

    scores = (q_t.float() @ k_t.float().T) * scale
    attn = torch.softmax(scores, dim=-1)
    ref = attn @ v_t.float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()

    d = parse_kernel_duration()
    print(
        f"flash_shard  N_CHUNKS={N_CHUNKS} DHt={DHt} vDHt={vDHt}  pcc={pcc:.4f}",
        flush=True,
    )
    print(
        f"  device kernel: {d['cycles']} cyc  {d['us']:.1f} us  "
        f"({d['us'] / N_CHUNKS:.2f} us/chunk)",
        flush=True,
    )
    print(f"  per-risc cyc:  {d['per_risc']}", flush=True)


if __name__ == "__main__":
    main()
