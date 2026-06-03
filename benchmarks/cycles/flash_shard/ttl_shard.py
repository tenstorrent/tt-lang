# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""tt-lang side of the flash-shard cycle A/B.

Runs the ttl.ops.flash_mla online-softmax shard on one core over its K/V slice
and reports the device kernel duration from the Tracy device profiler. Pairs
with ``metal.py`` (the ``compute_sdpa_chunk`` baseline); ``__main__`` runs both.

Run standalone on hardware with the profiler enabled:

    TT_METAL_DEVICE_PROFILER=1 python3 -m benchmarks.cycles.flash_shard.ttl
"""

import torch

import ttnn
from ttl.ops.flash_mla import make_flash_shard

from benchmarks.common import (
    clear_profile_log,
    parse_kernel_duration,
    read_device_profiler,
)

from . import shapes
from .shapes import DHt, PNHt, SCALE, Sk_chunk_t, TILE, WORKER_L1, vDHt


def _dram(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def run(n_chunks=shapes.N_CHUNKS):
    """Run the shard once with the profiler on; return cycles/us/per_risc/pcc."""
    torch.manual_seed(7)
    PN, D, vD = PNHt * TILE, DHt * TILE, vDHt * TILE
    S = Sk_chunk_t * n_chunks * TILE

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
            Sk_chunk_t=Sk_chunk_t, N_CHUNKS=n_chunks, scale=SCALE,
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

    scores = (q_t.float() @ k_t.float().T) * SCALE
    ref = torch.softmax(scores, dim=-1) @ v_t.float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()

    d = parse_kernel_duration()
    d["pcc"] = pcc
    return d


def main():
    d = run()
    shapes.print_result("flash_shard (ttl)", d, shapes.N_CHUNKS)


if __name__ == "__main__":
    main()
