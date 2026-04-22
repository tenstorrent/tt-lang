# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-kernel profiling harness.

Runs one (shape, block, parts) case once so TTLANG_PERF_DUMP /
TTLANG_AUTO_PROFILE stay under the 125-signpost-per-core budget.

Edit CASE below and run via run-test.sh --perf (or --auto-profile) --hw.
"""

import torch
import ttnn

from ksplit_kernel import make_kernel as make_ksplit_kernel
from summa_kernel import make_kernel as make_summa_kernel


# 4k³ current pick: highest-signal miss (1.38 vs ttnn).
CASE = ("4k³ (8,4,8)/(8,11,1) 88c iter=6",
        4096, 4096, 4096, (8, 4, 8), (8, 11, 1))


def padded_dims(M, N, block_cfg, part_cfg):
    TILE = 32
    bm, bn, _ = block_cfg
    Mp, Np, _ = part_cfg
    Mt, Nt = M // TILE, N // TILE
    Mb, Nb = Mt // bm, Nt // bn
    m_span = -(-Mb // Mp)
    n_span = -(-Nb // Np)
    return (Mp * m_span * bm * TILE, Np * n_span * bn * TILE)


def to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def pad_2d(t, rows, cols):
    r, c = t.shape
    if r == rows and c == cols:
        return t
    return torch.nn.functional.pad(t, (0, cols - c, 0, rows - r), value=0.0)


def main():
    label, M, K, N, block_cfg, part_cfg = CASE
    M_pad, N_pad = padded_dims(M, N, block_cfg, part_cfg)
    Kp = part_cfg[2]

    default_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(
        device_id=0,
        worker_l1_size=default_l1 - 131072,
    )
    try:
        torch.manual_seed(0)
        a_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
        w_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02

        a_k = to_dev(pad_2d(a_t, M_pad, K), device)
        w_k = to_dev(pad_2d(w_t, K, N_pad), device)
        out_k = to_dev(torch.zeros(M_pad, N_pad, dtype=torch.bfloat16), device)

        make_fn = make_summa_kernel if Kp == 1 else make_ksplit_kernel
        fn = make_fn(M_pad, K, N_pad, block_cfg, part_cfg)

        print(f"Profiling: {label}", flush=True)
        fn(a_k, w_k, out_k)
        ttnn.synchronize_device(device)

        for t in (a_k, w_k, out_k):
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
