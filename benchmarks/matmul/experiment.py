# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Probe hand-picked (shape, block, parts) candidates.

Use to validate the planner's picks against alternatives, or to isolate
kernel bugs by forcing a specific plan. Prints per-candidate PCC +
median timing so regressions are easy to spot.

Edit EXPERIMENTS at the bottom to add cases; run via run-test.sh --hw.
"""

import time

import torch
import ttnn

from ksplit_kernel import make_kernel as make_ksplit_kernel
from summa_kernel import make_kernel as make_summa_kernel


WARMUP_RUNS = 2
TIMED_RUNS = 3


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


def padded_dims(M, N, block_cfg, part_cfg):
    TILE = 32
    bm, bn, _ = block_cfg
    Mp, Np, _ = part_cfg
    Mt, Nt = M // TILE, N // TILE
    Mb, Nb = Mt // bm, Nt // bn
    m_span = -(-Mb // Mp)
    n_span = -(-Nb // Np)
    return (Mp * m_span * bm * TILE, Np * n_span * bn * TILE)


def run_case(device, label, M, K, N, block_cfg, part_cfg):
    M_pad, N_pad = padded_dims(M, N, block_cfg, part_cfg)
    Kp = part_cfg[2]

    torch.manual_seed(0)
    a_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    ref = a_t.float() @ w_t.float()

    a_k = to_dev(pad_2d(a_t, M_pad, K), device)
    w_k = to_dev(pad_2d(w_t, K, N_pad), device)
    out_k = to_dev(torch.zeros(M_pad, N_pad, dtype=torch.bfloat16), device)

    make_fn = make_summa_kernel if Kp == 1 else make_ksplit_kernel
    fn = make_fn(M_pad, K, N_pad, block_cfg, part_cfg)

    for _ in range(WARMUP_RUNS):
        fn(a_k, w_k, out_k)
    ttnn.synchronize_device(device)

    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        fn(a_k, w_k, out_k)
        ttnn.synchronize_device(device)
        times.append(time.perf_counter() - t0)

    result = ttnn.to_torch(out_k).float()[:M, :N]
    pcc = torch.corrcoef(
        torch.stack([result.flatten(), ref.flatten()]))[0, 1].item()
    max_err = (result - ref).abs().max().item()

    for t in (a_k, w_k, out_k):
        ttnn.deallocate(t)

    print(
        f"{label:<50}  {block_cfg}/{part_cfg}  "
        f"t={min(times)*1000:>7.3f}ms  pcc={pcc:.6f}  max_err={max_err:.2e}",
        flush=True,
    )


# Profile revealed 4k³ is compute-bound (not DRAM-bound), and ttnn uses 130
# cores while our planner caps at 88 (pad budget rejects 110c). Test whether
# relaxing pad to 1.3 and adding cores closes the gap.
EXPERIMENTS = [
    # 4k³: 88c current vs 110c via pad=1.29 (what baseline SUMMA picks).
    ("4k³    (8,4,8)/(8,11,1)  88c iter=6 pad=1.03 current",
     4096, 4096, 4096, (8, 4, 8), (8, 11, 1)),
    ("4k³    (8,4,8)/(10,11,1) 110c iter=6 pad=1.29",
     4096, 4096, 4096, (8, 4, 8), (10, 11, 1)),
    ("4k³    (8,4,8)/(8,13,1)  104c iter=6 pad=1.22",
     4096, 4096, 4096, (8, 4, 8), (8, 13, 1)),
    ("4k³    (8,4,8)/(10,13,1) 130c iter=6 pad=1.52",
     4096, 4096, 4096, (8, 4, 8), (10, 13, 1)),

    # 4k×8k×4k: same picks at longer K — iter should amortize gather better.
    ("4k8k4k (8,4,8)/(8,11,1)  88c iter=6 pad=1.03 current",
     4096, 8192, 4096, (8, 4, 8), (8, 11, 1)),
    ("4k8k4k (8,4,8)/(10,11,1) 110c iter=6 pad=1.29",
     4096, 8192, 4096, (8, 4, 8), (10, 11, 1)),

    # 2k³: try more-core variants to see if core-count is the lever here too.
    ("2k³    (8,4,8)/(8,6,2)  96c Kp=2 iter=3 current",
     2048, 2048, 2048, (8, 4, 8), (8, 6, 2)),
    ("2k³    (8,4,8)/(10,8,1) 80c iter=2 pad=1.25",
     2048, 2048, 2048, (8, 4, 8), (10, 8, 1)),
    ("2k³    (8,4,8)/(10,13,1)130c iter=2 pad=1.63",
     2048, 2048, 2048, (8, 4, 8), (10, 13, 1)),
]


def main():
    default_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(
        device_id=0,
        worker_l1_size=default_l1 - 131072,
    )
    try:
        for args in EXPERIMENTS:
            try:
                run_case(device, *args)
            except Exception as e:
                print(f"{args[0]:<50}  FAIL: {e}", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
