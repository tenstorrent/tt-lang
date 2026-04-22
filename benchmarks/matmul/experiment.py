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
        f"{label:<40}  {block_cfg}/{part_cfg}  "
        f"t={min(times)*1000:>7.3f}ms  pcc={pcc:.6f}  max_err={max_err:.2e}",
        flush=True,
    )


EXPERIMENTS = [
    # Tight candidates around the new tiebreak on 4k³ (all (8,4,8) Kp=1,
    # model ties them on throughput — pad/cores vary).
    ("4k³  (8,4,8)/(8,11,1)  88c pad=1.03", 4096, 4096, 4096, (8, 4, 8), (8, 11, 1)),
    ("4k³  (8,4,8)/(8,12,1)  96c pad=1.12", 4096, 4096, 4096, (8, 4, 8), (8, 12, 1)),
    ("4k³  (8,4,8)/(9,11,1)  99c pad=1.16", 4096, 4096, 4096, (8, 4, 8), (9, 11, 1)),
    ("4k³  (8,4,8)/(8,13,1) 104c pad=1.22", 4096, 4096, 4096, (8, 4, 8), (8, 13, 1)),
    ("4k³  (8,4,8)/(9,12,1) 108c pad=1.27", 4096, 4096, 4096, (8, 4, 8), (9, 12, 1)),
    ("4k³  (8,4,8)/(10,12,1) 120c pad=1.40", 4096, 4096, 4096, (8, 4, 8), (10, 12, 1)),

    # 8k³: test new 104c pad=1.02 pick vs previous 117c pad=1.14
    ("8k³  (8,4,8)/(8,13,1) 104c pad=1.02", 8192, 8192, 8192, (8, 4, 8), (8, 13, 1)),
    ("8k³  (8,4,8)/(9,13,1) 117c pad=1.14", 8192, 8192, 8192, (8, 4, 8), (9, 13, 1)),
    ("8k³  (8,4,8)/(10,13,1) 130c pad=1.27", 8192, 8192, 8192, (8, 4, 8), (10, 13, 1)),

    # 4k×8k×4k — follow same family
    ("4k8k4k  (8,4,8)/(8,11,1) 88c pad=1.03", 4096, 8192, 4096, (8, 4, 8), (8, 11, 1)),
    ("4k8k4k  (8,4,8)/(8,12,1) 96c pad=1.12", 4096, 8192, 4096, (8, 4, 8), (8, 12, 1)),
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
                print(f"{args[0]:<40}  FAIL: {e}", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
