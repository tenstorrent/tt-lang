# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sweep the ksplit/SUMMA matmul vs ttnn.matmul across shapes.

Each shape runs WARMUP_RUNS untimed passes to compile and warm caches,
then TIMED_RUNS timed passes with a short sleep between them. Best (min)
wall time is recorded; PCC is checked against a torch float reference
on unpadded output. Results go to OUTPUT_CSV for later plotting.

Dispatch: plans with K_parts == 1 run summa_kernel (no reduce_net);
plans with K_parts >= 2 run ksplit_kernel (gather partial blocks).

Requires ksplit_kernel.py, summa_kernel.py, and config.py in the same
directory as this script (use copy-file.sh to stage them on remote /tmp
before run-test.sh).
"""

import csv
import time
from pathlib import Path

import torch
import ttnn

from config import plan_matmul
from ksplit_kernel import make_kernel as make_ksplit_kernel
from plot import save_plot
from summa_kernel import make_kernel as make_summa_kernel


# Sorted by M*K*N (rough work size). Annotations in labels flag why a
# shape is interesting (short K, long K, full grid, etc.).
SHAPES = (
    (1024, 1024, 1024, "1k^3"),
    (1024, 2048, 1024, "1k x 2k x 1k"),
    (2048, 2048, 2048, "2k^3"),
    (3072, 1024, 3072, "3k x 1k x 3k (short K)"),
    (2560, 2048, 3072, "2.5k x 2k x 3k"),
    (2048, 4096, 2048, "2k x 4k x 2k"),
    (2560, 4096, 3072, "2.5k x 4k x 3k"),
    (2048, 8192, 2048, "2k x 8k x 2k (long K)"),
    (3072, 4096, 3072, "3k x 4k x 3k"),
    (1024, 16384, 2560, "1k x 16k x 2.5k (tall K)"),
    (5120, 2048, 5120, "5k x 2k x 5k (short K)"),
    (2560, 8192, 3072, "2.5k x 8k x 3k (120 cores)"),
    (4096, 4096, 4096, "4k^3"),
    (2560, 8192, 3328, "2.5k x 8k x 3.3k (130 cores)"),
    (6144, 2048, 6144, "6k x 2k x 6k (short K)"),
    (4096, 8192, 4096, "4k x 8k x 4k"),
    (2560, 16384, 3328, "2.5k x 16k x 3.3k"),
    (2560, 32768, 3328, "2.5k x 32k x 3.3k"),
    (8192, 8192, 8192, "8k^3"),
    (10240, 8192, 13312, "10k x 8k x 13k (130 cores, 4x4)"),
    (5120, 32768, 6656, "5k x 32k x 6.5k"),
    (10240, 16384, 13312, "10k x 16k x 13k"),
)

WARMUP_RUNS = 3
TIMED_RUNS = 5
SLEEP_BETWEEN_MS = 10
L1_BUDGET_REDUCTION_BYTES = 131072  # 128 KiB headroom for tt-metal state
OUTPUT_CSV = Path("/tmp/ksplit_sweep.csv")

FP32_ACC = True
TTNN_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4 if FP32_ACC else ttnn.MathFidelity.HiFi2,
    fp32_dest_acc_en=FP32_ACC,
    packer_l1_acc=True,
)

FIELDS = (
    "label",
    "M",
    "K",
    "N",
    "bm",
    "bn",
    "bk",
    "Mp",
    "Np",
    "Kp",
    "cores",
    "iter_per_core",
    "pad",
    "ksplit_ms",
    "ttnn_ms",
    "ratio",
    "pcc",
)


def to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def pad_2d(t, rows, cols):
    r, c = t.shape
    if r == rows and c == cols:
        return t
    return torch.nn.functional.pad(t, (0, cols - c, 0, rows - r), value=0.0)


def time_runs(thunk, cleanup, device):
    """Warmup + timed runs. `thunk()` returns a value passed to `cleanup`.

    Returns the fastest observed wall time in seconds.
    """
    for _ in range(WARMUP_RUNS):
        cleanup(thunk())
    ttnn.synchronize_device(device)

    times = []
    for _ in range(TIMED_RUNS):
        time.sleep(SLEEP_BETWEEN_MS / 1000)
        t0 = time.perf_counter()
        result = thunk()
        ttnn.synchronize_device(device)
        times.append(time.perf_counter() - t0)
        cleanup(result)
    return min(times)


def bench_shape(device, label, M, K, N):
    plan = plan_matmul(M, K, N)
    M_pad, N_pad = plan.padded_dims

    torch.manual_seed(0)
    a_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    ref = a_t.float() @ w_t.float()

    a_k = to_dev(pad_2d(a_t, M_pad, K), device)
    w_k = to_dev(pad_2d(w_t, K, N_pad), device)
    out_k = to_dev(torch.zeros(M_pad, N_pad, dtype=torch.bfloat16), device)

    _, _, Kp = plan.part_cfg
    make_kernel = make_summa_kernel if Kp == 1 else make_ksplit_kernel
    fn = make_kernel(M_pad, K, N_pad, plan.block_cfg, plan.part_cfg)
    ksplit_s = time_runs(
        thunk=lambda: fn(a_k, w_k, out_k),
        cleanup=lambda _r: None,
        device=device,
    )

    result = ttnn.to_torch(out_k).float()[:M, :N]
    pcc = torch.corrcoef(torch.stack([result.flatten(), ref.flatten()]))[0, 1].item()

    a_ref = to_dev(a_t, device)
    w_ref = to_dev(w_t, device)
    ttnn_s = time_runs(
        thunk=lambda: ttnn.matmul(a_ref, w_ref, compute_kernel_config=TTNN_CFG),
        cleanup=ttnn.deallocate,
        device=device,
    )

    for t in (a_k, w_k, out_k, a_ref, w_ref):
        ttnn.deallocate(t)

    bm, bn, bk = plan.block_cfg
    Mp, Np, Kp = plan.part_cfg
    return {
        "label": label,
        "M": M,
        "K": K,
        "N": N,
        "bm": bm,
        "bn": bn,
        "bk": bk,
        "Mp": Mp,
        "Np": Np,
        "Kp": Kp,
        "cores": plan.cores,
        "iter_per_core": plan.iters_per_core,
        "pad": round(plan.pad_ratio, 4),
        "ksplit_ms": round(ksplit_s * 1000, 4),
        "ttnn_ms": round(ttnn_s * 1000, 4),
        "ratio": round(ksplit_s / ttnn_s, 4),
        "pcc": round(pcc, 6),
    }


def main():
    default_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(
        device_id=0,
        worker_l1_size=default_l1 - L1_BUDGET_REDUCTION_BYTES,
    )
    results = []
    try:
        for M, K, N, label in SHAPES:
            try:
                r = bench_shape(device, label, M, K, N)
            except Exception as e:
                print(f"{label:<32}  FAIL: {e}", flush=True)
                continue
            print(
                f"{label:<32}  "
                f"ksplit={r['ksplit_ms']:>8.3f}ms  ttnn={r['ttnn_ms']:>8.3f}ms  "
                f"ratio={r['ratio']:.3f}  pcc={r['pcc']:.4f}  "
                f"({r['bm']},{r['bn']},{r['bk']})/"
                f"({r['Mp']},{r['Np']},{r['Kp']}) cores={r['cores']}",
                flush=True,
            )
            results.append(r)
    finally:
        ttnn.close_device(device)

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nwrote {len(results)} rows to {OUTPUT_CSV}", flush=True)

    save_plot(results, path=str(OUTPUT_CSV.with_suffix(".png")))


if __name__ == "__main__":
    main()
