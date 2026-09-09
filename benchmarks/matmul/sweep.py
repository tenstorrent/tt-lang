# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sweep the ksplit/SUMMA matmul vs ttnn.matmul across shapes.

Each shape runs WARMUP_RUNS untimed passes to compile and warm caches,
then TIMED_RUNS device-profiled passes. Best (min) TT-Metal device kernel
duration is recorded; PCC is checked against a torch float reference
on unpadded output. Results go to OUTPUT_CSV for later plotting.

Dispatch: plans with K_parts == 1 run summa_kernel (no reduce_net);
plans with K_parts >= 2 run ksplit_kernel (gather partial blocks).

Run from the repository root with `python -m benchmarks.matmul.sweep`.
Writes CSV, a ratio figure, and JSON provenance; failures retain diagnostics
in JSON and exit nonzero without publishing a partial figure.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import ttnn

from benchmarks.device_timing import latest_kernel_duration, read_device_profile
from benchmarks.matmul.config import plan_matmul
from benchmarks.matmul.ksplit_kernel import make_kernel as make_ksplit_kernel
from benchmarks.matmul.plot import save_plot
from benchmarks.matmul.summa_kernel import make_kernel as make_summa_kernel
from benchmarks.provenance import collect_provenance
from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

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
    return to_dram(t.contiguous(), device)


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


def device_runs(thunk, cleanup, device):
    """Use TT-Metal's device-kernel metric for both implementations."""
    for _iteration in range(WARMUP_RUNS):
        cleanup(thunk())
    ttnn.synchronize_device(device)
    samples = []
    previous_run_id = -1
    for _iteration in range(TIMED_RUNS):
        result = thunk()
        ttnn.synchronize_device(device)
        profiler_log = read_device_profile(device)
        device_ids = device.get_device_ids()
        if len(device_ids) != 1:
            raise ValueError("matmul sweep requires exactly one physical device")
        duration = latest_kernel_duration(profiler_log, device_ids)
        run_id = duration["per_device"][str(device_ids[0])]["run_host_id"]
        if run_id <= previous_run_id:
            raise ValueError("stale device-profiler sample")
        previous_run_id = run_id
        samples.append(duration)
        cleanup(result)
    return min(sample["us"] for sample in samples) / 1e6, samples


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
    worker_grid = device.compute_with_storage_grid_size()
    if plan.part_cfg[1] * Kp > worker_grid.x or plan.part_cfg[0] > worker_grid.y:
        raise ValueError(f"plan {plan.part_cfg} exceeds device grid {worker_grid}")
    make_kernel = make_summa_kernel if Kp == 1 else make_ksplit_kernel
    fn = make_kernel(M_pad, K, N_pad, plan.block_cfg, plan.part_cfg)
    ksplit_s, ksplit_samples = device_runs(
        thunk=lambda: fn(a_k, w_k, out_k),
        cleanup=lambda _r: None,
        device=device,
    )

    result = ttnn.to_torch(out_k).float()[:M, :N]
    assert_pcc(ref, result, threshold=0.99)
    pcc = torch.corrcoef(torch.stack([result.flatten(), ref.flatten()]))[0, 1].item()

    a_ref = to_dev(a_t, device)
    w_ref = to_dev(w_t, device)
    native_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=FP32_ACC,
        packer_l1_acc=True,
    )
    native_output = ttnn.matmul(a_ref, w_ref, compute_kernel_config=native_config)
    assert_pcc(ref, ttnn.to_torch(native_output).float(), threshold=0.99)
    ttnn.deallocate(native_output)
    ttnn_s, ttnn_samples = device_runs(
        thunk=lambda: ttnn.matmul(a_ref, w_ref, compute_kernel_config=native_config),
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
        "device_samples": {"ttlang": ksplit_samples, "ttnn": ttnn_samples},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--shape-index",
        type=int,
        action="append",
        help="run only these zero-based SHAPES entries (repeatable)",
    )
    arguments = parser.parse_args()
    if os.getenv("TT_METAL_DEVICE_PROFILER") != "1":
        parser.error("requires TT_METAL_DEVICE_PROFILER=1 before Python starts")
    if os.getenv("TT_METAL_PROFILER_MID_RUN_DUMP") != "1":
        parser.error("requires TT_METAL_PROFILER_MID_RUN_DUMP=1")
    if not os.getenv("TT_METAL_PROFILER_DIR"):
        parser.error("requires a run-specific TT_METAL_PROFILER_DIR")
    for variable in (
        "TT_METAL_PROFILER_ACCUMULATE",
        "TTLANG_COMPILE_ONLY",
        "TTLANG_AUTO_PROFILE",
        "TTLANG_PERF_DUMP",
        "TTLANG_SIGNPOST_PROFILE",
    ):
        if os.getenv(variable, "0") not in ("", "0"):
            parser.error(f"unset {variable} before timing")
    selected_shapes = (
        SHAPES
        if arguments.shape_index is None
        else tuple(SHAPES[index] for index in arguments.shape_index)
    )
    source_directory = Path(__file__).parent
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": collect_provenance(
            [
                *sorted(source_directory.glob("*.py")),
                source_directory.parent / "device_timing.py",
            ]
        ),
        "profiler_directory": os.environ["TT_METAL_PROFILER_DIR"],
        "measurement": {
            "warmup": WARMUP_RUNS,
            "runs": TIMED_RUNS,
            "statistic": "minimum_ttmetal_device_kernel_duration",
            "dtype": "bf16",
            "layout": "TILE",
            "memory": "DRAM",
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": FP32_ACC,
        },
        "requested_shapes": selected_shapes,
    }
    default_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(
        device_id=arguments.device_id,
        worker_l1_size=default_l1 - L1_BUDGET_REDUCTION_BYTES,
    )
    worker_grid = device.compute_with_storage_grid_size()
    metadata.update(
        arch=str(device.arch()),
        device_id=arguments.device_id,
        worker_grid=[worker_grid.x, worker_grid.y],
        worker_l1_size=default_l1 - L1_BUDGET_REDUCTION_BYTES,
    )
    results = []
    try:
        for M, K, N, label in selected_shapes:
            print(f"Running {label}", flush=True)
            r = bench_shape(device, label, M, K, N)
            print(
                f"{label:<32}  "
                f"ksplit={r['ksplit_ms']:>8.3f}ms  ttnn={r['ttnn_ms']:>8.3f}ms  "
                f"ratio={r['ratio']:.3f}  pcc={r['pcc']:.4f}  "
                f"({r['bm']},{r['bn']},{r['bk']})/"
                f"({r['Mp']},{r['Np']},{r['Kp']}) cores={r['cores']}",
                flush=True,
            )
            results.append(r)
    except BaseException as error:
        metadata["failure"] = {"label": label, "error": str(error)}
        raise
    finally:
        ttnn.close_device(device)
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["completed_rows"] = results
        arguments.csv.with_suffix(
            ".failed.json" if "failure" in metadata else ".json"
        ).write_text(json.dumps(metadata, indent=2) + "\n")

    with arguments.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in FIELDS} for row in results)
    print(f"\nwrote {len(results)} rows to {arguments.csv}", flush=True)

    save_plot(results, path=str(arguments.csv.with_suffix(".png")))


if __name__ == "__main__":
    main()
