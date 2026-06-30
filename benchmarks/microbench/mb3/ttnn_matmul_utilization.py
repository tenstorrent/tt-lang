# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-node TTNN matmul utilization validation.

This is the GEMM-report comparison point for matmul diagnostics. It uses TTNN's
MatmulMultiCoreReuseMultiCastProgramConfig on a 1x1 grid, then computes the same
ideal-cycle ratio used by tt-metal's GEMM FLOPS benchmark.

    python -m benchmarks.microbench.mb3.ttnn_matmul_utilization --mt 8 --nt 8 --kt 8
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

import argparse
from itertools import product
from pathlib import Path

import torch
import ttnn

from benchmarks.common import measure_pcc
from benchmarks.microbench import harness, profiler
from benchmarks.microbench.harness import TILE

MATMUL_CYCLES_PER_TILE = {"lofi": 16, "hifi2": 32, "hifi4": 64}
DTYPES = {
    "bf16": (ttnn.bfloat16, torch.bfloat16),
}
FIDELITIES = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi4": ttnn.MathFidelity.HiFi4,
}

# Same preference order as tt-metal's GEMM FLOPS benchmark.
SUBBLOCK_HW_CHOICES = (
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),
    (7, 1),
    (1, 7),
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),
    (5, 1),
    (1, 5),
    (2, 2),
    (4, 1),
    (1, 4),
    (3, 1),
    (1, 3),
    (2, 1),
    (1, 2),
    (1, 1),
)

CSV_FIELDS = (
    "mt",
    "nt",
    "kt",
    "dtype",
    "fidelity",
    "in0_block_w",
    "out_subblock_h",
    "out_subblock_w",
    "arch",
    "freq_mhz",
    "ideal_cycles",
    "trisc1_kernel_us",
    "trisc_max_us",
    "unpack_us",
    "pack_us",
    "trisc1_cycles",
    "trisc_max_cycles",
    "trisc1_utilization_pct",
    "trisc_max_utilization_pct",
    "pcc",
)


def parse_csv_ints(value):
    return [int(item) for item in str(value).split(",") if item]


def choose_subblock(mt, nt, fp32_dest_acc=False):
    for out_subblock_h, out_subblock_w in SUBBLOCK_HW_CHOICES:
        if fp32_dest_acc and out_subblock_h * out_subblock_w > 4:
            continue
        if mt % out_subblock_h == 0 and nt % out_subblock_w == 0:
            return out_subblock_h, out_subblock_w
    return 1, 1


def make_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt", default="8", help="per-core M tiles")
    parser.add_argument("--nt", default="8", help="per-core N tiles")
    parser.add_argument("--kt", default="8,16,32", help="K tiles")
    parser.add_argument("--dtype", default="bf16", choices=tuple(DTYPES))
    parser.add_argument("--fidelity", default="hifi4", choices=tuple(FIDELITIES))
    parser.add_argument("--in0-block-w-div", type=int, default=1)
    parser.add_argument("--out-subblock-h", type=int)
    parser.add_argument("--out-subblock-w", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmarks/microbench/results/ttnn_matmul_utilization.csv"),
    )
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=os.getenv("TTLANG_COMPILE_ONLY") == "1",
    )
    return parser


def flush_profiler(device, csv_path):
    ttnn.ReadDeviceProfiler(device)
    if csv_path.exists():
        csv_path.unlink()


def mean_zone(zones):
    if len(zones) == 1:
        return zones[0]
    averaged = dict(zones[0])
    for field in ("trisc_max_us", "unpack_us", "math_us", "pack_us"):
        values = [zone[field] for zone in zones if zone.get(field) is not None]
        averaged[field] = sum(values) / len(values) if values else None
    averaged["noc_active_in_zone"] = any(
        zone.get("noc_active_in_zone") for zone in zones
    )
    return averaged


def cycles(microseconds, freq_mhz):
    if microseconds is None:
        return None
    return round(microseconds * freq_mhz, 2)


def utilization_pct(ideal_cycles, actual_cycles):
    if actual_cycles is None or actual_cycles <= 0:
        return None
    return round(100.0 * ideal_cycles / actual_cycles, 2)


def run_config(device, args, mt, nt, kt):
    if kt % args.in0_block_w_div != 0:
        raise ValueError("kt must be divisible by --in0-block-w-div")

    out_subblock_h, out_subblock_w = choose_subblock(mt, nt)
    if args.out_subblock_h is not None:
        out_subblock_h = args.out_subblock_h
    if args.out_subblock_w is not None:
        out_subblock_w = args.out_subblock_w
    if mt % out_subblock_h != 0 or nt % out_subblock_w != 0:
        raise ValueError("output subblock must divide mt and nt")

    ttnn_dtype, torch_dtype = DTYPES[args.dtype]
    torch.manual_seed(args.seed)
    lhs = torch.ones(1, 1, mt * TILE, kt * TILE, dtype=torch_dtype)
    rhs = torch.randn(1, 1, kt * TILE, nt * TILE, dtype=torch_dtype)
    lhs_device = ttnn.from_torch(
        lhs,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rhs_device = ttnn.from_torch(
        rhs,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in0_block_w = kt // args.in0_block_w_div
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=mt,
        out_block_w=nt,
        per_core_M=mt,
        per_core_N=nt,
        transpose_mcast=False,
        fused_activation=None,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=FIDELITIES[args.fidelity],
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )

    def run_matmul():
        return ttnn.matmul(
            lhs_device,
            rhs_device,
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn_dtype,
            compute_kernel_config=compute_kernel_config,
        )

    output_device = None
    csv_path = profiler.find_profiler_csv()
    for _warmup_index in range(args.warmup):
        warmup_output = run_matmul()
        ttnn.synchronize_device(device)
        ttnn.deallocate(warmup_output)
    flush_profiler(device, csv_path)

    zones = []
    for run_index in range(args.runs):
        output_device = run_matmul()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        zones.append(profiler.summarize_zone(csv_path, "TRISC-KERNEL"))
        if csv_path.exists():
            csv_path.unlink()
        if run_index + 1 < args.runs:
            ttnn.deallocate(output_device)
            output_device = None

    zone = mean_zone(zones)
    actual = ttnn.to_torch(output_device).float()
    golden = lhs.float() @ rhs.float()
    pcc = measure_pcc(golden, actual)

    ttnn.deallocate(lhs_device)
    ttnn.deallocate(rhs_device)
    ttnn.deallocate(output_device)

    ideal_cycles = mt * nt * kt * MATMUL_CYCLES_PER_TILE[args.fidelity]
    freq_mhz = zone["freq_mhz"]
    trisc1_kernel_us = zone["math_us"]
    trisc1_cycles = cycles(trisc1_kernel_us, freq_mhz)
    trisc_max_cycles = cycles(zone["trisc_max_us"], freq_mhz)
    return {
        "mt": mt,
        "nt": nt,
        "kt": kt,
        "dtype": args.dtype,
        "fidelity": args.fidelity,
        "in0_block_w": in0_block_w,
        "out_subblock_h": out_subblock_h,
        "out_subblock_w": out_subblock_w,
        "arch": zone["arch"],
        "freq_mhz": freq_mhz,
        "ideal_cycles": ideal_cycles,
        "trisc1_kernel_us": trisc1_kernel_us,
        "trisc_max_us": zone["trisc_max_us"],
        "unpack_us": zone["unpack_us"],
        "pack_us": zone["pack_us"],
        "trisc1_cycles": trisc1_cycles,
        "trisc_max_cycles": trisc_max_cycles,
        "trisc1_utilization_pct": utilization_pct(ideal_cycles, trisc1_cycles),
        "trisc_max_utilization_pct": utilization_pct(ideal_cycles, trisc_max_cycles),
        "pcc": round(pcc, 6),
    }


def main():
    args = make_arg_parser().parse_args()
    if args.compile_only:
        print("compile-only: nothing to execute without a device.")
        return

    rows = []
    device = ttnn.open_device(device_id=args.device_id)
    try:
        for mt, nt, kt in product(
            parse_csv_ints(args.mt), parse_csv_ints(args.nt), parse_csv_ints(args.kt)
        ):
            row = run_config(device, args, mt, nt, kt)
            rows.append(row)
            print(
                f"mt={mt} nt={nt} kt={kt} | "
                f"trisc1={row['trisc1_kernel_us']} µs | "
                f"util={row['trisc1_utilization_pct']}% | pcc={row['pcc']}",
                flush=True,
            )
            if row["pcc"] < 0.999:
                print(f"  WARN: pcc {row['pcc']:.4f} < 0.999", flush=True)
    finally:
        ttnn.close_device(device)

    if not args.no_csv and rows:
        output_csv = harness.write_csv(
            rows,
            args.csv,
            CSV_FIELDS,
            rows[0]["arch"],
            args.dtype,
            args.fidelity,
            "single_node",
        )
        print(f"wrote {len(rows)} rows to {output_csv}", flush=True)


if __name__ == "__main__":
    main()
