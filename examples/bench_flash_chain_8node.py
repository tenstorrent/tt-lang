# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the 8-node Flash chain example.

The timed region covers only the operation launch plus device synchronization.
Tensor creation, golden computation, host transfers, and PCC checks are outside
the steady-state measurement.
"""

import argparse
import os
import statistics
import sys
import time

if "--device-profiler" in sys.argv:
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
    if "--keep-profiler-files" not in sys.argv:
        os.environ.setdefault("TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES", "1")

import torch
import ttnn

from flash_chain_8node import (
    DHt,
    HEAD_DIM,
    HEAD_DIM_V,
    KERNEL_CONFIG_BUFFER_SIZE,
    N_CHUNKS,
    NNODES,
    PCC_THRESHOLD,
    PNHt,
    Q_ROWS,
    SCALE,
    SEQ,
    Sk_chunk_t,
    TILE,
    _compute_abs_error_metrics,
    _compute_pcc,
    _to_dram,
    flash_chain_8node,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device-profiler",
        action="store_true",
        help="Report tt-metal device profiler durations for measured iterations.",
    )
    parser.add_argument(
        "--keep-profiler-files",
        action="store_true",
        help="Keep tt-metal profiler files when --device-profiler is enabled.",
    )
    return parser.parse_args()


def measure_once(device, run_once):
    start = time.perf_counter()
    run_once()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) * 1000.0


def summarize(values):
    if not values:
        return "count=0"
    if len(values) == 1:
        stdev = 0.0
    else:
        stdev = statistics.stdev(values)
    return (
        f"count={len(values)} "
        f"mean_ms={statistics.mean(values):.6f} "
        f"median_ms={statistics.median(values):.6f} "
        f"min_ms={min(values):.6f} "
        f"max_ms={max(values):.6f} "
        f"stdev_ms={stdev:.6f}"
    )


DEVICE_FW_DURATION_KEY = "DEVICE FW DURATION [ns]"
DEVICE_KERNEL_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def duration_us(program, name):
    analysis = program.program_analyses_results.get(name)
    if analysis is None:
        return None
    return analysis.duration / 1000.0


def collect_device_profile(device):
    ttnn.ReadDeviceProfiler(device)
    latest_data = ttnn.get_latest_programs_perf_data() or {}
    profile_rows = []
    for device_id in sorted(latest_data):
        programs = sorted(
            list(latest_data[device_id]),
            key=lambda program: (
                program.program_execution_uid.runtime_id,
                program.program_execution_uid.trace_id,
                program.program_execution_uid.trace_id_counter,
            ),
        )
        for program in programs:
            profile_rows.append(
                (
                    device_id,
                    program.program_execution_uid,
                    program.core_count,
                    duration_us(program, DEVICE_FW_DURATION_KEY),
                    duration_us(program, DEVICE_KERNEL_DURATION_KEY),
                )
            )
    return profile_rows


def summarize_us(values):
    values = [value for value in values if value is not None]
    if not values:
        return "count=0"
    if len(values) == 1:
        stdev = 0.0
    else:
        stdev = statistics.stdev(values)
    return (
        f"count={len(values)} "
        f"mean_us={statistics.mean(values):.6f} "
        f"median_us={statistics.median(values):.6f} "
        f"min_us={min(values):.6f} "
        f"max_us={max(values):.6f} "
        f"sum_us={sum(values):.6f} "
        f"stdev_us={stdev:.6f}"
    )


def print_device_profile(profile_rows):
    for index, (device_id, uid, core_count, fw_us, kernel_us) in enumerate(
        profile_rows
    ):
        fw_text = "n/a" if fw_us is None else f"{fw_us:.6f}"
        kernel_text = "n/a" if kernel_us is None else f"{kernel_us:.6f}"
        print(
            "device_profile_row "
            f"index={index} device_id={device_id} "
            f"uid_runtime={uid.runtime_id} uid_trace={uid.trace_id} "
            f"uid_counter={uid.trace_id_counter} cores={core_count} "
            f"fw_us={fw_text} kernel_us={kernel_text}"
        )
    print(
        "device_profile_fw "
        + summarize_us([fw_us for _, _, _, fw_us, _ in profile_rows])
    )
    print(
        "device_profile_kernel "
        + summarize_us([kernel_us for _, _, _, _, kernel_us in profile_rows])
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    max_worker_l1_size = ttnn.device.get_max_worker_l1_unreserved_size()
    worker_l1_size = max_worker_l1_size - KERNEL_CONFIG_BUFFER_SIZE
    device = ttnn.open_device(device_id=0, worker_l1_size=worker_l1_size)
    try:
        q_torch = torch.randn(Q_ROWS, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        k_torch = torch.randn(SEQ, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        v_torch = torch.randn(SEQ, HEAD_DIM_V, dtype=torch.bfloat16) * 0.1

        q_ref = q_torch.float().unsqueeze(0).unsqueeze(0)
        k_ref = k_torch.float().unsqueeze(0).unsqueeze(0)
        v_ref = v_torch.float().unsqueeze(0).unsqueeze(0)
        o_ref = (
            torch.nn.functional.scaled_dot_product_attention(
                q_ref,
                k_ref,
                v_ref,
                scale=SCALE,
            )
            .squeeze(0)
            .squeeze(0)
            .to(torch.bfloat16)
        )

        q_dram = _to_dram(device, q_torch)
        k_dram = _to_dram(device, k_torch)
        v_dram = _to_dram(device, v_torch)
        final_dram = _to_dram(
            device,
            torch.zeros(Q_ROWS, HEAD_DIM_V, dtype=torch.bfloat16),
        )

        def run_once():
            flash_chain_8node(q_dram, k_dram, v_dram, final_dram)

        print(
            "flash_chain8_bench "
            f"tile={TILE} pnh_t={PNHt} dh_t={DHt} vdh_t={HEAD_DIM_V // TILE} "
            f"sk_chunk_t={Sk_chunk_t} n_chunks={N_CHUNKS} nnodes={NNODES} "
            f"seq={SEQ} head_dim={HEAD_DIM} head_dim_v={HEAD_DIM_V} "
            f"warmup_iters={args.warmup_iters} bench_iters={args.bench_iters} "
            f"device_profiler={args.device_profiler}"
        )

        cold_first_ms = measure_once(device, run_once)
        print(f"cold_first_ms={cold_first_ms:.6f}")

        warmup_times = [
            measure_once(device, run_once) for _ in range(args.warmup_iters)
        ]
        print(f"warmup {summarize(warmup_times)}")
        if args.device_profiler:
            collect_device_profile(device)

        steady_times = [measure_once(device, run_once) for _ in range(args.bench_iters)]
        print(f"steady {summarize(steady_times)}")
        if args.device_profiler:
            print_device_profile(collect_device_profile(device))

        out = ttnn.to_torch(final_dram).reshape(Q_ROWS, HEAD_DIM_V).to(torch.bfloat16)
        pcc = _compute_pcc(o_ref, out)
        mean_abs, max_abs = _compute_abs_error_metrics(o_ref, out)
        print(f"pcc={pcc:.6f}")
        print(f"mean_abs={mean_abs:.6e}")
        print(f"max_abs={max_abs:.6e}")
        if pcc < PCC_THRESHOLD:
            raise AssertionError(f"PCC={pcc}, threshold={PCC_THRESHOLD}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
