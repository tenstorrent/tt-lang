# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Read TT-Metal's standard device-kernel metric, without parsing profiler CSVs."""

import os
import statistics
from pathlib import Path
from uuid import uuid4


def read_device_profile(device):
    """Retain each synchronous profiler dump without reprocessing earlier samples."""
    import ttnn

    ttnn.ReadDeviceProfiler(device)
    log = Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs/profile_log_device.csv"
    archived = log.with_name(f"profile_log_device_{uuid4().hex}.csv")
    # ReadDeviceProfiler closes the CSV; the next dump recreates its header.
    log.rename(archived)
    return archived


def latest_kernel_duration(csv_path, device_ids, *, aggregation="max", program_count=1):
    """Return the latest program sequence's duration on each participating device.

    TT-Metal defines device_kernel_duration as the earliest kernel start to
    latest kernel end across a device's workers. Device clocks are independent;
    compare durations, never timestamps, across devices.
    Multiple programs include the device-clock interval between launches.
    """
    if aggregation not in ("mean", "max"):
        raise ValueError(f"unsupported device aggregation: {aggregation}")
    if not device_ids:
        raise ValueError("at least one participating device is required")
    if program_count < 1:
        raise ValueError("program_count must be positive")
    from tracy.device_post_proc_config import default_setup
    from tracy.process_device_log import import_log_run_stats

    setup = default_setup()
    setup.deviceInputLog = str(Path(csv_path))
    setup.timerAnalysis = {
        "device_kernel_duration": setup.timerAnalysis["device_kernel_duration"]
    }
    processed = import_log_run_stats(setup)
    frequency_mhz = processed["deviceInfo"]["freq"]
    if frequency_mhz <= 0:
        raise ValueError(f"invalid profiler clock: {frequency_mhz}")
    per_device = {}
    for device_id in device_ids:
        operations = processed["devices"][device_id]["cores"]["DEVICE"]["riscs"][
            "TENSIX"
        ]["ops"]
        if not operations:
            raise ValueError(f"no profiled operations for device {device_id}")
        if len(operations) < program_count:
            raise ValueError(f"missing programs on device {device_id}")
        intervals = []
        for operation in operations[-program_count:]:
            series = operation["analysis"]["device_kernel_duration"]["series"]
            if len(series) != 1:
                raise ValueError(f"expected one kernel interval on device {device_id}")
            interval = series[0]
            start_zone, end_zone = interval["duration_type"]
            if start_zone["run_host_id"] != end_zone["run_host_id"]:
                raise ValueError("kernel interval spans different program launches")
            intervals.append(interval)
        cycles = int(intervals[-1]["duration_cycles"])
        if program_count > 1:
            for previous, current in zip(intervals, intervals[1:]):
                if (
                    current["duration_type"][0]["run_host_id"]
                    <= previous["duration_type"][1]["run_host_id"]
                    or current["start_cycle"] < previous["end_cycle"]
                ):
                    raise ValueError("program sequence is not ordered")
            cycles = int(intervals[-1]["end_cycle"] - intervals[0]["start_cycle"])
        per_device[str(device_id)] = {
            "run_host_id": int(start_zone["run_host_id"]),
            "cycles": cycles,
        }
    device_cycles = [device["cycles"] for device in per_device.values()]
    cycles = (
        statistics.mean(device_cycles) if aggregation == "mean" else max(device_cycles)
    )
    return {
        "profiler_log": str(Path(csv_path)),
        "cycles": cycles,
        "us": cycles / frequency_mhz,
        "frequency_mhz": frequency_mhz,
        "device_aggregation": aggregation,
        "program_count": program_count,
        "timing_scope": "first_kernel_start_to_last_kernel_end",
        "mean_device_us": statistics.mean(device_cycles) / frequency_mhz,
        "max_device_us": max(device_cycles) / frequency_mhz,
        "per_device": per_device,
    }
