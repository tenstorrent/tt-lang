# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-RISC DeviceZoneScopedN cycle readback for the calibration microbenchmarks.

Reads tt-metal's ``profile_log_device.csv``, pairs ``ZONE_START``/``ZONE_END``
for a named zone per RISC thread, and converts cycles to microseconds using
``CHIP_FREQ[MHz]`` from the CSV header. The conversion ``µs = cycles / freq_mhz``
is the repo's canonical one (see ``ttl._src.perf_summary``); reporting absolute
microseconds is required because the cost-model weights are absolute (the LLK
profiler counts are not µs-convertible, only their ratios are).

A ``DeviceZoneScopedN`` placed in a compute kernel records on each compute RISC,
so one zone yields the unpack (``TRISC_0``), math (``TRISC_1``), and pack
(``TRISC_2``) split directly. ``BRISC``/``NCRISC`` activity inside the zone means
the data-movement kernels were not idle -- the isolation gate.

Enable profiling with ``TT_METAL_DEVICE_PROFILER=1`` and
``TT_METAL_PROFILER_MID_RUN_DUMP=1``; flush with ``ttnn.ReadDeviceProfiler(device)``
before reading.
"""

import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

# profile_log_device.csv column indices (0-based), matching
# ttl._src.perf_summary.parse_kernel_durations.
_COL_CORE_X = 1
_COL_CORE_Y = 2
_COL_THREAD = 3
_COL_TIMESTAMP = 5
_COL_RUN_ID = 7
_COL_ZONE = 10
_COL_ZONE_TYPE = 11

UNPACK_THREAD = "TRISC_0"
MATH_THREAD = "TRISC_1"
PACK_THREAD = "TRISC_2"
COMPUTE_THREADS = (UNPACK_THREAD, MATH_THREAD, PACK_THREAD)
DATAMOVEMENT_THREADS = ("BRISC", "NCRISC")


def find_profiler_csv():
    """Locate ``profile_log_device.csv`` (``TTLANG_PROFILE_CSV`` or ``TT_METAL_HOME``)."""
    if "TTLANG_PROFILE_CSV" in os.environ:
        return Path(os.environ["TTLANG_PROFILE_CSV"])
    tt_metal_home = os.environ.get("TT_METAL_HOME")
    if not tt_metal_home:
        raise RuntimeError(
            "Set TT_METAL_HOME or TTLANG_PROFILE_CSV to locate the profiler CSV"
        )
    return (
        Path(tt_metal_home)
        / "generated"
        / "profiler"
        / ".logs"
        / "profile_log_device.csv"
    )


def parse_chip_info(csv_path):
    """Return ``(arch, freq_mhz)`` from the CSV header line."""
    with open(csv_path) as file:
        header = file.readline().strip()
    arch = "unknown"
    freq_mhz = 1000
    match = re.search(r"ARCH:\s*(\w+)", header)
    if match:
        arch = match.group(1)
    match = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header)
    if match:
        freq_mhz = int(match.group(1))
    return arch, freq_mhz


def zone_durations_cycles(csv_path, zone_name):
    """Pair ``ZONE_START``/``ZONE_END`` for ``zone_name`` per RISC thread.

    Returns ``{thread: [duration_cycles, ...]}`` aggregated across cores and
    runs (one entry per kernel invocation per core).
    """
    starts = {}
    durations = defaultdict(list)
    with open(csv_path) as file:
        file.readline()  # arch / frequency header
        file.readline()  # column-name header
        for line in file:
            parts = line.split(",")
            if len(parts) <= _COL_ZONE_TYPE:
                continue
            if parts[_COL_ZONE].strip() != zone_name:
                continue
            thread = parts[_COL_THREAD].strip()
            zone_type = parts[_COL_ZONE_TYPE].strip()
            try:
                timestamp = int(parts[_COL_TIMESTAMP].strip())
                run_id = int(parts[_COL_RUN_ID].strip())
                core_x = int(parts[_COL_CORE_X].strip())
                core_y = int(parts[_COL_CORE_Y].strip())
            except (ValueError, IndexError):
                continue
            key = (run_id, thread, core_x, core_y)
            if zone_type == "ZONE_START":
                starts[key] = timestamp
            elif zone_type == "ZONE_END" and key in starts:
                durations[thread].append(timestamp - starts.pop(key))
    return dict(durations)


def zone_durations_us(csv_path, zone_name, freq_mhz=None):
    """Return ``{thread: [duration_us, ...]}`` for ``zone_name``."""
    if freq_mhz is None:
        _, freq_mhz = parse_chip_info(csv_path)
    cycles = zone_durations_cycles(csv_path, zone_name)
    return {
        thread: [count / freq_mhz for count in counts]
        for thread, counts in cycles.items()
    }


def summarize_zone(csv_path, zone_name, reduce=statistics.median):
    """Summarize one named zone: per-RISC µs, TRISC-max, and the isolation gate.

    ``reduce`` collapses the per-invocation samples (median by default -- on-device
    zone cycles are stable run-to-run, so warmup invocations do not skew it).
    ``noc_active_in_zone`` is True if BRISC/NCRISC recorded the zone, which fails
    the single-core isolation requirement.
    """
    arch, freq_mhz = parse_chip_info(csv_path)
    per_invocation = zone_durations_us(csv_path, zone_name, freq_mhz)
    per_thread = {
        thread: reduce(samples) for thread, samples in per_invocation.items() if samples
    }
    compute = [per_thread[t] for t in COMPUTE_THREADS if t in per_thread]
    return {
        "arch": arch,
        "freq_mhz": freq_mhz,
        "unpack_us": per_thread.get(UNPACK_THREAD),
        "math_us": per_thread.get(MATH_THREAD),
        "pack_us": per_thread.get(PACK_THREAD),
        "trisc_max_us": max(compute) if compute else None,
        "noc_active_in_zone": any(
            thread in per_thread for thread in DATAMOVEMENT_THREADS
        ),
        "per_thread_us": per_thread,
    }
