# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""tt-lang side of the microbenchmarks: run a compiled ``@ttl.operation``
through real tt-metal dispatch and return warm, per-core device-profiler kernel
durations.

Unlike ``harness.py`` (handwritten kernels via ``ttnn.generic_op`` on one
compute core), this drives the tt-lang compiler: the operation compiles on the
first call and re-dispatches warm. The device profiler auto-records whole-kernel
zones (``BRISC-KERNEL``/``NCRISC-KERNEL``/``TRISC-KERNEL``) per RISC per core;
tt-lang user code cannot insert named zones, so whole-kernel duration is the
unit of measure.

Measurement is single-pass and syncs once per phase: warmup dispatches (zones
discarded), then ``runs`` measured dispatches with one ``synchronize_device``
and one ``ReadDeviceProfiler`` afterward, so the device sync stays outside the
dispatch loop. The optional wall time brackets the same measured dispatches plus
that single sync, matching ``benchmarks.common.time_runs``.
"""

import statistics
import time
from collections import defaultdict

import ttnn

from benchmarks.microbench import profiler

# profile_log_device.csv column indices, shared with profiler.py.
_COL_CORE_X = profiler._COL_CORE_X
_COL_CORE_Y = profiler._COL_CORE_Y
_COL_THREAD = profiler._COL_THREAD
_COL_TIMESTAMP = profiler._COL_TIMESTAMP
_COL_RUN_ID = profiler._COL_RUN_ID
_COL_ZONE = profiler._COL_ZONE
_COL_ZONE_TYPE = profiler._COL_ZONE_TYPE

DM_THREADS = ("BRISC", "NCRISC")


def _kernel_durations_by_core(csv_path):
    """Pair ZONE_START/ZONE_END of the auto-recorded ``*-KERNEL`` zones.

    Returns ``{(core_x, core_y): {thread: [duration_cycles, ...]}}`` aggregated
    across runs. Unlike ``perf_summary.parse_kernel_durations`` this keeps the
    core coordinate, so a two-core pipe can attribute sender versus receiver.
    """
    starts = {}
    durations = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as file:
        file.readline()  # arch / frequency header
        file.readline()  # column-name header
        for line in file:
            parts = line.split(",")
            if len(parts) <= _COL_ZONE_TYPE:
                continue
            if not parts[_COL_ZONE].strip().endswith("-KERNEL"):
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
                durations[(core_x, core_y)][thread].append(timestamp - starts.pop(key))
    return durations


def _median_us(durations, freq_mhz):
    """Collapse ``{core: {thread: [cycles]}}`` to ``{core: {thread: median_us}}``."""
    return {
        core: {
            thread: statistics.median(samples) / freq_mhz
            for thread, samples in threads.items()
            if samples
        }
        for core, threads in durations.items()
    }


def run_operation(device, operation, io_tensors, *, warmup, runs, wall=False):
    """Dispatch ``operation(*io_tensors)`` warm and summarize device-profiler zones.

    Returns ``(per_core_us, arch, freq_mhz, wall_s)`` where ``per_core_us`` maps
    ``(core_x, core_y) -> {thread: median_us}`` over the ``*-KERNEL`` zones and
    ``wall_s`` is the wall time of the ``runs`` measured dispatches plus the
    single trailing sync (or ``None`` when ``wall`` is False).
    """
    csv_path = profiler.find_profiler_csv()
    for _ in range(warmup):
        operation(*io_tensors)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)  # flush warmup zones...
    if csv_path.exists():
        csv_path.unlink()  # ...and discard them

    start = time.perf_counter() if wall else None
    for _ in range(runs):
        operation(*io_tensors)
    ttnn.synchronize_device(device)
    wall_s = (time.perf_counter() - start) if wall else None
    ttnn.ReadDeviceProfiler(device)

    arch, freq_mhz = profiler.parse_chip_info(csv_path)
    per_core = _median_us(_kernel_durations_by_core(csv_path), freq_mhz)
    if csv_path.exists():
        csv_path.unlink()
    return per_core, arch, freq_mhz, wall_s


def dm_max_us(per_core_us, core):
    """Max of BRISC/NCRISC kernel µs on ``core`` -- its data-movement cost.

    The tt-lang compiler may place the pipe send on either data-movement RISC,
    so report the slower of the two rather than assume an assignment.
    """
    threads = per_core_us.get(core, {})
    dm = [threads[name] for name in DM_THREADS if name in threads]
    return max(dm) if dm else None


def physical_core(device, logical):
    """Physical (NoC) coords of a logical worker core, matching profiler CSV keys.

    The device profiler logs physical core coordinates, so a logical ``(x, y)``
    must be translated before indexing the per-core summary.
    """
    coord = device.worker_core_from_logical_core(ttnn.CoreCoord(*logical))
    return (coord.x, coord.y)
