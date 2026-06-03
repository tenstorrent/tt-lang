# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device-profiler (Tracy) cycle accounting for single-core benchmarks.

A cycles benchmark runs one kernel once with ``TT_METAL_DEVICE_PROFILER=1`` and
reads per-zone start/end timestamps from ``profile_log_device.csv``. Tracy is
the hardware profiler, so the counts are exact: one run, no averaging.

The headline metric is the device kernel duration -- per core, the span from
the earliest kernel-start to the latest kernel-end across its RISCs, maxed over
cores (tt-metal's "DEVICE KERNEL DURATION"). ``per_risc`` carries each RISC's
kernel span so a phase reads as reader- (NCRISC), compute- (TRISC*), or
writer-bound (BRISC). Cycles are the primary, frequency-free metric; the
microsecond view divides by the chip clock and is only for human comparison.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

# card1 (Blackhole) AI clock; override with CYCLES_CHIP_FREQ_MHZ. Affects only
# the derived microsecond view -- the cycle counts and the A/B ratio do not.
CHIP_FREQ_MHZ = float(os.environ.get("CYCLES_CHIP_FREQ_MHZ", "1350"))

_KERNEL_SUFFIX = "-KERNEL"


def profile_log_path() -> Path:
    home = os.environ.get("TT_METAL_HOME", "")
    if not home:
        raise ValueError("TT_METAL_HOME not set; cannot locate profile_log_device.csv")
    return Path(home) / "generated/profiler/.logs/profile_log_device.csv"


def clear_profile_log() -> None:
    """Drop any stale device log so a run's CSV is single-variant."""
    path = profile_log_path()
    if path.exists():
        path.unlink()


def read_device_profiler(device) -> None:
    """Flush device-side profiler buffers into profile_log_device.csv."""
    import ttnn

    ttnn.ReadDeviceProfiler(device)


def parse_kernel_duration(csv_path=None) -> dict:
    """Device kernel duration from a Tracy device-profiler CSV.

    Returns ``{"cycles", "us", "per_risc"}`` where ``cycles`` is the maximum
    over cores of that core's kernel span (latest ``*-KERNEL`` end minus
    earliest ``*-KERNEL`` start) and ``per_risc`` maps each RISC to its largest
    kernel span across cores.
    """
    path = Path(csv_path) if csv_path else profile_log_path()
    if not path.exists():
        raise FileNotFoundError(
            f"profile log not found: {path} (run with TT_METAL_DEVICE_PROFILER=1)"
        )

    core_start: dict = {}
    core_end: dict = {}
    per_risc: dict = defaultdict(int)
    active = defaultdict(list)

    with open(path) as f:
        next(f, None)  # device descriptor line
        next(f, None)  # column header
        for row in csv.reader(f):
            if len(row) < 12:
                continue
            core = (row[1].strip(), row[2].strip())
            risc = row[3].strip()
            stamp = row[5].strip()
            zone = row[10].strip()
            ztype = row[11].strip()
            if not stamp.lstrip("-").isdigit():
                continue
            stamp = int(stamp)
            key = (core, risc)
            if ztype == "ZONE_START":
                active[key].append((zone, stamp))
            elif ztype == "ZONE_END" and active[key]:
                name, start = active[key].pop()
                if name != zone or not zone.endswith(_KERNEL_SUFFIX):
                    continue
                core_start[core] = min(core_start.get(core, start), start)
                core_end[core] = max(core_end.get(core, stamp), stamp)
                per_risc[risc] = max(per_risc[risc], stamp - start)

    if not core_start:
        raise ValueError(f"no *-KERNEL zones in {path}")

    cycles = max(core_end[c] - core_start[c] for c in core_start)
    return {"cycles": cycles, "us": cycles / CHIP_FREQ_MHZ, "per_risc": dict(per_risc)}
