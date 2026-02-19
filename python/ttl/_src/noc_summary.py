# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
NOC event profiler summary tool.

Parses NOC trace JSON and device profiler CSV produced by tt-metal
when TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 is set.

Usage:
    python -m ttl._src.noc_summary                    # default: $TT_METAL_HOME/generated/profiler/.logs/
    python -m ttl._src.noc_summary --path /tmp/        # override path
    python -m ttl._src.noc_summary --path /tmp/ --json  # machine-readable output
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ProgramSummary:
    program_id: int = 0
    # Core grid
    source_cores: Set[Tuple[int, int]] = field(default_factory=set)
    # Byte counts
    dram_bytes_read: int = 0
    dram_bytes_written: int = 0
    l1_bytes_read: int = 0
    l1_bytes_written: int = 0
    # Event counts
    dram_read_count: int = 0
    dram_write_count: int = 0
    l1_read_count: int = 0
    l1_write_count: int = 0
    read_barrier_count: int = 0
    write_barrier_count: int = 0
    # Transfer sizes seen
    transfer_sizes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    # NOC usage
    noc_read_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    noc_write_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Zone timing (per-core kernel durations)
    brisc_durations: List[int] = field(default_factory=list)
    ncrisc_durations: List[int] = field(default_factory=list)
    # Destination cores (for L1 traffic)
    l1_destinations: Set[Tuple[int, int]] = field(default_factory=set)
    dram_destinations: Set[Tuple[int, int]] = field(default_factory=set)
    # Multicast events (L1 pipe multicast)
    multicast_count: int = 0
    multicast_bytes: int = 0
    # Semaphore events
    semaphore_count: int = 0
    # Raw timestamps for duration
    min_timestamp: Optional[int] = None
    max_timestamp: Optional[int] = None


def _infer_grid_shape(cores: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Infer grid shape from set of core coordinates."""
    if not cores:
        return (0, 0)
    xs = sorted(set(x for x, _ in cores))
    ys = sorted(set(y for _, y in cores))
    return (len(xs), len(ys))


def _format_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _format_cycles(cycles: int, freq_mhz: int) -> str:
    us = cycles / freq_mhz
    if us >= 1000:
        return f"{cycles:,} cycles ({us / 1000:.2f} ms)"
    return f"{cycles:,} cycles ({us:.1f} us)"


def parse_chip_info(logs_path: Path) -> Tuple[str, int, int]:
    """Parse arch, frequency, and max cores from profile CSV header."""
    csv_path = logs_path / "profile_log_device.csv"
    if not csv_path.exists():
        return "unknown", 1000, 0

    with open(csv_path) as f:
        header = f.readline().strip()

    arch = "unknown"
    freq = 1000
    max_cores = 0

    m = re.search(r"ARCH:\s*(\w+)", header)
    if m:
        arch = m.group(1)
    m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header)
    if m:
        freq = int(m.group(1))
    m = re.search(r"Max Compute Cores:\s*(\d+)", header)
    if m:
        max_cores = int(m.group(1))

    return arch, freq, max_cores


def _collect_compute_cores_from_zones(events: list) -> Set[Tuple[int, int]]:
    """Collect all cores that have zone events (these are compute cores)."""
    cores = set()
    for ev in events:
        if "zone" in ev:
            cores.add((ev["sx"], ev["sy"]))
    return cores


def parse_noc_trace(filepath: Path) -> ProgramSummary:
    """Parse a single NOC trace JSON file into a ProgramSummary."""
    with open(filepath) as f:
        events = json.load(f)

    summary = ProgramSummary()

    if not events:
        return summary

    summary.program_id = events[0].get("run_host_id", 0)

    # First pass: collect compute cores from zone events
    compute_cores = _collect_compute_cores_from_zones(events)

    # Track zone starts for duration calculation
    zone_starts: Dict[Tuple[str, int, int], int] = {}

    for ev in events:
        sx, sy = ev.get("sx", -1), ev.get("sy", -1)
        ts = ev.get("timestamp", 0)

        # Track timestamp range
        if summary.min_timestamp is None or ts < summary.min_timestamp:
            summary.min_timestamp = ts
        if summary.max_timestamp is None or ts > summary.max_timestamp:
            summary.max_timestamp = ts

        # Zone events: track kernel durations
        if "zone" in ev:
            zone = ev["zone"]
            phase = ev.get("zone_phase", "")
            proc = ev.get("proc", "")

            if zone in ("BRISC-KERNEL", "NCRISC-KERNEL"):
                key = (zone, sx, sy)
                if phase == "ZONE_START":
                    zone_starts[key] = ts
                elif phase == "ZONE_END" and key in zone_starts:
                    duration = ts - zone_starts[key]
                    if "BRISC" in zone:
                        summary.brisc_durations.append(duration)
                    else:
                        summary.ncrisc_durations.append(duration)
            continue

        # NOC events
        event_type = ev.get("type", "")
        num_bytes = ev.get("num_bytes", 0)
        noc = ev.get("noc", "")

        summary.source_cores.add((sx, sy))

        # Semaphore events (pipe synchronization)
        if "SEMAPHORE" in event_type:
            summary.semaphore_count += 1
            continue

        # Barriers
        if "BARRIER" in event_type:
            if "READ" in event_type and "START" in event_type:
                summary.read_barrier_count += 1
            elif "WRITE" in event_type and "START" in event_type:
                summary.write_barrier_count += 1
            continue

        # Multicast: classify by whether targets are compute cores (L1) or DRAM
        is_multicast = "mcast_start_x" in ev
        if is_multicast:
            mcast_dst = (ev.get("mcast_start_x", -1), ev.get("mcast_start_y", -1))
            is_l1_mcast = mcast_dst in compute_cores
            summary.transfer_sizes[num_bytes] += 1
            summary.noc_write_counts[noc] += 1
            if is_l1_mcast:
                summary.multicast_count += 1
                summary.multicast_bytes += num_bytes
            else:
                summary.dram_bytes_written += num_bytes
                summary.dram_write_count += 1
            continue

        # Unicast read/write
        dx, dy = ev.get("dx", -1), ev.get("dy", -1)
        is_l1 = (dx, dy) in compute_cores and dx >= 0

        if "READ" in event_type:
            summary.transfer_sizes[num_bytes] += 1
            summary.noc_read_counts[noc] += 1
            if is_l1:
                summary.l1_bytes_read += num_bytes
                summary.l1_read_count += 1
                summary.l1_destinations.add((dx, dy))
            else:
                summary.dram_bytes_read += num_bytes
                summary.dram_read_count += 1
                if dx >= 0:
                    summary.dram_destinations.add((dx, dy))

        elif "WRITE" in event_type:
            summary.transfer_sizes[num_bytes] += 1
            summary.noc_write_counts[noc] += 1
            if is_l1:
                summary.l1_bytes_written += num_bytes
                summary.l1_write_count += 1
                summary.l1_destinations.add((dx, dy))
            else:
                summary.dram_bytes_written += num_bytes
                summary.dram_write_count += 1
                if dx >= 0:
                    summary.dram_destinations.add((dx, dy))

    return summary


def format_summary(
    summary: ProgramSummary, freq_mhz: int, name: Optional[str] = None
) -> str:
    """Format a ProgramSummary as human-readable text."""
    lines = []
    grid = _infer_grid_shape(summary.source_cores)
    n_cores = len(summary.source_cores)

    duration_cycles = 0
    if summary.min_timestamp is not None and summary.max_timestamp is not None:
        duration_cycles = summary.max_timestamp - summary.min_timestamp

    label = f" ({name})" if name else ""
    lines.append(f"--- Program {summary.program_id}{label} ---")
    lines.append(f"grid: {grid[0]}x{grid[1]} ({n_cores} cores)")
    lines.append(f"duration: {_format_cycles(duration_cycles, freq_mhz)}")

    # DRAM traffic
    total_dram = summary.dram_bytes_read + summary.dram_bytes_written
    lines.append(
        f"  DRAM read:      {_format_bytes(summary.dram_bytes_read):>10}  "
        f"({summary.dram_read_count} transfers)"
    )
    lines.append(
        f"  DRAM write:     {_format_bytes(summary.dram_bytes_written):>10}  "
        f"({summary.dram_write_count} transfers)"
    )

    # L1 traffic
    total_l1 = summary.l1_bytes_read + summary.l1_bytes_written
    if total_l1 > 0:
        lines.append(
            f"  L1 read:        {_format_bytes(summary.l1_bytes_read):>10}  "
            f"({summary.l1_read_count} transfers)"
        )
        lines.append(
            f"  L1 write:       {_format_bytes(summary.l1_bytes_written):>10}  "
            f"({summary.l1_write_count} transfers)"
        )
        if summary.l1_destinations:
            l1_grid = _infer_grid_shape(summary.l1_destinations)
            lines.append(
                f"  L1 targets:     {len(summary.l1_destinations)} cores "
                f"({l1_grid[0]}x{l1_grid[1]})"
            )

    # Multicast (L1 pipe traffic between compute cores)
    if summary.multicast_count > 0:
        lines.append(
            f"  L1 multicast:   {_format_bytes(summary.multicast_bytes):>10}  "
            f"({summary.multicast_count} transfers, pipe)"
        )

    # Bandwidth
    if duration_cycles > 0:
        duration_s = duration_cycles / (freq_mhz * 1e6)
        total_bytes = total_dram + total_l1 + summary.multicast_bytes
        bw_gbs = total_bytes / duration_s / 1e9
        lines.append(f"  effective BW:   {bw_gbs:.1f} GB/s (total payload / duration)")

    # Transfer sizes
    nonzero_sizes = {k: v for k, v in summary.transfer_sizes.items() if k > 0}
    if nonzero_sizes:
        if len(nonzero_sizes) == 1:
            size, count = next(iter(nonzero_sizes.items()))
            lines.append(f"  transfer size:  {_format_bytes(size)} (uniform)")
        else:
            parts = [
                f"{_format_bytes(s)}x{c}" for s, c in sorted(nonzero_sizes.items())
            ]
            lines.append(f"  transfer sizes: {', '.join(parts)}")

    # Barriers
    total_reads = summary.dram_read_count + summary.l1_read_count
    total_writes = (
        summary.dram_write_count + summary.l1_write_count + summary.multicast_count
    )
    if summary.read_barrier_count > 0 or summary.write_barrier_count > 0:
        read_ratio = ""
        if summary.read_barrier_count > 0 and total_reads > 0:
            ratio = total_reads / summary.read_barrier_count
            read_ratio = f" (1 per {ratio:.0f} reads)"
        write_ratio = ""
        if summary.write_barrier_count > 0 and total_writes > 0:
            ratio = total_writes / summary.write_barrier_count
            write_ratio = f" (1 per {ratio:.0f} writes)"
        lines.append(
            f"  barriers:       {summary.read_barrier_count} read{read_ratio}, "
            f"{summary.write_barrier_count} write{write_ratio}"
        )

    # Semaphores (pipe synchronization)
    if summary.semaphore_count > 0:
        lines.append(f"  semaphores:     {summary.semaphore_count} events")

    # NOC split
    read_nocs = ", ".join(
        f"{noc}={c}" for noc, c in sorted(summary.noc_read_counts.items())
    )
    write_nocs = ", ".join(
        f"{noc}={c}" for noc, c in sorted(summary.noc_write_counts.items())
    )
    if read_nocs or write_nocs:
        lines.append(f"  noc reads:      {read_nocs}")
        lines.append(f"  noc writes:     {write_nocs}")

    # DRAM channels hit
    if summary.dram_destinations:
        lines.append(f"  DRAM channels:  {len(summary.dram_destinations)}")

    # Kernel durations (from zone events)
    if summary.brisc_durations or summary.ncrisc_durations:
        lines.append(f"  kernel time:")
        if summary.brisc_durations:
            mn, mx = min(summary.brisc_durations), max(summary.brisc_durations)
            lines.append(
                f"    BRISC:  {_format_cycles(mn, freq_mhz)} - "
                f"{_format_cycles(mx, freq_mhz)}"
            )
        if summary.ncrisc_durations:
            mn, mx = min(summary.ncrisc_durations), max(summary.ncrisc_durations)
            lines.append(
                f"    NCRISC: {_format_cycles(mn, freq_mhz)} - "
                f"{_format_cycles(mx, freq_mhz)}"
            )

    return "\n".join(lines)


def format_json_summary(summary: ProgramSummary, freq_mhz: int) -> dict:
    """Format a ProgramSummary as a JSON-serializable dict."""
    grid = _infer_grid_shape(summary.source_cores)
    duration_cycles = 0
    if summary.min_timestamp is not None and summary.max_timestamp is not None:
        duration_cycles = summary.max_timestamp - summary.min_timestamp

    return {
        "program_id": summary.program_id,
        "grid": list(grid),
        "num_cores": len(summary.source_cores),
        "duration_cycles": duration_cycles,
        "duration_us": round(duration_cycles / freq_mhz, 2),
        "dram_bytes_read": summary.dram_bytes_read,
        "dram_bytes_written": summary.dram_bytes_written,
        "l1_bytes_read": summary.l1_bytes_read,
        "l1_bytes_written": summary.l1_bytes_written,
        "multicast_bytes": summary.multicast_bytes,
        "dram_read_count": summary.dram_read_count,
        "dram_write_count": summary.dram_write_count,
        "l1_read_count": summary.l1_read_count,
        "l1_write_count": summary.l1_write_count,
        "read_barriers": summary.read_barrier_count,
        "write_barriers": summary.write_barrier_count,
        "dram_channels": len(summary.dram_destinations),
        "transfer_sizes": {
            str(k): v for k, v in sorted(summary.transfer_sizes.items()) if k > 0
        },
    }


def run(
    logs_path: Path, output_json: bool = False, names: Optional[List[str]] = None
) -> Optional[str]:
    """Main entry point. Returns formatted string or None if no data."""
    arch, freq_mhz, max_cores = parse_chip_info(logs_path)

    trace_files = sorted(glob.glob(str(logs_path / "noc_trace_*.json")))
    if not trace_files:
        return None

    summaries = []
    for tf in trace_files:
        summaries.append(parse_noc_trace(Path(tf)))

    if output_json:
        json_programs = []
        for i, s in enumerate(summaries):
            d = format_json_summary(s, freq_mhz)
            if names and i < len(names):
                d["name"] = names[i]
            json_programs.append(d)
        result = {
            "arch": arch,
            "freq_mhz": freq_mhz,
            "max_compute_cores": max_cores,
            "programs": json_programs,
        }
        return json.dumps(result, indent=2)

    lines = []
    lines.append("=== NOC PROFILER SUMMARY ===")
    lines.append(f"arch: {arch}, freq: {freq_mhz} MHz, max_compute_cores: {max_cores}")
    lines.append(
        f"programs: {len(summaries)} (listed in dispatch order, includes ttnn ops)"
    )
    lines.append("")

    for i, summary in enumerate(summaries):
        name = names[i] if names and i < len(names) else None
        lines.append(format_summary(summary, freq_mhz, name=name))
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize NOC event profiler traces")
    parser.add_argument(
        "--path",
        default=None,
        help="Path to profiler logs directory "
        "(default: $TT_METAL_HOME/generated/profiler/.logs/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of text",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="Comma-separated kernel names in dispatch order "
        "(e.g. 'my_kernel,ttnn.multiply')",
    )
    args = parser.parse_args()

    if args.path:
        logs_path = Path(args.path)
    else:
        tt_metal_home = os.environ.get("TT_METAL_HOME", "")
        if not tt_metal_home:
            print(
                "Error: TT_METAL_HOME not set and --path not provided", file=sys.stderr
            )
            sys.exit(1)
        logs_path = Path(tt_metal_home) / "generated" / "profiler" / ".logs"

    if not logs_path.exists():
        print(f"Error: {logs_path} does not exist", file=sys.stderr)
        sys.exit(1)

    names = [n.strip() for n in args.names.split(",")] if args.names else None
    result = run(logs_path, output_json=args.json, names=names)
    if result is None:
        print(f"No noc_trace_*.json files found in {logs_path}", file=sys.stderr)
        sys.exit(1)

    print(result)


if __name__ == "__main__":
    main()
