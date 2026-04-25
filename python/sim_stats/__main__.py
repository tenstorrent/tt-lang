# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Post-processing tool for simulator trace files.

Reads a JSON Lines trace file produced by ttlang-sim --trace and derives the
same statistics that --show-stats reports during a live simulation run:

  Tensor Access Statistics   -- reads/writes and tile counts per tensor name
  Pipe Transfer Statistics   -- sends/receives and tile counts per pipe
  Dataflow Buffer Statistics -- reserves/waits and tile counts per DFB name,
                                broken down by core with a per-DFB subtotal

Usage:
    ttlang-sim-stats trace.jsonl
    ttlang-sim-stats --help
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator


def _core_from_kernel(kernel: str | None) -> str:
    """Extract the core identifier from a kernel name like 'core3-read'."""
    if kernel and "-" in kernel:
        return kernel.split("-", 1)[0]
    return kernel or "unknown"


def _core_sort_key(core: str) -> int:
    """Sort cores numerically: core0 < core1 < ... < core10."""
    try:
        return int(core.removeprefix("core"))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSON Lines file, skipping blank lines."""
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed line {lineno}: {exc}",
                    file=sys.stderr,
                )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_TensorStats = Dict[str, Dict[str, int]]
_PipeStats = Dict[str, Dict[str, int]]
_DfbStats = Dict[str, Dict[str, int]]
# dfb_name -> core_id -> {reserves, waits, tiles_reserved, tiles_waited}
_DfbPerCoreStats = Dict[str, Dict[str, Dict[str, int]]]


def _new_rw() -> Dict[str, int]:
    return {
        "reads": 0,
        "writes": 0,
        "tiles_read": 0,
        "tiles_written": 0,
        "local_l1_read": 0,
        "local_l1_write": 0,
        "remote_l1_read": 0,
        "remote_l1_write": 0,
        "dram_read": 0,
        "dram_write": 0,
    }


def _new_dfb() -> Dict[str, int]:
    return {"reserves": 0, "waits": 0, "tiles_reserved": 0, "tiles_waited": 0}


def _accumulate(
    path: Path,
) -> tuple[_TensorStats, _PipeStats, _DfbStats, _DfbPerCoreStats]:
    """Read events and return (tensor_stats, pipe_stats, dfb_stats, dfb_per_core)."""
    tensor_stats: _TensorStats = defaultdict(_new_rw)
    pipe_stats: _PipeStats = defaultdict(_new_rw)
    dfb_stats: _DfbStats = defaultdict(_new_dfb)
    dfb_per_core: _DfbPerCoreStats = defaultdict(lambda: defaultdict(_new_dfb))

    for ev in _iter_events(path):
        event = ev.get("event")

        match event:
            case "copy_end":
                tensor = ev.get("tensor")
                tiles = ev.get("tiles", 0)
                direction = ev.get("direction")
                local_l1 = ev.get("local_l1", 0)
                remote_l1 = ev.get("remote_l1", 0)
                dram = ev.get("dram", 0)
                if tensor and direction == "read":
                    tensor_stats[tensor]["reads"] += 1
                    tensor_stats[tensor]["tiles_read"] += tiles
                    tensor_stats[tensor]["local_l1_read"] += local_l1
                    tensor_stats[tensor]["remote_l1_read"] += remote_l1
                    tensor_stats[tensor]["dram_read"] += dram
                elif tensor and direction == "write":
                    tensor_stats[tensor]["writes"] += 1
                    tensor_stats[tensor]["tiles_written"] += tiles
                    tensor_stats[tensor]["local_l1_write"] += local_l1
                    tensor_stats[tensor]["remote_l1_write"] += remote_l1
                    tensor_stats[tensor]["dram_write"] += dram

            case "pipe_send":
                pipe = ev.get("pipe")
                tiles = ev.get("tiles", 0)
                if pipe:
                    pipe_stats[pipe]["writes"] += 1
                    pipe_stats[pipe]["tiles_written"] += tiles

            case "pipe_recv":
                pipe = ev.get("pipe")
                tiles = ev.get("tiles", 0)
                if pipe:
                    pipe_stats[pipe]["reads"] += 1
                    pipe_stats[pipe]["tiles_read"] += tiles

            case "dfb_reserve_end":
                dfb = ev.get("dfb")
                tiles = ev.get("tiles", 0)
                core = _core_from_kernel(ev.get("kernel"))
                if dfb:
                    dfb_stats[dfb]["reserves"] += 1
                    dfb_stats[dfb]["tiles_reserved"] += tiles
                    dfb_per_core[dfb][core]["reserves"] += 1
                    dfb_per_core[dfb][core]["tiles_reserved"] += tiles

            case "dfb_wait_end":
                dfb = ev.get("dfb")
                tiles = ev.get("tiles", 0)
                core = _core_from_kernel(ev.get("kernel"))
                if dfb:
                    dfb_stats[dfb]["waits"] += 1
                    dfb_stats[dfb]["tiles_waited"] += tiles
                    dfb_per_core[dfb][core]["waits"] += 1
                    dfb_per_core[dfb][core]["tiles_waited"] += tiles

            case _:
                pass

    return (
        dict(tensor_stats),
        dict(pipe_stats),
        dict(dfb_stats),
        {k: dict(v) for k, v in dfb_per_core.items()},
    )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_W = 84  # total table width for DFB section


def _print_tensor_stats(stats: _TensorStats) -> None:
    has_locality = any(
        s["local_l1_read"]
        + s["local_l1_write"]
        + s["remote_l1_read"]
        + s["remote_l1_write"]
        + s["dram_read"]
        + s["dram_write"]
        > 0
        for s in stats.values()
    )

    if has_locality:
        # Columns: Tensor | Local L1 R | Local L1 W | Remote L1 R | Remote L1 W | DRAM R | DRAM W | Tiles R | Tiles W
        width = 132
        header = (
            f"{'Tensor':<20}"
            f" {'Local L1 R':>12} {'Local L1 W':>12}"
            f" {'Remote L1 R':>13} {'Remote L1 W':>13}"
            f" {'DRAM R':>9} {'DRAM W':>9}"
            f" {'Tiles R':>9} {'Tiles W':>9}"
        )
    else:
        width = 64
        header = f"{'Tensor':<20} {'Reads':>8} {'Writes':>8} {'Tiles Read':>12} {'Tiles Written':>12}"

    print("\n" + "=" * width)
    print("Tensor Access Statistics")
    print("=" * width)
    print(header)
    print("-" * width)

    totals: Dict[str, int] = {
        "reads": 0,
        "writes": 0,
        "tiles_read": 0,
        "tiles_written": 0,
        "local_l1_read": 0,
        "local_l1_write": 0,
        "remote_l1_read": 0,
        "remote_l1_write": 0,
        "dram_read": 0,
        "dram_write": 0,
    }

    for name, s in sorted(stats.items()):
        for key in totals:
            totals[key] += s[key]
        if has_locality:
            print(
                f"{name:<20}"
                f" {s['local_l1_read']:>12} {s['local_l1_write']:>12}"
                f" {s['remote_l1_read']:>13} {s['remote_l1_write']:>13}"
                f" {s['dram_read']:>9} {s['dram_write']:>9}"
                f" {s['tiles_read']:>9} {s['tiles_written']:>9}"
            )
        else:
            print(
                f"{name:<20} {s['reads']:>8} {s['writes']:>8}"
                f" {s['tiles_read']:>12} {s['tiles_written']:>12}"
            )

    print("-" * width)
    if has_locality:
        print(
            f"{'TOTAL':<20}"
            f" {totals['local_l1_read']:>12} {totals['local_l1_write']:>12}"
            f" {totals['remote_l1_read']:>13} {totals['remote_l1_write']:>13}"
            f" {totals['dram_read']:>9} {totals['dram_write']:>9}"
            f" {totals['tiles_read']:>9} {totals['tiles_written']:>9}"
        )
    else:
        print(
            f"{'TOTAL':<20} {totals['reads']:>8} {totals['writes']:>8}"
            f" {totals['tiles_read']:>12} {totals['tiles_written']:>12}"
        )
    print("=" * width)


def _print_pipe_stats(stats: _PipeStats) -> None:
    print("\n" + "=" * 74)
    print("Pipe Transfer Statistics")
    print("=" * 74)
    print(
        f"{'Pipe':<30} {'Reads':>8} {'Writes':>8} {'Tiles Read':>12} {'Tiles Written':>12}"
    )
    print("-" * 74)

    total_reads = total_writes = total_tiles_read = total_tiles_written = 0

    for name, s in sorted(stats.items()):
        reads, writes = s["reads"], s["writes"]
        tiles_read, tiles_written = s["tiles_read"], s["tiles_written"]
        total_reads += reads
        total_writes += writes
        total_tiles_read += tiles_read
        total_tiles_written += tiles_written
        print(f"{name:<30} {reads:>8} {writes:>8} {tiles_read:>12} {tiles_written:>12}")

    print("-" * 74)
    print(
        f"{'TOTAL':<30} {total_reads:>8} {total_writes:>8}"
        f" {total_tiles_read:>12} {total_tiles_written:>12}"
    )
    print("=" * 74)


def _dfb_row(dfb: str, core: str, s: Dict[str, int]) -> str:
    """Format one DFB stats row.  dfb is blank for continuation lines."""
    return (
        f"{dfb:<20} {core:<12}"
        f" {s['reserves']:>10} {s['waits']:>10}"
        f" {s['tiles_reserved']:>16} {s['tiles_waited']:>14}"
    )


def _print_dfb_stats(
    dfb_stats: _DfbStats,
    dfb_per_core: _DfbPerCoreStats,
) -> None:
    print("\n" + "=" * _W)
    print("Dataflow Buffer Statistics")
    print("=" * _W)
    print(
        f"{'DFB':<20} {'Core':<12}"
        f" {'Reserves':>10} {'Waits':>10}"
        f" {'Tiles Reserved':>16} {'Tiles Waited':>14}"
    )
    print("-" * _W)

    grand = _new_dfb()

    for dfb_name in sorted(dfb_stats):
        cores = dfb_per_core.get(dfb_name, {})
        first = True
        for core in sorted(cores, key=_core_sort_key):
            s = cores[core]
            label = dfb_name if first else ""
            print(_dfb_row(label, core, s))
            first = False

        # Per-DFB subtotal (only printed when there are multiple cores)
        subtotal = dfb_stats[dfb_name]
        if len(cores) > 1:
            print(_dfb_row("", "TOTAL", subtotal))

        # Accumulate into grand total
        grand["reserves"] += subtotal["reserves"]
        grand["waits"] += subtotal["waits"]
        grand["tiles_reserved"] += subtotal["tiles_reserved"]
        grand["tiles_waited"] += subtotal["tiles_waited"]

    print("-" * _W)
    print(_dfb_row("TOTAL", "", grand))
    print("=" * _W)


def print_stats_from_trace(path: Path) -> None:
    """Compute and print statistics derived from a trace file."""
    tensor_stats, pipe_stats, dfb_stats, dfb_per_core = _accumulate(path)

    if not tensor_stats and not pipe_stats and not dfb_stats:
        print("\nNo statistics found in trace.")
        print(
            "Hint: regenerate the trace with ttlang-sim --trace and at least the "
            "'copy', 'dfb', or 'pipe' categories enabled (all are on by default)."
        )
        return

    if tensor_stats:
        _print_tensor_stats(tensor_stats)
    if pipe_stats:
        _print_pipe_stats(pipe_stats)
    if dfb_stats:
        _print_dfb_stats(dfb_stats, dfb_per_core)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ttlang-sim-stats",
        description=(
            "Derive simulator statistics from a ttlang-sim trace file.\n\n"
            "Produces the same tables as running ttlang-sim with --show-stats,\n"
            "but reads from a pre-recorded JSON Lines trace instead of a live run.\n"
            "DFB statistics include a per-core breakdown with a per-DFB subtotal."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "trace",
        metavar="FILE",
        help="JSON Lines trace file produced by ttlang-sim --trace",
    )
    args = parser.parse_args()

    path = Path(args.trace).resolve()
    if not path.exists():
        print(f"Error: trace file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print_stats_from_trace(path)


if __name__ == "__main__":
    main()
