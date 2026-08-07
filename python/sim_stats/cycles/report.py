# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Rendering and JSON serialization for cycle estimates (pure over CycleEstimate)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from .types import CycleEstimate, KernelEstimate, NodeEstimate
from ..utils import abbrev_count, node_sort_key

_TOOL = "tt-lang-sim-cycles"
_SCHEMA_VERSION = 1
_MIN_WIDTH = 78
_NUM_W = 10  # numeric column width (fits headers + abbreviated values)
_LABEL_PAD = 16  # gap between the label column and the first numeric column
# label + 3 numeric cols (each led by a space) + two-space gap + widest bound.
_ROW_TAIL = 3 * (_NUM_W + 1) + 2 + len("compute")


def _short_bound(bound: str) -> str:
    """ "compute-bound" -> "compute"; the table header already says "Bound"."""
    return bound.split("-", 1)[0]


def _label_width(labels: list[str], header: str) -> int:
    """Label column sized to the longest label (or header), plus a small pad."""
    longest = max((len(x) for x in labels), default=0)
    return max(len(header), longest) + _LABEL_PAD


def _row(
    label: str, compute: float, movement: float, cycles: float, bound: str, label_w: int
) -> str:
    return (
        f"{label:<{label_w}} {abbrev_count(compute):>{_NUM_W}} "
        f"{abbrev_count(movement):>{_NUM_W}} {abbrev_count(cycles):>{_NUM_W}}  {bound}"
    )


def _header(estimate: CycleEstimate, unit: str, label_w: int, width: int) -> None:
    print("\n" + "=" * width)
    print("Cycle Estimate — ideal-peak model")
    print(f"hw-profile: {estimate.profile_name}")
    if estimate.peak_compute_flops_per_cyc > 0.0:
        clock = float(estimate.profile.get("clock_ghz", 1.0))
        cores = int(estimate.profile.get("tensix_cores", 0))
        tflops = estimate.peak_compute_flops_per_cyc * clock / 1000.0
        print(
            f"  {'peak compute':<13}:  {tflops:.1f} TF/s   ({cores} Tensix, bf16 HiFi4)"
        )
        print(f"  {'ridge AI':<13}:  {estimate.ridge_ai:.0f} FLOP/B")
    if not estimate.profile.get("noc_bw"):
        print(
            "WARNING: profile has no noc_bw — movement modeled as free (latency only)"
        )
    print("=" * width)  # title block / tables separator
    print(
        f"{unit:<{label_w}} {'Compute':>{_NUM_W}} "
        f"{'Movement':>{_NUM_W}} {'Cycles':>{_NUM_W}}  Type"
    )
    print("." * width)


def _human_bytes(n: float) -> str:
    """Bytes as a compact binary magnitude (1024-based), matching tt-metal's
    perf_summary so the two tools' MB/GB labels line up directly (e.g. 48.0 MB)."""
    for unit, divisor in (
        ("B", 1.0),
        ("KB", 1024.0),
        ("MB", 1024.0**2),
        ("GB", 1024.0**3),
    ):
        if abs(n) / divisor < 1024.0:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / divisor:.1f} {unit}"
    return f"{n / 1024.0**3:.1f} GB"


def _stats_footer(estimate: CycleEstimate, width: int) -> None:
    """Nodes rollup, optional shared-memory block, and the program summary.

    A pure read of the pre-computed ``estimate`` — no cycle math happens here.
    """
    active = [n for n in estimate.nodes if n.cycles > 0.0]

    # Nodes — per-type rollup, then the per-node max and utilization.
    print("-" * width)
    print("Nodes")
    print("." * width)
    print(f"{'Type':<10}{'Nodes':>8}{'Avg Cycles':>14}{'Max':>14}   Max node")
    by_bound: dict[str, list[NodeEstimate]] = {}
    for n in active:
        by_bound.setdefault(n.bound, []).append(n)
    for bound in ("compute", "movement"):  # always show both types
        rows = by_bound.get(bound, [])
        count = len(rows)
        if rows:
            avg = sum(n.cycles for n in rows) / count
            max_cy = max(n.cycles for n in rows)
            max_node = sorted(
                (n.node for n in rows if n.cycles == max_cy), key=node_sort_key
            )[0]
            avg_s, max_s = abbrev_count(avg), abbrev_count(max_cy)
        else:
            # Empty type: dashes rather than 0.00, matching the "Max node" column.
            avg_s = max_s = max_node = "-"
        print(f"{bound:<10}{count:>8}{avg_s:>14}{max_s:>14}   {max_node}")
    idle = estimate.total_nodes - estimate.active_nodes
    print(
        f"  {'per-node max':<13}:  {abbrev_count(estimate.node_bound)}"
        f"   ({estimate.node_bound_reason})"
    )
    print(
        f"  {'active nodes':<13}:  {estimate.active_nodes} / {estimate.total_nodes}"
        f"   ({idle} idle)"
    )

    # Memory (shared) — only when the profile models an aggregate ceiling.
    agg_bw = float(estimate.profile.get("memory_aggregate_bw", 0.0))
    if agg_bw > 0.0:
        clock = float(estimate.profile.get("clock_ghz", 1.0))
        gbps = agg_bw * clock
        print("-" * width)
        print("Memory (shared)")
        print("." * width)
        print(f"  {'read':<13}:  {_human_bytes(estimate.memory_read_bytes)}")
        print(f"  {'write':<13}:  {_human_bytes(estimate.memory_write_bytes)}")
        print(
            f"  {'bandwidth':<13}:  {agg_bw:g} B/cyc   "
            f"({gbps:g} GB/s @ {clock:.1f} GHz)"
        )
        print(f"  {'floor':<13}:  {abbrev_count(estimate.memory_floor)}")

    # Program — the answer. `bound` is the resource that set it
    # (compute | movement | memory); AI and roof utilization need a compute peak.
    print("-" * width)
    print("Program")
    print("." * width)
    print(f"  {'cycles':<13}:  {abbrev_count(estimate.program_cycles)}")
    has_roofline = estimate.peak_compute_flops_per_cyc > 0.0
    if has_roofline:
        if estimate.total_memory_bytes == 0.0:  # no memory traffic → AI undefined
            ai_line = "n/a (no memory traffic)"
        else:
            ai_line = f"{estimate.arithmetic_intensity:.0f} FLOP/B"
        print(f"  {'AI':<13}:  {ai_line}")
    bound = (
        "memory" if estimate.program_bound == "memory" else estimate.node_bound_reason
    )
    print(f"  {'bound':<13}:  {bound}")
    if has_roofline:
        print(f"  {'compute util':<13}:  {estimate.compute_roof_pct:.0f}%")
        print(f"  {'memory  util':<13}:  {estimate.memory_roof_pct:.0f}%")
    print("=" * width)
    if sum(k.compute_cycles for k in estimate.kernels) == 0.0:
        print(
            "note: compute path is 0 — the trace has no compute_op events "
            "(compute category filtered out, or a pre-instrumentation trace); "
            "movement-only estimate."
        )


def print_detailed(estimate: CycleEstimate) -> None:
    """Detailed per-kernel view — complete, includes zero rows."""
    label_w = _label_width([ke.kernel for ke in estimate.kernels], "Kernel")
    width = max(_MIN_WIDTH, label_w + _ROW_TAIL)
    _header(estimate, "Kernel", label_w, width)
    for ke in estimate.kernels:
        print(
            _row(
                ke.kernel,
                ke.compute_cycles,
                ke.movement_cycles,
                ke.cycles,
                _short_bound(ke.bound),
                label_w,
            )
        )
    _stats_footer(estimate, width)


def print_summary(estimate: CycleEstimate, include_zero: bool = False) -> None:
    """Per-node rollup (the default view).

    Each node's columns are the max over its kernels (concurrent RISCs), matching
    the program combiner — a pure read of ``estimate.nodes``.
    """
    label_w = _label_width([n.node for n in estimate.nodes], "Node")
    width = max(_MIN_WIDTH, label_w + _ROW_TAIL)
    _header(estimate, "Node", label_w, width)
    for n in sorted(estimate.nodes, key=lambda n: node_sort_key(n.node)):
        if not include_zero and n.cycles == 0.0:
            continue
        print(_row(n.node, n.compute, n.movement, n.cycles, n.bound, label_w))
    _stats_footer(estimate, width)


def write_json(path: Path, estimate: CycleEstimate) -> None:
    """Serialize the full, self-describing estimate (for analysis reuse)."""
    payload = {
        "tool": _TOOL,
        "schema_version": _SCHEMA_VERSION,
        "model": "ideal-peak",
        "profile": estimate.profile,
        "program_cycles": estimate.program_cycles,
        "program_bound": estimate.program_bound,
        "memory_floor": estimate.memory_floor,
        "total_memory_bytes": estimate.total_memory_bytes,
        "memory_read_bytes": estimate.memory_read_bytes,
        "memory_write_bytes": estimate.memory_write_bytes,
        "node_bound": estimate.node_bound,
        "node_bound_reason": estimate.node_bound_reason,
        "node_fill_drain": estimate.node_fill_drain,
        "peak_compute_flops_per_cyc": estimate.peak_compute_flops_per_cyc,
        "ridge_ai": estimate.ridge_ai,
        "arithmetic_intensity": estimate.arithmetic_intensity,
        "roofline_bound": estimate.roofline_bound,
        "compute_roof_pct": estimate.compute_roof_pct,
        "memory_roof_pct": estimate.memory_roof_pct,
        "total_nodes": estimate.total_nodes,
        "active_nodes": estimate.active_nodes,
        "nodes": [asdict(n) for n in estimate.nodes],
        "kernels": [asdict(k) for k in estimate.kernels],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_estimate(path: Path | str) -> CycleEstimate:
    """Load a saved report JSON back into a CycleEstimate, with validation.

    Raises FileNotFoundError if the file is missing, or ValueError if it is not a
    tt-lang-sim-cycles report (including the common mistake of passing a raw
    JSON-Lines trace instead of a saved report).
    """
    p = Path(path).resolve()
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"report file not found: {p}") from None

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(
            f"{p} is not a cycle report: not a single JSON object "
            "(a raw --trace file is JSON Lines, not a report)"
        ) from None

    if not isinstance(raw, dict):
        raise ValueError(f"{p} is not a tt-lang-sim-cycles report (not a JSON object)")

    # Give the decoded JSON a concrete type so the reads below are not "unknown".
    data = cast("dict[str, Any]", raw)
    if data.get("tool") != _TOOL or "kernels" not in data:
        raise ValueError(
            f"{p} is not a tt-lang-sim-cycles report (missing tool marker or kernels)"
        )

    try:
        raw_kernels: list[dict[str, Any]] = data["kernels"]
        kernels = [KernelEstimate(**k) for k in raw_kernels]
        raw_nodes: list[dict[str, Any]] = data.get("nodes", [])
        nodes = [NodeEstimate(**n) for n in raw_nodes]
        profile: dict[str, Any] = data.get("profile", {})
        return CycleEstimate(
            profile_name=str(profile.get("name", data.get("profile_name", "?"))),
            profile=profile,
            program_cycles=float(data["program_cycles"]),
            total_nodes=int(data.get("total_nodes", 0)),
            active_nodes=int(data.get("active_nodes", 0)),
            kernels=kernels,
            program_bound=str(data.get("program_bound", "per-node")),
            memory_floor=float(data.get("memory_floor", 0.0)),
            total_memory_bytes=float(data.get("total_memory_bytes", 0.0)),
            memory_read_bytes=float(data.get("memory_read_bytes", 0.0)),
            memory_write_bytes=float(data.get("memory_write_bytes", 0.0)),
            nodes=nodes,
            node_bound=float(data.get("node_bound", 0.0)),
            node_bound_reason=str(data.get("node_bound_reason", "compute")),
            node_fill_drain=float(data.get("node_fill_drain", 0.0)),
            peak_compute_flops_per_cyc=float(
                data.get("peak_compute_flops_per_cyc", 0.0)
            ),
            ridge_ai=float(data.get("ridge_ai", 0.0)),
            arithmetic_intensity=float(data.get("arithmetic_intensity", 0.0)),
            roofline_bound=str(data.get("roofline_bound", "compute")),
            compute_roof_pct=float(data.get("compute_roof_pct", 0.0)),
            memory_roof_pct=float(data.get("memory_roof_pct", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed cycle report {p}: {exc}") from None
