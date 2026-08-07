# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Data types for cycle estimation from simulator traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    tick: int
    event: str
    kernel: str | None
    data: dict[str, Any]


@dataclass(frozen=True)
class HardwareProfile:
    """Static hardware spec: the rates a trace can't provide.

    Field meanings + provenance: docs/development/CycleEstimator.md. Profile *data*
    is JSON under ``hw_profiles/``, loaded by :mod:`model`
    (``resolve_profile`` / ``load_profile_json``).
    """

    name: str
    compute_rate: dict[tuple[str, str], float]
    compute_rate_default: float
    noc_bw: dict[str, float]
    noc_latency: dict[str, float]
    clock_ghz: (
        float  # load-bearing: also converts memory_aggregate_gbps -> B/cyc at load
    )
    bytes_per_tile: float
    dm_engines: int = 1
    memory_aggregate_bw: float = (
        0.0  # B/cyc; JSON stores gbps, converted ÷clock at load
    )
    tensix_cores: int = 0  # chip-wide Tensix count; 0 → compute roof unavailable

    def rate_for(self, op_type: str, dtype: str = "") -> float:
        """Peak tiles/cycle: exact ``(op_type, dtype)`` → ``(op_type, "")`` → default."""
        for key in ((op_type, dtype), (op_type, "")):
            if key in self.compute_rate:
                return self.compute_rate[key]
        return self.compute_rate_default

    def bandwidth_for(self, locality: str) -> float:
        """Peak bytes/cycle for a locality, or 0.0 if unknown."""
        return self.noc_bw.get(locality, 0.0)

    def latency_for(self, locality: str) -> float:
        """Fixed per-transfer latency in cycles for a locality, or 0.0 if unknown."""
        return self.noc_latency.get(locality, 0.0)

    def summary(self) -> dict[str, Any]:
        """Serializable snapshot embedded in a report for reproducibility."""
        return {
            "name": self.name,
            "clock_ghz": self.clock_ghz,
            "bytes_per_tile": self.bytes_per_tile,
            "compute_rate_default": self.compute_rate_default,
            "noc_bw": dict(self.noc_bw),
            "noc_latency": dict(self.noc_latency),
            "memory_aggregate_bw": self.memory_aggregate_bw,
            "tensix_cores": self.tensix_cores,
        }


@dataclass(frozen=True)
class OpWork:
    """A single operation extracted from the trace (per-op work record)."""

    kind: str  # "compute" | "movement"
    op_type: str  # e.g. "matmul", "add", "exp", "copy"
    dtype: str = ""  # e.g. "bf16", "fp32" (compute ops)
    tiles: int = 0  # work in tiles (compute tiles, or tiles moved)
    locality: str = ""  # "local_l1" | "remote_l1" | "dram" (movement ops)
    direction: str = ""  # "read" | "write" (movement ops)


@dataclass
class KernelWork:
    """Per-kernel collection of op records extracted from the trace."""

    kernel: str
    ops: list[OpWork] = field(default_factory=list[OpWork])


@dataclass(frozen=True)
class KernelEstimate:
    """Per-kernel cycle decomposition (a rendered result row)."""

    kernel: str
    node: str
    role: str
    compute_cycles: float
    movement_cycles: float
    cycles: float
    bound: str


@dataclass(frozen=True)
class NodeEstimate:
    """Per-node rollup row: the max over a node's kernels (concurrent RISCs)."""

    node: str
    compute: float
    movement: float
    cycles: float
    bound: str  # "compute" | "movement"


@dataclass(frozen=True)
class CycleEstimate:
    """Canonical estimate result: the intermediate that render + JSON share.

    Produced fresh from a trace (:func:`model.build_estimate`) or loaded back from
    a saved JSON report (:func:`report.load_estimate`). All views (summary /
    detailed / JSON) are pure functions of this.
    """

    profile_name: str
    profile: dict[str, Any]  # resolved rates, embedded for reproducibility
    program_cycles: float
    total_nodes: int
    active_nodes: int
    kernels: list[KernelEstimate] = field(default_factory=list[KernelEstimate])
    program_bound: str = "per-node"  # "per-node" | "memory"
    memory_floor: float = 0.0
    total_memory_bytes: float = 0.0
    memory_read_bytes: float = 0.0
    memory_write_bytes: float = 0.0
    nodes: list[NodeEstimate] = field(default_factory=list[NodeEstimate])
    node_bound: float = 0.0  # max over nodes of per-node cycles (throughput)
    node_bound_reason: str = "compute"  # slowest node's bound ("compute"|"movement")
    node_fill_drain: float = (
        0.0  # informational only; NOT in program_cycles (crude, can overshoot)
    )
    # Roofline: board constants (peak/ridge) + program position (AI, roof-%).
    peak_compute_flops_per_cyc: float = 0.0
    ridge_ai: float = 0.0
    arithmetic_intensity: float = 0.0
    roofline_bound: str = "compute"  # "compute" | "memory"
    compute_roof_pct: float = 0.0
    memory_roof_pct: float = 0.0


# Profile *data* is JSON under hw_profiles/, loaded by :mod:`model`
# (resolve_profile / load_profile_json). This module holds only the schema.
