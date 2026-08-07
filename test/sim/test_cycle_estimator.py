# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the sim_stats cycle estimator (analytical ideal-peak model).

Covers: per-op cost, kernel/program combiners, invariants, the compute path
(against synthetic compute_op events), report rendering, JSON round-trip, and
hardware-profile loading.
"""

import json

import pytest

from python.sim_stats.cycles.model import (
    FLOP_PER_MATMUL_TILE,
    build_estimate,
    memory_bytes_by_direction,
    kernel_cycles,
    kernel_paths,
    load_profile_json,
    op_cycles,
    per_node_fill_drain_bound,
    per_node_rollup,
    program_breakdown,
    program_cycles,
    resolve_profile,
    total_memory_bytes,
    total_matmul_flop,
)
from python.sim_stats.cycles.parse import extract_kernel_work
from python.sim_stats.cycles.report import (
    load_estimate,
    print_detailed,
    print_summary,
    write_json,
)
from python.sim_stats.cycles.types import (
    HardwareProfile,
    KernelEstimate,
    KernelWork,
    OpWork,
    TraceEvent,
)


def _hw() -> HardwareProfile:
    """Deterministic test profile (zero latency for clean arithmetic)."""
    return HardwareProfile(
        name="test",
        compute_rate={("matmul", "bf16"): 2.0},
        compute_rate_default=1.0,
        noc_bw={"local_l1": 8.0, "remote_l1": 4.0, "dram": 2.0},
        noc_latency={"local_l1": 0.0, "remote_l1": 0.0, "dram": 0.0},
        clock_ghz=1.0,
        bytes_per_tile=2.0,
    )


# ---------------------------------------------------------------------------
# Movement path
# ---------------------------------------------------------------------------


def test_extract_kernel_work_emits_movement_op_per_locality() -> None:
    events = [
        TraceEvent(0, "kernel_start", "node0-read", {}),
        TraceEvent(
            5,
            "copy_end",
            "node0-read",
            {"tiles": 4, "local_l1": 1, "remote_l1": 2, "dram": 1},
        ),
        TraceEvent(6, "kernel_end", "node0-read", {}),
    ]

    work = extract_kernel_work(events)
    kw = work["node0-read"]

    assert [(o.locality, o.tiles) for o in kw.ops] == [
        ("local_l1", 1),
        ("remote_l1", 2),
        ("dram", 1),
    ]
    assert all(o.kind == "movement" for o in kw.ops)


def test_movement_op_cost_is_tiles_times_bytes_over_bandwidth() -> None:
    hw = _hw()
    op = OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")
    # bytes = 4 tiles * 2 B/tile = 8; bw(dram) = 2 -> 4.0 cycles
    assert op_cycles(op, hw) == 4.0


def test_movement_cost_monotonic_in_tiles() -> None:
    hw = _hw()
    one = op_cycles(
        OpWork(kind="movement", op_type="copy", tiles=4, locality="dram"), hw
    )
    two = op_cycles(
        OpWork(kind="movement", op_type="copy", tiles=8, locality="dram"), hw
    )
    assert two == 2 * one


def test_kernel_cycles_is_max_of_compute_and_movement_paths() -> None:
    hw = _hw()
    kw = KernelWork(
        kernel="node0-compute",
        ops=[
            OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10),  # 5
            OpWork(kind="movement", op_type="copy", tiles=4, locality="dram"),  # 4
        ],
    )
    # max(compute_path=5, movement_path=4) = 5
    assert kernel_cycles(kw, hw) == 5.0


def test_zero_work_zero_cycles() -> None:
    hw = _hw()
    kw = KernelWork(kernel="node0-compute")
    assert kernel_cycles(kw, hw) == 0.0
    assert program_cycles([kw], hw) == 0.0


# ---------------------------------------------------------------------------
# Invariants / regression fixtures
#
# These assert structural PROPERTIES that hold for any valid input (bounds and
# guarantees, not exact placeholder arithmetic).
# ---------------------------------------------------------------------------


def _mixed_kernel() -> KernelWork:
    """A kernel with both a compute and a movement op (compute path dominates)."""
    return KernelWork(
        kernel="node0-compute",
        ops=[
            OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10),  # 5
            OpWork(kind="movement", op_type="copy", tiles=4, locality="dram"),  # 4
        ],
    )


def test_compute_cost_monotonic_in_tiles() -> None:
    # Doubling compute tiles doubles the compute cost (rate fixed).
    hw = _hw()
    one = op_cycles(
        OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10), hw
    )
    two = op_cycles(
        OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=20), hw
    )
    assert one == 5.0
    assert two == 2 * one


def test_kernel_cycles_never_double_counts() -> None:
    # Overlap (max), never additive (sum): max(c, m) <= kernel <= c + m, and
    # strictly less than the additive sum when both paths are non-zero.
    hw = _hw()
    kw = _mixed_kernel()
    compute_path, movement_path = 5.0, 4.0
    k = kernel_cycles(kw, hw)

    assert max(compute_path, movement_path) <= k <= compute_path + movement_path
    assert k < compute_path + movement_path


def test_kernel_cycles_non_negative() -> None:
    hw = _hw()
    assert kernel_cycles(_mixed_kernel(), hw) >= 0.0
    assert kernel_cycles(KernelWork(kernel="node0-read"), hw) >= 0.0


def test_kernel_cycles_deterministic() -> None:
    # Same (work, profile) -> identical output (pure function, no hidden state).
    hw = _hw()
    kw = _mixed_kernel()
    assert kernel_cycles(kw, hw) == kernel_cycles(kw, hw)


def test_program_cycles_is_max_across_parallel_nodes() -> None:
    # Distinct nodes are separate cores running in parallel -> max, not sum.
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10)],
        ),  # 5
        KernelWork(
            kernel="node1-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=16)],
        ),  # 8
    ]
    assert program_cycles(kernels, hw) == 8.0  # max(5, 8), not 13


def test_program_cycles_within_node_is_max_of_kernels() -> None:
    # Reader / compute / writer share one core's concurrent RISCs -> max.
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),  # 4
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10)],
        ),  # 5
        KernelWork(
            kernel="node0-write",
            ops=[OpWork(kind="movement", op_type="copy", tiles=2, locality="dram")],
        ),  # 2
    ]
    assert program_cycles(kernels, hw) == 5.0  # max(4, 5, 2)


def test_program_cycles_bounded_by_max_and_sum_of_kernels() -> None:
    # A program is no faster than its slowest kernel and no slower than running
    # every kernel serially.
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),  # 4
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10)],
        ),  # 5
    ]
    ks = [kernel_cycles(k, hw) for k in kernels]
    prog = program_cycles(kernels, hw)

    assert max(ks) <= prog <= sum(ks)


def test_hand_derived_simple_kernel_value() -> None:
    # Hand-derived golden: a read kernel moving 4 dram tiles.
    # bytes = 4 * 2 = 8; bw(dram) = 2 -> 4.0 cycles.
    hw = _hw()
    kw = KernelWork(
        kernel="node0-read",
        ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
    )
    assert kernel_cycles(kw, hw) == 4.0


# ---------------------------------------------------------------------------
# Aggregate DRAM bandwidth ceiling (program-level roofline)
# ---------------------------------------------------------------------------


def _hw_with_memory_ceiling(agg_bw: float) -> HardwareProfile:
    """Test profile with a shared aggregate-memory peak (bytes/cycle)."""
    return HardwareProfile(
        name="test-dram",
        compute_rate={("matmul", "bf16"): 2.0},
        compute_rate_default=1.0,
        noc_bw={"local_l1": 8.0, "remote_l1": 4.0, "dram": 2.0},
        noc_latency={"local_l1": 0.0, "remote_l1": 0.0, "dram": 0.0},
        clock_ghz=1.0,
        bytes_per_tile=2.0,
        memory_aggregate_bw=agg_bw,
    )


def _read_kernel(node: str, memory_tiles: int, local_tiles: int = 0) -> KernelWork:
    ops = [OpWork(kind="movement", op_type="copy", tiles=memory_tiles, locality="dram")]
    if local_tiles:
        ops.append(
            OpWork(
                kind="movement",
                op_type="copy",
                tiles=local_tiles,
                locality="local_l1",
            )
        )
    return KernelWork(kernel=f"{node}-read", ops=ops)


def test_total_memory_bytes_counts_only_memory_locality() -> None:
    # local_l1 / remote_l1 traffic never hits the DRAM controller.
    hw = _hw_with_memory_ceiling(2.0)
    kernels = [
        _read_kernel("node0", memory_tiles=10, local_tiles=100),
        _read_kernel("node1", memory_tiles=10, local_tiles=100),
    ]
    # 20 dram tiles * 2 B/tile = 40; local_l1 excluded.
    assert total_memory_bytes(kernels, hw) == 40.0


def test_memory_read_write_split_sums_to_total() -> None:
    hw = _hw_with_memory_ceiling(2.0)
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[
                OpWork(
                    kind="movement",
                    op_type="copy",
                    tiles=10,
                    locality="dram",
                    direction="read",
                ),
                OpWork(
                    kind="movement",
                    op_type="copy",
                    tiles=4,
                    locality="dram",
                    direction="write",
                ),
            ],
        )
    ]
    est = build_estimate(kernels, hw)
    assert est.memory_read_bytes == 20.0  # 10 tiles * 2 B
    assert est.memory_write_bytes == 8.0  # 4 tiles * 2 B
    assert (
        est.memory_read_bytes + est.memory_write_bytes == est.total_memory_bytes == 28.0
    )


def test_memory_direction_captured_on_op_from_trace() -> None:
    events = [
        TraceEvent(0, "kernel_start", "node0-read", {}),
        TraceEvent(
            5, "copy_end", "node0-read", {"tiles": 3, "dram": 3, "direction": "read"}
        ),
        TraceEvent(6, "kernel_end", "node0-read", {}),
    ]
    ops = extract_kernel_work(events)["node0-read"].ops
    assert [(o.locality, o.direction) for o in ops] == [("dram", "read")]


def test_memory_split_defaults_gracefully_without_direction() -> None:
    # Traces with no `direction` -> everything falls into read; write = 0;
    # sum still equals the total. (memory_bytes_by_direction and OpWork default.)
    hw = _hw_with_memory_ceiling(2.0)
    kernels = [_read_kernel("node0", memory_tiles=10)]  # _read_kernel sets no direction
    assert kernels[0].ops[0].direction == ""
    read, write = memory_bytes_by_direction(kernels, hw)
    assert (read, write) == (20.0, 0.0)
    est = build_estimate(kernels, hw)
    assert est.memory_write_bytes == 0.0
    assert est.memory_read_bytes == est.total_memory_bytes == 20.0


def test_memory_split_round_trips_through_json(tmp_path) -> None:
    hw = _hw_with_memory_ceiling(2.0)
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[
                OpWork(
                    kind="movement",
                    op_type="copy",
                    tiles=10,
                    locality="dram",
                    direction="read",
                ),
                OpWork(
                    kind="movement",
                    op_type="copy",
                    tiles=4,
                    locality="dram",
                    direction="write",
                ),
            ],
        )
    ]
    estimate = build_estimate(kernels, hw)
    p = tmp_path / "report.json"
    write_json(p, estimate)
    loaded = load_estimate(p)
    assert loaded.memory_read_bytes == 20.0
    assert loaded.memory_write_bytes == 8.0


def test_aggregate_memory_floor_engages_across_nodes() -> None:
    # Four parallel read nodes: per-node movement is small, but summed DRAM
    # traffic exceeds the shared GDDR6 pool -> the program is DRAM-bound.
    hw = _hw_with_memory_ceiling(2.0)
    kernels = [_read_kernel(f"node{i}", memory_tiles=10) for i in range(4)]
    # per-node movement = 10 tiles * 2 B / 2 (dram bw) = 10 cyc -> node_bound = 10.
    # total dram bytes = 4 * 10 * 2 = 80; floor = 80 / 2 = 40 > 10.
    prog, bound, memory_floor, node_bound = program_breakdown(kernels, hw)
    assert node_bound == 10.0
    assert memory_floor == 40.0
    assert bound == "memory"
    assert prog == 40.0
    assert program_cycles(kernels, hw) == 40.0

    est = build_estimate(kernels, hw)
    assert est.program_bound == "memory"
    assert est.program_cycles == 40.0
    assert est.memory_floor == 40.0


def test_tiny_workload_unaffected_by_memory_ceiling() -> None:
    # A single small read: the shared pool is far from saturated, so the
    # per-node bound wins and the floor is just diagnostic.
    hw = _hw_with_memory_ceiling(100.0)
    kernels = [_read_kernel("node0", memory_tiles=4)]
    prog, bound, memory_floor, node_bound = program_breakdown(kernels, hw)
    assert node_bound == 4.0  # 4 tiles * 2 B / 2 (dram bw)
    assert memory_floor == 4 * 2 / 100.0  # 0.08
    assert memory_floor < node_bound
    assert bound == "per-node"
    assert prog == 4.0


def test_zero_aggregate_bw_is_backward_compatible() -> None:
    # memory_aggregate_bw = 0.0 (the default) -> no ceiling, legacy behavior.
    hw = _hw()  # no memory_aggregate_bw set -> 0.0
    assert hw.memory_aggregate_bw == 0.0
    kernels = [_read_kernel(f"node{i}", memory_tiles=10) for i in range(4)]
    prog, bound, memory_floor, node_bound = program_breakdown(kernels, hw)
    assert memory_floor == 0.0
    assert bound == "per-node"
    assert prog == node_bound == 10.0  # unchanged from the pre-ceiling model
    assert build_estimate(kernels, hw).program_bound == "per-node"


def test_ceiling_fields_round_trip_through_json(tmp_path) -> None:
    # program_bound / memory_floor / total_memory_bytes all survive write -> load.
    hw = _hw_with_memory_ceiling(2.0)
    kernels = [_read_kernel(f"node{i}", memory_tiles=10) for i in range(4)]
    estimate = build_estimate(kernels, hw)
    assert estimate.total_memory_bytes == 80.0  # 4 * 10 tiles * 2 B/tile

    p = tmp_path / "report.json"
    write_json(p, estimate)
    loaded = load_estimate(p)

    assert loaded.program_bound == "memory"
    assert loaded.memory_floor == 40.0
    assert loaded.program_cycles == 40.0
    assert loaded.total_memory_bytes == 80.0


def test_old_report_without_ceiling_fields_loads_with_defaults(tmp_path) -> None:
    # A pre-ceiling report (schema had no program_bound/memory_floor/total_memory_bytes)
    # must still load, defaulting the new fields.
    p = tmp_path / "old_report.json"
    p.write_text(
        json.dumps(
            {
                "tool": "tt-lang-sim-cycles",
                "program_cycles": 42.0,
                "total_nodes": 1,
                "active_nodes": 1,
                "profile": {"name": "legacy"},
                "kernels": [],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_estimate(p)
    assert loaded.program_cycles == 42.0
    assert loaded.program_bound == "per-node"
    assert loaded.memory_floor == 0.0
    assert loaded.total_memory_bytes == 0.0


def test_memory_floor_equals_traffic_over_aggregate_bw() -> None:
    hw = _hw_with_memory_ceiling(2.0)
    kernels = [_read_kernel(f"node{i}", memory_tiles=10) for i in range(4)]
    estimate = build_estimate(kernels, hw)
    assert estimate.memory_floor == estimate.total_memory_bytes / hw.memory_aggregate_bw


def test_memory_floor_tie_with_node_bound_stays_per_node() -> None:
    # floor exactly == node_bound: the strict ">" means per-node wins the tie
    # (the aggregate ceiling only takes over when it is genuinely higher).
    hw = _hw_with_memory_ceiling(2.0)
    kernels = [_read_kernel("node0", memory_tiles=10)]
    # node_bound = 10*2/2 = 10; total dram = 20; floor = 20/2 = 10 == node_bound.
    _prog, bound, memory_floor, node_bound = program_breakdown(kernels, hw)
    assert memory_floor == node_bound == 10.0
    assert bound == "per-node"


def test_empty_program_is_zero_and_per_node() -> None:
    # No kernels at all: zero cycles, no ceiling engaged, no crash.
    hw = _hw_with_memory_ceiling(2.0)
    prog, bound, memory_floor, node_bound = program_breakdown([], hw)
    assert (prog, bound, memory_floor, node_bound) == (0.0, "per-node", 0.0, 0.0)
    est = build_estimate([], hw)
    assert est.program_cycles == 0.0
    assert est.active_nodes == 0
    assert est.total_memory_bytes == 0.0


def test_summary_shows_memory_block_only_with_a_ceiling(capsys) -> None:
    kernels = [_read_kernel(f"node{i}", memory_tiles=10) for i in range(4)]

    # With a ceiling: the Memory (shared) block renders read/write + bandwidth + floor
    # (read/write mirror tt-metal perf_summary; total is implicit).
    print_summary(build_estimate(kernels, _hw_with_memory_ceiling(2.0)))
    out = capsys.readouterr().out
    assert "Memory (shared)" in out
    assert "read" in out and "write" in out and "bandwidth" in out and "floor" in out
    assert "B/cyc" in out and "GB/s" in out
    assert "per-node max" in out
    assert "Program" in out
    # No equations leaked into the render (labeled values only).
    memory_block = out.split("Memory (shared)")[1].split("Program")[0]
    assert "÷" not in memory_block and "max(" not in memory_block

    # Without a ceiling (bw=0): no Memory block, footer shape unchanged.
    print_summary(build_estimate(kernels, _hw()))
    out = capsys.readouterr().out
    assert "Memory (shared)" not in out
    assert "Program" in out


def test_per_node_max_reason_matches_slowest_node() -> None:
    # node1's compute (16 tiles / rate 2 = 8) is the slowest node -> compute.
    hw = _hw_with_memory_ceiling(2.0)
    kernels = [
        _read_kernel("node0", memory_tiles=1),  # movement 1 cyc
        KernelWork(
            kernel="node1-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=16)],
        ),  # compute 8 cyc, dominates
    ]
    estimate = build_estimate(kernels, hw)
    assert estimate.node_bound == 8.0
    assert estimate.node_bound_reason == "compute"


def test_per_node_rollup_maxes_over_a_nodes_kernels() -> None:
    # A node's compute/movement/cycles are the max over its kernels.
    hw = _hw_with_memory_ceiling(2.0)
    kernels = [
        _read_kernel("node0", memory_tiles=4),  # movement 4 cyc
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10)],
        ),  # compute 5 cyc
    ]
    nodes = per_node_rollup(build_estimate(kernels, hw).kernels)
    assert len(nodes) == 1
    n = nodes[0]
    assert (n.node, n.compute, n.movement, n.cycles, n.bound) == (
        "node0",
        5.0,
        4.0,
        5.0,
        "compute",
    )


# ---------------------------------------------------------------------------
# Tier-1 pipeline fill/drain
# ---------------------------------------------------------------------------


def _ke(kernel: str, cycles: float) -> KernelEstimate:
    """A KernelEstimate stub with only the fields fill/drain reads."""
    return KernelEstimate(
        kernel=kernel,
        node="node0",
        role="",
        compute_cycles=0.0,
        movement_cycles=0.0,
        cycles=cycles,
        bound="compute-bound",
    )


def _write_kernel(node: str, items: int) -> KernelWork:
    """A write-role kernel with `items` movement ops (the pipeline-item count N)."""
    return KernelWork(
        kernel=f"{node}-write",
        ops=[OpWork(kind="movement", op_type="copy", tiles=1) for _ in range(items)],
    )


def test_fill_drain_large_n_recovers_node_bound() -> None:
    # Stages 4/10/2 -> max=10, sum=16. With large N the correction (sum-max)/N
    # -> 0, so the per-node bound recovers the throughput max (10).
    ke = [_ke("node0-read", 4.0), _ke("node0-compute", 10.0), _ke("node0-write", 2.0)]
    fd = per_node_fill_drain_bound(ke, [_write_kernel("node0", items=100_000)])
    assert fd == pytest.approx(10.0, abs=1e-3)


def test_fill_drain_small_n_adds_serial_correction() -> None:
    # N=1 -> no pipelining -> node_time = sum of stages (serial).
    ke = [_ke("node0-read", 4.0), _ke("node0-compute", 10.0), _ke("node0-write", 2.0)]
    fd = per_node_fill_drain_bound(ke, [_write_kernel("node0", items=1)])
    assert fd == 16.0  # 10 + (16 - 10) / 1

    # No write kernel -> N defaults to 1 (serial), same result.
    assert per_node_fill_drain_bound(ke, []) == 16.0


def test_fill_drain_moderate_n_matches_formula() -> None:
    # N=4: 10 + (16 - 10)/4 = 11.5.
    ke = [_ke("node0-read", 4.0), _ke("node0-compute", 10.0), _ke("node0-write", 2.0)]
    fd = per_node_fill_drain_bound(ke, [_write_kernel("node0", items=4)])
    assert fd == pytest.approx(11.5)


def test_fill_drain_does_not_change_memory_bound_program() -> None:
    # Per-node fill/drain is real but non-binding when the aggregate DRAM floor
    # dominates: program stays == memory_floor, memory.
    hw = _hw_with_memory_ceiling(2.0)
    kernels = []
    for i in range(4):
        node = f"node{i}"
        kernels += [
            KernelWork(
                kernel=f"{node}-read",
                ops=[
                    OpWork(kind="movement", op_type="copy", tiles=100, locality="dram")
                ],
            ),  # 100*2/2 = 100 cyc, 200 B dram
            KernelWork(
                kernel=f"{node}-compute",
                ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=2)],
            ),  # 2/2 = 1 cyc
            KernelWork(
                kernel=f"{node}-write",
                ops=[
                    OpWork(
                        kind="movement", op_type="copy", tiles=2, locality="local_l1"
                    )
                ],
            ),  # 2*2/8 = 0.5 cyc, N=1
        ]
    est = build_estimate(kernels, hw)
    # Per-node: stages 100/1/0.5, N=1 -> node_time 101.5; node_bound 100 -> fd = 1.5.
    assert est.node_bound == 100.0
    assert est.node_fill_drain == 1.5
    # DRAM: 4 * 100 tiles * 2 B = 800 B; floor = 800 / 2 = 400 > 101.5.
    assert est.memory_floor == 400.0
    assert est.program_bound == "memory"
    assert est.program_cycles == 400.0  # unchanged by fill/drain


def test_node_fill_drain_round_trips_through_json(tmp_path) -> None:
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=20)],
        ),
        KernelWork(
            kernel="node0-write",
            ops=[OpWork(kind="movement", op_type="copy", tiles=2, locality="dram")],
        ),
    ]
    estimate = build_estimate(kernels, hw)
    p = tmp_path / "report.json"
    write_json(p, estimate)
    assert load_estimate(p).node_fill_drain == estimate.node_fill_drain


def test_empty_bound_row_shows_dash_not_zero(capsys) -> None:
    # All nodes are movement-bound -> the empty "compute" row shows "-", not 0.00.
    hw = _hw()
    kernels = [_read_kernel(f"node{i}", memory_tiles=4) for i in range(2)]
    print_summary(build_estimate(kernels, hw))
    out = capsys.readouterr().out

    compute_row = next(ln for ln in out.splitlines() if ln.startswith("compute"))
    assert "0.00" not in compute_row
    assert compute_row.rstrip().endswith("-")  # Avg/Max/Max-node all "-"


def test_extract_kernel_work_reads_compute_op_events() -> None:
    events = [
        TraceEvent(0, "kernel_start", "node0-compute", {}),
        TraceEvent(
            2,
            "compute_op",
            "node0-compute",
            {"op_type": "matmul", "dtype": "bf16", "tiles": 10},
        ),
        TraceEvent(3, "kernel_end", "node0-compute", {}),
    ]

    work = extract_kernel_work(events)
    ops = work["node0-compute"].ops

    assert len(ops) == 1
    assert ops[0].kind == "compute"
    assert ops[0].op_type == "matmul"
    assert ops[0].dtype == "bf16"
    assert ops[0].tiles == 10


def test_kernel_cycles_from_trace_is_max_of_compute_and_movement() -> None:
    # End-to-end through the parser: a kernel that both computes and moves data.
    events = [
        TraceEvent(0, "kernel_start", "node0-compute", {}),
        TraceEvent(
            1,
            "compute_op",
            "node0-compute",
            {"op_type": "matmul", "dtype": "bf16", "tiles": 10},  # 10/2 = 5
        ),
        TraceEvent(
            2, "copy_end", "node0-compute", {"tiles": 4, "dram": 4}
        ),  # 4*2/2 = 4
        TraceEvent(3, "kernel_end", "node0-compute", {}),
    ]

    work = extract_kernel_work(events)
    kw = work["node0-compute"]
    kinds = sorted(o.kind for o in kw.ops)

    assert kinds == ["compute", "movement"]
    assert kernel_cycles(kw, _hw()) == 5.0  # max(5, 4)


def test_compute_op_with_zero_tiles_is_ignored() -> None:
    events = [
        TraceEvent(0, "kernel_start", "node0-compute", {}),
        TraceEvent(1, "compute_op", "node0-compute", {"op_type": "add", "tiles": 0}),
        TraceEvent(2, "kernel_end", "node0-compute", {}),
    ]

    work = extract_kernel_work(events)
    assert work["node0-compute"].ops == []


def test_compute_op_missing_dtype_falls_back_to_default_rate() -> None:
    # op_type + tiles only (the minimum-viable contract); dtype defaults to "".
    # rate_for("matmul", "") misses the (matmul, bf16) entry -> default rate 1.0.
    hw = _hw()
    events = [
        TraceEvent(0, "kernel_start", "node0-compute", {}),
        TraceEvent(1, "compute_op", "node0-compute", {"op_type": "matmul", "tiles": 4}),
        TraceEvent(2, "kernel_end", "node0-compute", {}),
    ]

    work = extract_kernel_work(events)
    # 4 tiles / default rate 1.0 = 4.0
    assert kernel_cycles(work["node0-compute"], hw) == 4.0


# ---------------------------------------------------------------------------
# rate_for + report rendering
# ---------------------------------------------------------------------------


def test_rate_for_is_dtype_optional() -> None:
    hw = HardwareProfile(
        name="t",
        compute_rate={("matmul", "bf16"): 8.0, ("add", ""): 32.0},
        compute_rate_default=2.0,
        noc_bw={},
        noc_latency={},
        clock_ghz=1.0,
        bytes_per_tile=1.0,
    )
    assert hw.rate_for("matmul", "bf16") == 8.0  # exact (op, dtype)
    assert hw.rate_for("matmul", "fp32") == 2.0  # no (matmul, "") -> default
    assert hw.rate_for("add", "bf16") == 32.0  # op-type-only entry serves any dtype
    assert hw.rate_for("add") == 32.0  # dtype omitted -> (add, "")
    assert hw.rate_for("exp") == 2.0  # unknown op -> default


def test_kernel_paths_splits_compute_and_movement() -> None:
    hw = _hw()
    kw = KernelWork(
        kernel="node0-compute",
        ops=[
            OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10),  # 5
            OpWork(kind="movement", op_type="copy", tiles=4, locality="dram"),  # 4
        ],
    )
    assert kernel_paths(kw, hw) == (5.0, 4.0)


def test_detailed_report_shows_decomposition_and_program_total(capsys) -> None:
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10)],
        ),
    ]

    print_detailed(build_estimate(kernels, hw))
    out = capsys.readouterr().out

    assert "ideal-peak model" in out
    assert "node0-compute" in out
    assert "node0-read" in out
    assert "Program" in out


def test_detailed_report_notes_empty_compute_path(capsys) -> None:
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),
    ]

    print_detailed(build_estimate(kernels, hw))
    out = capsys.readouterr().out

    assert "compute path is 0" in out


def test_summary_rolls_up_per_node_and_reports_utilization(capsys) -> None:
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10)],
        ),
        KernelWork(kernel="node1-read", ops=[]),  # idle node
        KernelWork(kernel="node1-compute", ops=[]),
    ]

    print_summary(build_estimate(kernels, hw))
    out = capsys.readouterr().out

    assert "Node" in out
    assert "node0" in out
    assert "1 / 2" in out
    assert "per-node max" in out
    assert "Memory (shared)" not in out  # profile has no aggregate ceiling
    assert "node1" not in out  # idle node hidden by default


def test_json_write_then_load_round_trip(tmp_path) -> None:
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),
    ]
    estimate = build_estimate(kernels, hw)

    p = tmp_path / "report.json"
    write_json(p, estimate)
    loaded = load_estimate(p)

    assert loaded.profile_name == estimate.profile_name
    assert loaded.program_cycles == estimate.program_cycles
    assert [k.kernel for k in loaded.kernels] == [k.kernel for k in estimate.kernels]
    assert loaded.kernels[0].movement_cycles == 4.0


def test_load_estimate_rejects_non_report(tmp_path) -> None:
    p = tmp_path / "other.json"
    p.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_estimate(p)


def test_load_estimate_rejects_jsonl_trace(tmp_path) -> None:
    # A raw trace is JSON Lines (multiple objects), not a single report object.
    p = tmp_path / "trace.jsonl"
    p.write_text(
        '{"event": "kernel_start"}\n{"event": "kernel_end"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_estimate(p)


# ---------------------------------------------------------------------------
# Custom hardware profile loading (--hw-profile <name|path.json>)
# ---------------------------------------------------------------------------


def _write_profile(path, **overrides) -> None:
    data = {
        "name": "custom",
        "compute_rate": [["matmul", "bf16", 8.0]],
        "compute_rate_default": 4.0,
        "noc_bw": {"dram": 2.0},
        "noc_latency": {"dram": 1.0},
        "clock_ghz": 1.0,
        "bytes_per_tile": 2048.0,
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_profile_json_round_trip(tmp_path) -> None:
    p = tmp_path / "custom.json"
    _write_profile(p)

    hw = load_profile_json(p)

    assert hw.name == "custom"
    assert hw.rate_for("matmul", "bf16") == 8.0  # listed
    assert hw.rate_for("add", "bf16") == 4.0  # falls back to default
    assert hw.bandwidth_for("dram") == 2.0
    assert hw.latency_for("dram") == 1.0
    assert hw.bytes_per_tile == 2048.0
    # JSON omits memory_aggregate_gbps -> defaults to 0.0 (no aggregate ceiling).
    assert hw.memory_aggregate_bw == 0.0


def test_resolve_profile_accepts_builtin_name_and_json_path(tmp_path) -> None:
    assert resolve_profile("wormhole_n300").name == "wormhole_n300"

    p = tmp_path / "mine.json"
    _write_profile(p, name="mine")
    assert resolve_profile(str(p)).name == "mine"


def test_resolve_profile_family_alias() -> None:
    # A board family resolves to its single bundled profile.
    assert resolve_profile("wormhole").name == "wormhole_n300"
    assert resolve_profile("blackhole").name == "blackhole_p100a"
    # An unknown name still raises.
    with pytest.raises(ValueError, match="unknown hardware profile"):
        resolve_profile("grayskull")


def test_load_profile_json_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_profile_json(tmp_path / "nope.json")


def test_load_profile_json_all_fields_optional(tmp_path) -> None:
    # Every field is optional: an empty profile loads with sane defaults.
    p = tmp_path / "min.json"
    p.write_text('{"name": "min"}', encoding="utf-8")
    hw = load_profile_json(p)
    assert hw.clock_ghz == 1.0
    assert hw.bytes_per_tile == 2048.0
    assert hw.compute_rate_default == 1.0
    assert hw.noc_bw == {}  # no noc_bw -> movement is free (report flags it)


def test_load_profile_json_malformed_raises(tmp_path) -> None:
    # Bad *values* still fail: non-numeric field, non-positive clock, wrong type.
    for bad in ('{"clock_ghz": "fast"}', '{"bytes_per_tile": 0}', '{"noc_bw": "x"}'):
        p = tmp_path / "bad.json"
        p.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            load_profile_json(p)


# ---------------------------------------------------------------------------
# Roofline (per-board compute peak + program position)
# ---------------------------------------------------------------------------


def _hw_roofline() -> HardwareProfile:
    """Test profile with a compute peak (tensix_cores) and a DRAM ceiling."""
    return HardwareProfile(
        name="test-roof",
        compute_rate={("matmul", ""): 0.5},
        compute_rate_default=1.0,
        noc_bw={"local_l1": 8.0, "remote_l1": 4.0, "dram": 2.0},
        noc_latency={"local_l1": 0.0, "remote_l1": 0.0, "dram": 0.0},
        clock_ghz=1.0,
        bytes_per_tile=2.0,
        memory_aggregate_bw=256.0,
        tensix_cores=4,
    )


def _matmul_kernel(node: str, tiles: int) -> KernelWork:
    return KernelWork(
        kernel=f"{node}-compute",
        ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=tiles)],
    )


def test_total_matmul_flop_counts_matmul_only() -> None:
    kernels = [
        _matmul_kernel("node0", tiles=3),
        KernelWork(
            kernel="node0-eltwise",
            ops=[OpWork(kind="compute", op_type="add", dtype="bf16", tiles=99)],
        ),
    ]
    # add is not matmul -> excluded; 3 tiles * FLOP/tile.
    assert total_matmul_flop(kernels) == 3 * FLOP_PER_MATMUL_TILE


def test_peak_compute_and_ridge_are_board_constants() -> None:
    # Independent of the program: build with no kernels.
    est = build_estimate([], _hw_roofline())
    assert est.peak_compute_flops_per_cyc == 4 * FLOP_PER_MATMUL_TILE * 0.5
    assert est.ridge_ai == est.peak_compute_flops_per_cyc / 256.0


def test_datasheet_profiles_expose_expected_ridge() -> None:
    # Board constants match the documented roofline (WH ~228, BH ~370 FLOP/B).
    for name, ridge in (("wormhole_n300", 228), ("blackhole_p100a", 370)):
        est = build_estimate([], resolve_profile(name))
        assert est.peak_compute_flops_per_cyc > 0.0
        assert round(est.ridge_ai) == ridge


def test_arithmetic_intensity_is_matmul_flop_over_memory_bytes() -> None:
    hw = _hw_roofline()
    kernels = [
        _matmul_kernel("node0", tiles=8),
        _read_kernel("node0", memory_tiles=100),
    ]
    est = build_estimate(kernels, hw)
    flop = 8 * FLOP_PER_MATMUL_TILE
    memory_bytes = 100 * 2.0
    assert est.arithmetic_intensity == flop / memory_bytes


def test_roofline_bound_classifies_by_ai_vs_ridge() -> None:
    hw = _hw_roofline()  # ridge = 512 FLOP/B
    # High AI (little DRAM traffic) -> compute region.
    compute_side = build_estimate(
        [_matmul_kernel("node0", tiles=8), _read_kernel("node0", memory_tiles=1)], hw
    )
    assert compute_side.arithmetic_intensity >= compute_side.ridge_ai
    assert compute_side.roofline_bound == "compute"
    # Low AI (lots of DRAM traffic) -> memory (dram) region.
    memory_side = build_estimate(
        [_matmul_kernel("node0", tiles=1), _read_kernel("node0", memory_tiles=1000)], hw
    )
    assert memory_side.arithmetic_intensity < memory_side.ridge_ai
    assert memory_side.roofline_bound == "memory"


def test_roofline_absent_when_profile_lacks_tensix_cores() -> None:
    # _hw() has no tensix_cores -> no compute peak, roofline block suppressed.
    est = build_estimate([_matmul_kernel("node0", tiles=8)], _hw())
    assert est.peak_compute_flops_per_cyc == 0.0
    assert est.ridge_ai == 0.0


def test_roofline_fields_round_trip_through_json(tmp_path) -> None:
    hw = _hw_roofline()
    est = build_estimate(
        [_matmul_kernel("node0", tiles=8), _read_kernel("node0", memory_tiles=100)], hw
    )
    p = tmp_path / "report.json"
    write_json(p, est)
    loaded = load_estimate(p)
    assert loaded.peak_compute_flops_per_cyc == est.peak_compute_flops_per_cyc
    assert loaded.ridge_ai == est.ridge_ai
    assert loaded.arithmetic_intensity == est.arithmetic_intensity
    assert loaded.roofline_bound == est.roofline_bound


def test_summary_renders_roofline_block_with_tensix_cores(capsys) -> None:
    kernels = [
        _matmul_kernel("node0", tiles=8),
        _read_kernel("node0", memory_tiles=100),
    ]
    print_summary(build_estimate(kernels, _hw_roofline()))
    out = capsys.readouterr().out
    assert "peak compute" in out
    assert "ridge AI" in out
    assert "compute util" in out
    assert "util" in out


def test_summary_omits_roofline_without_tensix_cores(capsys) -> None:
    # _hw() has no tensix_cores -> no compute peak, so AI/util lines are suppressed.
    print_summary(build_estimate([_matmul_kernel("node0", tiles=8)], _hw()))
    out = capsys.readouterr().out
    assert "compute util" not in out
    assert "memory util" not in out
    assert "peak compute" not in out


def test_roofline_bound_is_compute_when_no_memory_traffic() -> None:
    # Matmul with zero DRAM-locality movement: infinite AI -> compute-bound,
    # not memory-bound (regression for the total_bytes == 0 guard).
    est = build_estimate([_matmul_kernel("node0", tiles=8)], _hw_roofline())
    assert est.total_memory_bytes == 0.0
    assert est.roofline_bound == "compute"
