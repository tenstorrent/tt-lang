# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Benchmark workers retain private artifacts and accept explicit fidelity names."""

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.all_gather_minimal_matmul import __main__ as benchmark


def test_private_worker_directory(monkeypatch, tmp_path):
    arguments = SimpleNamespace(
        implementation="ttlang",
        json=tmp_path / "report.json",
        device_aggregation="mean",
        worker_timeout=180,
    )

    def run_worker(command, *, env, check, timeout):
        directory = Path(env["TT_METAL_PROFILER_DIR"])
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert check and timeout == 180
        Path(command[-1]).write_text(json.dumps({"measurements": {}}))

    monkeypatch.setattr(benchmark.subprocess, "run", run_worker)
    benchmark.run_isolated_variants(arguments)
    assert json.loads(arguments.json.read_text())["variants"] == {
        "ttlang": {"measurements": {}}
    }


def test_fidelity_allowlist():
    assert set(benchmark.MATH_FIDELITIES) == {"HiFi2", "HiFi4"}
    with pytest.raises(KeyError):
        benchmark.MATH_FIDELITIES["__class__"]


def test_collective_only_returns_workload(monkeypatch):
    monkeypatch.setattr(benchmark, "to_dram", lambda *arguments, **options: object())
    monkeypatch.setattr(
        benchmark.ttnn, "ShardTensorToMesh", lambda *arguments, **options: None
    )
    monkeypatch.setattr(
        benchmark.ttnn, "ReplicateTensorToMesh", lambda *arguments: None
    )
    monkeypatch.setattr(
        benchmark,
        "make_column_all_gather",
        lambda *arguments, **options: lambda *tensors: None,
    )
    config = SimpleNamespace(
        m_tiles=2,
        k_tiles_per_device=1,
        device_count=2,
        k_block_tiles=1,
        mesh_shape=(2, 1),
        m_workers=2,
        m_block_tiles=1,
    )
    arguments = SimpleNamespace(
        collective_only="activation",
        activation_all_gather="all_to_all",
        dtype="bf16",
        seed=0,
    )
    workloads, validate = benchmark.create_collective_workload(
        object(), config, arguments
    )
    assert isinstance(workloads["ttlang"], benchmark.Workload)
    assert workloads["ttlang"].program_count == 1
    assert workloads["ttlang"].run() is not None


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_replicated_component_reuses_operation_and_tensors(monkeypatch, dtype):
    from examples.all_gather_minimal_matmul.replicated.operation import (
        ActivationAllGatherConfig,
        ReplicatedAllGatherMatmul,
    )

    calls = []
    config = benchmark.AllGatherMinimalMatmulConfig(
        mesh_shape=(2, 1),
        m_tiles=2,
        k_tiles_per_device=1,
        n_tiles_per_device=2,
        worker_grid=(2, 2),
    )
    collective_config = ActivationAllGatherConfig((2, 1), 2, 1, 2, 1, 1, "all_to_all")
    operation = ReplicatedAllGatherMatmul(
        collective_config,
        lambda *tensors: calls.append(("collective", tensors)),
        lambda *tensors: calls.append(("matmul", tensors)),
    )
    monkeypatch.setattr(
        benchmark,
        "make_replicated_all_gather_matmul_operation",
        lambda *arguments, **options: operation,
    )
    monkeypatch.setattr(
        benchmark, "to_dram", lambda tensor, *arguments, **options: tensor
    )
    monkeypatch.setattr(
        benchmark.ttnn, "ShardTensorToMesh", lambda *arguments, **options: None
    )
    monkeypatch.setattr(
        benchmark.ttnn, "ReplicateTensorToMesh", lambda *arguments: None
    )
    workloads, validate = benchmark.create_workloads(
        object(), config, dtype, 0, "ttlang", 0, variant="replicated"
    )
    full = workloads["ttlang"]
    full.run()
    result = full.activation_collective.run()
    full.run()
    assert [name for name, tensors in calls] == [
        "collective",
        "matmul",
        "collective",
        "collective",
        "matmul",
    ]
    for call_index in (2, 3):
        assert all(
            actual is expected
            for actual, expected in zip(calls[call_index][1], calls[0][1])
        )
    assert result is calls[0][1][1]
    assert full.program_count == 2
    assert full.activation_collective.config == benchmark.asdict(collective_config)


@pytest.mark.parametrize("program_count", [1, 2])
def test_timer_uses_workload_program_count(monkeypatch, program_count):
    observed = []
    mesh = SimpleNamespace(get_device_ids=lambda: [0])
    arguments = SimpleNamespace(
        warmup=0, samples=1, trace=False, device_aggregation="mean"
    )
    workload = benchmark.Workload(
        lambda: object(), None, lambda output: None, program_count
    )
    monkeypatch.setattr(
        benchmark,
        "measured_invocation",
        lambda run, cleanup, device, trace: benchmark.nullcontext(run()),
    )
    monkeypatch.setattr(benchmark, "read_device_profile", lambda device: "unused.csv")

    def duration(log, device_ids, *, aggregation, program_count):
        observed.append(program_count)
        return {"cycles": 1350, "us": 1.0, "per_device": {0: {"run_host_id": 1}}}

    monkeypatch.setattr(benchmark, "latest_kernel_duration", duration)
    benchmark.benchmark(
        {"ttlang": workload}, lambda name, output, gathered: {}, mesh, arguments
    )
    assert observed == [program_count]


def test_trace_replay_retains_output_until_trace_release(monkeypatch):
    events = []
    output = object()
    mesh = object()
    monkeypatch.setattr(
        benchmark.ttnn,
        "begin_trace_capture",
        lambda device, cq_id: events.append("begin") or 7,
    )
    monkeypatch.setattr(
        benchmark.ttnn,
        "end_trace_capture",
        lambda device, trace_id, cq_id: events.append("end"),
    )
    monkeypatch.setattr(
        benchmark.ttnn,
        "execute_trace",
        lambda device, trace_id, cq_id, blocking: events.append("replay"),
    )
    monkeypatch.setattr(
        benchmark.ttnn,
        "synchronize_device",
        lambda device: events.append("sync"),
    )
    monkeypatch.setattr(
        benchmark.ttnn,
        "release_trace",
        lambda device, trace_id: events.append("release"),
    )

    def run():
        events.append("run")
        return output

    def cleanup(result):
        assert result is output
        events.append("cleanup")

    with benchmark.measured_invocation(run, cleanup, mesh, trace=True) as result:
        assert result is output
        events.append("measure")
    assert events == [
        "begin",
        "run",
        "end",
        "replay",
        "sync",
        "measure",
        "release",
        "cleanup",
    ]


@pytest.mark.parametrize(
    "discovered,requested,expected",
    [
        ((2, 2), None, ((2, 1), 0, (2, 2))),
        ((1, 4), None, ((1, 2), 1, (1, 4))),
        ((2, 2), (1, 2), ((1, 2), 1, (2, 2))),
        ((2, 2), (4, 1), ((4, 1), 0, (4, 1))),
        ((2, 2), (1, 4), ((1, 4), 1, (1, 4))),
        ((4, 8), (1, 8), ((1, 8), 1, (4, 8))),
        ((4, 8), (1, 16), ((1, 16), 1, (1, 32))),
    ],
)
def test_mesh_selection(discovered, requested, expected):
    assert benchmark.mesh_selection(discovered, requested) == expected


@pytest.mark.parametrize(
    "requested", ["2x2", "1x1", "0x4", "-1x4", "4", "1x2x2", "ax2"]
)
def test_reject_non_line_mesh(requested):
    with pytest.raises(benchmark.argparse.ArgumentTypeError):
        benchmark.participant_shape(requested)


def test_reject_unavailable_participants():
    with pytest.raises(ValueError, match="exceeds discovered mesh"):
        benchmark.mesh_selection((2, 2), (8, 1))


@pytest.mark.parametrize(
    "requested,fabric_config,expected",
    [
        (None, "auto", "1d"),
        ((2, 1), "auto", "1d"),
        ((4, 1), "auto", "1d-ring"),
        ((1, 8), "auto", "1d-ring"),
        ((4, 1), "1d", "1d"),
    ],
)
def test_select_fabric_config(requested, fabric_config, expected):
    assert benchmark.select_fabric_config(requested, fabric_config) == expected


@pytest.mark.parametrize(
    "fabric_config,expected",
    [
        ("1d", benchmark.ttnn.Topology.Linear),
        ("1d-ring", benchmark.ttnn.Topology.Ring),
    ],
)
def test_native_topology(fabric_config, expected):
    assert benchmark.topology_for_fabric_config(fabric_config) == expected


def test_fabric_reliability_allowlist():
    assert set(benchmark.FABRIC_RELIABILITY_MODES) == {"relaxed", "strict"}


def test_native_subblock_validation_accepts_four_by_one():
    benchmark_case = benchmark.BenchmarkCase(
        mesh_shape=(4, 1),
        m_tiles=96,
        k_tiles_per_device=40,
        n_tiles_per_device=40,
        m_block_tiles=4,
        k_block_tiles=8,
        n_block_tiles=5,
        worker_grid=(12, 9),
        transpose=True,
    )

    assert benchmark.native_subblock(benchmark_case) == (2, 1)
    assert benchmark.validate_native_subblock(benchmark_case, (4, 1), True) == (
        4,
        1,
    )
    with pytest.raises(ValueError, match="native subblock must divide"):
        benchmark.validate_native_subblock(benchmark_case, (2, 2), True)


def test_native_case_accepts_upstream_edge_blocks():
    benchmark_case = benchmark.BenchmarkCase(
        mesh_shape=(4, 1),
        m_tiles=96,
        k_tiles_per_device=40,
        n_tiles_per_device=40,
        m_block_tiles=8,
        k_block_tiles=8,
        n_block_tiles=8,
        worker_grid=(12, 9),
        transpose=True,
    )

    assert benchmark_case.grid == (12, 9)
    assert benchmark_case.m_workers == 12
    assert benchmark_case.n_workers == 9
    with pytest.raises(ValueError, match="N block count must be divisible"):
        benchmark_case.make_ttlang_config(reuse_activation=False)
