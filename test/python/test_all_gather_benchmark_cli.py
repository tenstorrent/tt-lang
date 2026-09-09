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
