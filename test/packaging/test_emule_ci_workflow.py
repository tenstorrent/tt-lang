# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PULL = ROOT / "scripts/emule-ci/pull.sh"
IMAGE = "ghcr.io/tenstorrent/tt-lang/experiment@sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def no_real_docker(tmp_path, monkeypatch):
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\necho unexpected-docker-invocation >&2\nexit 99\n")
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])


def test_workflow_is_manual_trusted_and_bounded():
    workflow = yaml.load(
        (ROOT / ".github/workflows/tt-lang-emule-benchmark.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert list(workflow["on"]) == ["workflow_dispatch"]
    assert (
        workflow["on"]["workflow_dispatch"]["inputs"]["company_policy_approved"][
            "default"
        ]
        == "false"
    )
    serial, shards = workflow["jobs"]["serial"], workflow["jobs"]["shards"]
    assert "inputs.company_policy_approved" in serial["if"]
    assert "github.repository == 'tenstorrent/tt-lang'" in serial["if"]
    assert "refs/heads/kostas/tt-lang-emule-hosted-ci-experiment" in serial["if"]
    assert shards["strategy"]["max-parallel"] == "3"
    assert shards["strategy"]["matrix"]["shard"] == ["1", "2", "3"]
    assert "needs.serial.outputs.plan_ready" in shards["if"]
    plan = next(step for step in serial["steps"] if step.get("id") == "plan")
    assert "--public-manifest" in plan["run"]
    assert "cp " not in plan["run"]
    for job in (serial, shards):
        assert job["runs-on"] == "ubuntu-24.04"
        assert job["permissions"] == {"contents": "read", "packages": "read"}
        assert int(job["timeout-minutes"]) <= 360
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout"):
                assert step["with"]["persist-credentials"] == "false"
            if step.get("uses", "").startswith("actions/upload-artifact"):
                assert "emule-public/" in step["with"]["path"]
                assert step["with"]["retention-days"] == "1"
                assert "steps.bounded.outcome == 'success'" in step["if"]
            if "scripts/emule-ci/worker.py" in step.get("run", ""):
                assert "--disk-floor-gib 2" in step["run"]


@pytest.mark.parametrize("exit_code", [0, 7])
def test_pull_keeps_logs_private_and_preserves_failure(tmp_path, exit_code):
    docker = tmp_path / "docker"
    docker.write_text(f"#!/bin/sh\necho private-runtime-error\nexit {exit_code}\n")
    docker.chmod(0o755)
    output = tmp_path / "result"
    result = subprocess.run(
        ["bash", str(PULL), IMAGE, str(output)],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code
    assert "private-runtime-error" not in result.stdout + result.stderr
    assert "private-runtime-error" in (output / "private/pull.log").read_text()
    assert (output / "public/pull-exit-code.txt").read_text().strip() == str(exit_code)
    for path in (output / "public").iterdir():
        assert all(word.isdigit() for word in path.read_text().split())
    assert not (output / "private").stat().st_mode & 0o077


@pytest.mark.parametrize(
    "image", ["latest", "local:v1", "ghcr.io/outsider/x@sha256:" + "a" * 64]
)
def test_pull_rejects_unapproved_or_mutable_reference(tmp_path, image):
    output = tmp_path / "result"
    result = subprocess.run(
        ["bash", str(PULL), image, str(output)], capture_output=True
    )
    assert result.returncode != 0
    assert not output.exists()


def test_pull_preserves_existing_evidence(tmp_path):
    marker = tmp_path / "marker"
    marker.write_text("keep")
    result = subprocess.run(
        ["bash", str(PULL), IMAGE, str(tmp_path)], capture_output=True
    )
    assert result.returncode != 0
    assert marker.read_text() == "keep"


def test_pull_disk_floor_kills_a_process_ignoring_term(tmp_path):
    marker = tmp_path / "docker-pid"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "while True: time.sleep(0.1)\n"
    )
    docker.chmod(0o755)
    df = tmp_path / "df"
    df.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, time\n"
        "for _ in range(100):\n"
        f"    if pathlib.Path({str(marker)!r}).exists(): break\n"
        "    time.sleep(0.01)\n"
        "print('Filesystem 1024-blocks Used Available Capacity Mounted')\n"
        "print('disk 100 99 1 99% /')\n"
    )
    df.chmod(0o755)
    output = tmp_path / "result"
    result = subprocess.run(
        ["bash", str(PULL), IMAGE, str(output)], capture_output=True, timeout=10
    )
    assert result.returncode == 95
    assert (output / "public/pull-exit-code.txt").read_text().strip() == "95"
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)
