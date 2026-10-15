# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import re
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow_definition():
    return yaml.load(
        (ROOT / ".github/workflows/tt-lang-emule-benchmark.yml").read_text(),
        Loader=yaml.BaseLoader,
    )


def workflow_step(name):
    return next(
        step
        for step in workflow_definition()["jobs"]["serial"]["steps"]
        if step.get("name") == name
    )


@pytest.mark.parametrize(
    "overrides,allowed",
    [
        ({}, True),
        ({"APPROVED_SHA": ""}, False),
        ({"APPROVED_SHA": "c" * 40}, False),
        ({"SELECTED_SHA": "main", "APPROVED_SHA": "main"}, False),
        ({"SOURCE_REPOSITORY": ""}, False),
        ({"SOURCE_REPOSITORY": "outsider/emulator"}, False),
        ({"SOURCE_REPOSITORY": "tenstorrent/runtime\ninjected"}, False),
        ({"SOURCE_REPOSITORY": "tenstorrent/runtime;id"}, False),
        ({"SOURCE_REPOSITORY": "https://github.com/tenstorrent/runtime"}, False),
    ],
)
def test_approval_gate_binds_reviewed_candidate_and_company_source(overrides, allowed):
    authorize = workflow_definition()["jobs"]["authorize"]
    assert authorize["permissions"] == {}
    assert len(authorize["steps"]) == 1
    step = authorize["steps"][0]
    assert "uses" not in step
    assert step["env"] == {
        "SELECTED_SHA": "${{ github.sha }}",
        "APPROVED_SHA": "${{ vars.TTLANG_EMULE_APPROVED_SHA }}",
        "SOURCE_REPOSITORY": "${{ vars.TTLANG_EMULE_SOURCE_REPOSITORY }}",
    }
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        env={
            **os.environ,
            "SELECTED_SHA": "b" * 40,
            "APPROVED_SHA": "b" * 40,
            "SOURCE_REPOSITORY": "tenstorrent/runtime",
            **overrides,
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is allowed


@pytest.fixture(autouse=True)
def no_real_docker(tmp_path, monkeypatch):
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\necho unexpected-docker-invocation >&2\nexit 99\n")
    docker.chmod(0o755)
    df = tmp_path / "df"
    df.write_text(
        "#!/bin/sh\n"
        "echo 'Filesystem 1024-blocks Used Available Capacity Mounted'\n"
        "echo 'disk 20000000 1 19999999 1% /'\n"
    )
    df.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])


def test_workflow_is_one_manual_trusted_runner_local_job():
    workflow = workflow_definition()
    assert list(workflow["on"]) == ["workflow_dispatch"]
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"owner_approval_recorded"}
    assert inputs["owner_approval_recorded"]["default"] == "false"
    assert workflow["permissions"] == {}
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    assert set(workflow["jobs"]) == {"authorize", "serial"}
    authorize = workflow["jobs"]["authorize"]
    assert "inputs.owner_approval_recorded" in authorize["if"]
    assert "github.repository == 'tenstorrent/tt-lang'" in authorize["if"]
    assert "refs/heads/kostas/tt-lang-emule-hosted-ci-experiment" in authorize["if"]
    job = workflow["jobs"]["serial"]
    assert job["needs"] == "authorize"
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["permissions"] == {"contents": "read"}
    assert int(job["timeout-minutes"]) <= 360
    assert job["defaults"]["run"] == {"shell": "bash", "working-directory": "compiler"}
    assert "strategy" not in job
    steps = job["steps"]
    checkouts = [step for step in steps if "uses" in step]
    assert len(checkouts) == 2
    public, private = checkouts
    for step in checkouts:
        assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", step["uses"])
        assert step["with"]["persist-credentials"] == "false"
    assert public["with"]["ref"] == "${{ github.sha }}"
    assert public["with"]["fetch-depth"] == "0"
    assert public["with"]["path"] == "compiler"
    assert private["with"] == {
        "repository": "${{ vars.TTLANG_EMULE_SOURCE_REPOSITORY }}",
        "ref": "${{ steps.stack.outputs.emulator_sha }}",
        "token": "${{ secrets.TTLANG_EMULE_SOURCE_READ_TOKEN }}",
        "path": "private-runtime",
        "fetch-depth": "1",
        "persist-credentials": "false",
        "submodules": "false",
        "lfs": "false",
    }
    verify = workflow_step(
        "Verify sources and absence of retained checkout credentials"
    )
    build = workflow_step("Measure the cold runner-local runtime build")
    measure = workflow_step(
        "Run the complete compiler test inventory serially with fresh caches"
    )
    report = workflow_step("Report the public per-suite summary")
    assert (
        steps.index(private)
        < steps.index(verify)
        < steps.index(build)
        < steps.index(measure)
        < steps.index(report)
    )
    assert report["if"] == "always()"
    assert "--report-summary" in report["run"]
    assert '"$GITHUB_STEP_SUMMARY"' in report["run"]
    assert "emule-serial/summary.json" in report["run"]
    assert "private" not in report["run"]
    assert build["env"]["TTLANG_EMULE_BUILD_ONLY"] == "1"
    assert build["env"]["TTLANG_EMULE_REBUILD"] == "1"
    assert (
        build["env"]["TTLANG_EMULE_RUNTIME_SOURCE_DIR"]
        == "${{ github.workspace }}/private-runtime"
    )
    assert build["env"]["DOCKER_BUILD_RECORD_UPLOAD"] == "false"
    assert build["env"]["DOCKER_BUILD_SUMMARY"] == "false"
    assert int(build["timeout-minutes"]) <= 120
    assert "--kill-after=30s 6900" in build["run"]
    assert "scripts/tt-lang-emule-container.sh" in build["run"]
    assert "--disk-floor-gib 2" in measure["run"]
    assert "--manifest" not in measure["run"]
    assert "--plan" not in measure["run"]
    assert "continue-on-error" not in measure
    raw = (ROOT / ".github/workflows/tt-lang-emule-benchmark.yml").read_text()
    assert raw.count("secrets.") == 1
    for forbidden in (
        "upload-artifact",
        "download-artifact",
        "actions/cache",
        "cache-to",
        "cache-from",
        "docker push",
        "docker/login",
        "packages:",
        "pull.sh",
        "TTLANG_EMULE_APPROVED_IMAGE",
        "TTLANG_EMULE_APPROVED_SHARDS",
    ):
        assert forbidden not in raw


def test_stack_step_reads_exact_manifest_pin(tmp_path):
    # CI checks out shallowly, so the real baseline may be absent; validate
    # against a fixture history whose baseline is an ancestor of HEAD.
    repo = tmp_path / "compiler"
    git = [
        "git",
        "-C",
        str(repo),
        "-c",
        "user.name=tt-lang tests",
        "-c",
        "user.email=tt-lang-tests@example.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    manifest = json.loads((ROOT / "config/tt-lang-emule-stack.json").read_text())
    (repo / "scripts").mkdir(parents=True)
    (repo / "config").mkdir()
    (repo / "scripts/tt-lang-emule-stack.py").write_text(
        (ROOT / "scripts/tt-lang-emule-stack.py").read_text()
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "add", "scripts"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "baseline"], check=True)
    baseline = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest["compiler"]["base_commit"] = baseline
    (repo / "config/tt-lang-emule-stack.json").write_text(json.dumps(manifest))
    subprocess.run([*git, "add", "config"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "manifest"], check=True)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            workflow_step("Read the pinned emulator revision")["run"],
        ],
        cwd=repo,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    pin = manifest["emulator"]["commit"]
    assert output.read_text() == f"emulator_sha={pin}\n"


@pytest.mark.parametrize(
    "key",
    [
        None,
        "http.https://github.com/.extraheader",
        "credential.helper",
        "core.sshCommand",
        "include.path",
    ],
)
def test_retained_checkout_credentials_stop_before_source_execution(tmp_path, key):
    for name in ("compiler", "private-runtime"):
        subprocess.run(["git", "init", "-q", str(tmp_path / name)], check=True)
    if key:
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path / "private-runtime"),
                "config",
                key,
                "AUTH_SENTINEL",
            ],
            check=True,
        )
    marker = tmp_path / "validated"
    python = tmp_path / "python3"
    python.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    python.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            workflow_step(
                "Verify sources and absence of retained checkout credentials"
            )["run"],
        ],
        env={**os.environ, "GITHUB_WORKSPACE": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (key is None)
    assert marker.exists() == (key is None)
    assert "AUTH_SENTINEL" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "available_kib,exit_code", [(1, 0), (3000000, 0), (3000000, 7)]
)
def test_build_preserves_failures_and_checks_disk_before_launch(
    tmp_path, available_kib, exit_code
):
    marker = tmp_path / "build-started"
    timeout = tmp_path / "timeout"
    timeout.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit {exit_code}\n")
    timeout.chmod(0o755)
    df = tmp_path / "df"
    df.write_text(
        "#!/bin/sh\n"
        "echo 'Filesystem 1024-blocks Used Available Capacity Mounted'\n"
        f"echo 'disk 4000000 1 {available_kib} 1% /'\n"
    )
    df.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            workflow_step("Measure the cold runner-local runtime build")["run"],
        ],
        env={**os.environ, "RUNNER_TEMP": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    expected = exit_code if available_kib >= 2097152 else 1
    assert result.returncode == expected
    assert marker.exists() == (available_kib >= 2097152)
    assert f"Runtime build: exit={expected} seconds=" in result.stdout


@pytest.mark.parametrize("present", [False, True])
def test_report_step_publishes_only_the_validated_summary(tmp_path, present):
    runner_temp = tmp_path / "runner"
    (runner_temp / "emule-serial").mkdir(parents=True)
    step_summary = tmp_path / "step-summary.md"
    if present:
        (runner_temp / "emule-serial/summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "publication_blocked": True,
                    "status": "failed",
                    "stages": {"preflight": {"exit_code": 22, "seconds": 3.5}},
                    "failure_reason": "range_lockstep_api_missing",
                    "samples": [{"private": "/private/host/path"}],
                }
            )
        )
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            workflow_step("Report the public per-suite summary")["run"],
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_STEP_SUMMARY": str(step_summary),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    published = step_summary.read_text()
    assert published == result.stdout
    if present:
        assert "Preflight failure: `range_lockstep_api_missing`" in published
        assert "Stage `preflight`: exit 22, 4 s" in published
        assert "no validated summary" in published
    else:
        assert "produced no summary" in published
    assert "/private" not in published


def test_report_step_publishes_nothing_from_an_invalid_summary(tmp_path):
    runner_temp = tmp_path / "runner"
    (runner_temp / "emule-serial").mkdir(parents=True)
    step_summary = tmp_path / "step-summary.md"
    (runner_temp / "emule-serial/summary.json").write_text(
        json.dumps({"schema_version": 1, "status": "Traceback /private/path"})
    )
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            workflow_step("Report the public per-suite summary")["run"],
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_STEP_SUMMARY": str(step_summary),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert not step_summary.exists() or step_summary.read_text() == ""
    assert "/private/path" not in result.stdout + result.stderr


def test_inline_workflow_shell_syntax():
    for job in workflow_definition()["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
