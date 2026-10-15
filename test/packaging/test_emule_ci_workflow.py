# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


WORKFLOW = ".github/workflows/tt-lang-emule-inventory.yml"


def workflow_definition():
    return yaml.load((ROOT / WORKFLOW).read_text(), Loader=yaml.BaseLoader)


def workflow_step(name):
    return next(
        step
        for step in workflow_definition()["jobs"]["inventory"]["steps"]
        if step.get("name") == name
    )


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


STEP_NAMES = [
    "Checkout the compiler revision",
    "Read the pinned emulator revision",
    "Free runner disk space",
    "Validate the emulator source configuration",
    "Snapshot the runner temporary directory",
    "Checkout the pinned private source without retaining credentials",
    "Verify sources and absence of retained checkout credentials",
    "Build the runner-local runtime image",
    "Run the compiler test inventory",
    "Report the public per-suite summary",
]
SECRET_STEPS = {
    "Validate the emulator source configuration",
    "Checkout the pinned private source without retaining credentials",
}


def test_workflow_runs_only_merged_main_code_in_one_job():
    workflow = workflow_definition()
    assert workflow["on"] == {
        "push": {"branches": ["main"]},
        "workflow_dispatch": {},
    }
    # PR-authored code must never run next to the private source or credential.
    assert not {
        "pull_request",
        "pull_request_target",
        "workflow_call",
        "workflow_run",
        "merge_group",
    } & set(workflow["on"])
    assert workflow["permissions"] == {}
    assert workflow["concurrency"] == {
        "group": "tt-lang-emule-inventory-${{ github.ref }}",
        "cancel-in-progress": "false",
    }
    assert set(workflow["jobs"]) == {"inventory"}
    job = workflow["jobs"]["inventory"]
    assert job["if"] == (
        "github.repository == 'tenstorrent/tt-lang' && "
        "github.ref == 'refs/heads/main'"
    )
    # A literal environment: only it, restricted to main, holds the secrets.
    assert job["environment"] == "tt-lang-emule"
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["permissions"] == {"contents": "read"}
    assert job["defaults"]["run"] == {"shell": "bash", "working-directory": "compiler"}
    assert not {"needs", "strategy", "continue-on-error", "services"} & set(job)
    steps = job["steps"]
    assert [step["name"] for step in steps] == STEP_NAMES
    assert steps[-1]["if"] == "always()"
    assert all("continue-on-error" not in step for step in steps)
    assert all("if" not in step for step in steps[:-1])
    # Every step is bounded, and the bounds fit the job limit.
    assert sum(int(step["timeout-minutes"]) for step in steps) <= int(
        job["timeout-minutes"]
    )
    assert int(job["timeout-minutes"]) == 360


def test_secrets_reach_only_validation_and_private_checkout():
    raw = (ROOT / WORKFLOW).read_text()
    secrets = {
        "secrets.TTLANG_EMULE_SOURCE_REPOSITORY",
        "secrets.TTLANG_EMULE_SOURCE_SSH_KEY",
    }
    found = re.findall(r"secrets\.[A-Za-z0-9_]+", raw)
    assert set(found) == secrets and len(found) == 2 * len(secrets)
    for step in workflow_definition()["jobs"]["inventory"]["steps"]:
        mentions = "secrets." in yaml.safe_dump(step)
        assert mentions == (step["name"] in SECRET_STEPS), step["name"]
    for forbidden in (
        "vars.",
        "github.token",
        "GITHUB_TOKEN",
        "upload-artifact",
        "download-artifact",
        "actions/cache",
        "cache-to",
        "cache-from",
        "docker push",
        "docker login",
        "docker/login",
        "packages:",
        "pull.sh",
        "owner_approval_recorded",
        "TTLANG_EMULE_APPROVED_SHA",
        "TTLANG_EMULE_SOURCE_READ_TOKEN",
        "TTLANG_EMULE_APPROVED_IMAGE",
        "TTLANG_EMULE_APPROVED_SHARDS",
    ):
        assert forbidden not in raw, forbidden


def test_checkouts_are_pinned_and_keep_no_credentials():
    steps = workflow_definition()["jobs"]["inventory"]["steps"]
    actions = [step for step in steps if "uses" in step]
    assert len(actions) == 2
    public, private = actions
    for step in actions:
        assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", step["uses"])
        assert step["with"]["persist-credentials"] == "false"
    assert public["with"] == {
        "ref": "${{ github.sha }}",
        "path": "compiler",
        "fetch-depth": "0",
        "persist-credentials": "false",
    }
    assert private["with"] == {
        "repository": "${{ secrets.TTLANG_EMULE_SOURCE_REPOSITORY }}",
        "ref": "${{ steps.stack.outputs.emulator_sha }}",
        "ssh-key": "${{ secrets.TTLANG_EMULE_SOURCE_SSH_KEY }}",
        "path": "private-runtime",
        "fetch-depth": "1",
        "persist-credentials": "false",
        "submodules": "false",
        "lfs": "false",
    }
    validate = workflow_step("Validate the emulator source configuration")
    assert validate["env"] == {
        "SOURCE_SSH_KEY": "${{ secrets.TTLANG_EMULE_SOURCE_SSH_KEY }}",
        "SOURCE_REPOSITORY": "${{ secrets.TTLANG_EMULE_SOURCE_REPOSITORY }}",
    }


@pytest.mark.parametrize(
    "overrides,allowed",
    [
        ({}, True),
        ({"SOURCE_SSH_KEY": ""}, False),
        ({"SOURCE_REPOSITORY": ""}, False),
        ({"SOURCE_REPOSITORY": "outsider/emulator"}, False),
        ({"SOURCE_REPOSITORY": "tenstorrent/"}, False),
        ({"SOURCE_REPOSITORY": "tenstorrent/runtime\ninjected"}, False),
        ({"SOURCE_REPOSITORY": "tenstorrent/runtime\n"}, False),
        ({"SOURCE_REPOSITORY": "tenstorrent/runtime;id"}, False),
        ({"SOURCE_REPOSITORY": "https://github.com/tenstorrent/runtime"}, False),
    ],
)
def test_source_configuration_is_validated_without_echoing_it(overrides, allowed):
    env = {
        "SOURCE_SSH_KEY": "KEY_SENTINEL",
        "SOURCE_REPOSITORY": "tenstorrent/runtime",
        **overrides,
    }
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            workflow_step("Validate the emulator source configuration")["run"],
        ],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is allowed
    output = result.stdout + result.stderr
    assert "KEY_SENTINEL" not in output
    assert "runtime" not in output and "outsider" not in output


def test_worker_accepts_the_hosted_invocation():
    run = workflow_step("Run the compiler test inventory")["run"]
    command = shlex.split(run.replace("\\\n", " "))
    assert command[:2] == ["python3", "scripts/emule-ci/worker.py"]
    spec = importlib.util.spec_from_file_location(
        "emule_ci_worker", ROOT / "scripts/emule-ci/worker.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    options = worker.arguments(command[2:])
    assert options.image == "$TTLANG_EMULE_IMAGE"
    assert options.worker == "inventory"
    assert str(options.output) == "$RUNNER_TEMP/emule-inventory"
    assert (options.cpus, options.memory_gib, options.disk_floor_gib) == (4, 12, 2)
    # The worker deadline ends before the step limit.
    assert options.timeout_seconds == 13200
    assert options.timeout_seconds < 60 * int(
        workflow_step("Run the compiler test inventory")["timeout-minutes"]
    )
    build = workflow_step("Build the runner-local runtime image")
    assert build["env"] == {
        "TTLANG_EMULE_BUILD_ONLY": "1",
        "TTLANG_EMULE_REBUILD": "1",
        "TTLANG_EMULE_RUNTIME_SOURCE_DIR": "${{ github.workspace }}/private-runtime",
        "DOCKER_BUILD_SUMMARY": "false",
        "DOCKER_BUILD_RECORD_UPLOAD": "false",
    }
    assert "--kill-after=30s 6900" in build["run"]
    assert 6900 + 30 < 60 * int(build["timeout-minutes"])


def test_stack_step_reads_exact_manifest_pin(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            workflow_step("Read the pinned emulator revision")["run"],
        ],
        cwd=ROOT,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((ROOT / "config/tt-lang-emule-stack.json").read_text())
    assert output.read_text() == f"emulator_sha={manifest['emulator']['commit']}\n"


def run_step(name, env):
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", workflow_step(name)["run"]],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "key",
    [
        None,
        "http.https://github.com/.extraheader",
        "credential.helper",
        "core.sshCommand",
        "include.path",
        "includeIf.gitdir:/runner/.path",
        "runner-temp-leftover",
    ],
)
def test_retained_checkout_credentials_stop_before_source_execution(tmp_path, key):
    workspace = tmp_path / "workspace"
    runner_temp = tmp_path / "runner-temp"
    (runner_temp / "_runner_file_commands").mkdir(parents=True)
    for name in ("compiler", "private-runtime"):
        subprocess.run(["git", "init", "-q", str(workspace / name)], check=True)
    env = {"GITHUB_WORKSPACE": str(workspace), "RUNNER_TEMP": str(runner_temp)}
    snapshot = run_step("Snapshot the runner temporary directory", env)
    assert snapshot.returncode == 0, snapshot.stderr
    # Files written after the snapshot inside existing directories are not leftovers.
    (runner_temp / "_runner_file_commands/set_output_1").write_text("")
    if key == "runner-temp-leftover":
        (runner_temp / "0f1e2d3c_known_hosts").write_text("AUTH_SENTINEL")
    elif key:
        subprocess.run(
            ["git", "-C", str(workspace / "private-runtime"), "config", key]
            + ["AUTH_SENTINEL"],
            check=True,
        )
    marker = tmp_path / "validated"
    python = tmp_path / "python3"
    python.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    python.chmod(0o755)
    result = run_step(
        "Verify sources and absence of retained checkout credentials", env
    )
    assert (result.returncode == 0) == (key is None), result.stderr
    assert marker.exists() == (key is None)
    output = result.stdout + result.stderr
    assert "AUTH_SENTINEL" not in output and "known_hosts" not in output


def test_missing_runner_temp_snapshot_fails_closed(tmp_path):
    workspace = tmp_path / "workspace"
    for name in ("compiler", "private-runtime"):
        subprocess.run(["git", "init", "-q", str(workspace / name)], check=True)
    marker = tmp_path / "validated"
    python = tmp_path / "python3"
    python.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    python.chmod(0o755)
    result = run_step(
        "Verify sources and absence of retained checkout credentials",
        {"GITHUB_WORKSPACE": str(workspace), "RUNNER_TEMP": str(tmp_path)},
    )
    assert result.returncode != 0
    assert not marker.exists()


@pytest.mark.parametrize(
    "available_kib,exit_code", [(1, 0), (3000000, 0), (3000000, 7)]
)
def test_build_preserves_failures_and_checks_disk_before_launch(
    tmp_path, available_kib, exit_code
):
    marker = tmp_path / "build-started"
    timeout = tmp_path / "timeout"
    timeout.write_text(
        f"#!/bin/sh\ntouch '{marker}'\n"
        "echo BUILD_SENTINEL\necho BUILD_SENTINEL >&2\n"
        f"exit {exit_code}\n"
    )
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
            workflow_step("Build the runner-local runtime image")["run"],
        ],
        env={**os.environ, "RUNNER_TEMP": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    expected = exit_code if available_kib >= 2097152 else 1
    assert result.returncode == expected
    assert marker.exists() == (available_kib >= 2097152)
    assert f"Runtime build: exit={expected} seconds=" in result.stdout
    # The build output stays in a runner-local file.
    assert "BUILD_SENTINEL" not in result.stdout + result.stderr
    log = tmp_path / "runtime-build.log"
    assert (log.exists() and log.read_text()) == (
        available_kib >= 2097152 and "BUILD_SENTINEL\nBUILD_SENTINEL\n"
    )


PREFLIGHT_FAILURE = {
    "schema_version": 1,
    "status": "failed",
    "stages": {"preflight": {"exit_code": 22, "seconds": 3.5}},
    "failure_reason": "range_lockstep_api_missing",
    "samples": [{"private": "/private/host/path"}],
}


@pytest.mark.parametrize(
    "summary,returncode",
    [
        (None, 0),
        (PREFLIGHT_FAILURE, 0),
        ({"schema_version": 1, "status": "Traceback /private/path"}, 2),
    ],
)
def test_report_step_publishes_only_the_validated_summary(
    tmp_path, summary, returncode
):
    runner_temp = tmp_path / "runner"
    (runner_temp / "emule-inventory").mkdir(parents=True)
    step_summary = tmp_path / "step-summary.md"
    if summary:
        (runner_temp / "emule-inventory/summary.json").write_text(json.dumps(summary))
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
    assert result.returncode == returncode, result.stderr
    published = step_summary.read_text() if step_summary.exists() else ""
    assert "/private" not in published + result.stdout + result.stderr
    if returncode:
        # An invalid summary appends nothing; pipefail fails the step.
        assert published == ""
        return
    assert published == result.stdout
    if summary:
        assert "Preflight failure: `range_lockstep_api_missing`" in published
        assert "Stage `preflight`: exit 22, 4 s" in published
        assert "no validated summary" in published
        assert "| Suite |" not in published
    else:
        assert "produced no summary" in published


def test_inline_workflow_shell_syntax():
    for job in workflow_definition()["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
