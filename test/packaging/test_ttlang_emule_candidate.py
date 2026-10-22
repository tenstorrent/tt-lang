# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STACK_TOOL = REPO_ROOT / "scripts" / "tt-lang-emule-stack.py"
MANIFEST = REPO_ROOT / "config" / "tt-lang-emule-stack.json"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tt-lang-emule-candidate.yml"


def load_workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def candidate_step(workflow, step_id):
    return next(
        step
        for step in workflow["jobs"]["candidate"]["steps"]
        if step.get("id") == step_id
    )


def make_emulator(tmp_path, metal_commit):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = tmp_path / "emulator"
    target = manifest["targets"][manifest["default_target"]]
    descriptor = source / target["cluster_descriptor"]
    descriptor.parent.mkdir(parents=True)
    descriptor.touch()
    (source / "tt-metal-pin.txt").write_text(
        f"# Exact compatible revision.\n{metal_commit}\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "Candidate emulator",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source, commit


def run_prepare(source, commit, *arguments):
    return subprocess.run(
        [
            "python3",
            str(STACK_TOOL),
            "--manifest",
            str(MANIFEST),
            "prepare",
            "--compiler-source",
            str(REPO_ROOT),
            "--emulator-source",
            str(source),
            "--emulator-commit",
            commit,
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_candidate_uses_checkout_head_and_its_exact_metal_pin(tmp_path):
    metal_commit = "a" * 40
    source, emulator_commit = make_emulator(tmp_path, metal_commit)
    output = tmp_path / "candidate.json"

    result = run_prepare(source, emulator_commit, "--output", str(output))

    assert result.returncode == 0, result.stderr
    candidate = json.loads(output.read_text(encoding="utf-8"))
    compiler_head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert candidate["compiler"]["base_commit"] == compiler_head
    assert candidate["emulator"]["commit"] == emulator_commit
    assert candidate["metal"]["commit"] == metal_commit


def test_candidate_rejects_a_symbolic_emulator_revision(tmp_path):
    source, _ = make_emulator(tmp_path, "b" * 40)

    result = run_prepare(source, "main")

    assert result.returncode == 1
    assert "must be a full lowercase commit SHA" in result.stderr


@pytest.mark.parametrize(
    "payload", ["{sha}; touch {marker}", "$(touch {marker})", "--help"]
)
def test_candidate_rejects_malicious_compiler_revisions(tmp_path, payload):
    source, emulator_commit = make_emulator(tmp_path, "b" * 40)
    marker = tmp_path / "injected-command"
    output = tmp_path / "candidate.json"
    compiler_commit = payload.format(sha="a" * 40, marker=marker)

    result = run_prepare(
        source,
        emulator_commit,
        f"--compiler-commit={compiler_commit}",
        "--output",
        str(output),
    )

    assert result.returncode == 1
    assert "compiler commit must be a full lowercase commit SHA" in result.stderr
    assert not marker.exists()
    assert not output.exists()


def test_candidate_rejects_a_checkout_at_another_commit(tmp_path):
    source, emulator_commit = make_emulator(tmp_path, "c" * 40)

    result = run_prepare(source, "d" * 40)

    assert result.returncode == 1
    assert emulator_commit in result.stderr
    assert "expected" in result.stderr


def test_candidate_rejects_dirty_pin_without_leaving_output(tmp_path):
    source, emulator_commit = make_emulator(tmp_path, "e" * 40)
    (source / "tt-metal-pin.txt").write_text("f" * 40 + "\n", encoding="utf-8")
    output = tmp_path / "candidate.json"

    result = run_prepare(source, emulator_commit, "--output", str(output))

    assert result.returncode == 1
    assert "must not contain local changes" in result.stderr
    assert not output.exists()


def test_candidate_workflow_validates_without_publishing():
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    workflow = load_workflow()
    # PyYAML reads the bare `on` key as the boolean true.
    triggers = workflow.get("on", workflow.get(True))

    assert "workflow_dispatch" in triggers
    assert "pull_request" in triggers
    assert workflow["jobs"]["candidate"]["runs-on"] == "mlir-large-runner-lang"
    assert workflow["permissions"]["packages"] == "read"
    assert "TTLANG_EMULE_SOURCE_REPOSITORY" in workflow_text
    assert "TTLANG_EMULE_SOURCE_TOKEN" in workflow_text
    assert "docker push" not in workflow_text
    assert "packages: write" not in workflow_text
    full_history_checkouts = [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("with", {}).get("fetch-depth") == 0
    ]
    assert len(full_history_checkouts) == 2

    prepare_step = candidate_step(workflow, "prepare-candidate")
    assert prepare_step["env"]["EMULATOR_COMMIT"] == "${{ inputs.emulator_commit }}"
    assert '--emulator-commit "$EMULATOR_COMMIT"' in prepare_step["run"]
    assert "inputs.emulator_commit" not in prepare_step["run"]


def test_candidate_installs_once_and_runs_reference_programs_without_rebuilding(
    tmp_path,
):
    workflow = load_workflow()
    install_step = candidate_step(workflow, "install-candidate")
    reference_step = candidate_step(workflow, "reference-suite")

    installer = tmp_path / "scripts" / "install-tt-lang-emule.sh"
    launcher = tmp_path / "bin" / "tt-lang-sim"
    for executable, operation, rebuild in (
        (installer, "install", "1"),
        (launcher, "run", "0"),
    ):
        executable.parent.mkdir()
        executable.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "{operation}:${{TTLANG_EMULE_REBUILD:-0}}:$*" >> calls\n'
            f'test "${{TTLANG_EMULE_REBUILD:-0}}" = "{rebuild}"\n',
            encoding="utf-8",
        )
        executable.chmod(0o755)

    for step in (install_step, reference_step):
        environment = os.environ.copy()
        environment.pop("TTLANG_EMULE_REBUILD", None)
        for key, value in step["env"].items():
            environment[key] = value.replace("${{ github.run_id }}", "12345")
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    assert (tmp_path / "calls").read_text(encoding="utf-8").splitlines() == [
        "install:1:",
        "run:0:--backend=emule examples/compiler_only_external_call.py",
        "run:0:--backend=emule examples/eltwise_add.py",
        "run:0:--backend=emule examples/single_node_matmul.py",
        "run:0:--backend=emule examples/broadcast.py",
        "run:0:--backend=emule examples/matmul_acc.py",
    ]
