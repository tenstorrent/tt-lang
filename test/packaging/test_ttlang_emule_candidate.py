# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE = REPO_ROOT / "scripts" / "prepare-tt-lang-emule-candidate.py"
MANIFEST = REPO_ROOT / "config" / "tt-lang-emule-stack.json"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tt-lang-emule-candidate.yml"


def make_emulator(tmp_path, metal_commit):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = tmp_path / "emulator"
    descriptor = source / manifest["target"]["cluster_descriptor"]
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
            str(PREPARE),
            "--manifest",
            str(MANIFEST),
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
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" in workflow
    assert "runs-on: mlir-large-runner-lang" in workflow
    assert "TTLANG_EMULE_SOURCE_REPOSITORY" in workflow
    assert "TTLANG_EMULE_SOURCE_TOKEN" in workflow
    assert "packages: read" in workflow
    assert "docker push" not in workflow
    assert "packages: write" not in workflow
