# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run-tt-lang-emule-tests.py"
PLAN = REPO_ROOT / "config" / "tt-lang-emule-test-plan.json"


def run_plan(*arguments, plan=PLAN, environment=None):
    env = os.environ.copy()
    if environment:
        env.update(environment)
    return subprocess.run(
        ["python3", str(RUNNER), "--plan", str(plan), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_inventory_separates_compiler_runtime_and_unqualified_execution():
    result = run_plan("--list")

    assert result.returncode == 0, result.stderr
    assert "compiler-core\tcompiler\tcompiler_only\trunnable" in result.stdout
    assert "emule-reference\truntime\temulated_execution\trunnable" in result.stdout
    assert "python-pytest-device-execution\tmixed\tinventory_only" in result.stdout
    assert "me2e-device-execution\tmixed\tinventory_only" in result.stdout


def test_default_dry_run_uses_compiler_targets_and_reference_suite(tmp_path):
    result = run_plan(
        "--dry-run", "--build-dir", str(tmp_path), "--runtime-image", "runtime:tested"
    )

    assert result.returncode == 0, result.stderr
    assert f"cmake --build {tmp_path}" in result.stdout
    assert "--target check-ttlang" in result.stdout
    assert "--target check-ttlang-python-lit" in result.stdout
    assert (
        "run-tt-lang-emule-examples.py --runtime-image runtime:tested" in result.stdout
    )
    assert "python-pytest-device-execution" not in result.stdout


def test_inventory_only_phase_is_rejected():
    result = run_plan("--phase", "me2e-device-execution", "--dry-run")

    assert result.returncode == 2
    assert "inventory-only phases cannot run" in result.stderr


def test_plan_rejects_runnable_phase_without_a_command(tmp_path):
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    del plan["phases"][0]["command"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(plan), encoding="utf-8")

    result = run_plan("--list", plan=invalid)

    assert result.returncode == 2
    assert "command must be a non-empty string list" in result.stderr
