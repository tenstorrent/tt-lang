# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run-tt-lang-emule-examples.py"
MANIFEST = REPO_ROOT / "config" / "tt-lang-emule-examples.json"


def write_launcher(tmp_path):
    launcher = tmp_path / "launcher"
    launcher.write_text(
        """#!/usr/bin/env python3
import os
import pathlib
import sys

with pathlib.Path(os.environ["LAUNCH_LOG"]).open("a", encoding="utf-8") as log:
    log.write("\\t".join(sys.argv[1:]) + "\\n")
name = pathlib.Path(sys.argv[1]).stem
sys.exit(int(os.environ.get("FAIL_" + name.upper().replace("-", "_"), "0")))
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def run_examples(tmp_path, *arguments, environment=None):
    launcher = write_launcher(tmp_path)
    log = tmp_path / "launch.log"
    env = os.environ.copy()
    env["LAUNCH_LOG"] = str(log)
    if environment:
        env.update(environment)
    result = subprocess.run(
        [
            "python3",
            str(RUNNER),
            "--manifest",
            str(MANIFEST),
            "--launcher",
            str(launcher),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, lines


def test_reference_suite_covers_four_operation_classes(tmp_path):
    result, lines = run_examples(tmp_path, "--runtime-image", "runtime:tested")

    assert result.returncode == 0, result.stderr
    assert len(lines) == 4
    assert any("examples/eltwise_add.py" in line for line in lines)
    assert any("examples/single_node_matmul.py" in line for line in lines)
    assert any(
        "examples/spec/block/elementwise_broadcast_reduce.py" in line for line in lines
    )
    assert any("examples/matmul_acc.py" in line for line in lines)
    assert all("--backend\temule" in line for line in lines)
    assert all("--runtime-image\truntime:tested" in line for line in lines)


def test_named_selection_preserves_requested_order(tmp_path):
    result, lines = run_examples(tmp_path, "--example", "reduce", "--example", "add")

    assert result.returncode == 0, result.stderr
    assert "elementwise_broadcast_reduce.py" in lines[0]
    assert "eltwise_add.py" in lines[1]


def test_failure_stops_the_suite_by_default(tmp_path):
    result, lines = run_examples(tmp_path, environment={"FAIL_ELTWISE_ADD": "9"})

    assert result.returncode == 1
    assert len(lines) == 1
    assert "Failed examples: add=9" in result.stderr


def test_manifest_rejects_paths_outside_the_repository(tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["examples"][0]["path"] = "../outside.py"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(RUNNER), "--manifest", str(invalid), "--list"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must be a safe relative path" in result.stderr
