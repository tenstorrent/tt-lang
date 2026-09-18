# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STACK_TOOL = REPO_ROOT / "scripts" / "tt-lang-emule-stack.py"
STACK_MANIFEST = REPO_ROOT / "config" / "tt-lang-emule-stack.json"


def run_stack(*arguments):
    return subprocess.run(
        [
            "python3",
            str(STACK_TOOL),
            "--manifest",
            str(STACK_MANIFEST),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def write_emulator_stack(tmp_path):
    manifest = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))
    emulator = tmp_path / "emulator"
    descriptor = emulator / manifest["target"]["cluster_descriptor"]
    descriptor.parent.mkdir(parents=True)
    descriptor.touch()
    (emulator / "tt-metal-pin.txt").write_text(
        f"# Tested Metal revision.\n{manifest['metal']['commit']}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(emulator)], check=True)
    subprocess.run(["git", "-C", str(emulator), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(emulator),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "Test emulator",
        ],
        check=True,
    )
    emulator_commit = subprocess.run(
        ["git", "-C", str(emulator), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest["emulator"]["commit"] = emulator_commit
    test_manifest = tmp_path / "stack.json"
    test_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    return emulator, test_manifest


def test_manifest_emits_exact_runtime_inputs():
    result = run_stack("emit")

    assert result.returncode == 0, result.stderr
    values = dict(line.split("\t", 1) for line in result.stdout.splitlines())
    assert (
        values["TTLANG_EMULE_STACK_MANIFEST_SHA256"]
        == hashlib.sha256(STACK_MANIFEST.read_bytes()).hexdigest()
    )
    assert values["TTLANG_EMULE_REPOSITORY"] == ""
    assert len(values["TTLANG_EMULE_COMMIT"]) == 40
    assert len(values["TTLANG_METAL_COMMIT"]) == 40
    assert values["TTLANG_EMULE_PLATFORM"] == "linux/amd64"
    assert values["TTLANG_EMULE_MESH_DEVICE"] == "P150"
    assert "@sha256:" in values["TTLANG_EMULE_BASE_IMAGE"]


def test_manifest_accepts_the_current_compiler_checkout():
    result = run_stack("validate", "--compiler-source", str(REPO_ROOT))

    assert result.returncode == 0, result.stderr
    assert "Validated tt-lang/emule stack" in result.stdout


def test_manifest_rejects_a_non_digest_base_image(tmp_path):
    manifest = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))
    manifest["runtime"]["base_image"] = "ubuntu:latest"
    invalid_manifest = tmp_path / "invalid-stack.json"
    invalid_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            "python3",
            str(STACK_TOOL),
            "--manifest",
            str(invalid_manifest),
            "emit",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "must use an exact sha256 digest" in result.stderr


def test_manifest_validates_the_emulator_metal_pin(tmp_path):
    emulator, test_manifest = write_emulator_stack(tmp_path)

    result = subprocess.run(
        [
            "python3",
            str(STACK_TOOL),
            "--manifest",
            str(test_manifest),
            "validate",
            "--emulator-source",
            str(emulator),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    (emulator / "tt-metal-pin.txt").write_text("0" * 40 + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            "python3",
            str(STACK_TOOL),
            "--manifest",
            str(test_manifest),
            "validate",
            "--emulator-source",
            str(emulator),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "emulator pins Metal" in result.stderr
