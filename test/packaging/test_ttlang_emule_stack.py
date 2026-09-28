# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STACK_TOOL = REPO_ROOT / "scripts" / "tt-lang-emule-stack.py"
STACK_MANIFEST = REPO_ROOT / "config" / "tt-lang-emule-stack.json"
TARGET_PROFILES = [
    (
        "p150",
        "blackhole-p150",
        "cluster_descriptors/blackhole_P150_unharvested.yaml",
        "P150",
    ),
    ("p100", "blackhole-p100", "cluster_descriptors/blackhole_P100.yaml", "P100"),
    (
        "p150-harvested",
        "blackhole-p150-harvested",
        "cluster_descriptors/blackhole_P150.yaml",
        "P150",
    ),
    (
        "p300",
        "blackhole-p300",
        "cluster_descriptors/blackhole_P300_both_mmio.yaml",
        "P300",
    ),
    (
        "p150x4",
        "blackhole-p150x4",
        "cluster_descriptors/blackhole_4xP150.yaml",
        "P150x4",
    ),
    (
        "p150x8",
        "blackhole-p150x8",
        "cluster_descriptors/blackhole_8xP150.yaml",
        "P150x8",
    ),
    (
        "p150x8-unharvested",
        "blackhole-p150x8-unharvested",
        "cluster_descriptors/blackhole_8xP150_unharvested.yaml",
        "P150x8",
    ),
    (
        "galaxy",
        "blackhole-galaxy",
        "cluster_descriptors/blackhole_galaxy.yaml",
        "BHGLX",
    ),
    ("n150", "wormhole-n150", "cluster_descriptors/wormhole_N150.yaml", "N150"),
    ("n300", "wormhole-n300", "cluster_descriptors/wormhole_N300.yaml", "N300"),
    ("q1", "quasar-q1", "cluster_descriptors/quasar_Q1.yaml", ""),
]


def run_stack(*arguments, manifest=STACK_MANIFEST):
    return subprocess.run(
        [
            "python3",
            str(STACK_TOOL),
            "--manifest",
            str(manifest),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def write_emulator_stack(tmp_path):
    manifest = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))
    emulator = tmp_path / "emulator"
    for target in manifest["targets"].values():
        descriptor = emulator / target["cluster_descriptor"]
        descriptor.parent.mkdir(parents=True, exist_ok=True)
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
    assert "TTLANG_EMULE_REPOSITORY" not in values
    assert "TTLANG_EMULE_PLATFORM" not in values
    assert "TTLANG_EMULE_ALLOCATOR_MODE" not in values
    assert len(values["TTLANG_EMULE_COMMIT"]) == 40
    assert len(values["TTLANG_METAL_COMMIT"]) == 40
    assert values["TTLANG_EMULE_TARGET_ID"] == "p150"
    assert values["TTLANG_EMULE_TARGET"] == "blackhole-p150"
    assert values["TTLANG_EMULE_MESH_DEVICE"] == "P150"
    assert "@sha256:" in values["TTLANG_EMULE_BASE_IMAGE"]


def test_manifest_contains_all_supported_targets():
    manifest = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))

    assert set(manifest["targets"]) == {profile[0] for profile in TARGET_PROFILES}
    assert "mesh_device" not in manifest["targets"]["q1"]


@pytest.mark.parametrize("target, name, descriptor, mesh", TARGET_PROFILES)
def test_manifest_emits_selected_target(target, name, descriptor, mesh):
    result = run_stack("--target", target, "emit")

    assert result.returncode == 0, result.stderr
    values = dict(line.split("\t", 1) for line in result.stdout.splitlines())
    assert values["TTLANG_EMULE_TARGET_ID"] == target
    assert values["TTLANG_EMULE_TARGET"] == name
    assert values["TTLANG_EMULE_CLUSTER_DESCRIPTOR"] == descriptor
    assert values["TTLANG_EMULE_MESH_DEVICE"] == mesh
    default_result = run_stack("emit")
    assert default_result.returncode == 0, default_result.stderr
    defaults = dict(line.split("\t", 1) for line in default_result.stdout.splitlines())
    target_keys = {
        "TTLANG_EMULE_TARGET_ID",
        "TTLANG_EMULE_TARGET",
        "TTLANG_EMULE_CLUSTER_DESCRIPTOR",
        "TTLANG_EMULE_MESH_DEVICE",
    }
    assert {key: value for key, value in values.items() if key not in target_keys} == {
        key: value for key, value in defaults.items() if key not in target_keys
    }


@pytest.mark.parametrize("target", ["unknown", ""])
@pytest.mark.parametrize("command", ["emit", "validate"])
def test_manifest_rejects_unknown_and_empty_selected_targets(target, command):
    result = run_stack("--target", target, command)

    assert result.returncode == 1
    assert "unknown target" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("default_target", ["unknown", ""])
def test_manifest_rejects_invalid_default_target(tmp_path, default_target):
    manifest = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))
    manifest["default_target"] = default_target
    invalid_manifest = tmp_path / "invalid-stack.json"
    invalid_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_stack("emit", manifest=invalid_manifest)

    assert result.returncode == 1
    assert "default_target" in result.stderr


@pytest.mark.parametrize("profile", [{}, None])
def test_manifest_rejects_empty_target_profiles(tmp_path, profile):
    manifest = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))
    manifest["targets"]["p100"] = profile
    invalid_manifest = tmp_path / "invalid-stack.json"
    invalid_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_stack("emit", manifest=invalid_manifest)

    assert result.returncode == 1
    assert "targets.p100" in result.stderr


@pytest.mark.parametrize("mesh", ["", None, 1, False, [], {}])
def test_manifest_rejects_invalid_present_mesh_device(tmp_path, mesh):
    manifest = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))
    manifest["targets"]["q1"]["mesh_device"] = mesh
    invalid_manifest = tmp_path / "invalid-stack.json"
    invalid_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_stack("emit", manifest=invalid_manifest)

    assert result.returncode == 1
    assert "targets.q1.mesh_device must be a non-empty string" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "descriptor", ["/tmp/cluster.yaml", "../cluster.yaml", "a/../b"]
)
def test_manifest_rejects_unsafe_descriptor_paths_in_any_profile(tmp_path, descriptor):
    manifest = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))
    manifest["targets"]["p100"]["cluster_descriptor"] = descriptor
    invalid_manifest = tmp_path / "invalid-stack.json"
    invalid_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_stack("emit", manifest=invalid_manifest)

    assert result.returncode == 1
    assert (
        "targets.p100.cluster_descriptor must be a safe relative path" in result.stderr
    )


@pytest.mark.parametrize(
    "target, descriptor",
    [(target, descriptor) for target, _, descriptor, _ in TARGET_PROFILES],
)
def test_manifest_validates_the_selected_emulator_descriptor(
    tmp_path, target, descriptor
):
    emulator, test_manifest = write_emulator_stack(tmp_path)
    result = run_stack(
        "--target",
        target,
        "validate",
        "--emulator-source",
        str(emulator),
        manifest=test_manifest,
    )
    assert result.returncode == 0, result.stderr

    (emulator / descriptor).unlink()
    other_result = run_stack(
        "--target",
        "p100" if target == "p150" else "p150",
        "validate",
        "--emulator-source",
        str(emulator),
        manifest=test_manifest,
    )
    assert other_result.returncode == 0, other_result.stderr
    result = run_stack(
        "--target",
        target,
        "validate",
        "--emulator-source",
        str(emulator),
        manifest=test_manifest,
    )
    assert result.returncode == 1
    assert "emulator target descriptor is missing" in result.stderr
    assert descriptor in result.stderr


def test_manifest_accepts_a_compiler_descendant_of_its_baseline(tmp_path):
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    subprocess.run(["git", "init", "-q", str(compiler)], check=True)
    (compiler / "tracked").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(compiler), "add", "tracked"], check=True)
    commit = [
        "git",
        "-C",
        str(compiler),
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-q",
        "-m",
    ]
    subprocess.run([*commit, "Baseline"], check=True)
    baseline = subprocess.run(
        ["git", "-C", str(compiler), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (compiler / "tracked").write_text("descendant\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(compiler), "add", "tracked"], check=True)
    subprocess.run([*commit, "Descendant"], check=True)

    manifest = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))
    manifest["compiler"]["base_commit"] = baseline
    test_manifest = tmp_path / "stack.json"
    test_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_stack(
        "validate",
        "--compiler-source",
        str(compiler),
        manifest=test_manifest,
    )

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
