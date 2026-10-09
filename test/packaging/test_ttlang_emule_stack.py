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


def run_stack(*arguments, manifest=STACK_MANIFEST, input_text=None):
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
        input=input_text,
    )


@pytest.fixture
def image_inspection():
    stack = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))
    return [
        {
            "Id": "sha256:" + "a" * 64,
            "Config": {
                "Labels": {
                    "org.opencontainers.image.source": stack["compiler"]["repository"],
                    "io.tenstorrent.tt-lang.compiler.commit": stack["compiler"][
                        "base_commit"
                    ],
                    "io.tenstorrent.tt-lang.emule.commit": stack["emulator"]["commit"],
                    "io.tenstorrent.tt-lang.metal.repository": stack["metal"][
                        "repository"
                    ],
                    "io.tenstorrent.tt-lang.metal.commit": stack["metal"]["commit"],
                    "io.tenstorrent.tt-lang.runtime.base-image": stack["runtime"][
                        "base_image"
                    ],
                    "io.tenstorrent.tt-lang.runtime.manifest-sha256": hashlib.sha256(
                        STACK_MANIFEST.read_bytes()
                    ).hexdigest(),
                    "io.tenstorrent.tt-lang.runtime.platform": "linux/amd64",
                    "io.tenstorrent.tt-lang.target.name": stack["target"]["name"],
                    "io.tenstorrent.tt-lang.target.cluster-descriptor": stack["target"][
                        "cluster_descriptor"
                    ],
                    "io.tenstorrent.tt-lang.target.mesh-device": stack["target"][
                        "mesh_device"
                    ],
                }
            },
        }
    ]


@pytest.mark.parametrize("image_id", ["sha256:" + "a" * 64, "a" * 64])
def test_image_validation_emits_the_immutable_image_id(image_inspection, image_id):
    image_inspection[0]["Id"] = image_id
    result = run_stack("validate-image", input_text=json.dumps(image_inspection))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == image_inspection[0]["Id"]


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize(
    "label",
    [
        "org.opencontainers.image.source",
        "io.tenstorrent.tt-lang.compiler.commit",
        "io.tenstorrent.tt-lang.emule.commit",
        "io.tenstorrent.tt-lang.metal.repository",
        "io.tenstorrent.tt-lang.metal.commit",
        "io.tenstorrent.tt-lang.runtime.base-image",
        "io.tenstorrent.tt-lang.runtime.manifest-sha256",
        "io.tenstorrent.tt-lang.runtime.platform",
        "io.tenstorrent.tt-lang.target.name",
        "io.tenstorrent.tt-lang.target.cluster-descriptor",
        "io.tenstorrent.tt-lang.target.mesh-device",
    ],
)
def test_image_validation_rejects_missing_or_mismatched_labels(
    image_inspection, label, missing
):
    labels = image_inspection[0]["Config"]["Labels"]
    if missing:
        del labels[label]
    else:
        labels[label] = "different"

    result = run_stack("validate-image", input_text=json.dumps(image_inspection))

    assert result.returncode == 1
    assert "image provenance mismatch" in result.stderr
    assert label in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "inspection",
    [
        None,
        {},
        [],
        [{}, {}],
        [None],
        [{}],
        [{"Config": {}}],
        [{"Config": {"Labels": []}}],
    ],
)
def test_image_validation_rejects_malformed_inspection(inspection):
    result = run_stack("validate-image", input_text=json.dumps(inspection))

    assert result.returncode == 1
    assert "tt-lang-emule-stack:" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_image_validation_rejects_invalid_json():
    result = run_stack("validate-image", input_text="not image inspection JSON")

    assert result.returncode == 1
    assert "cannot read image inspection" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "image_id", [None, "", "runtime:latest", "a" * 12, "sha256:abc"]
)
def test_image_validation_rejects_an_unusable_image_id(image_inspection, image_id):
    image_inspection[0]["Id"] = image_id

    result = run_stack("validate-image", input_text=json.dumps(image_inspection))

    assert result.returncode == 1
    assert "image.Id" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "option,label,value",
    [
        ("--emulator-commit", "io.tenstorrent.tt-lang.emule.commit", "b" * 40),
        ("--metal-commit", "io.tenstorrent.tt-lang.metal.commit", "c" * 40),
        (
            "--metal-repository",
            "io.tenstorrent.tt-lang.metal.repository",
            "https://github.com/tenstorrent/experimental-metal.git",
        ),
        (
            "--base-image",
            "io.tenstorrent.tt-lang.runtime.base-image",
            "candidate@sha256:" + "d" * 64,
        ),
        ("--platform", "io.tenstorrent.tt-lang.runtime.platform", "linux/amd64/v2"),
    ],
)
def test_image_validation_checks_resolved_runtime_overrides(
    image_inspection, option, label, value
):
    result = run_stack(
        "validate-image", option, value, input_text=json.dumps(image_inspection)
    )
    assert result.returncode == 1
    assert label in result.stderr

    image_inspection[0]["Config"]["Labels"][label] = value
    result = run_stack(
        "validate-image", option, value, input_text=json.dumps(image_inspection)
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == image_inspection[0]["Id"]


def test_image_validation_checks_the_selected_manifest(image_inspection, tmp_path):
    candidate = json.loads(STACK_MANIFEST.read_text(encoding="utf-8"))
    candidate["compiler"]["base_commit"] = "e" * 40
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = run_stack(
        "validate-image",
        manifest=candidate_path,
        input_text=json.dumps(image_inspection),
    )
    assert result.returncode == 1
    assert "compiler.commit" in result.stderr
    assert "manifest-sha256" in result.stderr

    labels = image_inspection[0]["Config"]["Labels"]
    labels["io.tenstorrent.tt-lang.compiler.commit"] = "e" * 40
    labels["io.tenstorrent.tt-lang.runtime.manifest-sha256"] = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    result = run_stack(
        "validate-image",
        manifest=candidate_path,
        input_text=json.dumps(image_inspection),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == image_inspection[0]["Id"]


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
    assert "TTLANG_EMULE_REPOSITORY" not in values
    assert "TTLANG_EMULE_PLATFORM" not in values
    assert "TTLANG_EMULE_ALLOCATOR_MODE" not in values
    assert len(values["TTLANG_EMULE_COMMIT"]) == 40
    assert len(values["TTLANG_METAL_COMMIT"]) == 40
    assert values["TTLANG_EMULE_MESH_DEVICE"] == "P150"
    assert "@sha256:" in values["TTLANG_EMULE_BASE_IMAGE"]


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
