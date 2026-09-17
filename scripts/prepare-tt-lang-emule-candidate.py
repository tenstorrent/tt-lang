#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "tt-lang-emule-stack.json"
STACK_TOOL = REPO_ROOT / "scripts" / "tt-lang-emule-stack.py"
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


class CandidateError(RuntimeError):
    pass


def run_git(source, *arguments):
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CandidateError(f"git {' '.join(arguments)} failed in {source}: {detail}")
    return result.stdout.strip()


def require_sha(value, name):
    if SHA_PATTERN.fullmatch(value) is None:
        raise CandidateError(f"{name} must be a full lowercase commit SHA")
    return value


def first_pin(path):
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                return require_sha(value, "emulator Metal pin")
    except OSError as error:
        raise CandidateError(
            f"cannot read emulator Metal pin {path}: {error}"
        ) from error
    raise CandidateError(f"emulator Metal pin {path} has no commit")


def load_manifest(path):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateError(f"cannot read stack manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise CandidateError("stack manifest schema_version must be 1")
    for section in ("compiler", "emulator", "metal", "runtime", "target"):
        if not isinstance(manifest.get(section), dict):
            raise CandidateError(f"stack manifest {section} must be an object")
    return manifest


def render_candidate(manifest, compiler_commit, emulator_commit, metal_commit):
    manifest["compiler"]["base_commit"] = compiler_commit
    manifest["emulator"]["commit"] = emulator_commit
    manifest["metal"]["commit"] = metal_commit
    return json.dumps(manifest, indent=2) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an exact compiler/emulator/Metal stack candidate"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--compiler-source", type=Path, default=REPO_ROOT)
    parser.add_argument("--compiler-commit")
    parser.add_argument("--emulator-source", type=Path, required=True)
    parser.add_argument("--emulator-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="require the input manifest to already match the candidate",
    )
    return parser.parse_args()


def main():
    arguments = parse_args()
    temporary_output = None
    try:
        manifest = load_manifest(arguments.manifest)
        compiler_head = run_git(arguments.compiler_source, "rev-parse", "HEAD")
        compiler_commit = require_sha(
            arguments.compiler_commit or compiler_head, "compiler commit"
        )
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                str(arguments.compiler_source),
                "merge-base",
                "--is-ancestor",
                compiler_commit,
                compiler_head,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if ancestor.returncode != 0:
            raise CandidateError(
                f"compiler commit {compiler_commit} is not in the compiler checkout"
            )

        emulator_commit = require_sha(arguments.emulator_commit, "emulator commit")
        emulator_head = run_git(arguments.emulator_source, "rev-parse", "HEAD")
        if emulator_head != emulator_commit:
            raise CandidateError(
                f"emulator checkout is at {emulator_head}, expected {emulator_commit}"
            )
        emulator_status = run_git(
            arguments.emulator_source, "status", "--porcelain", "--untracked-files=all"
        )
        if emulator_status:
            raise CandidateError("emulator checkout must not contain local changes")
        metal_commit = first_pin(arguments.emulator_source / "tt-metal-pin.txt")
        descriptor = arguments.emulator_source / manifest["target"].get(
            "cluster_descriptor", ""
        )
        if not descriptor.is_file():
            raise CandidateError(f"emulator target descriptor is missing: {descriptor}")

        rendered = render_candidate(
            manifest, compiler_commit, emulator_commit, metal_commit
        )
        if arguments.check:
            if arguments.output is not None:
                raise CandidateError("--check and --output cannot be used together")
            if arguments.manifest.read_text(encoding="utf-8") != rendered:
                raise CandidateError("stack manifest does not match the candidate")
            output = arguments.manifest
            validation_manifest = output
        else:
            output = arguments.output or arguments.manifest
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                delete=False,
            ) as candidate_file:
                candidate_file.write(rendered)
                temporary_output = Path(candidate_file.name)
            validation_manifest = temporary_output

        validation = subprocess.run(
            [
                sys.executable,
                str(STACK_TOOL),
                "--manifest",
                str(validation_manifest),
                "validate",
                "--compiler-source",
                str(arguments.compiler_source),
                "--emulator-source",
                str(arguments.emulator_source),
                "--quiet",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if validation.returncode != 0:
            raise CandidateError(
                validation.stderr.strip() or "candidate stack validation failed"
            )
        if temporary_output is not None:
            os.replace(temporary_output, output)
            temporary_output = None
        print(
            f"Prepared stack candidate: compiler={compiler_commit} "
            f"emulator={emulator_commit} metal={metal_commit} output={output}"
        )
        return 0
    except (CandidateError, OSError, UnicodeDecodeError) as error:
        print(f"tt-lang emule candidate: {error}", file=sys.stderr)
        return 1
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
