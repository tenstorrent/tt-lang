#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
IMAGE_PATTERN = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")


class StackError(RuntimeError):
    pass


def require_mapping(value, name):
    if not isinstance(value, dict):
        raise StackError(f"{name} must be an object")
    return value


def require_string(mapping, key, name):
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise StackError(f"{name}.{key} must be a non-empty string")
    return value


def require_sha(mapping, key, name):
    value = require_string(mapping, key, name)
    if SHA_PATTERN.fullmatch(value) is None:
        raise StackError(f"{name}.{key} must be a full lowercase commit SHA")
    return value


def require_repository(mapping, name):
    value = require_string(mapping, "repository", name)
    if not value.startswith("https://github.com/tenstorrent/") or not value.endswith(
        ".git"
    ):
        raise StackError(f"{name}.repository must be a Tenstorrent GitHub URL")
    return value


def load_stack(path, target_id=None):
    try:
        manifest_bytes = path.read_bytes()
        stack = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackError(f"cannot read stack manifest {path}: {error}") from error

    if not isinstance(stack, dict):
        raise StackError("stack manifest must be an object")
    if stack.get("schema_version") != 2:
        raise StackError("schema_version must be 2")

    compiler = require_mapping(stack.get("compiler"), "compiler")
    emulator = require_mapping(stack.get("emulator"), "emulator")
    metal = require_mapping(stack.get("metal"), "metal")
    runtime = require_mapping(stack.get("runtime"), "runtime")
    targets = require_mapping(stack.get("targets"), "targets")
    default_target = require_string(stack, "default_target", "stack")
    for name, profile in targets.items():
        if re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) is None:
            raise StackError(
                "target identifiers must use lowercase letters, digits or hyphens"
            )
        profile = require_mapping(profile, f"targets.{name}")
        descriptor = require_string(profile, "cluster_descriptor", f"targets.{name}")
        descriptor_path = PurePosixPath(descriptor)
        if descriptor_path.is_absolute() or ".." in descriptor_path.parts:
            raise StackError(
                f"targets.{name}.cluster_descriptor must be a safe relative path"
            )
        require_string(profile, "name", f"targets.{name}")
        require_string(profile, "mesh_device", f"targets.{name}")
    if default_target not in targets:
        raise StackError("default_target must name a target in targets")
    if target_id is None:
        target_id = default_target
    if target_id not in targets:
        raise StackError(f"unknown target '{target_id}'; choose {', '.join(targets)}")
    target = targets[target_id]

    base_image = require_string(runtime, "base_image", "runtime")
    if IMAGE_PATTERN.fullmatch(base_image) is None:
        raise StackError("runtime.base_image must use an exact sha256 digest")
    values = {
        "TTLANG_EMULE_STACK_MANIFEST_SHA256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "TTLANG_COMPILER_REPOSITORY": require_repository(compiler, "compiler"),
        "TTLANG_COMPILER_BASE_COMMIT": require_sha(compiler, "base_commit", "compiler"),
        "TTLANG_EMULE_COMMIT": require_sha(emulator, "commit", "emulator"),
        "TTLANG_METAL_REPOSITORY": require_repository(metal, "metal"),
        "TTLANG_METAL_COMMIT": require_sha(metal, "commit", "metal"),
        "TTLANG_EMULE_BASE_IMAGE": base_image,
        "TTLANG_EMULE_TARGET_ID": target_id,
        "TTLANG_EMULE_TARGET": target["name"],
        "TTLANG_EMULE_CLUSTER_DESCRIPTOR": target["cluster_descriptor"],
        "TTLANG_EMULE_MESH_DEVICE": target["mesh_device"],
    }
    return values


def run_git(source, *arguments):
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise StackError(f"git {' '.join(arguments)} failed in {source}: {detail}")
    return result.stdout.strip()


def first_pin(path):
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                return value
    except OSError as error:
        raise StackError(f"cannot read emulator Metal pin {path}: {error}") from error
    raise StackError(f"emulator Metal pin {path} has no commit")


def validate_sources(values, compiler_source, emulator_source):
    if compiler_source is not None:
        compiler_head = run_git(compiler_source, "rev-parse", "HEAD")
        result = subprocess.run(
            [
                "git",
                "-C",
                str(compiler_source),
                "merge-base",
                "--is-ancestor",
                values["TTLANG_COMPILER_BASE_COMMIT"],
                compiler_head,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise StackError(
                "compiler checkout does not contain the manifest compiler baseline"
            )

    if emulator_source is not None:
        emulator_head = run_git(emulator_source, "rev-parse", "HEAD")
        if emulator_head != values["TTLANG_EMULE_COMMIT"]:
            raise StackError(
                "emulator checkout is at "
                f"{emulator_head}, expected {values['TTLANG_EMULE_COMMIT']}"
            )
        descriptor = emulator_source / values["TTLANG_EMULE_CLUSTER_DESCRIPTOR"]
        if not descriptor.is_file():
            raise StackError(f"emulator target descriptor is missing: {descriptor}")
        emulator_pin = first_pin(emulator_source / "tt-metal-pin.txt")
        if emulator_pin != values["TTLANG_METAL_COMMIT"]:
            raise StackError(
                f"emulator pins Metal {emulator_pin}, expected "
                f"{values['TTLANG_METAL_COMMIT']}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and inspect the compiler-backed emulator stack"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="stack manifest to inspect"
    )
    parser.add_argument(
        "--target", help="emulated hardware profile (default: manifest default)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("emit", help="emit tab-separated values for the launcher")
    validate = subparsers.add_parser("validate", help="validate source checkouts")
    validate.add_argument("--compiler-source", type=Path)
    validate.add_argument("--emulator-source", type=Path)
    validate.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_args()
    try:
        values = load_stack(arguments.manifest, arguments.target)
        if arguments.command == "emit":
            for key, value in values.items():
                print(f"{key}\t{value}")
            return 0

        validate_sources(values, arguments.compiler_source, arguments.emulator_source)
        if not arguments.quiet:
            print(
                "Validated tt-lang/emule stack: "
                f"{values['TTLANG_EMULE_TARGET']} "
                f"emule={values['TTLANG_EMULE_COMMIT']} "
                f"metal={values['TTLANG_METAL_COMMIT']}"
            )
        return 0
    except StackError as error:
        print(f"tt-lang-emule-stack: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
