#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "tt-lang-emule-examples.json"
DEFAULT_LAUNCHER = REPO_ROOT / "bin" / "tt-lang-sim"
REQUIRED_CATEGORIES = {
    "elementwise",
    "matrix_multiplication",
    "reduction",
    "fusion",
}


class SuiteError(RuntimeError):
    pass


def require_string(mapping, key, context):
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SuiteError(f"{context}.{key} must be a non-empty string")
    return value


def load_suite(path, repo_root):
    try:
        suite = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SuiteError(f"cannot read example manifest {path}: {error}") from error
    if not isinstance(suite, dict) or suite.get("schema_version") != 1:
        raise SuiteError("example manifest schema_version must be 1")
    entries = suite.get("examples")
    if not isinstance(entries, list) or not entries:
        raise SuiteError("example manifest examples must be a non-empty list")

    examples = []
    names = set()
    categories = set()
    for index, entry in enumerate(entries):
        context = f"examples[{index}]"
        if not isinstance(entry, dict):
            raise SuiteError(f"{context} must be an object")
        name = require_string(entry, "name", context)
        category = require_string(entry, "category", context)
        relative_path = require_string(entry, "path", context)
        correctness = require_string(entry, "correctness", context)
        path_parts = PurePosixPath(relative_path)
        if path_parts.is_absolute() or ".." in path_parts.parts:
            raise SuiteError(f"{context}.path must be a safe relative path")
        if name in names:
            raise SuiteError(f"duplicate example name: {name}")
        if correctness != "embedded_torch_reference":
            raise SuiteError(f"{context}.correctness is not supported")
        example_path = repo_root / path_parts
        if not example_path.is_file():
            raise SuiteError(f"example does not exist: {relative_path}")
        names.add(name)
        categories.add(category)
        examples.append(
            {
                "name": name,
                "category": category,
                "relative_path": relative_path,
                "path": example_path,
            }
        )

    missing_categories = REQUIRED_CATEGORIES - categories
    if missing_categories:
        raise SuiteError(
            "example manifest is missing required categories: "
            + ", ".join(sorted(missing_categories))
        )
    return examples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the compiler-backed emulation reference examples"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--runtime-image")
    parser.add_argument(
        "--example", action="append", default=[], help="run only this named example"
    )
    parser.add_argument("--list", action="store_true", help="list examples and exit")
    parser.add_argument(
        "--keep-going", action="store_true", help="continue after a failed example"
    )
    return parser.parse_args()


def main():
    arguments = parse_args()
    try:
        examples = load_suite(arguments.manifest, REPO_ROOT)
        available = {example["name"]: example for example in examples}
        unknown = sorted(set(arguments.example) - set(available))
        if unknown:
            raise SuiteError("unknown examples: " + ", ".join(unknown))
        if arguments.example:
            selected = [available[name] for name in arguments.example]
        else:
            selected = examples

        if arguments.list:
            for example in selected:
                print(
                    f"{example['name']}\t{example['category']}\t"
                    f"{example['relative_path']}"
                )
            return 0
        if not arguments.launcher.is_file() or not os.access(
            arguments.launcher, os.X_OK
        ):
            raise SuiteError(f"launcher does not exist: {arguments.launcher}")

        failures = []
        for example in selected:
            command = [
                str(arguments.launcher),
                str(example["path"]),
                "--backend",
                "emule",
            ]
            if arguments.runtime_image:
                command.extend(["--runtime-image", arguments.runtime_image])
            print(
                f"=== RUN {example['name']} ({example['category']}) ===",
                flush=True,
            )
            try:
                result = subprocess.run(command, check=False, cwd=REPO_ROOT)
            except OSError as error:
                raise SuiteError(
                    f"cannot run launcher {arguments.launcher}: {error}"
                ) from error
            if result.returncode == 0:
                print(f"=== PASS {example['name']} ===", flush=True)
                continue
            failures.append((example["name"], result.returncode))
            print(
                f"=== FAIL {example['name']} (exit {result.returncode}) ===",
                file=sys.stderr,
                flush=True,
            )
            if not arguments.keep_going:
                break

        if failures:
            print(
                "Failed examples: "
                + ", ".join(f"{name}={code}" for name, code in failures),
                file=sys.stderr,
            )
            return 1
        return 0
    except SuiteError as error:
        print(f"tt-lang emule examples: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
