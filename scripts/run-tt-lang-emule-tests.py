#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = REPO_ROOT / "config" / "tt-lang-emule-test-plan.json"
VALID_LAYERS = {"compiler", "runtime", "mixed"}
RUNNABLE_MODES = {"compiler_only", "emulated_execution"}
VALID_MODES = RUNNABLE_MODES | {"inventory_only"}
PLACEHOLDERS = {
    "{repo_root}": lambda build_dir: str(REPO_ROOT),
    "{build_dir}": lambda build_dir: str(build_dir),
}


class PlanError(RuntimeError):
    pass


def require_string(mapping, key, context):
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise PlanError(f"{context}.{key} must be a non-empty string")
    return value


def load_plan(path):
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot read test plan {path}: {error}") from error
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise PlanError("test plan schema_version must be 1")
    entries = plan.get("phases")
    if not isinstance(entries, list) or not entries:
        raise PlanError("test plan phases must be a non-empty list")

    phases = []
    names = set()
    for index, entry in enumerate(entries):
        context = f"phases[{index}]"
        if not isinstance(entry, dict):
            raise PlanError(f"{context} must be an object")
        name = require_string(entry, "name", context)
        layer = require_string(entry, "layer", context)
        mode = require_string(entry, "mode", context)
        if name in names:
            raise PlanError(f"duplicate phase name: {name}")
        if layer not in VALID_LAYERS:
            raise PlanError(f"{context}.layer is not supported: {layer}")
        if mode not in VALID_MODES:
            raise PlanError(f"{context}.mode is not supported: {mode}")

        command = entry.get("command")
        environment = entry.get("environment", {})
        reason = entry.get("reason")
        if mode == "inventory_only":
            if command is not None:
                raise PlanError(
                    f"{context} inventory-only phase must not have a command"
                )
            if not isinstance(reason, str) or not reason:
                raise PlanError(f"{context} inventory-only phase requires a reason")
        else:
            if (
                not isinstance(command, list)
                or not command
                or not all(
                    isinstance(argument, str) and argument for argument in command
                )
            ):
                raise PlanError(f"{context}.command must be a non-empty string list")
            if reason is not None:
                raise PlanError(f"{context} runnable phase must not have a reason")
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise PlanError(f"{context}.environment must contain string values")

        names.add(name)
        phases.append(
            {
                "name": name,
                "layer": layer,
                "mode": mode,
                "command": command,
                "environment": environment,
                "reason": reason,
            }
        )
    return phases


def expand_command(command, build_dir):
    expanded = []
    for argument in command:
        value = argument
        for placeholder, resolver in PLACEHOLDERS.items():
            value = value.replace(placeholder, resolver(build_dir))
        if "{" in value or "}" in value:
            raise PlanError(f"unknown command placeholder in: {argument}")
        expanded.append(value)
    return expanded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the supported compiler and emulated-execution test layers"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--runtime-image")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_args()
    try:
        phases = load_plan(arguments.plan)
        available = {phase["name"]: phase for phase in phases}
        unknown = sorted(set(arguments.phase) - set(available))
        if unknown:
            raise PlanError("unknown phases: " + ", ".join(unknown))
        selected = (
            [available[name] for name in arguments.phase]
            if arguments.phase
            else [phase for phase in phases if phase["mode"] in RUNNABLE_MODES]
        )

        if arguments.list:
            for phase in phases:
                detail = phase["reason"] or "runnable"
                print(f"{phase['name']}\t{phase['layer']}\t{phase['mode']}\t{detail}")
            return 0
        blocked = [
            phase["name"] for phase in selected if phase["mode"] == "inventory_only"
        ]
        if blocked:
            raise PlanError("inventory-only phases cannot run: " + ", ".join(blocked))

        needs_build = any(
            any("{build_dir}" in argument for argument in phase["command"])
            for phase in selected
        )
        build_dir = arguments.build_dir.resolve()
        if needs_build and not build_dir.is_dir() and not arguments.dry_run:
            raise PlanError(f"configured build directory does not exist: {build_dir}")

        failures = []
        for phase in selected:
            command = expand_command(phase["command"], build_dir)
            if phase["mode"] == "emulated_execution" and arguments.runtime_image:
                command.extend(["--runtime-image", arguments.runtime_image])
            print(f"=== RUN {phase['name']} ({phase['mode']}) ===", flush=True)
            print("+ " + shlex.join(command), flush=True)
            if arguments.dry_run:
                continue
            environment = os.environ.copy()
            environment.update(phase["environment"])
            try:
                result = subprocess.run(
                    command, check=False, cwd=REPO_ROOT, env=environment
                )
            except OSError as error:
                raise PlanError(f"cannot run {phase['name']}: {error}") from error
            if result.returncode == 0:
                print(f"=== PASS {phase['name']} ===", flush=True)
                continue
            failures.append((phase["name"], result.returncode))
            print(
                f"=== FAIL {phase['name']} (exit {result.returncode}) ===",
                file=sys.stderr,
                flush=True,
            )
            if not arguments.keep_going:
                break

        if failures:
            print(
                "Failed phases: "
                + ", ".join(f"{name}={code}" for name, code in failures),
                file=sys.stderr,
            )
            return 1
        return 0
    except PlanError as error:
        print(f"tt-lang emule test plan: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
