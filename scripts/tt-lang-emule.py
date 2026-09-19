#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and run the Docker simulator from a source checkout."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / ".ttlang-sim" / "emule.json"
SETTINGS = {
    "image": "TTLANG_EMULE_IMAGE",
    "source": "TTLANG_EMULE_RUNTIME_SOURCE_DIR",
    "source_url": "TTLANG_EMULE_RUNTIME_SOURCE_URL",
    "jobs": "TTLANG_EMULE_JOBS",
}
SUITES = ("mlir", "bindings", "packaging", "pytest", "me2e", "python-lit")


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args():
    action_flags = {
        "--setup": "setup",
        "--smoke-test": "smoke",
        "--examples": "examples",
        "--test": "test",
    }
    arguments = sys.argv[1:]
    action_flag = arguments[0] if arguments and arguments[0] in action_flags else None
    if action_flag:
        arguments = [action_flags[action_flag], *arguments[1:]]
    parser = argparse.ArgumentParser(
        prog="tt-lang-sim --backend=emule",
        description="Configure and test the compiler-backed simulator.",
        epilog="Run programs with: tt-lang-sim --backend=emule SCRIPT.py [-- ARGS...]",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_command(name, help):
        options = {"help": help}
        if action_flag:
            options["prog"] = f"tt-lang-sim --backend=emule {action_flag}"
        return commands.add_parser(name, **options)

    setup = add_command("setup", "save a runtime choice after a smoke test")
    runtime = setup.add_mutually_exclusive_group()
    runtime.add_argument("--source", type=Path, help="exact local emulator checkout")
    runtime.add_argument(
        "--source-url", help="Git repository containing the pinned emulator"
    )
    runtime.add_argument("--image", help="existing local or published runtime image")
    setup.add_argument(
        "--jobs", type=positive_integer, help="compiler build parallelism"
    )

    add_command("smoke", "run the compiler-to-emulator acceptance test")
    add_command("examples", "run the four reference examples")
    tests = add_command("test", "run all six compiler suites through Docker")
    tests.add_argument(
        "--suite",
        action="append",
        choices=SUITES,
        help="run only this suite (repeatable)",
    )
    tests.add_argument(
        "--reports-dir", type=Path, help="parent directory for a new report folder"
    )
    return parser.parse_args(arguments)


def load_settings():
    if not CONFIG_PATH.exists():
        return {}
    settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or settings.get("schema_version") != 1:
        raise ValueError(f"invalid configuration: {CONFIG_PATH}")
    if set(settings) - set(SETTINGS) - {"schema_version"}:
        raise ValueError(f"unknown configuration fields: {CONFIG_PATH}")
    for key in SETTINGS:
        if key not in settings:
            continue
        value = settings[key]
        if key == "jobs":
            if type(value) is not int or value < 1:
                raise ValueError("configured jobs must be a positive integer")
        elif not isinstance(value, str) or not value.strip() or "\0" in value:
            raise ValueError(f"configured {key} must be a non-empty string")
    return settings


def make_environment(settings):
    environment = os.environ.copy()
    # Any explicit runtime selection replaces the saved selection as a group.
    runtime_override = any(
        environment.get(SETTINGS[key]) for key in ("source", "source_url", "image")
    )
    for key, variable in SETTINGS.items():
        if key in settings and (key == "jobs" or not runtime_override):
            if not environment.get(variable):
                environment[variable] = str(settings[key])
    return environment


def run_command(command, environment):
    return subprocess.run(command, env=environment, check=False).returncode


def validate_environment(environment):
    jobs = environment.get("TTLANG_EMULE_JOBS")
    if jobs:
        try:
            number = int(jobs)
        except ValueError:
            raise ValueError("TTLANG_EMULE_JOBS must be a positive integer") from None
        if number < 1:
            raise ValueError("TTLANG_EMULE_JOBS must be a positive integer")


def prepare_image(environment):
    image = environment.get("TTLANG_EMULE_IMAGE")
    if not image:
        return
    # A source override or rebuild can use IMAGE as a local output tag.
    if (
        environment.get("TTLANG_EMULE_RUNTIME_SOURCE_DIR")
        or environment.get("TTLANG_EMULE_RUNTIME_SOURCE_URL")
        or environment.get("TTLANG_EMULE_REBUILD") == "1"
    ):
        return
    docker = environment.get("TTLANG_EMULE_DOCKER", "docker")
    if subprocess.run(
        [docker, "info"],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise ValueError(
            "cannot connect to Docker; start Docker Desktop or the Docker daemon"
        )
    inspection = subprocess.run(
        [docker, "image", "inspect", image],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if inspection.returncode == 0:
        return
    detail = (inspection.stderr or "").strip()
    if inspection.returncode != 1 or not detail.startswith(
        "Error response from daemon: No such image:"
    ):
        raise ValueError(
            f"cannot inspect runtime image {image} (exit {inspection.returncode}); "
            f"no download attempted: {detail or 'Docker returned no diagnostic'}"
        )
    print(f"Downloading simulator runtime {image}", flush=True)
    if run_command([docker, "pull", "--platform", "linux/amd64", image], environment):
        raise ValueError(
            "runtime image could not be downloaded; check its name and registry access, or use setup --source/--source-url"
        )


def launcher(*arguments):
    return [
        str(REPO_ROOT / "bin" / "tt-lang-sim"),
        "--backend",
        "emule",
        *map(str, arguments),
    ]


def save_settings(settings):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=CONFIG_PATH.parent, delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            json.dump({"schema_version": 1, **settings}, output, indent=2)
            output.write("\n")
            output.close()
            temporary.replace(CONFIG_PATH)
        finally:
            temporary.unlink(missing_ok=True)


def setup(arguments, settings):
    selected = next(
        (
            key
            for key in ("source", "source_url", "image")
            if getattr(arguments, key) is not None
        ),
        None,
    )
    environment = make_environment(settings)
    if selected:
        for key in ("source", "source_url", "image"):
            settings.pop(key, None)
            environment.pop(SETTINGS[key], None)
        value = getattr(arguments, selected)
        if selected == "source":
            value = str(value.expanduser().resolve(strict=True))
        if not str(value).strip():
            raise ValueError(f"--{selected.replace('_', '-')} must not be empty")
        settings[selected] = value
        environment[SETTINGS[selected]] = value
    if arguments.jobs is not None:
        settings["jobs"] = arguments.jobs
        environment[SETTINGS["jobs"]] = str(arguments.jobs)
    validate_environment(environment)
    source = environment.get(SETTINGS["source"])
    if source:
        manifest = environment.get(
            "TTLANG_EMULE_STACK_MANIFEST",
            str(REPO_ROOT / "config" / "tt-lang-emule-stack.json"),
        )
        result = run_command(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "tt-lang-emule-stack.py"),
                "--manifest",
                manifest,
                "validate",
                "--compiler-source",
                str(REPO_ROOT),
                "--emulator-source",
                source,
            ],
            environment,
        )
        if result:
            return result
    prepare_image(environment)
    result = run_command(launcher("--smoke-test"), environment)
    if result == 0:
        # Persist the settings actually used, including explicit environment overrides.
        settings = {
            key: environment[variable]
            for key, variable in SETTINGS.items()
            if environment.get(variable)
        }
        if "source" in settings:
            settings["source"] = str(Path(settings["source"]).expanduser().resolve())
        if "jobs" in settings:
            settings["jobs"] = int(settings["jobs"])
        save_settings(settings)
        print(f"Simulator ready. Configuration: {CONFIG_PATH}", flush=True)
    return result


def run_tests(arguments, environment):
    parent = (
        (arguments.reports_dir or REPO_ROOT / ".ttlang-sim" / "reports")
        .expanduser()
        .resolve()
    )
    parent.mkdir(parents=True, exist_ok=True)
    reports = Path(
        tempfile.mkdtemp(
            prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=parent
        )
    )
    environment["TTLANG_EMULE_REPORT_DIR"] = str(reports)
    command = launcher(
        REPO_ROOT / "scripts" / "run-tt-lang-emule-suite.py",
        "--",
        "--reports-dir",
        "/ttlang-reports",
    )
    for suite in arguments.suite or []:
        command.extend(["--suite", suite])
    revision = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    environment["TTLANG_EMULE_COMPILER_SHA"] = revision
    environment["TTLANG_EMULE_COMPILER_DIRTY"] = "1" if status else "0"
    provenance = {
        "compiler_commit": revision,
        "compiler_dirty": bool(status),
        "suites": arguments.suite or list(SUITES),
        "runtime_image": environment.get("TTLANG_EMULE_IMAGE"),
        "source_directory": str(REPO_ROOT),
    }
    (reports / "invocation.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Test reports: {reports}", flush=True)
    try:
        return run_command(command, environment)
    finally:
        print(f"Test reports: {reports}", flush=True)


def main():
    # The shell launcher has already consumed its options and the separator.
    launch_arguments = sys.argv[2:] if sys.argv[1:2] == ["--launch"] else None
    arguments = None if launch_arguments is not None else parse_args()
    try:
        settings = load_settings()
        if arguments is not None and arguments.command == "setup":
            return setup(arguments, settings)
        environment = make_environment(settings)
        validate_environment(environment)
        if launch_arguments is not None:
            if len(launch_arguments) < 2:
                raise ValueError("the emule backend requires a script path")
            if not Path(launch_arguments[1]).is_file():
                raise ValueError(f"script not found: {launch_arguments[1]}")
            prepare_image(environment)
            return run_command(launch_arguments, environment)
        prepare_image(environment)
        if arguments.command == "smoke":
            return run_command(launcher("--smoke-test"), environment)
        if arguments.command == "examples":
            return run_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run-tt-lang-emule-examples.py"),
                    "--keep-going",
                ],
                environment,
            )
        return run_tests(arguments, environment)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"tt-lang-sim emule: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
