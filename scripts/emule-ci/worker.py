#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Local-only, bounded emulator worker. No image builds, pulls or publication."""

import argparse
from enum import IntEnum
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
ENVIRONMENT_KEYS = {
    "TTLANG_EMULE_TARGET_NAME",
    "TTLANG_EMULE_EXPECTED_LLVM_SHA",
    "TT_METAL_MOCK_CLUSTER_DESC_PATH",
    "TT_METAL_ALLOCATOR_MODE_HYBRID",
    "MESH_DEVICE",
    "TTLANG_EMULE_COMPILER_SHA",
    "TTLANG_EMULE_SOURCE_FINGERPRINT",
    "TTLANG_EMULE_INSTALL",
}


class PreflightFailure(IntEnum):
    RUNTIME_IMPORT_FAILED = 20
    CPU_ONLY_TORCH_REQUIRED = 21
    RANGE_LOCKSTEP_API_MISSING = 22
    PIP_CHECK_FAILED = 23
    PIP_CHECK_TIMED_OUT = 24


def bounded_integer(value, maximum):
    number = int(value)
    if not 1 <= number <= maximum:
        raise ValueError("resource limit out of range")
    return number


def safe_mount_path(value):
    if not Path(value).is_absolute() or any(c in value for c in ",\n\r"):
        raise ValueError("invalid bind mount path")
    return value


def proxy_command(arguments, environment):
    """Allow only the existing launcher's read-only Docker queries and runs."""
    docker = environment["TTLANG_CI_DOCKER"]
    if not Path(docker).is_absolute():
        raise ValueError("Docker executable must be absolute")
    image = environment["TTLANG_EMULE_IMAGE"]
    if not IMAGE_ID.fullmatch(image):
        raise ValueError("an inspected immutable image ID is required")
    if arguments == ["info"] or arguments == ["image", "inspect", "--", image]:
        return [docker, *arguments]
    name = environment["TTLANG_CI_CONTAINER"]
    token = environment["TTLANG_CI_TOKEN"]
    if not NAME.fullmatch(name) or not NAME.fullmatch(token):
        raise ValueError("invalid worker identity")
    cpus = bounded_integer(environment["TTLANG_CI_CPUS"], 4)
    memory = bounded_integer(environment["TTLANG_CI_MEMORY_GIB"], 12)
    build = environment["TTLANG_EMULE_BUILD_VOLUME"]
    cache = environment["TTLANG_EMULE_CACHE_VOLUME"]
    if build != token + "-build" or cache != token + "-cache":
        raise ValueError("volumes must belong to this worker")
    command = [
        docker,
        "run",
        "--pull=never",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--cpus={cpus}",
        f"--memory={memory}g",
        f"--memory-swap={memory}g",
        "--pids-limit=512",
        "--name",
        name,
        "--label",
        f"ttlang.emule-ci.owner={token}",
        "-e",
        "GIT_CONFIG_COUNT=1",
        "-e",
        "GIT_CONFIG_KEY_0=safe.directory",
        "-e",
        "GIT_CONFIG_VALUE_0=/workspace",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PIP_NO_INDEX=1",
        "-e",
        f"TTLANG_EMULE_CI_THREADS={cpus}",
        "-e",
        f"OMP_NUM_THREADS={cpus}",
        "-e",
        f"MKL_NUM_THREADS={cpus}",
        "-e",
        f"OPENBLAS_NUM_THREADS={cpus}",
        "-e",
        "TTLANG_EMULE_CI_CACHE=/tt-metal-cache",
    ]
    if arguments == ["preflight"]:
        source = safe_mount_path(environment["TTLANG_CI_SOURCE"])
        metal = "/opt/tt-emule-runtime/tt-metal"
        return command + [
            "--rm",
            "--platform=linux/amd64",
            "--entrypoint=/opt/ttlang-toolchain/venv/bin/python",
            "--mount",
            f"type=bind,src={source},dst=/workspace,readonly",
            "-e",
            f"TT_METAL_HOME={metal}",
            "-e",
            f"TT_METAL_RUNTIME_ROOT={metal}",
            "-e",
            f"PYTHONPATH={metal}/ttnn:{metal}/tools",
            image,
            "/workspace/scripts/emule-ci/worker.py",
            "--runtime-preflight",
        ]
    if not arguments or arguments[0] != "run":
        raise ValueError("Docker operation is not permitted")
    source = safe_mount_path(environment["TTLANG_CI_SOURCE"])
    common = safe_mount_path(environment["TTLANG_CI_GIT_COMMON"])
    results = safe_mount_path(environment["TTLANG_CI_RESULTS"])
    mounts = {
        f"type=bind,src={source},dst=/workspace": f"type=bind,src={source},dst=/workspace,readonly",
        f"type=bind,src={common},dst={common},readonly": f"type=bind,src={common},dst={common},readonly",
        f"type=volume,src={build},dst=/ttlang-build": f"type=volume,src={build},dst=/ttlang-build",
        f"type=volume,src={cache},dst=/tt-metal-cache": f"type=volume,src={cache},dst=/tt-metal-cache",
    }
    index = 1
    while index < len(arguments) and arguments[index] != image:
        option = arguments[index]
        if option in ("--rm", "-i"):
            command.append(option)
            index += 1
            continue
        if index + 1 >= len(arguments):
            raise ValueError("missing Docker option value")
        value = arguments[index + 1]
        if option == "--mount" and value in mounts:
            value = mounts[value]
        elif option == "--platform" and value == "linux/amd64":
            pass
        elif (
            option == "--entrypoint"
            and value == "/workspace/scripts/tt-lang-emule-entrypoint.sh"
        ):
            pass
        elif option == "--workdir" and value == "/workspace":
            pass
        elif (
            option == "-e"
            and value.partition("=")[0] in ENVIRONMENT_KEYS
            and "=" in value
        ):
            pass
        else:
            raise ValueError("unexpected Docker option or mount")
        command.extend([option, value])
        index += 2
    if index == len(arguments):
        raise ValueError("expected runtime image is missing")
    return command + [
        "--tmpfs",
        "/workspace/build:rw,exec,nosuid,size=512m",
        "--mount",
        f"type=bind,src={results},dst=/emule-results",
        *arguments[index:],
    ]


def parse_bytes(value):
    match = re.fullmatch(r"\s*([0-9.]+)\s*(B|kB|MB|GB|KiB|MiB|GiB)\s*", value)
    if not match:
        raise ValueError("invalid Docker size")
    multipliers = {
        "B": 1,
        "kB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    number = float(match[1]) * multipliers[match[2]]
    if not math.isfinite(number) or number < 0:
        raise ValueError("invalid Docker size")
    return int(number)


def numeric_stats(raw):
    stats = json.loads(raw)
    cpu = float(stats["CPUPerc"].removesuffix("%"))
    if not math.isfinite(cpu) or cpu < 0:
        raise ValueError("invalid CPU sample")
    return {
        "cpu_percent": cpu,
        "memory_bytes": parse_bytes(stats["MemUsage"].split("/")[0]),
    }


SUITES = ("mlir", "bindings", "packaging", "pytest", "me2e", "lit")
GROUP_STATUSES = {"passed", "failed", "infrastructure_error"}
CASE_OUTCOMES = {
    "passed",
    "failed",
    "skipped",
    "xfail",
    "unsupported",
    "incomplete",
    "not_run",
}
SUITE_COUNT_KEYS = {
    "planned_groups",
    "planned_cases",
    "groups",
    "complete_groups",
    "status_counts",
    "outcome_counts",
}


def counted(value, vocabulary):
    """A mapping from a fixed vocabulary to non-negative integers."""
    if (
        not isinstance(value, dict)
        or set(value) - vocabulary
        or any(type(count) is not int or count < 0 for count in value.values())
    ):
        raise ValueError("invalid benchmark counts")
    return value


def public_test_summary(value):
    """Validate the harness's small public contract; never copy private results."""
    counts = (
        "planned_group_count",
        "planned_case_count",
        "group_count",
        "complete_group_count",
        "generic_op_launches",
        "generic_op_returns",
    )
    keys = {
        *counts,
        "group_status_counts",
        "case_outcome_counts",
        "suite_counts",
        "wall_seconds",
        "runtime_execution_observed",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("unexpected benchmark summary fields")
    for key in counts:
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError("invalid benchmark count")
    counted(value["group_status_counts"], GROUP_STATUSES)
    counted(value["case_outcome_counts"], CASE_OUTCOMES)
    suites = value["suite_counts"]
    if not isinstance(suites, dict) or set(suites) != set(SUITES):
        raise ValueError("invalid benchmark suites")
    for suite in suites.values():
        if not isinstance(suite, dict) or set(suite) != SUITE_COUNT_KEYS:
            raise ValueError("invalid benchmark suite counts")
        for key in ("planned_groups", "planned_cases", "groups", "complete_groups"):
            if type(suite[key]) is not int or suite[key] < 0:
                raise ValueError("invalid benchmark suite counts")
        counted(suite["status_counts"], GROUP_STATUSES)
        counted(suite["outcome_counts"], CASE_OUTCOMES)
    seconds = value["wall_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid benchmark duration")
    if type(value["runtime_execution_observed"]) is not bool:
        raise ValueError("invalid execution observation")
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def format_counts(counts):
    return ", ".join(f"{key} {count}" for key, count in sorted(counts.items())) or "-"


def report_summary(summary):
    """Render a worker summary as Markdown from its validated public fields only.

    Nothing from private logs, commands, paths or test IDs is read: the suite
    table comes from the benchmark's public contract, and the status fields are
    the worker's fixed vocabulary.
    """
    if not isinstance(summary, dict) or summary.get("schema_version") != 1:
        raise ValueError("unexpected worker summary")
    status = summary.get("status")
    if status not in ("not_started", "passed", "failed", "error"):
        raise ValueError("unexpected worker status")
    lines = [f"Worker status: `{status}`"]
    reason = summary.get("failure_reason")
    if reason is not None:
        if reason != "preflight_failure" and reason not in {
            member.name.lower() for member in PreflightFailure
        }:
            raise ValueError("unexpected failure reason")
        lines.append(f"Preflight failure: `{reason}`")
    stages = summary.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("unexpected worker stages")
    for name in ("preflight", "install", "prepare", "smoke", "collect", "test"):
        if name in stages:
            stage = stages[name]
            if (
                not isinstance(stage, dict)
                or type(stage.get("exit_code")) is not int
                or type(stage.get("seconds")) not in (int, float)
            ):
                raise ValueError("unexpected worker stage")
            lines.append(
                f"Stage `{name}`: exit {stage['exit_code']}, {stage['seconds']:.0f} s"
            )
    if "smoke" in summary:
        smoke = public_test_summary(summary["smoke"])
        lines.append(
            "Representative smoke: runtime execution observed "
            f"{smoke['runtime_execution_observed']}, {smoke['wall_seconds']:.0f} s"
        )
    if "tests" not in summary:
        # Either the inventory never started or its summary was not readable;
        # neither is evidence about the suites, so no table is rendered.
        lines.append("Complete compiler inventory: no validated summary was recorded.")
        return "\n".join(lines) + "\n"
    tests = public_test_summary(summary["tests"])
    lines += [
        "",
        f"Complete compiler inventory: {tests['planned_case_count']} planned cases"
        f" in {tests['planned_group_count']} groups; {tests['complete_group_count']}"
        f" groups completed; runtime execution observed"
        f" {tests['runtime_execution_observed']} ({tests['generic_op_returns']}"
        f" of {tests['generic_op_launches']} generic_op calls returned);"
        f" {tests['wall_seconds']:.0f} s of test time.",
        "",
        "| Suite | Planned groups | Run | Complete | Planned cases"
        " | Case outcomes | Group status |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for suite in SUITES:
        counts = tests["suite_counts"][suite]
        lines.append(
            f"| {suite} | {counts['planned_groups']} | {counts['groups']}"
            f" | {counts['complete_groups']} | {counts['planned_cases']}"
            f" | {format_counts(counts['outcome_counts'])}"
            f" | {format_counts(counts['status_counts'])} |"
        )
    lines += [
        "",
        f"All suites: cases {format_counts(tests['case_outcome_counts'])};"
        f" groups {format_counts(tests['group_status_counts'])}.",
        "",
        "`unsupported` counts cases that declare themselves unsupported on the"
        " single-device emulator profile (`multi_device` pytest skips and lit"
        " `UNSUPPORTED`); `not_run` counts planned cases without a recorded"
        " outcome. Group status `infrastructure_error` marks incomplete coverage,"
        " timeouts or abnormal exits, not test verdicts.",
    ]
    return "\n".join(lines) + "\n"


def test_tools_command(environment):
    if environment.get("TT_METAL_EMULE_MODE") != "1":
        raise ValueError("test tools require the installed emulator environment")
    threads = bounded_integer(environment["TTLANG_EMULE_CI_THREADS"], 4)
    return [
        "cmake",
        "--build",
        "/ttlang-build",
        "--target",
        "ttlang-test-tools",
        "--parallel",
        str(threads),
    ]


def runtime_preflight():
    try:
        import torch
        import ttnn
        import pytest
        import lit
    except Exception:
        return PreflightFailure.RUNTIME_IMPORT_FAILED

    if torch.version.cuda is not None or torch.version.hip is not None:
        return PreflightFailure.CPU_ONLY_TORCH_REQUIRED
    if not callable(
        getattr(
            getattr(ttnn, "MemoryConfig", None),
            "experimental_set_range_lockstep_allocation",
            None,
        )
    ):
        return PreflightFailure.RANGE_LOCKSTEP_API_MISSING
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"], check=False, timeout=60
        )
    except subprocess.TimeoutExpired:
        return PreflightFailure.PIP_CHECK_TIMED_OUT
    except OSError:
        return PreflightFailure.PIP_CHECK_FAILED
    return 0 if result.returncode == 0 else PreflightFailure.PIP_CHECK_FAILED


def runtime_smoke(revision, runtime_id):
    import benchmark

    args = argparse.Namespace(
        source=Path("/workspace"),
        build=Path("/ttlang-build"),
        output=Path("/emule-results/smoke"),
        revision=revision,
        runtime_id=runtime_id,
        manifest=Path("/emule-results/smoke/manifest.json"),
        timeout=600,
        group_timeout=900,
    )
    path = "python/pipe/test_broadcast_2d.py"
    manifest = {
        "revision": revision,
        "runtime_id": runtime_id,
        "groups": [
            {
                "id": "pytest:" + path,
                "kind": "pytest",
                "path": path,
                "cases": [path + "::test_broadcast_2d"],
                "markers": {marker: [] for marker in benchmark.CLASSIFICATION_MARKERS},
            }
        ],
    }
    benchmark.validate_manifest(manifest)
    benchmark.validate_source(args.source, revision)
    env = benchmark.environment(args)
    args.output.mkdir(parents=True, exist_ok=False)
    benchmark.write_json(args.manifest, manifest)
    result = benchmark.run(args, env)
    evidence = benchmark.read_json(args.output / "results.json")
    groups = evidence["groups"]
    if (
        result != 0
        or len(groups) != 1
        or groups[0]["outcomes"] != {"passed": 1}
        or not benchmark.public_summary(evidence)["runtime_execution_observed"]
    ):
        return 1
    return 0


class Worker:
    def __init__(self, args):
        self.args = args
        self.source = Path(__file__).resolve().parents[2]
        self.output = args.output.resolve()
        self.private = self.output / "private"
        self.token = f"ttlang-ci-{args.worker}-{uuid.uuid4().hex[:12]}"
        # Reserve time for bounded Docker/process cleanup within a 45-minute run.
        self.deadline = (
            time.monotonic() + args.timeout_seconds - min(30, args.timeout_seconds / 2)
        )
        self.docker = shutil.which("docker")
        self.environment = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("TTLANG_", "TT_METAL_", "EMULE_", "PYTHON"))
        }
        self.environment.update(
            {
                "TTLANG_CI_HOST_PYTHON": sys.executable,
                "TTLANG_CI_DOCKER": self.docker or "",
                "TTLANG_CI_TOKEN": self.token,
                "TTLANG_CI_SOURCE": str(self.source),
                "TTLANG_CI_RESULTS": str(self.private / "container"),
                "TTLANG_CI_CPUS": str(args.cpus),
                "TTLANG_CI_MEMORY_GIB": str(args.memory_gib),
                "TTLANG_EMULE_BUILD_VOLUME": self.token + "-build",
                "TTLANG_EMULE_CACHE_VOLUME": self.token + "-cache",
                "TTLANG_EMULE_HOST_PYTHON": sys.executable,
                "TTLANG_EMULE_DOCKER": str(
                    self.source / "scripts/emule-ci/docker-bounded.sh"
                ),
            }
        )
        self.summary = {
            "schema_version": 1,
            "publication_blocked": True,
            "status": "not_started",
            "cpus": args.cpus,
            "memory_limit_bytes": args.memory_gib * 1024**3,
            "timeout_seconds": args.timeout_seconds,
            "disk_floor_bytes": args.disk_floor_gib * 1024**3,
            "low_disk": False,
            "stages": {},
            "samples": [],
        }

    def capture(self, command, timeout=30):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("worker deadline exceeded")
        result = subprocess.run(
            command,
            cwd=self.source,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=min(timeout, remaining),
        )
        with (self.private / "host.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps(command) + "\n" + result.stdout + result.stderr)
        return result

    def stop_owned_container(self, name):
        # Callers pass only this worker's uuid-named stage container. Build/cache
        # volumes remain available for inspection.
        try:
            subprocess.run(
                [self.docker, "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def sample(self, stage):
        sample = {
            "stage": stage,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "host_disk_free_bytes": shutil.disk_usage(self.output).free,
        }
        try:
            result = self.capture(
                [
                    self.docker,
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{json .}}",
                    self.environment["TTLANG_CI_CONTAINER"],
                ],
                timeout=3,
            )
            if result.returncode == 0:
                sample.update(numeric_stats(result.stdout))
        except (ValueError, KeyError, TimeoutError, subprocess.TimeoutExpired):
            pass
        self.summary["samples"].append(sample)
        if sample["host_disk_free_bytes"] < self.summary["disk_floor_bytes"]:
            self.summary["low_disk"] = True
            raise RuntimeError("host disk free space fell below the requested floor")

    def stage(self, name, command):
        self.environment["TTLANG_CI_CONTAINER"] = self.token + "-" + name
        start = time.monotonic()
        code = 125
        process = None
        try:
            with (self.private / f"{name}.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.source,
                    env=self.environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while process.poll() is None:
                    if time.monotonic() >= self.deadline:
                        code = 124
                        break
                    self.sample(name)
                    time.sleep(min(5, max(0, self.deadline - time.monotonic())))
                else:
                    code = process.returncode
        finally:
            # A disconnected Docker client can exit while its daemon-side
            # container continues running. Check ownership regardless of exit.
            self.stop_owned_container(self.environment["TTLANG_CI_CONTAINER"])
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
            self.summary["stages"][name] = {
                "exit_code": code,
                "seconds": round(time.monotonic() - start, 3),
            }
        return code

    def sizes(self, when):
        # The daemon sizes this worker's uuid-named volumes without another
        # container; before the first stage they do not exist yet.
        sizes = {"host_disk_free_bytes": shutil.disk_usage(self.output).free}
        self.summary[when] = sizes
        try:
            result = self.capture(
                [self.docker, "system", "df", "-v", "--format", "{{json .}}"],
                timeout=60,
            )
            for volume in json.loads(result.stdout)["Volumes"] or []:
                for kind in ("build", "cache"):
                    if volume["Name"] == f"{self.token}-{kind}":
                        sizes[kind + "_bytes"] = parse_bytes(volume["Size"])
        except (ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired):
            pass

    def prepare(self):
        if self.docker is None:
            raise RuntimeError("Docker executable unavailable")
        image = self.capture(
            [
                self.docker,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                "--",
                self.args.image,
            ]
        )
        image_id = image.stdout.strip()
        if image.returncode != 0 or not IMAGE_ID.fullmatch(image_id):
            raise RuntimeError("runtime image must already exist locally")
        self.environment["TTLANG_EMULE_IMAGE"] = image_id
        revision = self.capture(["git", "rev-parse", "HEAD"])
        common = self.capture(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"]
        )
        if (
            revision.returncode
            or not re.fullmatch(r"[0-9a-f]{40}", revision.stdout.strip())
            or common.returncode
        ):
            raise RuntimeError("source identity unavailable")
        self.revision = revision.stdout.strip()
        self.environment["TTLANG_CI_GIT_COMMON"] = common.stdout.strip()
        # CMake/tests may generate temporary files under source/build. Only add
        # its empty mountpoint when Git explicitly ignores it; never replace an
        # existing symlink or expose existing build products to the container.
        build_mountpoint = self.source / "build"
        ignored = self.capture(["git", "check-ignore", "-q", "build/"])
        if ignored.returncode or build_mountpoint.is_symlink():
            raise RuntimeError("source build mountpoint must be an ignored directory")
        build_mountpoint.mkdir(exist_ok=True)

    def execute(self):
        self.started = time.monotonic()
        if shutil.disk_usage(self.output).free < self.summary["disk_floor_bytes"]:
            self.summary["low_disk"] = True
            raise RuntimeError("insufficient host disk free space for this experiment")
        self.prepare()
        self.sizes("before")
        launcher = [
            "bash",
            str(self.source / "scripts/tt-lang-emule-container.sh"),
            "scripts/emule-ci/benchmark.py",
        ]
        identity = [
            "--source",
            "/workspace",
            "--build",
            "/ttlang-build",
            "--revision",
            self.revision,
            "--runtime-id",
            self.environment["TTLANG_EMULE_IMAGE"],
        ]
        self.summary["status"] = "failed"
        try:
            self.environment["TTLANG_CI_CONTAINER"] = self.token + "-preflight"
            preflight = self.stage(
                "preflight", proxy_command(["preflight"], self.environment)
            )
            if preflight:
                try:
                    reason = PreflightFailure(preflight).name.lower()
                except ValueError:
                    reason = "preflight_failure"
                self.summary["failure_reason"] = reason
                return 1
            if self.stage(
                "install",
                ["bash", str(self.source / "scripts/install-tt-lang-emule.sh")],
            ):
                return 1
            if self.stage(
                "prepare",
                [
                    "bash",
                    str(self.source / "scripts/tt-lang-emule-container.sh"),
                    "scripts/emule-ci/worker.py",
                    "--prepare-test-tools",
                ],
            ):
                return 1
            if self.stage(
                "smoke",
                [
                    "bash",
                    str(self.source / "scripts/tt-lang-emule-container.sh"),
                    "scripts/emule-ci/worker.py",
                    "--runtime-smoke",
                    "--revision",
                    self.revision,
                    "--runtime-id",
                    self.environment["TTLANG_EMULE_IMAGE"],
                ],
            ):
                return 1
            smoke_report = self.private / "container/smoke/summary.json"
            self.summary["smoke"] = public_test_summary(
                json.loads(smoke_report.read_text())
            )
            if not self.summary["smoke"]["runtime_execution_observed"]:
                return 1
            if self.stage(
                "collect",
                [
                    *launcher,
                    "collect",
                    *identity,
                    "--output",
                    "/emule-results/collection",
                ],
            ):
                return 1
            result = self.stage(
                "test",
                [
                    *launcher,
                    "run",
                    *identity,
                    "--manifest",
                    "/emule-results/collection/manifest.json",
                    "--output",
                    "/emule-results/run",
                ],
            )
            report = self.private / "container/run/summary.json"
            if report.exists():
                self.summary["tests"] = public_test_summary(
                    json.loads(report.read_text())
                )
            passed = result == 0 and self.summary.get("tests", {}).get(
                "runtime_execution_observed", False
            )
            self.summary["status"] = "passed" if passed else "failed"
            return 0 if passed else 1
        finally:
            self.sizes("after")


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        required=True,
        help="Already available local image or digest; never pulled",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New private experiment directory"
    )
    parser.add_argument("--worker", default="serial")
    parser.add_argument("--cpus", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--memory-gib", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--timeout-seconds", type=int, default=2700)
    parser.add_argument(
        "--disk-floor-gib", type=int, choices=range(1, 1025), default=10
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,15}", args.worker):
        parser.error("worker must be a short lowercase identifier")
    if not 1 <= args.timeout_seconds <= 18000:
        parser.error("timeout must be between 1 and 18000 seconds")
    if not args.image or args.image.startswith("-"):
        parser.error("invalid local image reference")
    return args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--runtime-preflight"]:
        return runtime_preflight()
    if argv[:1] == ["--runtime-smoke"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--revision", required=True)
        parser.add_argument("--runtime-id", required=True)
        args = parser.parse_args(argv[1:])
        return runtime_smoke(args.revision, args.runtime_id)
    if argv == ["--prepare-test-tools"]:
        try:
            return subprocess.run(
                test_tools_command(os.environ), check=False
            ).returncode
        except (KeyError, ValueError):
            print(
                "emule CI: test tools require the bounded runtime environment",
                file=sys.stderr,
            )
            return 2
    if argv[:1] == ["--docker-proxy"]:
        try:
            command = proxy_command(argv[1:], os.environ)
        except (KeyError, ValueError):
            print("emule CI: refused an unexpected Docker operation", file=sys.stderr)
            return 2
        os.execv(command[0], command)
    if argv[:1] == ["--report-summary"]:
        if len(argv) != 2:
            print("emule CI: --report-summary takes one summary path", file=sys.stderr)
            return 2
        try:
            print(report_summary(json.loads(Path(argv[1]).read_text())), end="")
        except (OSError, ValueError, TypeError, KeyError):
            print("emule CI: the worker summary is missing or invalid", file=sys.stderr)
            return 2
        return 0
    args = arguments(argv)
    os.umask(0o077)
    worker = Worker(args)
    if worker.output == worker.source or worker.source in worker.output.parents:
        print("emule CI: output must be outside the source checkout", file=sys.stderr)
        return 2
    try:
        worker.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        (worker.private / "container").mkdir(parents=True)
        # The host-private parent stays 0700. The bind-mounted child must allow
        # container root to write without CAP_DAC_OVERRIDE on native Linux;
        # only this child is visible inside the container.
        (worker.private / "container").chmod(0o777)
    except OSError:
        print("emule CI: output must be a new writable directory", file=sys.stderr)
        return 2

    def interrupt(*_):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        result = worker.execute()
    except (Exception, KeyboardInterrupt):
        with (worker.private / "error.log").open("x", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        worker.summary["status"] = "error"
        result = 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    write_json(worker.output / "summary.json", worker.summary)
    print(json.dumps(worker.summary, allow_nan=False))
    return result


if __name__ == "__main__":
    sys.exit(main())
