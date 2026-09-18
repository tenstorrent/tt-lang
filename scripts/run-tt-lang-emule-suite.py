#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run compiler test suites inside the configured tt-emule container."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_STACK = Path("/opt/tt-emule-runtime/stack.json")
SUITES = {
    "mlir": ("check-ttlang-mlir", "ttlang-mlir-report.xml"),
    "bindings": ("check-ttlang-python-bindings", "python-bindings-report.xml"),
    "packaging": ("check-ttlang-packaging", "packaging-pytest-report.xml"),
    "pytest": ("check-ttlang-pytest", "pytest-report.xml"),
    "me2e": ("check-ttlang-me2e", "me2e-report.xml"),
    "python-lit": ("check-ttlang-python-lit", "python-lit-report.xml"),
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def compiler_provenance():
    sha = os.environ.get("TTLANG_EMULE_COMPILER_SHA")
    dirty = os.environ.get("TTLANG_EMULE_COMPILER_DIRTY")
    if sha and dirty in ("0", "1"):
        return {"sha": sha, "dirty": dirty == "1", "source": "launcher"}
    source_dir = os.environ.get("TTLANG_EMULE_SOURCE_DIR", str(REPO_ROOT))
    try:
        sha = subprocess.check_output(
            ["git", "-C", source_dir, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", source_dir, "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"sha": sha, "dirty": bool(dirty), "source": "git"}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None, "source": "unavailable"}


def runtime_provenance():
    try:
        return json.loads(RUNTIME_STACK.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"unavailable": str(error)}


def junit_counts(report):
    root = ET.parse(report).getroot()
    if root.tag not in ("testsuite", "testsuites"):
        raise ValueError("expected a JUnit testsuite or testsuites root")
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError("JUnit report contains no test cases")
    counts = {"tests": len(cases), "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for case in cases:
        if case.find("error") is not None:
            counts["errors"] += 1
        elif case.find("failure") is not None:
            counts["failed"] += 1
        elif case.find("skipped") is not None:
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    return counts


def write_output(text, log):
    log.write(text)
    log.flush()
    print(text, end="", flush=True)


def run_command(command, log_path):
    with log_path.open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        except OSError as error:
            write_output(f"Cannot start suite: {error}\n", log)
            return 127, False
        try:
            for line in process.stdout:
                write_output(line, log)
            code = process.wait()
            return code, code == -signal.SIGINT
        except KeyboardInterrupt:
            write_output("\nInterrupted; stopping the active suite.\n", log)
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                remaining, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                remaining, _ = process.communicate()
            if remaining:
                write_output(remaining, log)
            return 130, True
        finally:
            process.stdout.close()


def run_suite(name, build_dir, reports_dir):
    target, report_name = SUITES[name]
    source_report = build_dir / "test" / report_name
    result = {"suite": name, "target": target, "log": f"{name}.log"}
    started = time.monotonic()
    print(f"=== RUN {target} ===", flush=True)
    # A failed build may never invoke the test runner or replace its old report.
    source_report.unlink(missing_ok=True)
    code, interrupted = run_command(
        ["cmake", "--build", str(build_dir), "--target", target],
        reports_dir / result["log"],
    )
    result.update(
        exit_code=code,
        duration_seconds=round(time.monotonic() - started, 3),
        status="interrupted" if interrupted else "failed",
        counts=None,
        junit=None,
    )
    if source_report.is_file():
        destination = reports_dir / report_name
        with source_report.open("rb") as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output)
        result["junit"] = report_name
        try:
            result["counts"] = junit_counts(destination)
        except (ET.ParseError, ValueError) as error:
            result["report_error"] = f"Invalid JUnit report: {error}"
    else:
        result["report_error"] = "Suite did not produce a JUnit report"
    if code == 0 and result["counts"] is not None and not interrupted:
        counts = result["counts"]
        if not counts["failed"] and not counts["errors"]:
            result["status"] = "passed"
    print(f"=== {result['status'].upper()} {target} (exit {code}) ===", flush=True)
    if result.get("report_error"):
        print(result["report_error"], file=sys.stderr, flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run compiler test suites through the configured tt-emule runtime"
    )
    parser.add_argument("--build-dir", type=Path, default=Path("/ttlang-build"))
    parser.add_argument("--reports-dir", type=Path, default=Path("/ttlang-reports"))
    parser.add_argument(
        "--suite",
        action="append",
        choices=SUITES,
        help="run this suite; repeat to select several (default: all six)",
    )
    return parser.parse_args()


def main():
    arguments = parse_args()
    summary = None
    try:
        if os.environ.get("TT_METAL_EMULE_MODE") != "1":
            raise ValueError(
                "TT_METAL_EMULE_MODE=1 is required; use tt-lang-sim --backend=emule --test"
            )
        for variable in (
            "TTLANG_COMPILE_ONLY",
            "TTLANG_SIM_ONLY",
            "TT_METAL_SIMULATOR",
        ):
            if variable in os.environ:
                raise ValueError(f"unset {variable} to test tt-emule execution")
        build_dir = arguments.build_dir.resolve()
        if not (build_dir / "CMakeCache.txt").is_file():
            raise ValueError(
                f"configured compiler build directory not found: {build_dir}"
            )
        if shutil.which("cmake") is None:
            raise ValueError("cmake is unavailable in the configured runtime")
        reports_dir = arguments.reports_dir.resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        selected = list(dict.fromkeys(arguments.suite or SUITES))
        expected_files = ["summary.json"]
        for name in selected:
            expected_files.extend((f"{name}.log", SUITES[name][1]))
        for name in expected_files:
            if (reports_dir / name).exists():
                raise ValueError(f"report already exists: {reports_dir / name}")
        initial_summary = {
            "schema_version": 1,
            "started_at": timestamp(),
            "compiler": compiler_provenance(),
            "runtime": runtime_provenance(),
            "requested_suites": selected,
            "suites": [],
            "status": "running",
        }
        summary_path = reports_dir / "summary.json"
        with summary_path.open("x", encoding="utf-8") as output:
            json.dump(initial_summary, output, indent=2)
            output.write("\n")
        summary = initial_summary
        for name in selected:
            result = run_suite(name, build_dir, reports_dir)
            summary["suites"].append(result)
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            if result["status"] == "interrupted":
                break
        statuses = [result["status"] for result in summary["suites"]]
        if "interrupted" in statuses:
            summary["status"] = "interrupted"
        else:
            summary["status"] = (
                "passed" if all(status == "passed" for status in statuses) else "failed"
            )
        summary["finished_at"] = timestamp()
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Reports: {reports_dir}", flush=True)
        if summary["status"] == "interrupted":
            return 130
        return 0 if summary["status"] == "passed" else 1
    except (OSError, ValueError, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        message = "interrupted" if interrupted else str(error)
        if summary is not None:
            summary.update(
                status="interrupted" if interrupted else "failed",
                finished_at=timestamp(),
                error=message,
            )
            try:
                summary_path.write_text(
                    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                )
            except OSError as report_error:
                print(f"Cannot finalize summary: {report_error}", file=sys.stderr)
        print(f"tt-lang emule suite: {message}", file=sys.stderr)
        return 130 if interrupted else 2


if __name__ == "__main__":
    sys.exit(main())
