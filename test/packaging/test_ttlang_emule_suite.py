# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run-tt-lang-emule-suite.py"
REPORTS = {
    "check-ttlang-mlir": "ttlang-mlir-report.xml",
    "check-ttlang-python-bindings": "python-bindings-report.xml",
    "check-ttlang-packaging": "packaging-pytest-report.xml",
    "check-ttlang-pytest": "pytest-report.xml",
    "check-ttlang-me2e": "me2e-report.xml",
    "check-ttlang-python-lit": "python-lit-report.xml",
}


@pytest.fixture
def suite_environment(tmp_path):
    build = tmp_path / "build"
    (build / "test").mkdir(parents=True)
    (build / "CMakeCache.txt").write_text("# configured build\n", encoding="utf-8")
    commands = tmp_path / "commands"
    binary = tmp_path / "bin"
    binary.mkdir()
    cmake = binary / "cmake"
    cmake.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, signal, sys, time\n"
        f"reports = {REPORTS!r}\n"
        "target = sys.argv[-1]\n"
        "with pathlib.Path(os.environ['COMMAND_LOG']).open('a') as output:\n"
        "    output.write(target + '\\n')\n"
        "print('running ' + target, flush=True)\n"
        "if os.environ.get('WAIT_TARGET') == target:\n"
        "    time.sleep(30)\n"
        "if os.environ.get('INTERRUPT_TARGET') == target:\n"
        "    os.kill(os.getpid(), signal.SIGINT)\n"
        "if os.environ.get('MISSING_REPORT') != target:\n"
        "    path = pathlib.Path(sys.argv[2]) / 'test' / reports[target]\n"
        "    path.write_text(os.environ.get('REPORT_XML', "
        "'<testsuite><testcase name=\"passed\"/></testsuite>'))\n"
        "sys.exit(9 if target == os.environ.get('FAIL_TARGET') else 0)\n",
        encoding="utf-8",
    )
    cmake.chmod(0o755)
    env = os.environ.copy()
    for key in ("TTLANG_COMPILE_ONLY", "TTLANG_SIM_ONLY", "TT_METAL_SIMULATOR"):
        env.pop(key, None)
    env.update(
        PATH=str(binary) + os.pathsep + env.get("PATH", ""),
        TT_METAL_EMULE_MODE="1",
        TTLANG_EMULE_COMPILER_SHA="a" * 40,
        TTLANG_EMULE_COMPILER_DIRTY="1",
        COMMAND_LOG=str(commands),
    )
    return build, tmp_path / "reports", commands, env


def run_suite(environment, *arguments):
    build, reports, _, env = environment
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--build-dir",
            str(build),
            "--reports-dir",
            str(reports),
            *arguments,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_all_six_suites_run_after_failure(suite_environment):
    _, reports, commands, env = suite_environment
    env["FAIL_TARGET"] = "check-ttlang-pytest"

    result = run_suite(suite_environment)

    assert result.returncode == 1, result.stderr
    assert commands.read_text().splitlines() == list(REPORTS)
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert len(summary["suites"]) == 6
    assert summary["suites"][3]["exit_code"] == 9
    assert summary["suites"][3]["status"] == "failed"
    assert summary["suites"][5]["status"] == "passed"
    assert summary["compiler"] == {"sha": "a" * 40, "dirty": True, "source": "launcher"}
    assert (reports / "pytest.log").read_text() == "running check-ttlang-pytest\n"
    assert "running check-ttlang-pytest" in result.stdout


def test_selected_suites_summarize_junit_counts(suite_environment):
    _, reports, commands, env = suite_environment
    env["REPORT_XML"] = (
        '<testsuites><testsuite tests="999"><testcase name="pass"/>'
        '<testcase name="skip"><skipped message="unsupported"/></testcase>'
        '<testcase name="xfail"><skipped type="pytest.xfail"/></testcase>'
        "</testsuite></testsuites>"
    )

    result = run_suite(suite_environment, "--suite", "me2e", "--suite", "bindings")

    assert result.returncode == 0, result.stderr
    assert commands.read_text().splitlines() == [
        "check-ttlang-me2e",
        "check-ttlang-python-bindings",
    ]
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["suites"][0]["counts"] == {
        "tests": 3,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 2,
    }
    assert not (reports / "pytest-report.xml").exists()


def test_runtime_provenance_records_built_image_stack(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("emule_suite_runner", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    path = tmp_path / "stack.json"
    stack = {"emulator_commit": "b" * 40, "metal_commit": "c" * 40}
    path.write_text(json.dumps(stack))
    monkeypatch.setattr(runner, "RUNTIME_STACK", path)

    assert runner.runtime_provenance() == stack

    path.unlink()
    assert "unavailable" in runner.runtime_provenance()


@pytest.mark.parametrize("stale", [False, True])
def test_missing_junit_does_not_pass_or_copy_stale_report(suite_environment, stale):
    build, reports, _, env = suite_environment
    env["MISSING_REPORT"] = "check-ttlang-mlir"
    if stale:
        (build / "test" / REPORTS["check-ttlang-mlir"]).write_text(
            '<testsuite><testcase name="old-pass"/></testsuite>'
        )

    result = run_suite(suite_environment, "--suite", "mlir")

    assert result.returncode == 1
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["suites"][0]["status"] == "failed"
    assert summary["suites"][0]["counts"] is None
    assert not (reports / REPORTS["check-ttlang-mlir"]).exists()
    assert "did not produce a JUnit report" in result.stderr


@pytest.mark.parametrize(
    "report_xml",
    [
        "not XML",
        "<testsuite/>",
        '<testsuite><testcase name="failure"><failure/></testcase></testsuite>',
        '<testsuite><testcase name="error"><error/></testcase></testsuite>',
    ],
)
def test_invalid_or_failing_junit_cannot_pass(suite_environment, report_xml):
    _, reports, _, env = suite_environment
    env["REPORT_XML"] = report_xml

    result = run_suite(suite_environment, "--suite", "packaging")

    assert result.returncode == 1
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["suites"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "variable,value",
    [
        ("TT_METAL_EMULE_MODE", "0"),
        ("TTLANG_COMPILE_ONLY", "1"),
        ("TTLANG_SIM_ONLY", "1"),
        ("TT_METAL_SIMULATOR", "/fake/libttsim.so"),
    ],
)
def test_rejects_non_emule_execution_before_running(suite_environment, variable, value):
    _, reports, commands, env = suite_environment
    env[variable] = value

    result = run_suite(suite_environment)

    assert result.returncode == 2
    assert variable in result.stderr
    assert not commands.exists()
    assert not reports.exists()


def test_missing_build_fails_cleanly(suite_environment):
    build, _, commands, _ = suite_environment
    (build / "CMakeCache.txt").unlink()

    result = run_suite(suite_environment)

    assert result.returncode == 2
    assert "configured compiler build directory not found" in result.stderr
    assert not commands.exists()


def test_missing_cmake_fails_cleanly(suite_environment):
    _, _, commands, env = suite_environment
    env["PATH"] = ""

    result = run_suite(suite_environment)

    assert result.returncode == 2
    assert "cmake is unavailable" in result.stderr
    assert not commands.exists()


def test_report_filesystem_error_finalizes_partial_summary(suite_environment):
    build, reports, commands, _ = suite_environment
    report_path = build / "test" / REPORTS["check-ttlang-python-bindings"]
    report_path.mkdir()

    result = run_suite(suite_environment)

    assert result.returncode == 2
    assert commands.read_text().splitlines() == ["check-ttlang-mlir"]
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["finished_at"]
    assert str(report_path) in summary["error"]
    assert len(summary["suites"]) == 1
    assert summary["suites"][0]["status"] == "passed"


def test_interrupt_saves_partial_summary_and_stops(suite_environment):
    _, reports, commands, env = suite_environment
    env["INTERRUPT_TARGET"] = "check-ttlang-python-bindings"

    result = run_suite(suite_environment)

    assert result.returncode == 130
    assert commands.read_text().splitlines() == list(REPORTS)[:2]
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["status"] == "interrupted"
    assert len(summary["suites"]) == 2


def test_keyboard_interrupt_stops_active_subprocess(suite_environment):
    build, reports, commands, env = suite_environment
    env["WAIT_TARGET"] = "check-ttlang-mlir"
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "--build-dir",
            str(build),
            "--reports-dir",
            str(reports),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not commands.exists() and process.poll() is None:
            assert time.monotonic() < deadline, "suite did not start"
            time.sleep(0.01)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    assert process.returncode == 130, stdout + stderr
    assert commands.read_text().splitlines() == ["check-ttlang-mlir"]
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["status"] == "interrupted"
    assert "stopping the active suite" in stdout


def test_existing_reports_are_not_overwritten(suite_environment):
    _, reports, commands, _ = suite_environment
    reports.mkdir()
    log = reports / "mlir.log"
    log.write_text("previous run")

    result = run_suite(suite_environment)

    assert result.returncode == 2
    assert "report already exists" in result.stderr
    assert log.read_text() == "previous run"
    assert not commands.exists()


def test_help_does_not_require_emulation_environment(suite_environment):
    _, _, commands, env = suite_environment
    env.pop("TT_METAL_EMULE_MODE")

    result = run_suite(suite_environment, "--help")

    assert result.returncode == 0
    assert "default: all six" in " ".join(result.stdout.split())
    assert not commands.exists()
