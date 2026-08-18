# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for aggregate hardware job result handling."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

CHECK_HARDWARE_JOB_RESULTS = (
    REPO_ROOT / ".github" / "scripts" / "check-hardware-job-results.py"
)


def _hardware_job(
    runner_name: str,
    *,
    setup_conclusion: str = "success",
    tests_complete: bool = True,
    failed_step_name: str | None = None,
    complete_runner_conclusion: str = "success",
) -> dict[str, object]:
    steps: list[dict[str, str]] = [
        {"name": "Set up job", "conclusion": "success"},
        {"name": "Set up runner", "conclusion": setup_conclusion},
    ]
    if setup_conclusion == "success":
        steps.append({"name": "Initialize containers", "conclusion": "success"})
        if failed_step_name is not None:
            steps.append({"name": failed_step_name, "conclusion": "failure"})
        steps.append(
            {
                "name": "Mark hardware tests complete",
                "conclusion": "success" if tests_complete else "skipped",
            }
        )
        steps.append(
            {
                "name": "Complete runner",
                "conclusion": complete_runner_conclusion,
            }
        )

    caller_job = "test-exabox" if runner_name == "galaxy-bh" else "test-hardware"
    return {
        "name": f"build / {caller_job} / Hardware Tests ({runner_name})",
        "steps": steps,
    }


def _run_check(
    jobs: list[dict[str, object]],
    *,
    expected_job_count: int = 2,
    optional_runners: list[str] | None = None,
    separate_pages: bool = False,
) -> subprocess.CompletedProcess[str]:
    if separate_pages:
        jobs_json = "\n".join(
            json.dumps({"total_count": len(jobs), "jobs": [job]}) for job in jobs
        )
    else:
        jobs_json = json.dumps([{"total_count": len(jobs), "jobs": jobs}])
    command = [
        sys.executable,
        str(CHECK_HARDWARE_JOB_RESULTS),
        "--expected-job-count",
        str(expected_job_count),
    ]
    for optional_runner in optional_runners or []:
        command.extend(["--optional-runner", optional_runner])
    return subprocess.run(
        command,
        input=jobs_json,
        text=True,
        capture_output=True,
        check=False,
    )


def test_both_hardware_jobs_succeed() -> None:
    result = _run_check(
        [_hardware_job("n150"), _hardware_job("galaxy-bh")],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("completed all hardware tests") == 2


def test_single_expected_hardware_job_succeeds() -> None:
    result = _run_check(
        [_hardware_job("n150")],
        expected_job_count=1,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "completed all hardware tests" in result.stdout


def test_paginated_hardware_jobs_succeed() -> None:
    result = _run_check(
        [_hardware_job("n150"), _hardware_job("galaxy-bh")],
        separate_pages=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_one_setup_failure_and_one_success_succeeds() -> None:
    result = _run_check(
        [
            _hardware_job("n150", setup_conclusion="cancelled", tests_complete=False),
            _hardware_job("galaxy-bh"),
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning::" in result.stdout
    assert "hardware tests did not run" in result.stdout


def test_both_setup_failures_fail() -> None:
    result = _run_check(
        [
            _hardware_job("n150", setup_conclusion="failure", tests_complete=False),
            _hardware_job(
                "galaxy-bh", setup_conclusion="cancelled", tests_complete=False
            ),
        ],
    )

    assert result.returncode == 1
    assert "No required hardware runner completed the test suite" in result.stdout


@pytest.mark.parametrize(
    "optional_job",
    [
        _hardware_job(
            "bh-loudbox-viommu",
            setup_conclusion="failure",
            tests_complete=False,
        ),
        _hardware_job(
            "bh-loudbox-viommu",
            tests_complete=False,
            failed_step_name="Build tt-lang",
        ),
        _hardware_job("bh-loudbox-viommu", tests_complete=False),
        _hardware_job("bh-loudbox-viommu", complete_runner_conclusion="failure"),
    ],
    ids=["setup", "test-step", "incomplete", "runner-completion"],
)
def test_optional_runner_failure_succeeds(optional_job: dict[str, object]) -> None:
    result = _run_check(
        [_hardware_job("n150"), optional_job],
        optional_runners=["bh-loudbox-viommu"],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "optional hardware job" in result.stdout


def test_optional_runner_does_not_replace_required_success() -> None:
    result = _run_check(
        [
            _hardware_job("n150", setup_conclusion="failure", tests_complete=False),
            _hardware_job("bh-loudbox-viommu"),
        ],
        optional_runners=["bh-loudbox-viommu"],
    )

    assert result.returncode == 1
    assert "No required hardware runner completed the test suite" in result.stdout


@pytest.mark.parametrize("failed_runner", ["n150", "galaxy-bh"])
def test_post_setup_failure_fails(failed_runner: str) -> None:
    other_runner = "galaxy-bh" if failed_runner == "n150" else "n150"
    result = _run_check(
        [
            _hardware_job(
                failed_runner,
                tests_complete=False,
                failed_step_name="Build tt-lang",
            ),
            _hardware_job(other_runner),
        ],
    )

    assert result.returncode == 1
    assert "Build tt-lang concluded failure" in result.stdout


def test_incomplete_job_after_successful_setup_fails() -> None:
    result = _run_check(
        [
            _hardware_job("n150", tests_complete=False),
            _hardware_job("galaxy-bh"),
        ],
    )

    assert result.returncode == 1
    assert "did not complete Mark hardware tests complete" in result.stdout


def test_runner_completion_failure_fails() -> None:
    result = _run_check(
        [
            _hardware_job("n150", complete_runner_conclusion="failure"),
            _hardware_job("galaxy-bh"),
        ],
    )

    assert result.returncode == 1
    assert "Complete runner concluded failure" in result.stdout


def test_missing_hardware_job_fails() -> None:
    result = _run_check(
        [_hardware_job("n150")],
    )

    assert result.returncode == 1
    assert "Expected 2 hardware jobs, found 1" in result.stdout
