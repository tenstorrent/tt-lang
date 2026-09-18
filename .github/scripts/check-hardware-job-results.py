# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Check aggregate hardware results without treating runner setup as a test failure."""

from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from typing import Any

HARDWARE_JOB_NAME_MARKER = "Hardware Tests ("
RUNNER_SETUP_STEP_NAME = "Set up runner"
TESTS_COMPLETE_STEP_NAME = "Mark hardware tests complete"
FAILED_STEP_CONCLUSIONS = {
    "action_required",
    "cancelled",
    "failure",
    "stale",
    "timed_out",
}


class HardwareJobResult(Enum):
    SUCCESS = "success"
    RUNNER_SETUP_FAILURE = "runner_setup_failure"
    POST_SETUP_FAILURE = "post_setup_failure"


def _load_jobs(jobs_json: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    payloads: list[Any] = []
    position = 0
    while position < len(jobs_json):
        while position < len(jobs_json) and jobs_json[position].isspace():
            position += 1
        if position == len(jobs_json):
            break
        payload, position = decoder.raw_decode(jobs_json, position)
        payloads.append(payload)

    if len(payloads) == 1 and isinstance(payloads[0], list):
        pages = payloads[0]
    else:
        pages = payloads

    jobs: list[dict[str, Any]] = []

    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("jobs"), list):
            raise ValueError("expected an Actions jobs response or a list of responses")
        jobs.extend(page["jobs"])

    return jobs


def _find_step(job: dict[str, Any], step_name: str) -> dict[str, Any] | None:
    return next(
        (step for step in job.get("steps", []) if step.get("name") == step_name),
        None,
    )


def _classify_job(job: dict[str, Any]) -> tuple[HardwareJobResult, str]:
    setup_step = _find_step(job, RUNNER_SETUP_STEP_NAME)
    if setup_step is None:
        return HardwareJobResult.POST_SETUP_FAILURE, "has no runner setup result"

    setup_conclusion = setup_step.get("conclusion")
    if setup_conclusion != "success":
        return (
            HardwareJobResult.RUNNER_SETUP_FAILURE,
            f"runner setup concluded {setup_conclusion or 'without a result'}",
        )

    failed_steps = [
        step
        for step in job.get("steps", [])
        if step.get("conclusion") in FAILED_STEP_CONCLUSIONS
        and step.get("name") != RUNNER_SETUP_STEP_NAME
    ]
    if failed_steps:
        failed_step = failed_steps[0]
        return (
            HardwareJobResult.POST_SETUP_FAILURE,
            f"{failed_step.get('name')} concluded {failed_step.get('conclusion')}",
        )

    completion_step = _find_step(job, TESTS_COMPLETE_STEP_NAME)
    if completion_step is not None and completion_step.get("conclusion") == "success":
        return HardwareJobResult.SUCCESS, "completed all hardware tests"

    return (
        HardwareJobResult.POST_SETUP_FAILURE,
        f"did not complete {TESTS_COMPLETE_STEP_NAME}",
    )


def _hardware_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [job for job in jobs if HARDWARE_JOB_NAME_MARKER in str(job.get("name", ""))]


def check_hardware_jobs(jobs: list[dict[str, Any]], expected_job_count: int) -> bool:
    hardware_jobs = _hardware_jobs(jobs)
    if len(hardware_jobs) != expected_job_count:
        print(
            "::error::Expected "
            f"{expected_job_count} hardware jobs, found {len(hardware_jobs)}."
        )
        return False

    successful_jobs = 0
    post_setup_failures = 0
    for job in sorted(hardware_jobs, key=lambda hardware_job: hardware_job["name"]):
        job_name = job["name"]
        result, description = _classify_job(job)
        if result is HardwareJobResult.SUCCESS:
            successful_jobs += 1
            print(f"{job_name}: {description}.")
        elif result is HardwareJobResult.RUNNER_SETUP_FAILURE:
            print(f"::warning::{job_name}: {description}; hardware tests did not run.")
        else:
            post_setup_failures += 1
            print(f"::error::{job_name}: {description}.")

    if post_setup_failures:
        return False
    if successful_jobs == 0:
        print("::error::No hardware runner completed the test suite.")
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check aggregate GitHub Actions hardware job results from stdin."
    )
    parser.add_argument(
        "--expected-job-count",
        type=int,
        required=True,
        help="Number of hardware matrix jobs expected in the workflow run.",
    )
    arguments = parser.parse_args()

    try:
        jobs = _load_jobs(sys.stdin.read())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    return 0 if check_hardware_jobs(jobs, arguments.expected_job_count) else 1


if __name__ == "__main__":
    raise SystemExit(main())
