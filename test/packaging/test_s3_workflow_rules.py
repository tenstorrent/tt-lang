# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Rules for the workflows that can reach the S3 PyPI bucket.

A job that can get AWS credentials (``id-token: write``) runs only on main,
dry runs included, and dispatch inputs reach shell steps through ``env:``.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import yaml

from conftest import REPO_ROOT

WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
# Every workflow with a job that can get AWS credentials for the bucket.
S3_WORKFLOWS = (
    "call-ttmetal-light-wheel.yml",
    "publish-s3-pypi.yml",
    "s3-pypi-ops.yml",
    "s3-wheel-maintenance.yml",
    "ttmetal-light-on-demand.yml",
)
# The XLA light wheel workflow takes the same tt-metal SHA input.
INPUT_WORKFLOWS = (*S3_WORKFLOWS, "ttmetal-light-xla-on-demand.yml")
MAIN_ONLY = "github.ref == 'refs/heads/main'"
INPUT_EXPRESSION = re.compile(r"\$\{\{[^}]*\binputs\.")


def _load(workflow_name: str) -> dict[str, Any]:
    return yaml.safe_load((WORKFLOWS_DIR / workflow_name).read_text())


def _can_get_aws_credentials(job: dict[str, Any], default_permissions: Any) -> bool:
    # Job permissions replace the workflow's; without them the job inherits.
    permissions = job.get("permissions", default_permissions)
    if permissions == "write-all":
        return True
    return isinstance(permissions, dict) and permissions.get("id-token") == "write"


def _split_top_level(expression: str, operator: str) -> list[str]:
    """Split an Actions expression on an operator outside parentheses."""
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(expression):
        if expression[index] == "(":
            depth += 1
        elif expression[index] == ")":
            depth -= 1
        elif depth == 0 and expression.startswith(operator, index):
            parts.append(expression[start:index].strip())
            start = index + len(operator)
            index = start
            continue
        index += 1
    parts.append(expression[start:].strip())
    return parts


def _requires_main(condition: Any) -> bool:
    """Whether a job condition is false on every ref other than main."""
    expression = str(condition or "").strip()
    if expression.startswith("${{") and expression.endswith("}}"):
        expression = expression[3:-2]
    if len(_split_top_level(expression, "||")) != 1:
        return False
    return MAIN_ONLY in _split_top_level(expression, "&&")


def test_every_workflow_that_can_reach_the_bucket_is_checked() -> None:
    reaching_workflows = sorted(
        path.name
        for path in WORKFLOWS_DIR.glob("*.yml")
        if "configure-tt-s3-credentials" in path.read_text()
        or "configure-aws-credentials" in path.read_text()
    )

    assert reaching_workflows == sorted(S3_WORKFLOWS)


@pytest.mark.parametrize("workflow_name", S3_WORKFLOWS)
def test_jobs_that_can_get_aws_credentials_run_only_on_main(
    workflow_name: str,
) -> None:
    workflow = _load(workflow_name)
    default_permissions = workflow.get("permissions")
    credential_jobs = [
        name
        for name, job in workflow["jobs"].items()
        if _can_get_aws_credentials(job, default_permissions)
    ]

    assert credential_jobs
    for name in credential_jobs:
        assert _requires_main(workflow["jobs"][name].get("if")), name


@pytest.mark.parametrize("workflow_name", INPUT_WORKFLOWS)
def test_dispatch_inputs_reach_shell_steps_through_env(workflow_name: str) -> None:
    workflow = _load(workflow_name)

    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            assert not INPUT_EXPRESSION.search(step.get("run", "")), (
                job_name,
                step.get("name"),
            )


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("${{ github.ref == 'refs/heads/main' }}", True),
        ("${{ always() && github.ref == 'refs/heads/main' }}", True),
        ("${{ (a || b) && github.ref == 'refs/heads/main' }}", True),
        ("${{ dry_run == 'true' || github.ref == 'refs/heads/main' }}", False),
        ("${{ a || b && github.ref == 'refs/heads/main' }}", False),
        ("${{ github.event_name == 'schedule' }}", False),
        (None, False),
    ],
)
def test_main_only_condition_check(condition: str | None, expected: bool) -> None:
    assert _requires_main(condition) is expected
