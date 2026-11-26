#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Dispatch private host mocks and publish an exact-SHA public commit status."""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid


PUBLIC_REPOSITORY = "tenstorrent/tt-lang"
PRIVATE_REPOSITORY = "tenstorrent/tt-lang-ops-and-models"
WORKFLOW = "k3-host-mock.yml"
STATUS_CONTEXT = "K3 host mock"
POLL_SECONDS = 20
WAIT_SECONDS = 50 * 60


def request(token, endpoint, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    api_request = urllib.request.Request(
        f"https://api.github.com/repos/{endpoint}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(api_request, timeout=30) as response:
            content = response.read()
            return json.loads(content) if content else {}
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"GitHub API returned HTTP {error.code}") from None


def wait_for_run(token, compiler_sha, correlation_id, deadline):
    title = f"K3 host mock | {compiler_sha} | {correlation_id}"
    run_id = None
    while time.monotonic() < deadline:
        if run_id is None:
            response = request(
                token,
                f"{PRIVATE_REPOSITORY}/actions/workflows/{WORKFLOW}/runs"
                "?event=workflow_dispatch&branch=main&per_page=100",
            )
            matching = [
                run
                for run in response["workflow_runs"]
                if run["display_title"] == title
                and run["head_branch"] == "main"
                and run["event"] == "workflow_dispatch"
            ]
            if len(matching) > 1:
                raise RuntimeError("ambiguous private run correlation")
            if matching:
                run_id = matching[0]["id"]
        if run_id is not None:
            run = request(token, f"{PRIVATE_REPOSITORY}/actions/runs/{run_id}")
            if run["display_title"] != title or run["head_branch"] != "main":
                raise RuntimeError("private run correlation changed")
            if run["status"] == "completed":
                return run["conclusion"] == "success"
        time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
    raise RuntimeError("private host-mock run exceeded the polling deadline")


def prepare_request():
    if (
        os.environ.get("GITHUB_REPOSITORY") != PUBLIC_REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME")
        not in {"push", "schedule", "workflow_dispatch"}
    ):
        raise RuntimeError("host-mock dispatch requires a trusted main workflow")
    compiler_sha = os.environ.get("K3_COMPILER_SHA") or os.environ["GITHUB_SHA"]
    if not re.fullmatch("[0-9a-f]{40}", compiler_sha):
        raise RuntimeError("compiler SHA must be a full lowercase SHA")
    public_token = os.environ["GH_TOKEN"]
    comparison = request(
        public_token, f"{PUBLIC_REPOSITORY}/compare/{compiler_sha}...main"
    )
    if comparison["status"] not in {"ahead", "identical"}:
        raise RuntimeError("compiler SHA is not an ancestor of public main")
    correlation_id = (
        f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-"
        f"{uuid.uuid4().hex}"
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"compiler_sha={compiler_sha}\ncorrelation_id={correlation_id}\n")
    publish(
        compiler_sha,
        "pending",
        "Host construction pending; device correctness/timing untested",
    )


def publish(compiler_sha, state, description):
    public_url = (
        f"https://github.com/{PUBLIC_REPOSITORY}/actions/runs/"
        f"{os.environ['GITHUB_RUN_ID']}"
    )
    request(
        os.environ["GH_TOKEN"],
        f"{PUBLIC_REPOSITORY}/statuses/{compiler_sha}",
        {
            "state": state,
            "context": STATUS_CONTEXT,
            "target_url": public_url,
            "description": description,
        },
    )


def dispatch_request():
    dispatch_token = os.environ["K3_DISPATCH_TOKEN"]
    compiler_sha = os.environ["K3_COMPILER_SHA"]
    correlation_id = os.environ["K3_CORRELATION_ID"]
    request(
        dispatch_token,
        f"{PRIVATE_REPOSITORY}/actions/workflows/{WORKFLOW}/dispatches",
        {
            "ref": "main",
            "inputs": {"compiler_sha": compiler_sha, "correlation_id": correlation_id},
        },
    )
    return wait_for_run(
        dispatch_token, compiler_sha, correlation_id, time.monotonic() + WAIT_SECONDS
    )


def finish_request():
    compiler_sha = os.environ["K3_COMPILER_SHA"]
    correlation_id = os.environ["K3_CORRELATION_ID"]
    succeeded = os.environ["K3_RUN_OUTCOME"] == "success"
    publish(
        compiler_sha,
        "success" if succeeded else "failure",
        (
            "5/5 host mocks passed; device correctness/timing untested"
            if succeeded
            else "Host mocks failed/incomplete; device correctness/timing untested"
        ),
    )
    summary = {
        "compiler_sha": compiler_sha,
        "correlation_id": correlation_id,
        "result": "success" if succeeded else "failure",
        "device_correctness": "untested",
        "device_timing": "untested",
    }
    print(json.dumps(summary))
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary_file:
        summary_file.write("```json\n" + json.dumps(summary, indent=2) + "\n```\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "dispatch", "finish"))
    operation = parser.parse_args().operation
    try:
        if operation == "prepare":
            prepare_request()
        elif operation == "dispatch":
            return 0 if dispatch_request() else 1
        else:
            finish_request()
    except (RuntimeError, OSError, KeyError, ValueError):
        print(
            "Host-mock request failed; inspect credential setup and private run state."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
