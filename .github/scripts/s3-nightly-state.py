#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Read and update the successful scheduled S3 wheel publish marker."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_BUCKET = "tenstorrent-pypi"
DEFAULT_KEY = "tt-lang/nightly-state.json"
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
HEAD_OBJECT_ERROR_RE = re.compile(
    r"An error occurred \((?P<code>[^)]+)\) " r"when calling the HeadObject operation"
)


def _sha(value: str | None) -> str:
    resolved = (
        value
        if value
        else subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    if not SHA_RE.fullmatch(resolved):
        raise RuntimeError(f"expected a full 40-character commit SHA, got {resolved!r}")
    return resolved.lower()


def _write_outputs(values: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    stream = Path(output_path).open("a") if output_path else sys.stdout
    try:
        for name, value in values.items():
            print(f"{name}={value}", file=stream)
    finally:
        if output_path:
            stream.close()


def _aws(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.environ.get("AWS", "aws"), *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_marker(bucket: str, key: str) -> dict[str, object] | None:
    head_result = _aws(
        "s3api",
        "head-object",
        "--bucket",
        bucket,
        "--key",
        key,
    )
    if head_result.returncode != 0:
        error_match = HEAD_OBJECT_ERROR_RE.search(head_result.stderr)
        if error_match and error_match.group("code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise RuntimeError(
            f"failed to inspect s3://{bucket}/{key}: {head_result.stderr.strip()}"
        )

    result = _aws(
        "s3",
        "cp",
        f"s3://{bucket}/{key}",
        "-",
        "--only-show-errors",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to read s3://{bucket}/{key}: {result.stderr.strip()}"
        )
    try:
        marker = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid nightly marker JSON: {error}") from error
    if not isinstance(marker, dict):
        raise RuntimeError("invalid nightly marker JSON: expected an object")
    return marker


def check(args: argparse.Namespace) -> int:
    ttlang_sha = _sha(args.sha)
    if args.event != "schedule":
        _write_outputs(
            {
                "publish-needed": "true",
                "ttlang-sha": ttlang_sha,
                "previous-sha": "",
            }
        )
        print("Non-scheduled publish: source-state skip is disabled.")
        return 0

    marker = _read_marker(args.bucket, args.key)
    previous_sha = "" if marker is None else str(marker.get("ttlang_sha", ""))
    if previous_sha and not SHA_RE.fullmatch(previous_sha):
        raise RuntimeError("invalid nightly marker JSON: ttlang_sha is not a full SHA")
    previous_sha = previous_sha.lower()
    publish_needed = previous_sha != ttlang_sha
    _write_outputs(
        {
            "publish-needed": str(publish_needed).lower(),
            "ttlang-sha": ttlang_sha,
            "previous-sha": previous_sha,
        }
    )
    if publish_needed:
        print(
            f"Scheduled publish required: previous={previous_sha or 'none'} current={ttlang_sha}"
        )
    else:
        print(f"Scheduled publish skipped: {ttlang_sha} was already published.")
    return 0


def record(args: argparse.Namespace) -> int:
    if args.event != "schedule":
        raise RuntimeError("nightly state is recorded only for schedule events")
    marker = {
        "ttlang_sha": _sha(args.sha),
        "version": args.version,
        "run_id": args.run_id or os.environ.get("GITHUB_RUN_ID", ""),
        "published_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    payload = json.dumps(marker, sort_keys=True) + "\n"
    result = _aws(
        "s3",
        "cp",
        "-",
        f"s3://{args.bucket}/{args.key}",
        "--content-type",
        "application/json",
        "--only-show-errors",
        input_text=payload,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to write s3://{args.bucket}/{args.key}: {result.stderr.strip()}"
        )
    print(f"Recorded scheduled publish state for {marker['ttlang_sha']}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "record"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--event", default=os.environ.get("GITHUB_EVENT_NAME", "")
        )
        subparser.add_argument("--sha")
        subparser.add_argument(
            "--bucket", default=os.environ.get("TTLANG_S3_BUCKET", DEFAULT_BUCKET)
        )
        subparser.add_argument("--key", default=DEFAULT_KEY)
    record_parser = subparsers.choices["record"]
    record_parser.add_argument("--version", required=True)
    record_parser.add_argument("--run-id")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return check(args) if args.command == "check" else record(args)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
