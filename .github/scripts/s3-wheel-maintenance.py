#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Plan or delete redundant and date-selected S3 wheel object versions."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable

BUCKET = "tenstorrent-pypi"
PREFIX = "tt-lang/"
DEV_DATE_RE = re.compile(r"\.dev(?P<date>[0-9]{8})(?:\+[^-]+)?-")
CONFIRMATIONS = {
    "deduplicate": "delete-duplicate-versions",
    "remove-dev-range": "delete-dev-versions",
}


@dataclasses.dataclass(frozen=True)
class ObjectVersion:
    key: str
    version_id: str
    last_modified: str
    size: int | None
    etag: str | None
    is_latest: bool
    is_delete_marker: bool = False


@dataclasses.dataclass(frozen=True)
class Deletion:
    version: ObjectVersion
    reason: str
    retained_version_id: str | None = None


def _bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _date(value: str) -> datetime.date | None:
    if value == "":
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM-DD, got {value!r}"
        ) from error


def _aws(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.environ.get("AWS", "aws"), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _run_aws_json(*args: str) -> dict[str, object]:
    result = _aws(*args, "--output", "json")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "AWS CLI command failed")
    try:
        value = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as error:
        raise RuntimeError(f"AWS CLI returned invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("AWS CLI returned non-object JSON")
    return value


def list_object_versions(bucket: str = BUCKET) -> list[ObjectVersion]:
    versions: list[ObjectVersion] = []
    key_marker: str | None = None
    version_id_marker: str | None = None

    while True:
        arguments = [
            "s3api",
            "list-object-versions",
            "--bucket",
            bucket,
            "--prefix",
            PREFIX,
        ]
        if key_marker is not None:
            arguments.extend(["--key-marker", key_marker])
        if version_id_marker is not None:
            arguments.extend(["--version-id-marker", version_id_marker])
        response = _run_aws_json(*arguments)

        for raw_version in response.get("Versions", []):
            if not isinstance(raw_version, dict):
                raise RuntimeError("invalid Versions entry from AWS")
            versions.append(
                ObjectVersion(
                    key=str(raw_version["Key"]),
                    version_id=str(raw_version["VersionId"]),
                    last_modified=str(raw_version["LastModified"]),
                    size=int(raw_version["Size"]),
                    etag=str(raw_version["ETag"]),
                    is_latest=bool(raw_version.get("IsLatest", False)),
                )
            )
        for raw_marker in response.get("DeleteMarkers", []):
            if not isinstance(raw_marker, dict):
                raise RuntimeError("invalid DeleteMarkers entry from AWS")
            versions.append(
                ObjectVersion(
                    key=str(raw_marker["Key"]),
                    version_id=str(raw_marker["VersionId"]),
                    last_modified=str(raw_marker["LastModified"]),
                    size=None,
                    etag=None,
                    is_latest=bool(raw_marker.get("IsLatest", False)),
                    is_delete_marker=True,
                )
            )

        if not response.get("IsTruncated", False):
            return versions
        key_marker_value = response.get("NextKeyMarker")
        version_marker_value = response.get("NextVersionIdMarker")
        if not key_marker_value or not version_marker_value:
            raise RuntimeError("truncated AWS response omitted continuation markers")
        key_marker = str(key_marker_value)
        version_id_marker = str(version_marker_value)


def _newest(versions: Iterable[ObjectVersion]) -> ObjectVersion:
    return max(
        versions,
        key=lambda version: (
            version.last_modified,
            version.is_latest,
            version.version_id,
        ),
    )


def duplicate_deletions(versions: Iterable[ObjectVersion]) -> list[Deletion]:
    groups: dict[tuple[str, int, str], list[ObjectVersion]] = collections.defaultdict(
        list
    )
    for version in versions:
        if (
            version.is_delete_marker
            or not version.key.endswith(".whl")
            or version.size is None
            or version.etag is None
        ):
            continue
        groups[(version.key, version.size, version.etag)].append(version)

    deletions: list[Deletion] = []
    for matching_versions in groups.values():
        if len(matching_versions) < 2:
            continue
        retained = _newest(matching_versions)
        for version in matching_versions:
            if version.version_id != retained.version_id:
                deletions.append(
                    Deletion(
                        version=version,
                        reason="duplicate",
                        retained_version_id=retained.version_id,
                    )
                )
    return sorted(
        deletions,
        key=lambda deletion: (
            deletion.version.key,
            deletion.version.last_modified,
            deletion.version.version_id,
        ),
    )


def _dev_date(key: str) -> datetime.date | None:
    key_path = PurePosixPath(key)
    if key_path.parent != PurePosixPath("tt-lang") or key_path.suffix != ".whl":
        return None
    match = DEV_DATE_RE.search(key_path.name)
    if match is None:
        return None
    try:
        return datetime.datetime.strptime(match.group("date"), "%Y%m%d").date()
    except ValueError:
        # PEP 440 dev releases are arbitrary integers, even when they contain
        # eight digits. Only calendar-date dev releases belong to month views.
        return None


def date_range_deletions(
    versions: Iterable[ObjectVersion],
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[Deletion]:
    deletions = []
    for version in versions:
        dev_date = _dev_date(version.key)
        if dev_date is not None and start_date <= dev_date <= end_date:
            deletions.append(
                Deletion(version=version, reason=f"dev-date={dev_date.isoformat()}")
            )
    return sorted(
        deletions,
        key=lambda deletion: (
            deletion.version.key,
            deletion.version.last_modified,
            deletion.version.version_id,
        ),
    )


def delete_versions(
    deletions: list[Deletion],
    bucket: str = BUCKET,
) -> None:
    for offset in range(0, len(deletions), 1000):
        batch = deletions[offset : offset + 1000]
        request = {
            "Objects": [
                {
                    "Key": deletion.version.key,
                    "VersionId": deletion.version.version_id,
                }
                for deletion in batch
            ],
            "Quiet": True,
        }
        response = _run_aws_json(
            "s3api",
            "delete-objects",
            "--bucket",
            bucket,
            "--delete",
            json.dumps(request, separators=(",", ":")),
        )
        errors = response.get("Errors", [])
        if errors:
            raise RuntimeError(f"S3 version deletion failed: {json.dumps(errors)}")


def _affected_months(deletions: Iterable[Deletion]) -> list[str]:
    months = set()
    for deletion in deletions:
        dev_date = _dev_date(deletion.version.key)
        if dev_date is not None:
            months.add(dev_date.strftime("%Y-%m"))
    return sorted(months)


def _months_in_range(
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[str]:
    months = []
    current_year = start_date.year
    current_month = start_date.month
    while (current_year, current_month) <= (end_date.year, end_date.month):
        months.append(f"{current_year:04d}-{current_month:02d}")
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1
    return months


def _write_summary(
    operation: str,
    dry_run: bool,
    deletions: list[Deletion],
    months: list[str],
) -> None:
    total_bytes = sum(deletion.version.size or 0 for deletion in deletions)
    lines = [
        "### S3 wheel maintenance",
        "",
        f"- Operation: `{operation}`",
        f"- Dry run: `{str(dry_run).lower()}`",
        f"- Object versions selected: `{len(deletions)}`",
        f"- Selected bytes: `{total_bytes}`",
    ]
    if months:
        month_label = (
            "Affected month views"
            if operation == "remove-dev-range"
            else "Duplicate wheel months"
        )
        lines.append(f"- {month_label}: `{','.join(months)}`")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a") as summary:
            print("\n".join(lines), file=summary)
    print("\n".join(lines))


def _write_outputs(refresh_months: list[str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a") as output:
            print(f"refresh-months={','.join(refresh_months)}", file=output)


def _validate(args: argparse.Namespace) -> None:
    if args.github_ref != "refs/heads/main":
        raise RuntimeError(
            f"S3 wheel maintenance is restricted to refs/heads/main (got {args.github_ref or 'unset'})"
        )
    if args.operation == "remove-dev-range":
        if args.start_date is None or args.end_date is None:
            raise RuntimeError("remove-dev-range requires --start-date and --end-date")
        if args.start_date > args.end_date:
            raise RuntimeError("start date must not be after end date")
    elif args.start_date is not None or args.end_date is not None:
        raise RuntimeError("date arguments are valid only for remove-dev-range")

    if not args.dry_run:
        required_confirmation = CONFIRMATIONS[args.operation]
        if args.confirm != required_confirmation:
            raise RuntimeError(
                f"live {args.operation} requires --confirm {required_confirmation}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("deduplicate", "remove-dev-range"))
    parser.add_argument("--dry-run", type=_bool, default=True)
    parser.add_argument("--start-date", type=_date)
    parser.add_argument("--end-date", type=_date)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--github-ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--bucket", default=os.environ.get("TTLANG_S3_BUCKET", BUCKET))
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        _validate(args)
        versions = list_object_versions(args.bucket)
        if args.operation == "deduplicate":
            deletions = duplicate_deletions(versions)
            months = _affected_months(deletions)
        else:
            deletions = date_range_deletions(versions, args.start_date, args.end_date)
            months = _months_in_range(args.start_date, args.end_date)

        for deletion in deletions:
            retained = (
                f" retain={deletion.retained_version_id}"
                if deletion.retained_version_id
                else ""
            )
            print(
                f"{'WOULD DELETE' if args.dry_run else 'DELETE'} "
                f"s3://{args.bucket}/{deletion.version.key}"
                f"?versionId={deletion.version.version_id} "
                f"reason={deletion.reason}{retained}"
            )

        if not args.dry_run:
            if deletions:
                delete_versions(deletions, args.bucket)
        refresh_months = (
            months if args.operation == "remove-dev-range" and not args.dry_run else []
        )
        _write_outputs(refresh_months)
        _write_summary(args.operation, args.dry_run, deletions, months)
        return 0
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
