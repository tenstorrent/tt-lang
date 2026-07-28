# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for S3 wheel object-version maintenance."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / ".github" / "scripts" / "s3-wheel-maintenance.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "s3-wheel-maintenance.yml"

spec = importlib.util.spec_from_file_location("s3_wheel_maintenance", SCRIPT)
assert spec is not None and spec.loader is not None
maintenance = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = maintenance
spec.loader.exec_module(maintenance)


def _version(
    key: str,
    version_id: str,
    *,
    modified: str,
    size: int = 100,
    etag: str = '"etag"',
    latest: bool = False,
    marker: bool = False,
):
    return maintenance.ObjectVersion(
        key=key,
        version_id=version_id,
        last_modified=modified,
        size=None if marker else size,
        etag=None if marker else etag,
        is_latest=latest,
        is_delete_marker=marker,
    )


def test_deduplicate_keeps_newest_identical_version_per_key() -> None:
    key = "tt-lang/tt_lang-1.2.3-py3-none-any.whl"
    versions = [
        _version(key, "old", modified="2026-07-01T00:00:00Z"),
        _version(key, "new", modified="2026-07-02T00:00:00Z", latest=True),
        _version(
            key,
            "different",
            modified="2026-07-03T00:00:00Z",
            etag='"different"',
        ),
        _version(
            key,
            "marker",
            modified="2026-07-04T00:00:00Z",
            marker=True,
        ),
        _version(
            "tt-lang/other-1.2.3-py3-none-any.whl",
            "other",
            modified="2026-07-01T00:00:00Z",
        ),
    ]

    deletions = maintenance.duplicate_deletions(versions)

    assert [deletion.version.version_id for deletion in deletions] == ["old"]
    assert deletions[0].retained_version_id == "new"


def test_date_range_is_inclusive_and_limited_to_top_level_dev_wheels() -> None:
    versions = [
        _version(
            "tt-lang/tt_lang-1.2.3.dev20260701-py3-none-any.whl",
            "start",
            modified="2026-07-01T00:00:00Z",
        ),
        _version(
            "tt-lang/tt_lang-1.2.3.dev20260731+light-py3-none-any.whl",
            "end",
            modified="2026-07-31T00:00:00Z",
        ),
        _version(
            "tt-lang/tt_lang-1.2.3.dev20260801-py3-none-any.whl",
            "outside",
            modified="2026-08-01T00:00:00Z",
        ),
        _version(
            "tt-lang/ttmetal/abc/tt_lang-1.2.3.dev20260715-py3-none-any.whl",
            "per-sha",
            modified="2026-07-15T00:00:00Z",
        ),
        _version(
            "tt-lang/tt_lang-1.2.3.dev20260701-py3-none-any.whl",
            "marker",
            modified="2026-07-02T00:00:00Z",
            marker=True,
        ),
    ]

    deletions = maintenance.date_range_deletions(
        versions,
        maintenance._date("2026-07-01"),
        maintenance._date("2026-07-31"),
    )

    assert [deletion.version.version_id for deletion in deletions] == [
        "start",
        "marker",
        "end",
    ]
    assert maintenance._affected_months(deletions) == ["2026-07"]


def test_unparsable_dev_date_is_not_selected() -> None:
    versions = [
        _version(
            "tt-lang/tt_lang-1.2.3.dev20261332-py3-none-any.whl",
            "impossible-date",
            modified="2026-07-01T00:00:00Z",
        ),
        _version(
            "tt-lang/tt_lang-1.2.3.dev20260715-py3-none-any.whl",
            "real-date",
            modified="2026-07-15T00:00:00Z",
        ),
    ]

    deletions = maintenance.date_range_deletions(
        versions,
        maintenance._date("2026-07-01"),
        maintenance._date("2026-07-31"),
    )

    assert [deletion.version.version_id for deletion in deletions] == ["real-date"]
    assert maintenance._affected_months(deletions) == ["2026-07"]


def test_unparsable_dev_date_does_not_break_deduplicate() -> None:
    key = "tt-lang/tt_lang-1.2.3.dev12345678-py3-none-any.whl"
    versions = [
        _version(key, "old", modified="2026-07-01T00:00:00Z"),
        _version(key, "new", modified="2026-07-02T00:00:00Z", latest=True),
    ]

    deletions = maintenance.duplicate_deletions(versions)

    assert [deletion.version.version_id for deletion in deletions] == ["old"]
    assert maintenance._affected_months(deletions) == []


def test_month_range_includes_every_month_across_year_boundary() -> None:
    assert maintenance._months_in_range(
        maintenance._date("2026-12-31"),
        maintenance._date("2027-02-01"),
    ) == ["2026-12", "2027-01", "2027-02"]


def test_list_object_versions_follows_version_markers(monkeypatch) -> None:
    responses = [
        {
            "Versions": [
                {
                    "Key": "tt-lang/a.whl",
                    "VersionId": "one",
                    "LastModified": "2026-07-01T00:00:00Z",
                    "Size": 1,
                    "ETag": '"one"',
                }
            ],
            "IsTruncated": True,
            "NextKeyMarker": "tt-lang/a.whl",
            "NextVersionIdMarker": "one",
        },
        {
            "Versions": [
                {
                    "Key": "tt-lang/b.whl",
                    "VersionId": "two",
                    "LastModified": "2026-07-02T00:00:00Z",
                    "Size": 2,
                    "ETag": '"two"',
                }
            ],
            "IsTruncated": False,
        },
    ]
    calls = []

    def fake_aws_json(*args):
        calls.append(args)
        return responses.pop(0)

    monkeypatch.setattr(maintenance, "_run_aws_json", fake_aws_json)

    versions = maintenance.list_object_versions("bucket")

    assert [version.version_id for version in versions] == ["one", "two"]
    assert "--key-marker" in calls[1]
    assert "--version-id-marker" in calls[1]


def test_delete_versions_batches_at_s3_limit(monkeypatch) -> None:
    calls = []

    def fake_aws_json(*args):
        calls.append(args)
        return {}

    monkeypatch.setattr(maintenance, "_run_aws_json", fake_aws_json)
    deletions = [
        maintenance.Deletion(
            _version(
                f"tt-lang/wheel-{index}.whl",
                str(index),
                modified="2026-07-01T00:00:00Z",
            ),
            "test",
        )
        for index in range(1001)
    ]

    maintenance.delete_versions(deletions, "bucket")

    assert len(calls) == 2
    first_request = json.loads(calls[0][calls[0].index("--delete") + 1])
    second_request = json.loads(calls[1][calls[1].index("--delete") + 1])
    assert len(first_request["Objects"]) == 1000
    assert len(second_request["Objects"]) == 1


def test_delete_versions_rejects_partial_s3_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        maintenance,
        "_run_aws_json",
        lambda *_: {
            "Errors": [
                {
                    "Key": "tt-lang/wheel.whl",
                    "VersionId": "version",
                    "Code": "AccessDenied",
                }
            ]
        },
    )
    deletions = [
        maintenance.Deletion(
            _version(
                "tt-lang/wheel.whl",
                "version",
                modified="2026-07-01T00:00:00Z",
            ),
            "test",
        )
    ]

    with pytest.raises(RuntimeError, match="S3 version deletion failed"):
        maintenance.delete_versions(deletions, "bucket")


def test_deduplicate_summary_does_not_imply_month_view_refresh(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    maintenance._write_summary("deduplicate", True, [], ["2026-07"])

    output = capsys.readouterr().out
    assert "Duplicate wheel months: `2026-07`" in output
    assert "Affected month views" not in output


def test_dry_run_does_not_delete_or_refresh(monkeypatch) -> None:
    key = "tt-lang/tt_lang-1.2.3.dev20260701-py3-none-any.whl"
    versions = [
        _version(key, "old", modified="2026-07-01T00:00:00Z"),
        _version(key, "new", modified="2026-07-02T00:00:00Z"),
    ]
    monkeypatch.setattr(maintenance, "list_object_versions", lambda _: versions)
    monkeypatch.setattr(
        maintenance,
        "delete_versions",
        lambda *_: (_ for _ in ()).throw(AssertionError("delete called")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "deduplicate",
            "--dry-run",
            "true",
            "--github-ref",
            "refs/heads/main",
        ],
    )

    assert maintenance.main() == 0


def test_live_date_removal_requires_confirmation_before_listing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        maintenance,
        "list_object_versions",
        lambda _: (_ for _ in ()).throw(AssertionError("listing called")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "remove-dev-range",
            "--dry-run",
            "false",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-07-31",
            "--github-ref",
            "refs/heads/main",
        ],
    )

    assert maintenance.main() == 1


def test_live_date_removal_deletes_and_refreshes_affected_month(
    monkeypatch,
    tmp_path: Path,
) -> None:
    versions = [
        _version(
            "tt-lang/tt_lang-1.2.3.dev20260701-py3-none-any.whl",
            "version",
            modified="2026-07-01T00:00:00Z",
        )
    ]
    deleted = []
    output_path = tmp_path / "output"
    monkeypatch.setattr(maintenance, "list_object_versions", lambda _: versions)
    monkeypatch.setattr(
        maintenance,
        "delete_versions",
        lambda deletions, bucket: deleted.extend(deletions),
    )
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "remove-dev-range",
            "--dry-run",
            "false",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-07-31",
            "--confirm",
            "delete-dev-versions",
            "--github-ref",
            "refs/heads/main",
        ],
    )

    assert maintenance.main() == 0
    assert [deletion.version.version_id for deletion in deleted] == ["version"]
    assert output_path.read_text() == "refresh-months=2026-07\n"


def test_live_date_removal_refreshes_range_after_prior_deletion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "output"
    monkeypatch.setattr(maintenance, "list_object_versions", lambda _: [])
    monkeypatch.setattr(
        maintenance,
        "delete_versions",
        lambda *_: (_ for _ in ()).throw(AssertionError("delete called")),
    )
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "remove-dev-range",
            "--dry-run",
            "false",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-08-31",
            "--confirm",
            "delete-dev-versions",
            "--github-ref",
            "refs/heads/main",
        ],
    )

    assert maintenance.main() == 0
    assert output_path.read_text() == "refresh-months=2026-07,2026-08\n"


def test_maintenance_workflow_is_dry_by_default_and_has_no_inline_shell() -> None:
    workflow = WORKFLOW.read_text()
    dry_run_input = workflow.split("      dry_run:", 1)[1].split("      confirm:", 1)[0]
    assert "default: true" in dry_run_input
    assert "run: |" not in workflow
    for line in workflow.splitlines():
        if line.lstrip().startswith("run:"):
            assert "${{ inputs." not in line
    assert "delete-duplicate-versions or delete-dev-versions" in workflow
    assert ".github/scripts/s3-wheel-maintenance.py" in workflow
    assert "id: maintenance" in workflow
    assert (
        "AFFECTED_MONTHS: ${{ steps.maintenance.outputs.refresh-months }}" in workflow
    )
    assert 'refresh-s3-wheel-views.sh --months "$AFFECTED_MONTHS"' in workflow
