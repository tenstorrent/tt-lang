# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for packaging helper scripts used by wheel workflows."""

from __future__ import annotations

import datetime
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_script(
    script: Path, *args: str, cwd: Path = REPO_ROOT
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_wheel(dist_dir: Path, filename: str, metadata: str) -> Path:
    wheel_path = dist_dir / filename
    dist_info = filename.split("-", 1)[0] + "-0.0.0.dist-info"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(f"{dist_info}/METADATA", metadata)
    return wheel_path


def test_check_wheel_ttnn_metadata_matches_requirement_name(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525-py3-none-any.whl",
        "Metadata-Version: 2.1\nRequires-Dist: ttnn-foo >= 1\n",
    )

    script = REPO_ROOT / ".github" / "scripts" / "check-wheel-ttnn-metadata.py"
    result = _run_script(script, "--mode", "pypi", "--dist-dir", str(dist_dir))

    assert result.returncode != 0
    assert "default wheel metadata must require ttnn" in result.stderr


def test_check_wheel_ttnn_metadata_rejects_external_payload(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel_path = _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525+light-py3-none-any.whl",
        "Metadata-Version: 2.1\n",
    )
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.writestr("ttnn/__init__.py", "")

    script = REPO_ROOT / ".github" / "scripts" / "check-wheel-ttnn-metadata.py"
    result = _run_script(script, "--mode", "external", "--dist-dir", str(dist_dir))

    assert result.returncode != 0
    assert "external wheel must not bundle a ttnn payload" in result.stderr


def test_check_light_metapackage_parses_requires_dist(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir,
        "tt_lang_light-0.71.0.dev20260525-py3-none-any.whl",
        (
            "Metadata-Version: 2.1\n"
            "Requires-Dist: tt-lang == 0.71.0.dev20260525+light ; "
            'python_version >= "3.12"\n'
        ),
    )

    script = REPO_ROOT / ".github" / "scripts" / "check-light-metapackage.py"
    result = _run_script(
        script,
        "--dist-dir",
        str(dist_dir),
        "--expect-ttlang-version",
        "0.71.0.dev20260525+light",
    )

    assert result.returncode == 0, result.stderr


def test_compute_nightly_version_uses_latest_reachable_tag(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "file.txt").write_text("first\n")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "first"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "v1.2.3"], cwd=tmp_path, check=True)
    (tmp_path / "file.txt").write_text("second\n")
    subprocess.run(["git", "commit", "-am", "second"], cwd=tmp_path, check=True)

    script = REPO_ROOT / ".github" / "scripts" / "compute-nightly-version.py"
    result = _run_script(script, cwd=tmp_path)
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"1.2.3.dev{today}"


def test_internal_wheel_metadata_fails_when_git_version_cannot_be_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(REPO_ROOT / "packaging"))
    import internal_wheel_metadata

    monkeypatch.delenv("TTLANG_PRETEND_VERSION", raising=False)

    def fail_git(*_args: object, **_kwargs: object) -> str:
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr(subprocess, "check_output", fail_git)

    with pytest.raises(SystemExit, match="failed to derive internal wheel version"):
        internal_wheel_metadata.get_version(REPO_ROOT)
