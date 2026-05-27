# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tt-lang-light metapackage metadata."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIGHT_ROOT = REPO_ROOT / "packaging" / "light"


def _run_egg_info(
    tmp_path: Path,
    env_updates: dict[str, str],
    env_removals: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in env_removals:
        environment.pop(name, None)
    environment.update(env_updates)

    egg_base = tmp_path / "egg-info"
    egg_base.mkdir()
    return subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(egg_base)],
        cwd=LIGHT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _requires_text(tmp_path: Path) -> str:
    matches = list((tmp_path / "egg-info").glob("*.egg-info/requires.txt"))
    assert len(matches) == 1
    return matches[0].read_text()


def test_light_metadata_pins_ttlang_version(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {"TTLANG_PRETEND_VERSION": "0.71.0.dev20260525"},
    )

    assert result.returncode == 0, result.stderr
    assert "tt-lang==0.71.0.dev20260525+light" in _requires_text(tmp_path)


def test_light_metadata_accepts_explicit_ttlang_version(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {
            "TTLANG_PRETEND_VERSION": "0.71.0.dev20260525",
            "TTLANG_LIGHT_TTLANG_VERSION": "0.71.0.dev20260524+light",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "tt-lang==0.71.0.dev20260524+light" in _requires_text(tmp_path)


def test_light_metadata_rejects_ttlang_without_light_label(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {
            "TTLANG_PRETEND_VERSION": "0.71.0.dev20260525",
            "TTLANG_LIGHT_TTLANG_VERSION": "0.71.0.dev20260524",
        },
    )

    assert result.returncode != 0
    assert "requires local version label +light" in result.stderr


def test_light_metadata_requires_pretend_version(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {},
        env_removals=("TTLANG_PRETEND_VERSION", "TTLANG_LIGHT_TTLANG_VERSION"),
    )

    assert result.returncode != 0
    assert "requires TTLANG_PRETEND_VERSION" in result.stderr


def test_light_metadata_rejects_final_version(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {"TTLANG_PRETEND_VERSION": "0.71.0"},
    )

    assert result.returncode != 0
    assert "requires a non-final version" in result.stderr
