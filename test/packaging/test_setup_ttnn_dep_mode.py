# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for setup.py's dynamic ttnn dependency metadata."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _requires_text(tmp_path: Path) -> str:
    matches = list((tmp_path / "egg-info").glob("*.egg-info/requires.txt"))
    assert len(matches) == 1
    return matches[0].read_text()


def _make_fake_tt_metal_install(tmp_path: Path) -> Path:
    tt_metal = tmp_path / "tt-metal"
    ttnn_package = tt_metal / "python_packages" / "ttnn" / "ttnn"
    ttnn_package.mkdir(parents=True)
    (ttnn_package / "__init__.py").write_text("")
    (ttnn_package / "_ttnn.so").write_bytes(b"")
    (ttnn_package / "_ttnncpp.so").write_bytes(b"")
    tracy_package = tt_metal / "python_packages" / "tools" / "tracy"
    tracy_package.mkdir(parents=True)
    (tracy_package / "__init__.py").write_text("")
    (tt_metal / "lib").mkdir()
    (tt_metal / "runtime").mkdir()
    (tt_metal / "tt_metal").mkdir()
    (tt_metal / "ttnn" / "cpp").mkdir(parents=True)
    return tt_metal


def test_default_metadata_requires_ttnn(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {"TTLANG_PRETEND_VERSION": "0.71.0.dev20260525"},
        env_removals=("TTLANG_TTNN_DEP_MODE",),
    )

    assert result.returncode == 0, result.stderr
    assert "ttnn==" in _requires_text(tmp_path)


def test_external_metadata_omits_ttnn(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {
            "TTLANG_TTNN_DEP_MODE": "external",
            "TTLANG_PRETEND_VERSION": "0.71.0.dev20260525",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "ttnn==" not in _requires_text(tmp_path)


def test_bundled_metadata_omits_ttnn_and_adds_ttnn_runtime_deps(
    tmp_path: Path,
) -> None:
    tt_metal = _make_fake_tt_metal_install(tmp_path)

    result = _run_egg_info(
        tmp_path,
        {
            "TTLANG_TTNN_DEP_MODE": "bundled",
            "TTLANG_PRETEND_VERSION": "0.71.0.dev20260525",
            "TTLANG_BUNDLED_TT_METAL_DIR": str(tt_metal),
        },
    )

    requirements = _requires_text(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "ttnn==" not in requirements
    assert "loguru>=0.6.0" in requirements
    assert "networkx>=3.1" in requirements


def test_external_metadata_requires_explicit_nonfinal_version(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {"TTLANG_TTNN_DEP_MODE": "external"},
        env_removals=("TTLANG_PRETEND_VERSION",),
    )

    assert result.returncode != 0
    assert "requires TTLANG_PRETEND_VERSION" in result.stderr


def test_external_metadata_rejects_final_version(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {
            "TTLANG_TTNN_DEP_MODE": "external",
            "TTLANG_PRETEND_VERSION": "0.71.0",
        },
    )

    assert result.returncode != 0
    assert "requires a non-final version" in result.stderr


def test_bundled_metadata_requires_tt_metal_root(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {
            "TTLANG_TTNN_DEP_MODE": "bundled",
            "TTLANG_PRETEND_VERSION": "0.71.0.dev20260525",
            "TTLANG_BUNDLED_TT_METAL_DIR": str(tmp_path / "missing"),
        },
    )

    assert result.returncode != 0
    assert "bundled tt-metal root is not a directory" in result.stderr


def test_invalid_dependency_mode_fails(tmp_path: Path) -> None:
    result = _run_egg_info(
        tmp_path,
        {
            "TTLANG_TTNN_DEP_MODE": "invalid",
            "TTLANG_PRETEND_VERSION": "0.71.0.dev20260525",
        },
    )

    assert result.returncode != 0
    assert "TTLANG_TTNN_DEP_MODE must be one of" in result.stderr
