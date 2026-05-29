# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for setup.py's dynamic ttnn dependency metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def test_default_metadata_requires_ttnn(
    run_egg_info: Callable[..., object],
    requires_text: Callable[[], str],
) -> None:
    result = run_egg_info(
        {"TTLANG_VERSION_OVERRIDE": "0.71.0.dev20260525"},
        env_removals=("TTLANG_TTNN_DEP_MODE",),
    )

    assert result.returncode == 0, result.stderr
    assert "ttnn==" in requires_text()


def test_external_metadata_omits_ttnn(
    run_egg_info: Callable[..., object],
    requires_text: Callable[[], str],
) -> None:
    result = run_egg_info(
        {
            "TTLANG_TTNN_DEP_MODE": "external",
            "TTLANG_VERSION_OVERRIDE": "0.71.0.dev20260525+light",
        },
    )

    requirements = requires_text()
    assert result.returncode == 0, result.stderr
    assert "ttnn==" not in requirements
    assert "loguru>=0.6.0" in requirements
    assert "networkx>=3.1" in requirements


def test_bundled_metadata_omits_ttnn_and_adds_ttnn_runtime_deps(
    run_egg_info: Callable[..., object],
    requires_text: Callable[[], str],
    make_fake_tt_metal_install: Callable[..., Path],
) -> None:
    tt_metal = make_fake_tt_metal_install()

    result = run_egg_info(
        {
            "TTLANG_TTNN_DEP_MODE": "bundled",
            "TTLANG_VERSION_OVERRIDE": "0.71.0.dev20260525",
            "TTLANG_BUNDLED_TT_METAL_DIR": str(tt_metal),
        },
    )

    requirements = requires_text()
    assert result.returncode == 0, result.stderr
    assert "ttnn==" not in requirements
    assert "loguru>=0.6.0" in requirements
    assert "networkx>=3.1" in requirements


def test_external_metadata_requires_explicit_nonfinal_version(
    run_egg_info: Callable[..., object],
) -> None:
    result = run_egg_info(
        {"TTLANG_TTNN_DEP_MODE": "external"},
        env_removals=("TTLANG_VERSION_OVERRIDE",),
    )

    assert result.returncode != 0
    assert "requires TTLANG_VERSION_OVERRIDE" in result.stderr


def test_external_metadata_rejects_final_version(
    run_egg_info: Callable[..., object],
) -> None:
    result = run_egg_info(
        {
            "TTLANG_TTNN_DEP_MODE": "external",
            "TTLANG_VERSION_OVERRIDE": "0.71.0",
        },
    )

    assert result.returncode != 0
    assert "requires a non-final version" in result.stderr


def test_external_metadata_allows_final_version_when_release_guard_passed(
    run_egg_info: Callable[..., object],
    requires_text: Callable[[], str],
) -> None:
    result = run_egg_info(
        {
            "TTLANG_TTNN_DEP_MODE": "external",
            "TTLANG_VERSION_OVERRIDE": "0.71.0+light",
            "TTLANG_ALLOW_FINAL_INTERNAL_VERSION": "true",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "ttnn==" not in requires_text()


def test_bundled_metadata_allows_final_version_when_release_guard_passed(
    run_egg_info: Callable[..., object],
    requires_text: Callable[[], str],
    make_fake_tt_metal_install: Callable[..., Path],
) -> None:
    tt_metal = make_fake_tt_metal_install()

    result = run_egg_info(
        {
            "TTLANG_TTNN_DEP_MODE": "bundled",
            "TTLANG_VERSION_OVERRIDE": "0.71.0",
            "TTLANG_ALLOW_FINAL_INTERNAL_VERSION": "true",
            "TTLANG_BUNDLED_TT_METAL_DIR": str(tt_metal),
        },
    )

    requirements = requires_text()
    assert result.returncode == 0, result.stderr
    assert "ttnn==" not in requirements
    assert "loguru>=0.6.0" in requirements


def test_external_metadata_requires_light_label(
    run_egg_info: Callable[..., object],
) -> None:
    result = run_egg_info(
        {
            "TTLANG_TTNN_DEP_MODE": "external",
            "TTLANG_VERSION_OVERRIDE": "0.71.0.dev20260525",
        },
    )

    assert result.returncode != 0
    assert "requires a +light local version" in result.stderr


def test_bundled_metadata_requires_tt_metal_root(
    run_egg_info: Callable[..., object],
    tmp_path: Path,
) -> None:
    result = run_egg_info(
        {
            "TTLANG_TTNN_DEP_MODE": "bundled",
            "TTLANG_VERSION_OVERRIDE": "0.71.0.dev20260525",
            "TTLANG_BUNDLED_TT_METAL_DIR": str(tmp_path / "missing"),
        },
    )

    assert result.returncode != 0
    assert "bundled tt-metal root is not a directory" in result.stderr


def test_invalid_dependency_mode_fails(
    run_egg_info: Callable[..., object],
) -> None:
    result = run_egg_info(
        {
            "TTLANG_TTNN_DEP_MODE": "invalid",
            "TTLANG_VERSION_OVERRIDE": "0.71.0.dev20260525",
        },
    )

    assert result.returncode != 0
    assert "TTLANG_TTNN_DEP_MODE must be one of" in result.stderr
