# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tt-lang-light metapackage metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from conftest import LIGHT_ROOT


def test_light_metadata_pins_ttlang_version(
    run_egg_info: Callable[..., object],
    requires_text: Callable[[], str],
) -> None:
    result = run_egg_info(
        {"TTLANG_VERSION_OVERRIDE": "0.71.0.dev20260525"},
        cwd=LIGHT_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "tt-lang==0.71.0.dev20260525+light" in requires_text()


def test_light_metadata_accepts_explicit_ttlang_version(
    run_egg_info: Callable[..., object],
    requires_text: Callable[[], str],
) -> None:
    result = run_egg_info(
        {
            "TTLANG_VERSION_OVERRIDE": "0.71.0.dev20260525",
            "TTLANG_LIGHT_TTLANG_VERSION": "0.71.0.dev20260524+light",
        },
        cwd=LIGHT_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "tt-lang==0.71.0.dev20260524+light" in requires_text()


def test_light_metadata_rejects_ttlang_without_light_label(
    run_egg_info: Callable[..., object],
) -> None:
    result = run_egg_info(
        {
            "TTLANG_VERSION_OVERRIDE": "0.71.0.dev20260525",
            "TTLANG_LIGHT_TTLANG_VERSION": "0.71.0.dev20260524",
        },
        cwd=LIGHT_ROOT,
    )

    assert result.returncode != 0
    assert "requires local version label +light" in result.stderr


def test_light_metadata_requires_version_override(
    run_egg_info: Callable[..., object],
) -> None:
    result = run_egg_info(
        {},
        env_removals=("TTLANG_VERSION_OVERRIDE", "TTLANG_LIGHT_TTLANG_VERSION"),
        cwd=LIGHT_ROOT,
    )

    assert result.returncode != 0
    assert "requires TTLANG_VERSION_OVERRIDE" in result.stderr


def test_light_metadata_rejects_final_version(
    run_egg_info: Callable[..., object],
) -> None:
    result = run_egg_info(
        {"TTLANG_VERSION_OVERRIDE": "0.71.0"},
        cwd=LIGHT_ROOT,
    )

    assert result.returncode != 0
    assert "requires a non-final version" in result.stderr


def test_light_metadata_rejects_base_version_mismatch(
    run_egg_info: Callable[..., object],
) -> None:
    # The metapackage is 0.71.0.devX but the explicit TTLANG_LIGHT_TTLANG_VERSION
    # uses base 0.72.0. The +light label passes the local-label check but the
    # bases diverge, so the metapackage would never resolve against the matching
    # tt-lang wheel — fail early.
    result = run_egg_info(
        {
            "TTLANG_VERSION_OVERRIDE": "0.71.0.dev20260525",
            "TTLANG_LIGHT_TTLANG_VERSION": "0.72.0.dev20260525+light",
        },
        cwd=LIGHT_ROOT,
    )

    assert result.returncode != 0
    assert "base version" in result.stderr
