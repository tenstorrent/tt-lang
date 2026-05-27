# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tt-lang-setup install-variant detection.

`tt-lang-setup` runs sfpi setup by default. For installs that do not ship
ttnn via pip (sim-only wheels and the +light core wheel) sfpi setup is
nonsensical and used to fail with a confusing "ttnn is not installed" error.
These tests pin the detection behavior.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SETUP_DIR = REPO_ROOT / "python" / "ttl" / "_setup"

# Load python/ttl/_setup/cli.py without triggering ttl/__init__.py (which
# imports compiler extensions not built in this test env). Register a synthetic
# parent package so cli.py's relative `from . import sfpi` resolves.
_pkg_name = "_ttlang_setup_under_test"
if _pkg_name not in sys.modules:
    _pkg = types.ModuleType(_pkg_name)
    _pkg.__path__ = [str(_SETUP_DIR)]
    sys.modules[_pkg_name] = _pkg
    _cli_spec = importlib.util.spec_from_file_location(
        f"{_pkg_name}.cli", _SETUP_DIR / "cli.py"
    )
    assert _cli_spec is not None and _cli_spec.loader is not None
    _cli_module = importlib.util.module_from_spec(_cli_spec)
    sys.modules[f"{_pkg_name}.cli"] = _cli_module
    _cli_spec.loader.exec_module(_cli_module)

cli = sys.modules[f"{_pkg_name}.cli"]


def _fake_distribution_with_version(version: str):
    class _FakeDist:
        pass

    fake = _FakeDist()
    fake.version = version  # type: ignore[attr-defined]
    return fake


def test_light_install_detected_when_version_has_light_local_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: _fake_distribution_with_version("1.1.1.dev20260527+light"),
    )
    assert cli._is_light_install() is True


def test_light_install_not_detected_for_plain_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: _fake_distribution_with_version("1.1.1.dev20260527"),
    )
    assert cli._is_light_install() is False


def test_light_install_not_detected_for_other_local_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: _fake_distribution_with_version("1.1.1+local"),
    )
    assert cli._is_light_install() is False


def test_light_install_returns_false_when_tt_lang_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(name: str):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", _raise)
    assert cli._is_light_install() is False
