# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared metadata helpers for internal tt-lang metapackages."""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

_NON_FINAL_VERSION_RE = re.compile(r"(?:\.dev|a|b|rc)\d+", re.IGNORECASE)


def get_version(repo_root: pathlib.Path) -> str:
    pretend = os.environ.get("TTLANG_PRETEND_VERSION", "").strip()
    if pretend:
        return pretend
    try:
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--match", "v[0-9]*", "--abbrev=0"],
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=str(repo_root),
            )
            .strip()
            .lstrip("v")
        )
        commits = subprocess.check_output(
            ["git", "rev-list", f"v{tag}..HEAD", "--count"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(repo_root),
        ).strip()
        base, sep, local = tag.partition("+")
        local_suffix = f"+{local}" if sep else ""
        if commits and commits != "0":
            return f"{base}.dev{commits}{local_suffix}"
        return f"{base}{local_suffix}"
    except Exception:
        return "0.2.0.dev0"


def require_non_final_internal_version(package_name: str, version: str) -> None:
    if not os.environ.get("TTLANG_PRETEND_VERSION", "").strip():
        raise SystemExit(
            f"{package_name} requires TTLANG_PRETEND_VERSION so internal wheels "
            "cannot be confused with PyPI release wheels"
        )
    if not _NON_FINAL_VERSION_RE.search(version):
        raise SystemExit(
            f"{package_name} requires a non-final version such as "
            "0.71.0.dev20260525 or 0.71.0rc1"
        )
