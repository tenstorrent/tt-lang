# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared metadata helpers for internal tt-lang metapackages."""

from __future__ import annotations

import os
import pathlib
import subprocess

from packaging.version import InvalidVersion, Version

VERSION_OVERRIDE_ENV = "TTLANG_VERSION_OVERRIDE"
ALLOW_FINAL_INTERNAL_VERSION_ENV = "TTLANG_ALLOW_FINAL_INTERNAL_VERSION"


def get_version_override() -> str:
    return os.environ.get(VERSION_OVERRIDE_ENV, "").strip()


def allow_final_internal_version() -> bool:
    return os.environ.get(ALLOW_FINAL_INTERNAL_VERSION_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def require_version_override(package_name: str) -> None:
    if not get_version_override():
        raise SystemExit(
            f"{package_name} requires {VERSION_OVERRIDE_ENV} so internal wheels "
            "cannot be confused with PyPI release wheels"
        )


def get_version(repo_root: pathlib.Path) -> str:
    version_override = get_version_override()
    if version_override:
        return version_override
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--match", "v[0-9]*", "--abbrev=0"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(repo_root),
        ).strip()
        commits = subprocess.check_output(
            ["git", "rev-list", f"{tag}..HEAD", "--count"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(repo_root),
        ).strip()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "failed to derive internal wheel version from git; set "
            f"{VERSION_OVERRIDE_ENV} explicitly"
        ) from error

    tag = tag.lstrip("v")
    base, sep, local = tag.partition("+")
    local_suffix = f"+{local}" if sep else ""
    if commits and commits != "0":
        return f"{base}.dev{commits}{local_suffix}"
    return f"{base}{local_suffix}"


def require_non_final_internal_version(package_name: str, version: str) -> None:
    require_version_override(package_name)

    try:
        parsed_version = Version(version)
    except InvalidVersion as error:
        raise SystemExit(f"{package_name} has invalid version {version!r}") from error

    if (
        not parsed_version.is_devrelease
        and not parsed_version.is_prerelease
        and not allow_final_internal_version()
    ):
        raise SystemExit(
            f"{package_name} requires a non-final version such as "
            "0.71.0.dev20260525 or 0.71.0rc1"
        )


def require_local_version_label(package_name: str, version: str, label: str) -> None:
    try:
        parsed_version = Version(version)
    except InvalidVersion as error:
        raise SystemExit(f"{package_name} has invalid version {version!r}") from error

    if parsed_version.local != label:
        raise SystemExit(
            f"{package_name} requires local version label +{label}; got {version}"
        )
