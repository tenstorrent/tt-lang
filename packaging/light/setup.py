#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the internal tt-lang-light metapackage. It ships no Python modules; it
# pins an internal tt-lang wheel built with TTLANG_TTNN_DEP_MODE=external. The
# pinned tt-lang wheel uses a +light local version and must stay on the internal
# S3 index; this metapackage is not suitable for public PyPI.

from __future__ import annotations

import os
import pathlib
import sys

from packaging.version import InvalidVersion, Version
from setuptools import setup
from setuptools.command.sdist import sdist as _sdist

PKG_ROOT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent.parent
sys.path.insert(0, str(PKG_ROOT.parent))
from internal_wheel_metadata import (  # noqa: E402
    get_version,
    require_local_version_label,
    require_non_final_internal_version,
    require_version_override,
)


class NoSdist(_sdist):
    """Reject source distribution builds; tt-lang-light only ships wheels."""

    def run(self):
        raise SystemExit("tt-lang-light only publishes wheels.")


def _ttlang_requirement(version: str) -> str:
    ttlang_version = os.environ.get("TTLANG_LIGHT_TTLANG_VERSION", "").strip()
    if not ttlang_version:
        ttlang_version = f"{version}+light"
    require_local_version_label("tt-lang-light", ttlang_version, "light")
    # The metapackage pins tt-lang==X+light by exact match. PEP 440
    # base_version drops dev/pre/post segments, so different dev dates within
    # the same MAJOR.MINOR.PATCH line are allowed (the existing
    # test_light_metadata_accepts_explicit_ttlang_version case depends on
    # this). A MAJOR/MINOR/PATCH mismatch, on the other hand, would produce a
    # pin that nothing on the index can satisfy — catch that here.
    try:
        ttlang_base = Version(ttlang_version).base_version
        metapackage_base = Version(version).base_version
    except InvalidVersion as error:
        raise SystemExit(f"tt-lang-light: failed to parse version: {error}") from error
    if ttlang_base != metapackage_base:
        raise SystemExit(
            "tt-lang-light: tt-lang dependency base version "
            f"({ttlang_base}) does not match metapackage base version "
            f"({metapackage_base}); set TTLANG_LIGHT_TTLANG_VERSION to align."
        )
    return f"tt-lang == {ttlang_version}"


require_version_override("tt-lang-light")
VERSION = get_version(REPO_ROOT)
require_non_final_internal_version("tt-lang-light", VERSION)


setup(
    name="tt-lang-light",
    version=VERSION,
    install_requires=[_ttlang_requirement(VERSION)],
    packages=[],
    cmdclass={"sdist": NoSdist},
    zip_safe=False,
)
