#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the internal tt-lang-light metapackage. It ships no Python modules; it
# pins an internal tt-lang wheel built with TTLANG_TTNN_DEP_MODE=external.

from __future__ import annotations

import os
import pathlib
import sys

from setuptools import setup
from setuptools.command.sdist import sdist as _sdist

PKG_ROOT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent.parent
sys.path.insert(0, str(PKG_ROOT.parent))
from internal_wheel_metadata import (  # noqa: E402
    get_version,
    require_non_final_internal_version,
)


class NoSdist(_sdist):
    """Reject source distribution builds; tt-lang-light only ships wheels."""

    def run(self):
        raise SystemExit("tt-lang-light only publishes wheels.")


def _ttlang_requirement(version: str) -> str:
    ttlang_version = os.environ.get("TTLANG_LIGHT_TTLANG_VERSION", "").strip()
    if not ttlang_version:
        ttlang_version = version
    return f"tt-lang == {ttlang_version}"


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
