# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tt-lang version detection.

Reads the version from the CMake-generated config module when available,
otherwise falls back to ``git describe`` for development checkouts.
"""

from __future__ import annotations

import subprocess


def _get_version() -> str:
    """Return the tt-lang version string."""
    try:
        from ttl.config import VERSION

        if VERSION and not VERSION.startswith("@"):
            return VERSION
    except (ImportError, AttributeError):
        pass

    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--match", "v[0-9]*", "--always"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            tag = result.stdout.strip()
            return tag.lstrip("v")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "unknown"


__version__ = _get_version()
