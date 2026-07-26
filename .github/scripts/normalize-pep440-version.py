#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate one PEP 440 version and print its canonical form."""

from __future__ import annotations

import sys

from packaging.version import InvalidVersion, Version


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <version>", file=sys.stderr)
        return 2
    try:
        version = Version(sys.argv[1])
    except InvalidVersion:
        print(f"Invalid PEP 440 version: {sys.argv[1]!r}", file=sys.stderr)
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
