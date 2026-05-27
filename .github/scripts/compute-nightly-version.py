#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Print a PEP 440 nightly version derived from release tags.

Reads `git tag --list 'v[0-9]*'`, picks the greatest stable vMAJOR.MINOR.PATCH
tag, and prints `MAJOR.MINOR.PATCH.devYYYYMMDD` using today's UTC date.

Usage: compute-nightly-version.py
"""

import datetime
import re
import subprocess
import sys


def main() -> int:
    tags = subprocess.check_output(
        ["git", "tag", "--list", "v[0-9]*"],
        text=True,
    ).splitlines()

    stable_versions = []
    for tag in tags:
        match = re.fullmatch(r"v([0-9]+)\.([0-9]+)\.([0-9]+)", tag)
        if match:
            stable_versions.append(tuple(int(part) for part in match.groups()))

    if not stable_versions:
        print("No stable vMAJOR.MINOR.PATCH tag found", file=sys.stderr)
        return 1

    major, minor, patch = max(stable_versions)
    nightly_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    print(f"{major}.{minor}.{patch}.dev{nightly_date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
