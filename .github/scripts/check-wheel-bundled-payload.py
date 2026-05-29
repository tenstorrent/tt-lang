#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify the bundled tt-lang wheel contains the expected ttnn payload.

Usage: check-wheel-bundled-payload.py --dist-dir <dir>
"""

import argparse
import glob
import sys
import zipfile

REQUIRED_PATHS = (
    "ttnn/__init__.py",
    "ttnn/_ttnn.so",
    "ttnn/build/lib/_ttnncpp.so",
    "ttnn/build/lib/libtt_metal.so",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", required=True)
    args = parser.parse_args()

    wheels = glob.glob(f"{args.dist_dir}/tt_lang-*.whl")
    if len(wheels) != 1:
        print(
            f"expected one tt-lang wheel in {args.dist_dir}, found {wheels}",
            file=sys.stderr,
        )
        return 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())

    missing = [path for path in REQUIRED_PATHS if path not in names]
    if missing:
        print(f"bundled wheel is missing: {missing}", file=sys.stderr)
        return 1

    print(f"{wheels[0]}: bundled ttnn payload present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
