#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify the tt-lang wheel METADATA Requires-Dist matches the ttnn dep mode.

The "pypi" mode must declare a Requires-Dist on ttnn. The "external" and
"bundled" modes must not.

Usage: check-wheel-ttnn-metadata.py --mode {pypi,external,bundled} --dist-dir <dir>
"""

import argparse
import glob
import sys
import zipfile

MODES = ("pypi", "external", "bundled")


def _read_metadata(wheel_path: str) -> str:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        return wheel.read(metadata_name).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--dist-dir", required=True)
    args = parser.parse_args()

    wheels = glob.glob(f"{args.dist_dir}/tt_lang-*.whl")
    if len(wheels) != 1:
        print(
            f"expected one tt-lang wheel in {args.dist_dir}, found {wheels}",
            file=sys.stderr,
        )
        return 1

    metadata = _read_metadata(wheels[0])
    has_ttnn = any(
        line.startswith("Requires-Dist: ttnn") for line in metadata.splitlines()
    )

    if args.mode in ("external", "bundled") and has_ttnn:
        print(f"{args.mode} wheel metadata must not require ttnn", file=sys.stderr)
        return 1
    if args.mode == "pypi" and not has_ttnn:
        print("default wheel metadata must require ttnn", file=sys.stderr)
        return 1

    print(f"{wheels[0]}: ttnn dependency present={has_ttnn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
