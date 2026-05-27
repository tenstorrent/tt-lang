#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify the tt-lang-light metapackage pins tt-lang at the expected version.

Usage: check-light-metapackage.py --dist-dir <dir> --expect-ttlang-version <ver>
"""

import argparse
import glob
import sys
import zipfile


def _metadata_for(pattern: str) -> str:
    wheels = glob.glob(pattern)
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel for {pattern}, found {wheels}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        return wheel.read(metadata_name).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", required=True)
    parser.add_argument("--expect-ttlang-version", required=True)
    args = parser.parse_args()

    metadata = _metadata_for(f"{args.dist_dir}/tt_lang_light-*.whl")
    normalized_lines = [line.replace(" ", "") for line in metadata.splitlines()]
    expected = f"Requires-Dist:tt-lang=={args.expect_ttlang_version}"

    if expected not in normalized_lines:
        print(
            f"tt-lang-light must require {expected!r}; metadata had no match",
            file=sys.stderr,
        )
        return 1

    print(f"tt-lang-light pins tt-lang=={args.expect_ttlang_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
