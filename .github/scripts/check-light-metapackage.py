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

from packaging.requirements import InvalidRequirement, Requirement


def _metadata_for(pattern: str) -> str:
    wheels = glob.glob(pattern)
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel for {pattern}, found {wheels}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_name = next(
            (name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")),
            None,
        )
        if metadata_name is None:
            raise SystemExit(f"{wheels[0]} has no .dist-info/METADATA entry")
        return wheel.read(metadata_name).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", required=True)
    parser.add_argument("--expect-ttlang-version", required=True)
    args = parser.parse_args()

    requirements = []
    for line in _metadata_for(f"{args.dist_dir}/tt_lang_light-*.whl").splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        try:
            requirements.append(Requirement(line.split(":", 1)[1].strip()))
        except InvalidRequirement as error:
            print(f"invalid Requires-Dist line: {line}: {error}", file=sys.stderr)
            return 1

    expected_specifier = f"=={args.expect_ttlang_version}"
    has_expected_pin = any(
        requirement.name.lower() == "tt-lang"
        and str(requirement.specifier) == expected_specifier
        for requirement in requirements
    )

    if not has_expected_pin:
        print(
            "tt-lang-light must require "
            f"tt-lang{expected_specifier}; metadata had no match",
            file=sys.stderr,
        )
        return 1

    print(f"tt-lang-light pins tt-lang=={args.expect_ttlang_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
