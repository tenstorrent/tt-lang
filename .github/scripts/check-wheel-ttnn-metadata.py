#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify the tt-lang wheel metadata matches the requested build mode.

The "pypi" mode must declare a Requires-Dist on ttnn. The "external" and
"bundled" modes must not. When --expect-tt-metal-commit is passed, the generated
ttl/config.py file must record the same tt-metal commit.

Usage: check-wheel-ttnn-metadata.py --mode {pypi,external,bundled}
    --dist-dir <dir> [--expect-tt-metal-commit <sha>]
"""

import argparse
import glob
import re
import sys
import zipfile

from packaging.requirements import InvalidRequirement, Requirement

MODES = ("pypi", "external", "bundled")
TT_METAL_COMMIT_RE = re.compile(r'^TT_METAL_COMMIT = "([^"]*)"$', re.MULTILINE)


def _read_metadata(wheel_path: str) -> str:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_name = next(
            (name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")),
            None,
        )
        if metadata_name is None:
            raise ValueError(f"{wheel_path} has no .dist-info/METADATA entry")
        return wheel.read(metadata_name).decode()


def _has_bundled_ttnn_payload(wheel_path: str) -> bool:
    with zipfile.ZipFile(wheel_path) as wheel:
        return any(name.startswith("ttnn/") for name in wheel.namelist())


def _tt_metal_commit(wheel_path: str) -> str:
    with zipfile.ZipFile(wheel_path) as wheel:
        try:
            config_py = wheel.read("ttl/config.py").decode()
        except KeyError as error:
            raise ValueError(f"{wheel_path} has no ttl/config.py entry") from error
    match = TT_METAL_COMMIT_RE.search(config_py)
    if not match:
        raise ValueError(f"{wheel_path} ttl/config.py has no TT_METAL_COMMIT")
    return match.group(1)


def _ttnn_requirement(metadata: str) -> Requirement | None:
    for line in metadata.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        try:
            requirement = Requirement(line.split(":", 1)[1].strip())
        except InvalidRequirement as error:
            raise ValueError(f"invalid Requires-Dist line: {line}: {error}") from error
        if requirement.name.lower() == "ttnn":
            return requirement
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--dist-dir", required=True)
    parser.add_argument("--expect-tt-metal-commit")
    parser.add_argument("--expect-ttnn-version")
    args = parser.parse_args()

    wheels = glob.glob(f"{args.dist_dir}/tt_lang-*.whl")
    if len(wheels) != 1:
        print(
            f"expected one tt-lang wheel in {args.dist_dir}, found {wheels}",
            file=sys.stderr,
        )
        return 1

    try:
        ttnn_requirement = _ttnn_requirement(_read_metadata(wheels[0]))
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    has_ttnn = ttnn_requirement is not None
    if args.mode in ("external", "bundled") and has_ttnn:
        print(f"{args.mode} wheel metadata must not require ttnn", file=sys.stderr)
        return 1
    if args.mode == "external" and _has_bundled_ttnn_payload(wheels[0]):
        print("external wheel must not bundle a ttnn payload", file=sys.stderr)
        return 1
    if args.mode == "pypi" and not has_ttnn:
        print("default wheel metadata must require ttnn", file=sys.stderr)
        return 1
    if args.expect_ttnn_version is not None:
        if args.mode != "pypi":
            print(
                "--expect-ttnn-version is valid only in pypi mode",
                file=sys.stderr,
            )
            return 1
        expected_specifier = f"=={args.expect_ttnn_version}"
        if str(ttnn_requirement.specifier) != expected_specifier:
            print(
                "tt-lang wheel ttnn dependency mismatch: "
                f"expected {expected_specifier}, got {ttnn_requirement.specifier}",
                file=sys.stderr,
            )
            return 1
    if args.expect_tt_metal_commit is not None:
        try:
            tt_metal_commit = _tt_metal_commit(wheels[0])
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1
        if tt_metal_commit != args.expect_tt_metal_commit:
            print(
                "tt-lang wheel tt-metal provenance mismatch: "
                f"expected {args.expect_tt_metal_commit}, got {tt_metal_commit}",
                file=sys.stderr,
            )
            return 1

    print(f"{wheels[0]}: ttnn dependency present={has_ttnn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
