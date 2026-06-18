# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Umbrella `tt-lang-setup` command — runs sfpi setup and tutorial copy."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys

from packaging.version import InvalidVersion, Version

from . import sfpi as _sfpi
from . import tutorials as _tutorials


def _is_sim_only_install() -> bool:
    return importlib.util.find_spec("ttl._sim_only_marker") is not None


def _is_light_install() -> bool:
    try:
        raw_version = importlib.metadata.version("tt-lang")
    except importlib.metadata.PackageNotFoundError:
        return False
    try:
        local = Version(raw_version).local
    except InvalidVersion:
        return False
    # PEP 440 local labels are dot-separated; this PR's convention is "+light"
    # exactly, so reject `+lightning`, `+light.dev1`, etc.
    return local == "light"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tt-lang-setup",
        description=(
            "Run all post-install setup steps: install the sfpi runtime "
            "ttnn needs, then copy bundled tutorials to a writable directory."
        ),
    )
    p.add_argument(
        "--no-sfpi",
        action="store_true",
        help="skip sfpi install (equivalent to running tt-lang-setup-tutorials only)",
    )
    p.add_argument(
        "--no-tutorials",
        action="store_true",
        help="skip tutorial copy (equivalent to running tt-lang-setup-sfpi only)",
    )
    p.add_argument(
        "-t",
        "--tutorials-target",
        default="tutorials",
        help="destination directory for tutorials (default: ./tutorials)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="reinstall sfpi and overwrite existing tutorial directories",
    )
    args = p.parse_args(argv)

    if _is_sim_only_install() and not args.no_sfpi:
        print("sim-only install (tt-lang-sim) detected; skipping sfpi install")
        args.no_sfpi = True
    elif _is_light_install() and not args.no_sfpi:
        print(
            "light install (tt-lang+light) detected; skipping sfpi install "
            "(sfpi is provided by the external tt-metal)"
        )
        args.no_sfpi = True

    if not args.no_sfpi:
        sfpi_argv = ["--force"] if args.force else []
        rc = _sfpi.main(sfpi_argv)
        if rc != 0:
            return rc

    if not args.no_tutorials:
        tut_argv = ["--target", args.tutorials_target]
        if args.force:
            tut_argv.append("--force")
        rc = _tutorials.main(tut_argv)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
