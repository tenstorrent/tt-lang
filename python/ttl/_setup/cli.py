# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Umbrella `tt-lang-setup` command — runs sfpi setup and tutorial copy."""

from __future__ import annotations

import argparse
import importlib.util
import sys

from . import host as _host
from . import tutorials as _tutorials


def _is_sim_only_install() -> bool:
    return importlib.util.find_spec("ttl._sim_only_marker") is not None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tt-lang-setup",
        description=(
            "Run all post-install setup steps: install the sfpi runtime "
            "ttnn needs, then copy bundled tutorials to a writable directory."
        ),
    )
    p.add_argument(
        "--no-host",
        action="store_true",
        help="skip sfpi install (equivalent to running tt-lang-setup-tutorials only)",
    )
    p.add_argument(
        "--no-tutorials",
        action="store_true",
        help="skip tutorial copy (equivalent to running tt-lang-setup-host only)",
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

    if _is_sim_only_install() and not args.no_host:
        print("sim-only install (tt-lang-sim) detected; skipping sfpi install")
        args.no_host = True

    if not args.no_host:
        host_argv = ["--force"] if args.force else []
        rc = _host.main(host_argv)
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
