#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Check the installed-state of the ttnn dependency after `pip install`.

  --mode external  -> ttnn must NOT be importable.
  --mode bundled   -> ttnn must be importable and bundled native libs present.
  --mode pypi      -> no check; exit 0.

Run inside the test venv so importlib resolves the installed tt-lang.

Usage: check-installed-ttnn.py --mode {pypi,external,bundled}
"""

import argparse
import importlib.util
import os
import pathlib
import re
import shlex
import subprocess
import sys

REQUIRED_BUNDLED_RELATIVE = (
    "_ttnn.so",
    "build/lib/_ttnncpp.so",
    "build/lib/libtt_metal.so",
)

_LDD_NOT_FOUND_RE = re.compile(r"=>\s*not found")
_LDD_ARROW_RE = re.compile(r"^\s*([^\s]+)\s*=>\s*(\S+)", re.MULTILINE)
_LDD_COMMAND_ENV = "TTLANG_LDD_COMMAND"


def check_external() -> int:
    if importlib.util.find_spec("ttnn") is not None:
        print("external wheel unexpectedly installed ttnn", file=sys.stderr)
        return 1
    return 0


def _ldd_command() -> list[str]:
    command = shlex.split(os.environ.get(_LDD_COMMAND_ENV, "ldd"))
    if not command:
        raise ValueError(f"{_LDD_COMMAND_ENV} must not be empty")
    return command


def check_bundled() -> int:
    import ttnn  # noqa: WPS433 - import here so external mode never imports it.

    ttnn_root = pathlib.Path(ttnn.__file__).resolve().parent
    missing = [
        str(ttnn_root / rel)
        for rel in REQUIRED_BUNDLED_RELATIVE
        if not (ttnn_root / rel).exists()
    ]
    if missing:
        print(f"bundled ttnn is missing files: {missing}", file=sys.stderr)
        return 1

    ttnn_extension = ttnn_root / "_ttnn.so"
    expected_ttnncpp = (ttnn_root / "build" / "lib" / "_ttnncpp.so").resolve()
    # Strip LD_LIBRARY_PATH so a stray ttnncpp on the loader path can't mask a
    # broken RUNPATH. ldd inherits everything else (PATH for the helper, etc.).
    ldd_env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    try:
        ldd_command = _ldd_command()
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    ldd_result = subprocess.run(
        [*ldd_command, str(ttnn_extension)],
        capture_output=True,
        text=True,
        env=ldd_env,
        check=False,
    )
    if ldd_result.returncode != 0:
        print(
            f"ldd failed for bundled ttnn extension:\n{ldd_result.stderr}",
            file=sys.stderr,
        )
        return 1
    if _LDD_NOT_FOUND_RE.search(ldd_result.stdout):
        print(
            f"bundled ttnn extension has unresolved libraries:\n{ldd_result.stdout}",
            file=sys.stderr,
        )
        return 1

    ttnncpp_target: pathlib.Path | None = None
    for match in _LDD_ARROW_RE.finditer(ldd_result.stdout):
        soname, target = match.group(1), match.group(2)
        # Skip the "not found" sentinel; the regex above would capture
        # target="not", which never resolves to expected_ttnncpp anyway,
        # but is also not a real soname target. Defensive.
        if target == "not":
            continue
        if pathlib.PurePath(soname).name == "_ttnncpp.so":
            ttnncpp_target = pathlib.Path(target).resolve()
            break
    if ttnncpp_target != expected_ttnncpp:
        print(
            "bundled _ttnn.so does not resolve _ttnncpp.so from "
            f"{expected_ttnncpp}; got {ttnncpp_target}\n{ldd_result.stdout}",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("pypi", "external", "bundled"),
    )
    args = parser.parse_args()

    if args.mode == "external":
        return check_external()
    if args.mode == "bundled":
        return check_bundled()
    return 0


if __name__ == "__main__":
    sys.exit(main())
