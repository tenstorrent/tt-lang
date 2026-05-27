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
import pathlib
import sys

REQUIRED_BUNDLED_RELATIVE = (
    "_ttnn.so",
    "build/lib/_ttnncpp.so",
    "build/lib/libtt_metal.so",
)


def check_external() -> int:
    if importlib.util.find_spec("ttnn") is not None:
        print("external wheel unexpectedly installed ttnn", file=sys.stderr)
        return 1
    return 0


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
