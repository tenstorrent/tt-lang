# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Copy bundled tt-lang tutorials out of the install for editing.

Tutorials are shipped inside the wheel under `ttl.tutorials.{elementwise,
matmul,broadcast}`. This script copies them out to a target directory
(default `./tutorials/`) so users can run/edit them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from importlib.resources import files
from pathlib import Path

SECTIONS = {
    "elementwise": "ttl.tutorials.elementwise",
    "matmul": "ttl.tutorials.matmul",
    "broadcast": "ttl.tutorials.broadcast",
}


def _section_path(pkg: str) -> Path:
    """Resolve the on-disk directory of a bundled tutorial package."""
    return Path(str(files(pkg)))


def _copy_section(src: Path, dst: Path, force: bool) -> int:
    if dst.exists() and not force:
        print(f"skip: {dst} exists (use --force to overwrite)")
        return 0
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    n = 0
    for f in src.iterdir():
        if f.suffix != ".py" or f.name == "__init__.py":
            continue
        shutil.copy2(f, dst / f.name)
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tt-lang-setup-tutorials",
        description="Copy bundled tt-lang tutorial scripts to a target directory.",
    )
    p.add_argument(
        "-t",
        "--target",
        default="tutorials",
        help="destination directory (default: ./tutorials)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing tutorial subdirectories",
    )
    args = p.parse_args(argv)

    target = Path(
        args.target
    ).resolve()  # noqa: SAST  user-supplied CLI destination, no trust boundary
    target.mkdir(parents=True, exist_ok=True)

    total = 0
    for name, pkg in SECTIONS.items():
        try:
            src = _section_path(pkg)
        except (ModuleNotFoundError, FileNotFoundError):
            sys.exit(f"missing bundled package {pkg}; reinstall tt-lang")
        n = _copy_section(src, target / name, args.force)
        if n:
            print(f"copied {n} script(s) -> {target / name}")
        total += n

    if total == 0:
        print(f"no tutorials copied (use --force to overwrite existing in {target})")
    else:
        print(f"done: {total} script(s) under {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
