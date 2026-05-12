# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Install the sfpi runtime that ttnn needs.

Runs after `pip install tt-lang` on a Linux x86_64 / aarch64 host (where
ttnn is pulled in as a hard dependency). Reads the sfpi version recorded
by the installed ttnn wheel, downloads the matching tarball from the sfpi
GitHub release, verifies its sha256, and extracts it into
`<ttnn>/runtime/sfpi/`.

No sudo: the install target is inside the user's venv.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from importlib.util import find_spec
from pathlib import Path

# Maps (uname.sysname, uname.machine) -> (filename suffix, hash key in sfpi-version).
_ARCH_MAP = {
    ("Linux", "x86_64"): ("x86_64_debian", "sfpi_x86_64_debian_txz_hash"),
    ("Linux", "aarch64"): ("aarch64_debian", "sfpi_aarch64_debian_txz_hash"),
}


def _ttnn_pkg_dir() -> Path:
    spec = find_spec("ttnn")
    if spec is None or not spec.submodule_search_locations:
        sys.exit(
            "ttnn is not installed; sfpi setup requires the device `tt-lang` "
            "wheel. For sim-only installs run `tt-lang-setup` (skips sfpi)."
        )
    return Path(next(iter(spec.submodule_search_locations)))


def _parse_sfpi_version_file(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^([a-z0-9_]+)='([^']*)'", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp, dst.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _safe_extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:xz") as tar:
        try:
            tar.extractall(path=dest, filter="data")
        except TypeError:
            tar.extractall(path=dest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tt-lang-setup-host",
        description="Install the sfpi runtime needed by ttnn (no sudo).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="reinstall sfpi even if already present",
    )
    args = p.parse_args(argv)

    ttnn_dir = _ttnn_pkg_dir()
    runtime_dir = ttnn_dir / "runtime"
    target = runtime_dir / "sfpi"

    sfpi_meta = ttnn_dir / "tt_metal" / "sfpi-version"
    if not sfpi_meta.exists():
        sys.exit(
            f"missing {sfpi_meta} in installed ttnn; this ttnn version is not "
            "supported by tt-lang-setup-host"
        )
    info = _parse_sfpi_version_file(sfpi_meta.read_text())
    version = info.get("sfpi_version")
    if not version:
        sys.exit(f"no sfpi_version line in {sfpi_meta}")

    arch_key = (os.uname().sysname, os.uname().machine)
    if arch_key not in _ARCH_MAP:
        sys.exit(
            f"unsupported platform {arch_key}; tt-lang-setup-host only "
            "handles Linux x86_64 / aarch64"
        )
    suffix, hash_key = _ARCH_MAP[arch_key]
    expected_hash = info.get(hash_key)
    if not expected_hash:
        sys.exit(f"no {hash_key} in {sfpi_meta}")

    bin_dir = target / "compiler" / "bin"
    if any(bin_dir.glob("riscv*-tt-elf-g++")) and not args.force:
        print(f"sfpi {version} already installed at {target}")
        return 0

    filename = f"sfpi_{version}_{suffix}.txz"
    url = (
        "https://github.com/tenstorrent/sfpi/releases/download/" f"{version}/{filename}"
    )

    with tempfile.TemporaryDirectory(prefix="tt-lang-sfpi-") as tmp:
        archive = Path(tmp) / filename
        _download(url, archive)
        actual = _sha256(archive)
        if actual != expected_hash:
            sys.exit(
                f"sha256 mismatch for {filename}: expected {expected_hash}, "
                f"got {actual}"
            )

        if target.exists():
            shutil.rmtree(target)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract(archive, runtime_dir)

    if not any((target / "compiler" / "bin").glob("riscv*-tt-elf-g++")):
        sys.exit(
            f"extraction completed but no riscv*-tt-elf-g++ under {target}; "
            "tarball layout may have changed"
        )

    print(f"sfpi {version} installed at {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
