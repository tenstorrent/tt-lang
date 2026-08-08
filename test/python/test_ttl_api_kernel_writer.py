# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for generated TT-Metal kernel source file writing."""

import os
from pathlib import Path

from ttl.ttl_api import _write_kernel_to_tmp


def _cleanup(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    for directory in (path.parent, path.parent.parent):
        try:
            directory.rmdir()
        except OSError:
            pass


def test_write_kernel_uses_xdist_worker_subdirectory(monkeypatch):
    source = "void kernel_main() {}\n"
    user = f"ttlang-test-{os.getpid()}"
    monkeypatch.setenv("USER", user)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")

    path = Path(_write_kernel_to_tmp("compute_fn", source))
    try:
        assert path.parent == Path("/tmp") / user / "gw3"
        assert path.read_text() == source
    finally:
        _cleanup(path)


def test_write_kernel_replaces_existing_file_atomically(monkeypatch):
    source = "void kernel_main() {}\n"
    user = f"ttlang-test-{os.getpid()}"
    monkeypatch.setenv("USER", user)

    path = Path(_write_kernel_to_tmp("compute_fn", source))
    original_replace = os.replace
    observed_temp_source = []

    def checked_replace(src, dst):
        assert Path(dst).read_text() == source
        observed_temp_source.append(Path(src).read_text())
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", checked_replace)
    try:
        assert Path(_write_kernel_to_tmp("compute_fn", source)) == path
        assert observed_temp_source == [source]
    finally:
        _cleanup(path)
