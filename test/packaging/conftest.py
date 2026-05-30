# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the packaging test suite.

Provides:
    run_egg_info             Run `python setup.py egg_info` with controlled env.
    requires_text            Read the requires.txt produced by `run_egg_info`.
    make_fake_tt_metal_install
                             Build a synthetic tt-metal install tree on tmp_path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIGHT_ROOT = REPO_ROOT / "packaging" / "light"


@pytest.fixture
def run_egg_info(
    tmp_path: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def _run(
        env_updates: dict[str, str],
        env_removals: tuple[str, ...] = (),
        cwd: Path = REPO_ROOT,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        for name in env_removals:
            environment.pop(name, None)
        environment.update(env_updates)

        egg_base = tmp_path / "egg-info"
        egg_base.mkdir(exist_ok=True)
        return subprocess.run(
            [sys.executable, "setup.py", "egg_info", "--egg-base", str(egg_base)],
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    return _run


@pytest.fixture
def requires_text(tmp_path: Path) -> Callable[[], str]:
    def _read() -> str:
        matches = list((tmp_path / "egg-info").glob("*.egg-info/requires.txt"))
        assert len(matches) == 1
        return matches[0].read_text()

    return _read


@pytest.fixture
def make_fake_tt_metal_install(tmp_path: Path) -> Callable[..., Path]:
    """Build a synthetic tt-metal install tree.

    With `with_libs=False` (default) produces only the directory layout that
    `_validate_bundled_tt_metal_root` checks. With `with_libs=True` also writes
    the lib/, runtime/, generated/, ttnn/, and tt_metal/ payload used by the
    bundled-wheel copy logic, including a SONAME symlink chain under lib/.
    """

    def _make(*, with_libs: bool = False) -> Path:
        tt_metal = tmp_path / "tt-metal"
        ttnn_package = tt_metal / "python_packages" / "ttnn" / "ttnn"
        ttnn_package.mkdir(parents=True)
        (ttnn_package / "__init__.py").write_text("")
        (ttnn_package / "_ttnn.so").write_bytes(b"")
        (ttnn_package / "_ttnncpp.so").write_bytes(b"")
        tracy_package = tt_metal / "python_packages" / "tools" / "tracy"
        tracy_package.mkdir(parents=True)
        (tracy_package / "__init__.py").write_text("")
        (tt_metal / "lib").mkdir()
        (tt_metal / "runtime").mkdir()
        (tt_metal / "tt_metal").mkdir()
        (tt_metal / "ttnn" / "cpp").mkdir(parents=True)

        if with_libs:
            for sub in ("operations", "examples"):
                (ttnn_package / sub).mkdir()
                (ttnn_package / sub / "__init__.py").write_text("")
            for library_name in (
                "_ttnn.so",
                "_ttnncpp.so",
                "libtt_metal.so",
                "libtt-umd.so.0",
                "libtt_stl.so",
                "libtracy.so.0.10.0",
                "libfmt.so.11",
            ):
                (tt_metal / "lib" / library_name).write_bytes(b"x")
            (tt_metal / "lib" / "libtt-umd.so").symlink_to("libtt-umd.so.0")
            for relpath in (
                "runtime/hw/firmware.hex",
                "runtime/sfpi/include/sfpi.h",
                "generated/fabric/mesh.yaml",
                "ttnn/api/ttnn/tensor/enum_types.hpp",
                "ttnn/cpp/ttnn/kernel/data.cpp",
                "tt_metal/api/tt-metalium/constants.hpp",
                "tt_metal/hw/kernel.cpp",
            ):
                target = tt_metal / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"x")
        return tt_metal

    return _make
