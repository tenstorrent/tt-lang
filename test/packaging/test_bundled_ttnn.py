# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for copying bundled ttnn artifacts into the tt-lang wheel tree."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from conftest import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "packaging"))

from bundled_ttnn import (  # noqa: E402
    copy_bundled_ttnn,
    stage_bundled_ttnn_python_packages,
)


def test_staged_metadata_discovers_ttnn_and_tracy_packages(
    tmp_path: Path,
    make_fake_tt_metal_install: Callable[..., Path],
) -> None:
    tt_metal = make_fake_tt_metal_install(with_libs=True)

    metadata = stage_bundled_ttnn_python_packages(
        tt_metal, tmp_path / "stage", tmp_path
    )

    assert "ttnn" in metadata.packages
    assert "ttnn.operations" in metadata.packages
    assert "ttnn.examples" not in metadata.packages
    assert "tracy" in metadata.packages
    assert metadata.package_dir["ttnn"] == "stage/ttnn"
    assert (tmp_path / "stage" / "ttnn" / "__init__.py").is_file()
    assert not (tmp_path / "stage" / "ttnn" / "_ttnn.so").exists()


def test_copy_bundled_ttnn_uses_pip_wheel_layout(
    tmp_path: Path,
    make_fake_tt_metal_install: Callable[..., Path],
) -> None:
    tt_metal = make_fake_tt_metal_install(with_libs=True)
    build_lib = tmp_path / "build-lib"

    copy_bundled_ttnn(tt_metal, build_lib)

    assert (build_lib / "ttnn" / "__init__.py").is_file()
    assert not (build_lib / "ttnn" / "examples").exists()
    assert not (build_lib / "ttnn" / "_ttnncpp.so").exists()
    assert (build_lib / "ttnn" / "_ttnn.so").is_file()
    assert (build_lib / "ttnn" / "build" / "lib" / "_ttnncpp.so").is_file()
    assert (build_lib / "ttnn" / "build" / "lib" / "libtt_metal.so").is_file()
    assert (build_lib / "ttnn" / "build" / "lib" / "libtt-umd.so").is_symlink()
    assert (build_lib / "ttnn" / "build" / "lib" / "libtt-umd.so").readlink() == Path(
        "libtt-umd.so.0"
    )
    assert (build_lib / "ttnn" / "runtime" / "hw" / "firmware.hex").is_file()
    assert not (build_lib / "ttnn" / "runtime" / "sfpi").exists()
    assert (build_lib / "ttnn" / "generated" / "fabric" / "mesh.yaml").is_file()
    assert (
        build_lib / "ttnn" / "ttnn" / "cpp" / "ttnn" / "kernel" / "data.cpp"
    ).is_file()
    assert (build_lib / "ttnn" / "tt_metal" / "hw" / "kernel.cpp").is_file()
    assert (build_lib / "tracy" / "__init__.py").is_file()
