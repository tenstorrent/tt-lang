# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for copying bundled ttnn artifacts into the tt-lang wheel tree."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packaging"))

from bundled_ttnn import (  # noqa: E402
    copy_bundled_ttnn,
    stage_bundled_ttnn_python_packages,
)


def _write(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_tt_metal_install(tmp_path: Path) -> Path:
    root = tmp_path / "tt-metal"
    _write(root / "python_packages" / "ttnn" / "ttnn" / "__init__.py", b"")
    _write(
        root / "python_packages" / "ttnn" / "ttnn" / "operations" / "__init__.py", b""
    )
    _write(root / "python_packages" / "ttnn" / "ttnn" / "examples" / "__init__.py", b"")
    _write(root / "python_packages" / "ttnn" / "ttnn" / "_ttnn.so")
    _write(root / "python_packages" / "ttnn" / "ttnn" / "_ttnncpp.so")
    _write(root / "python_packages" / "tools" / "tracy" / "__init__.py", b"")

    for library_name in (
        "_ttnn.so",
        "_ttnncpp.so",
        "libtt_metal.so",
        "libtt-umd.so.0",
        "libtt_stl.so",
        "libtracy.so.0.10.0",
        "libfmt.so.11",
    ):
        _write(root / "lib" / library_name)

    _write(root / "runtime" / "hw" / "firmware.hex")
    _write(root / "runtime" / "sfpi" / "include" / "sfpi.h")
    _write(root / "generated" / "fabric" / "mesh.yaml")
    _write(root / "ttnn" / "api" / "ttnn" / "tensor" / "enum_types.hpp")
    _write(root / "ttnn" / "cpp" / "ttnn" / "kernel" / "data.cpp")
    _write(root / "tt_metal" / "api" / "tt-metalium" / "constants.hpp")
    _write(root / "tt_metal" / "hw" / "kernel.cpp")
    return root


def test_staged_metadata_discovers_ttnn_and_tracy_packages(tmp_path: Path) -> None:
    tt_metal = _make_tt_metal_install(tmp_path)

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


def test_copy_bundled_ttnn_uses_pip_wheel_layout(tmp_path: Path) -> None:
    tt_metal = _make_tt_metal_install(tmp_path)
    build_lib = tmp_path / "build-lib"

    copy_bundled_ttnn(tt_metal, build_lib)

    assert (build_lib / "ttnn" / "__init__.py").is_file()
    assert not (build_lib / "ttnn" / "examples").exists()
    assert not (build_lib / "ttnn" / "_ttnncpp.so").exists()
    assert (build_lib / "ttnn" / "_ttnn.so").is_file()
    assert (build_lib / "ttnn" / "build" / "lib" / "_ttnncpp.so").is_file()
    assert (build_lib / "ttnn" / "build" / "lib" / "libtt_metal.so").is_file()
    assert (build_lib / "ttnn" / "runtime" / "hw" / "firmware.hex").is_file()
    assert not (build_lib / "ttnn" / "runtime" / "sfpi").exists()
    assert (build_lib / "ttnn" / "generated" / "fabric" / "mesh.yaml").is_file()
    assert (
        build_lib / "ttnn" / "ttnn" / "cpp" / "ttnn" / "kernel" / "data.cpp"
    ).is_file()
    assert (build_lib / "ttnn" / "tt_metal" / "hw" / "kernel.cpp").is_file()
    assert (build_lib / "tracy" / "__init__.py").is_file()
