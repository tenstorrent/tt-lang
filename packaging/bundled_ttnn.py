# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for bundling cached tt-metal/ttnn artifacts into tt-lang wheels."""

from __future__ import annotations

import fnmatch
import glob
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from setuptools import find_packages

TTNN_RUNTIME_REQUIREMENTS = (
    "loguru>=0.6.0",
    "networkx>=3.1",
    "graphviz>=0.20.3",
    "click>=8.1.7",
    "pandas>=2.0.3",
    "seaborn>=0.13.2",
)

_RUNTIME_PATTERNS = ("hw/**/*",)

_TTNN_PATTERNS = ("api/ttnn/tensor/enum_types.hpp",)

_TTNN_CPP_PATTERNS = (
    "ttnn/kernel/**/*",
    "ttnn/operations/**/kernels/**/*",
    "ttnn/operations/**/kernels_ng/**/*",
    "ttnn/operations/**/shared_with_host/**/*",
    "ttnn/operations/kernel_helper_functions/*",
    "ttnn/operations/ccl/**/*",
    "ttnn/operations/data_movement/**/*",
    "ttnn/operations/moreh/**/*",
    "ttnn/kernel/*",
    "ttnn/kernel_lib/*",
    "ttnn/operations/normalization/kernel_util/**/*",
)

_TT_METAL_PATTERNS = (
    "api/tt-metalium/buffer_constants.hpp",
    "api/tt-metalium/buffer_types.hpp",
    "api/tt-metalium/circular_buffer_constants.h",
    "api/tt-metalium/constants.hpp",
    "api/tt-metalium/dev_msgs.h",
    "api/tt-metalium/experimental/fabric/fabric_edm_types.hpp",
    "fabric/fabric_edm_packet_header.hpp",
    "api/tt-metalium/experimental/fabric/edm_fabric_counters.hpp",
    "core_descriptors/*.yaml",
    "fabric/hw/**/*",
    "fabric/mesh_graph_descriptors/*.yaml",
    "fabric/mesh_graph_descriptors/*.textproto",
    "fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
    "fabric/impl/kernels/tt_fabric_mux.cpp",
    "hw/**/*",
    "hostdevcommon/api/hostdevcommon/**/*",
    "impl/dispatch/kernels/**/*",
    "include/**/*",
    "kernels/**/*",
    "tt-llk/**/*",
    "tools/profiler/**/*",
    "soc_descriptors/*.yaml",
    "sfpi-version",
    "pre-compiled/**/*",
)

_LIB_PATTERNS = (
    "_ttnn.so",
    "_ttnncpp.so",
    "libtt_metal.so",
    "libtt-umd.so*",
    "libtt_stl.so",
    "libtracy.so*",
    "libfmt.so*",
)


@dataclass(frozen=True)
class BundledTTNNMetadata:
    packages: list[str]
    package_dir: dict[str, str]


def resolve_tt_metal_root(repo_root: Path) -> Path:
    raw_root = (
        os.environ.get("TTLANG_BUNDLED_TT_METAL_DIR", "").strip()
        or os.environ.get("TTLANG_EXTERNAL_TT_METAL_DIR", "").strip()
    )
    if raw_root:
        root = Path(raw_root).expanduser().resolve()
    else:
        toolchain_dir = os.environ.get("TTLANG_TOOLCHAIN_DIR", "").strip()
        root = (
            Path(toolchain_dir).expanduser().resolve() / "tt-metal"
            if toolchain_dir
            else repo_root / "toolchain" / "tt-metal"
        )

    _require_dir(root, "bundled tt-metal root")
    _require_dir(root / "python_packages" / "ttnn" / "ttnn", "ttnn package")
    _require_file(
        root / "python_packages" / "ttnn" / "ttnn" / "_ttnn.so",
        "ttnn extension",
    )
    _require_file(
        root / "python_packages" / "ttnn" / "ttnn" / "_ttnncpp.so",
        "ttnn cpp extension",
    )
    _require_dir(root / "lib", "tt-metal library directory")
    return root


def stage_bundled_ttnn_python_packages(
    tt_metal_root: Path, stage_root: Path, repo_root: Path
) -> BundledTTNNMetadata:
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)

    _copy_python_package(
        tt_metal_root / "python_packages" / "ttnn" / "ttnn",
        stage_root / "ttnn",
    )
    _copy_python_package(
        tt_metal_root / "python_packages" / "tools" / "tracy",
        stage_root / "tracy",
        required=False,
    )

    packages = find_packages(
        where=str(stage_root), exclude=("ttnn.examples", "ttnn.examples.*")
    )
    package_dir = {
        "ttnn": os.path.relpath(stage_root / "ttnn", repo_root),
    }
    if (stage_root / "tracy" / "__init__.py").is_file():
        package_dir["tracy"] = os.path.relpath(stage_root / "tracy", repo_root)

    return BundledTTNNMetadata(packages=packages, package_dir=package_dir)


def copy_bundled_ttnn(tt_metal_root: Path, build_lib: Path) -> None:
    ttnn_package = build_lib / "ttnn"
    if ttnn_package.exists():
        shutil.rmtree(ttnn_package)

    _copy_python_package(
        tt_metal_root / "python_packages" / "ttnn" / "ttnn",
        ttnn_package,
    )
    _copy_python_package(
        tt_metal_root / "python_packages" / "tools" / "tracy",
        build_lib / "tracy",
        required=False,
    )

    lib_dir = ttnn_package / "build" / "lib"
    _copy_patterns(
        tt_metal_root / "lib", lib_dir, _LIB_PATTERNS, preserve_symlinks=True
    )
    _require_file(lib_dir / "_ttnn.so", "bundled _ttnn.so")
    shutil.copy2(lib_dir / "_ttnn.so", ttnn_package / "_ttnn.so")
    (lib_dir / "_ttnn.so").unlink()

    _copy_patterns(
        tt_metal_root / "runtime",
        ttnn_package / "runtime",
        _RUNTIME_PATTERNS,
        required=False,
    )
    _copy_tree(tt_metal_root / "generated", ttnn_package / "generated", required=False)
    _copy_patterns(tt_metal_root / "ttnn", ttnn_package, _TTNN_PATTERNS, required=False)
    _copy_patterns(
        tt_metal_root / "ttnn" / "cpp",
        ttnn_package / "ttnn" / "cpp",
        _TTNN_CPP_PATTERNS,
    )
    _copy_patterns(
        tt_metal_root / "tt_metal",
        ttnn_package / "tt_metal",
        _TT_METAL_PATTERNS,
    )


def _copy_python_package(src_dir: Path, dst_dir: Path, required: bool = True) -> None:
    if not src_dir.is_dir():
        if required:
            raise RuntimeError(f"required package directory is missing: {src_dir}")
        return
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(
        src_dir,
        dst_dir,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "examples"),
    )


def _copy_tree(src_dir: Path, dst_dir: Path, required: bool = True) -> None:
    if not src_dir.is_dir():
        if required:
            raise RuntimeError(f"required directory is missing: {src_dir}")
        return
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(
        src_dir,
        dst_dir,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
    )


def _copy_patterns(
    src_dir: Path,
    dst_dir: Path,
    patterns: tuple[str, ...],
    exclude_files: set[str] | None = None,
    required: bool = True,
    preserve_symlinks: bool = False,
) -> None:
    if not src_dir.is_dir():
        if required:
            raise RuntimeError(f"required directory is missing: {src_dir}")
        return

    copied = 0
    for pattern in patterns:
        for src_path_text in glob.glob(str(src_dir / pattern), recursive=True):
            src_path = Path(src_path_text)
            if src_path.is_dir():
                continue
            if exclude_files is not None and src_path.name in exclude_files:
                continue
            relative_path = src_path.relative_to(src_dir)
            if _is_ignored_artifact(relative_path):
                continue
            dst_path = dst_dir / relative_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path, follow_symlinks=not preserve_symlinks)
            copied += 1

    if required and copied == 0:
        raise RuntimeError(f"no files matched required patterns in {src_dir}")


def _is_ignored_artifact(relative_path: Path) -> bool:
    parts = set(relative_path.parts)
    if "__pycache__" in parts or ".git" in parts:
        return True
    return any(fnmatch.fnmatch(relative_path.name, pattern) for pattern in ("*.pyc",))


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise RuntimeError(f"{label} is not a directory: {path}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is not a file: {path}")
