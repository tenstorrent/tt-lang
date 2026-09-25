# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Header lookup preserves installed resources and explicit include overrides."""

import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT


MODULE_PATH = REPO_ROOT / "python/ttl/_kernel_headers.py"
HEADER_DIRECTORY = Path("ttlang/Target/TTKernel/LLKs")


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("_kernel_headers", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_module(package_root: Path) -> Path:
    package_root.mkdir(parents=True)
    module_path = package_root / "_kernel_headers.py"
    shutil.copyfile(MODULE_PATH, module_path)
    return module_path


def _stage_headers(include_root: Path) -> None:
    shutil.copytree(
        REPO_ROOT / "include" / HEADER_DIRECTORY, include_root / HEADER_DIRECTORY
    )


def test_kernel_headers_source_tree_lookup():
    module = _load_module(MODULE_PATH)

    assert module.kernel_include_paths([]) == [str(REPO_ROOT / "include")]


@pytest.mark.parametrize("layout", ["wheel/site-packages", "install/python_packages"])
def test_kernel_headers_installed_lookup_without_source(tmp_path: Path, layout: str):
    package_root = tmp_path / layout / "ttl"
    module_path = _stage_module(package_root)
    _stage_headers(package_root / "include")
    module = _load_module(module_path)

    assert module.kernel_include_paths([]) == [str(package_root / "include")]


def test_kernel_headers_build_tree_lookup_preserves_module_symlink(tmp_path: Path):
    package_root = tmp_path / "build/python_packages/ttl"
    package_root.mkdir(parents=True)
    module_path = package_root / "_kernel_headers.py"
    module_path.symlink_to(MODULE_PATH)
    _stage_headers(package_root / "include")
    module = _load_module(module_path)

    assert module_path.resolve() == MODULE_PATH
    assert module.kernel_include_paths([]) == [str(package_root / "include")]


def test_kernel_headers_build_tree_falls_back_to_source(tmp_path: Path):
    package_root = tmp_path / "build/python_packages/ttl"
    package_root.mkdir(parents=True)
    module_path = package_root / "_kernel_headers.py"
    module_path.symlink_to(MODULE_PATH)
    module = _load_module(module_path)

    assert module.kernel_include_paths([]) == [str(REPO_ROOT / "include")]


def test_kernel_headers_preserve_explicit_override_order():
    module = _load_module(MODULE_PATH)
    include_paths = ["/emulator/overrides", "relative/user/headers"]

    assert module.kernel_include_paths(include_paths) == [
        "/emulator/overrides",
        "relative/user/headers",
        str(REPO_ROOT / "include"),
    ]
    assert include_paths == ["/emulator/overrides", "relative/user/headers"]


def test_kernel_headers_missing_resources_preserve_caller_paths(tmp_path: Path):
    module_path = _stage_module(tmp_path / "site-packages/ttl")
    module = _load_module(module_path)

    assert module.kernel_include_paths(["/user/headers"]) == ["/user/headers"]


@pytest.mark.parametrize(
    "header, declaration, call",
    [
        (
            "experimental_dfb_reset.h",
            "void reset_dfb_interfaces(uint32_t, uint32_t, uint32_t)",
            "experimental::reset_dfb_interfaces(0, 0, 0)",
        ),
        (
            "experimental_dfb_reconfiguration.h",
            "void reconfigure_dfb_interfaces(uint32_t)",
            "experimental::reconfigure_dfb_interfaces(0)",
        ),
    ],
)
def test_kernel_headers_allow_cpp_overrides(
    tmp_path: Path, header: str, declaration: str, call: str
):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    module = _load_module(MODULE_PATH)
    override_root = tmp_path / "overrides"
    header_path = override_root / HEADER_DIRECTORY / header
    header_path.parent.mkdir(parents=True)
    header_path.write_text(
        "#include <cstdint>\n"
        f"namespace experimental {{ inline {declaration} {{}} }}\n"
    )
    include_flags = [
        flag
        for root in module.kernel_include_paths([str(override_root)])
        for flag in ("-I", root)
    ]
    result = subprocess.run(
        [compiler, "-std=c++17", "-fsyntax-only", "-x", "c++", "-", *include_flags],
        input=f'#include "{HEADER_DIRECTORY / header}"\nvoid kernel() {{ {call}; }}\n',
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
