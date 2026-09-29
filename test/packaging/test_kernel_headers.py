# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Header lookup preserves installed resources and explicit include overrides."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest

from conftest import REPO_ROOT


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configured_package(tmp_path: Path) -> Path:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is unavailable")
    build_dir = tmp_path / "custom build-output"
    package_root = build_dir / "python_packages/ttl"
    package_root.mkdir(parents=True)
    config_path = build_dir / "generated/config.py"
    script = tmp_path / "configure.cmake"
    script.write_text('configure_file("${TEMPLATE}" "${OUTPUT}" @ONLY)\n')
    subprocess.run(
        [
            cmake,
            f"-DTEMPLATE={REPO_ROOT / 'python/ttl/config.py.in'}",
            f"-DOUTPUT={config_path}",
            "-DTTLANG_HAS_DEVICE_INT=0",
            "-DTTLANG_KERNEL_INCLUDE_DIR=header-resources",
            "-DTTLANG_KERNEL_HEADER_DIR=custom/LLKs",
            "-P",
            str(script),
        ],
        cwd=build_dir,
        check=True,
    )
    (package_root / "config.py").symlink_to(config_path)
    return package_root


def _stage_headers(config) -> None:
    header_dir = config.KERNEL_INCLUDE_DIR / config.KERNEL_HEADER_DIR
    header_dir.mkdir(parents=True)
    for name in ("experimental_dfb_reset.h", "experimental_dfb_reconfiguration.h"):
        (header_dir / name).write_text("// packaged header\n")


def test_kernel_headers_configured_build_tree(configured_package: Path):
    config = _load_module(configured_package / "config.py")
    _stage_headers(config)

    assert configured_package.is_relative_to(config.BUILD_DIR)
    assert Path(config.__file__).is_symlink()
    assert config.KERNEL_INCLUDE_DIR == configured_package / "header-resources"
    assert config.KERNEL_HEADER_DIR == Path("custom/LLKs")
    assert config.kernel_include_paths([]) == [str(config.KERNEL_INCLUDE_DIR)]


@pytest.mark.parametrize("layout", ["wheel/site-packages", "install/python_packages"])
def test_kernel_headers_relocated_lookup(
    configured_package: Path, tmp_path: Path, layout: str
):
    config = _load_module(configured_package / "config.py")
    _stage_headers(config)
    package_root = tmp_path / layout / "ttl"
    shutil.copytree(configured_package, package_root)
    installed = _load_module(package_root / "config.py")
    installed.BUILD_DIR = tmp_path / "absent-build"

    assert not installed.BUILD_DIR.exists()
    assert not Path(installed.__file__).is_symlink()
    assert installed.kernel_include_paths([]) == [
        str(package_root / "header-resources")
    ]


def test_kernel_headers_preserve_explicit_override_order(configured_package: Path):
    config = _load_module(configured_package / "config.py")
    _stage_headers(config)
    paths = ["/emulator/overrides", "relative/user/headers"]

    assert config.kernel_include_paths(paths) == [
        *paths,
        str(config.KERNEL_INCLUDE_DIR),
    ]
    assert paths == ["/emulator/overrides", "relative/user/headers"]


def test_kernel_headers_missing_resources_preserve_caller_paths(
    configured_package: Path,
):
    config = _load_module(configured_package / "config.py")

    assert config.kernel_include_paths(["/user/headers"]) == ["/user/headers"]


@pytest.mark.parametrize(
    "header, declaration, call",
    [
        (
            "experimental_dfb_reset.h",
            "void reset_dfb_interfaces(uint32_t, uint32_t, uint32_t)",
            "::experimental::reset_dfb_interfaces(0, 0, 0)",
        ),
        (
            "experimental_dfb_reconfiguration.h",
            "void reconfigure_dfb_interfaces(uint32_t)",
            "::experimental::reconfigure_dfb_interfaces(0)",
        ),
    ],
)
@pytest.mark.parametrize("conflicting_namespace", [False, True])
def test_kernel_headers_allow_cpp_overrides(
    configured_package: Path,
    tmp_path: Path,
    header: str,
    declaration: str,
    call: str,
    conflicting_namespace: bool,
):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    config = _load_module(configured_package / "config.py")
    _stage_headers(config)
    override_root = tmp_path / "overrides"
    header_path = override_root / config.KERNEL_HEADER_DIR / header
    header_path.parent.mkdir(parents=True)
    header_path.write_text(
        "#include <cstdint>\n"
        f"namespace experimental {{ inline {declaration} {{}} }}\n"
    )
    include_flags = [
        flag
        for root in config.kernel_include_paths([str(override_root)])
        for flag in ("-I", root)
    ]
    namespace_import = (
        "namespace ckernel { namespace experimental {} }\n" "using namespace ckernel;\n"
        if conflicting_namespace
        else ""
    )
    result = subprocess.run(
        [compiler, "-std=c++17", "-fsyntax-only", "-x", "c++", "-", *include_flags],
        input=(
            namespace_import
            + f'#include "{config.KERNEL_HEADER_DIR / header}"\n'
            + f"void kernel() {{ {call}; }}\n"
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("resources", ["complete", "missing", "source-symlink"])
def test_wheel_smoke_checks_installed_headers(
    configured_package: Path, tmp_path: Path, monkeypatch, capsys, resources: str
):
    config = _load_module(configured_package / "config.py")
    _stage_headers(config)
    installed_root = tmp_path / "site-packages/ttl"
    shutil.copytree(configured_package, installed_root)
    installed = _load_module(installed_root / "config.py")
    header = (
        installed.KERNEL_INCLUDE_DIR
        / installed.KERNEL_HEADER_DIR
        / "experimental_dfb_reset.h"
    )
    if resources != "complete":
        header.unlink()
    if resources == "source-symlink":
        header.symlink_to(
            config.KERNEL_INCLUDE_DIR
            / config.KERNEL_HEADER_DIR
            / "experimental_dfb_reset.h"
        )
    package = ModuleType("ttl")
    package.config = installed
    monkeypatch.setitem(sys.modules, "ttl", package)
    smoke = _load_module(REPO_ROOT / ".github/scripts/smoke-test-wheel.py")

    assert smoke.check_kernel_headers() == (0 if resources == "complete" else 1)
    if resources != "complete":
        assert "kernel header" in capsys.readouterr().err
