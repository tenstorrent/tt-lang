# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the external tt-metal environment helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT  # noqa: E402

MODULE_PATH = REPO_ROOT / "python" / "ttl" / "_setup" / "external_tt_metal.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("external_tt_metal", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detects_install_layout_and_emits_shell_exports(tmp_path: Path) -> None:
    module = _load_module()
    install_root = tmp_path / "tt-metal-install"
    (install_root / "python_packages" / "ttnn" / "ttnn").mkdir(parents=True)
    (install_root / "python_packages" / "tools").mkdir(parents=True)
    (install_root / "python_packages" / "ttnn" / "ttnn" / "_ttnn.so").touch()
    (install_root / "lib").mkdir()

    settings = module.detect_external_tt_metal(install_root)
    shell_exports = module.emit_shell_exports(settings)
    environment = module.environment_for_external_tt_metal(
        settings,
        {"PYTHONPATH": "existing-python", "LD_LIBRARY_PATH": "existing-lib"},
    )

    assert settings.tt_metal_home == install_root
    assert environment["TT_METAL_HOME"] == str(install_root)
    assert environment["TT_METAL_RUNTIME_ROOT"] == str(install_root)
    assert environment["PYTHONPATH"].startswith(
        f"{install_root}/python_packages/ttnn:{install_root}/python_packages/tools:"
    )
    assert environment["LD_LIBRARY_PATH"].startswith(f"{install_root}/lib:")
    assert f"export TT_METAL_HOME={install_root}" in shell_exports
    assert "python_packages/ttnn" in shell_exports


def test_detects_native_layout_with_explicit_build_dir(tmp_path: Path) -> None:
    module = _load_module()
    source_root = tmp_path / "tt-metal-src"
    build_dir = tmp_path / "tt-metal-build"
    (source_root / "ttnn" / "ttnn").mkdir(parents=True)
    (source_root / "tools").mkdir()
    (source_root / "tt_metal").mkdir()
    (build_dir / "lib").mkdir(parents=True)

    settings = module.detect_external_tt_metal(source_root, build_dir)

    assert settings.tt_metal_home == source_root
    assert settings.python_entries == (source_root / "ttnn", source_root / "tools")
    assert settings.library_entries[0] == build_dir / "lib"
    assert build_dir / "tt_metal" in settings.library_entries


def test_native_layout_requires_built_libraries(tmp_path: Path) -> None:
    module = _load_module()
    source_root = tmp_path / "tt-metal-src"
    (source_root / "ttnn" / "ttnn").mkdir(parents=True)
    (source_root / "tt_metal").mkdir()

    with pytest.raises(ValueError, match="has no built libraries"):
        module.detect_external_tt_metal(source_root)


def test_native_layout_rejects_missing_explicit_build_dir(tmp_path: Path) -> None:
    module = _load_module()
    source_root = tmp_path / "tt-metal-src"
    (source_root / "ttnn" / "ttnn").mkdir(parents=True)
    (source_root / "tt_metal").mkdir()

    with pytest.raises(ValueError, match="tt-metal build directory is not a directory"):
        module.detect_external_tt_metal(source_root, tmp_path / "missing-build")


def test_rejects_colon_in_path(tmp_path: Path) -> None:
    module = _load_module()
    bad_root = tmp_path / "tt:metal"
    bad_root.mkdir()

    with pytest.raises(ValueError, match="must not contain ':'"):
        module.detect_external_tt_metal(bad_root)


def test_command_form_runs_with_external_environment(tmp_path: Path) -> None:
    module = _load_module()
    install_root = tmp_path / "tt-metal-install"
    (install_root / "python_packages" / "ttnn" / "ttnn").mkdir(parents=True)
    (install_root / "python_packages" / "tools").mkdir(parents=True)
    (install_root / "python_packages" / "ttnn" / "ttnn" / "_ttnn.so").touch()
    (install_root / "lib").mkdir()

    check_environment = """
import os
import sys

install_root = sys.argv[1]
assert os.environ["TT_METAL_HOME"] == install_root
assert os.environ["TT_METAL_RUNTIME_ROOT"] == install_root
assert os.environ["PYTHONPATH"].split(":")[:2] == [
    f"{install_root}/python_packages/ttnn",
    f"{install_root}/python_packages/tools",
]
assert os.environ["LD_LIBRARY_PATH"].split(":")[0] == f"{install_root}/lib"
"""

    status = module.main(
        [
            "--tt-metal-dir",
            str(install_root),
            "--",
            sys.executable,
            "-c",
            check_environment,
            str(install_root),
        ]
    )

    assert status == 0
