# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tt-triage launcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import REPO_ROOT


@pytest.mark.parametrize("wheel_package", [False, True], ids=["runtime", "wheel"])
def test_launcher_runs_triage(tmp_path: Path, wheel_package: bool) -> None:
    if wheel_package:
        triage_directory = tmp_path / "triage"
        runtime_root = None
    else:
        runtime_root = tmp_path / "tt-metal"
        triage_directory = runtime_root / "tools" / "triage"
    triage_directory.mkdir(parents=True)
    (triage_directory / "__init__.py").write_text("")
    (triage_directory / "utils.py").write_text("VALUE = 'loaded'\n")
    (triage_directory / "triage.py").write_text(
        "import sys\n" "import utils\n" "print(utils.VALUE, sys.argv[1])\n"
    )
    environment = os.environ.copy()
    environment.pop("TTLANG_TOOLCHAIN_DIR", None)
    if runtime_root is not None:
        environment["TT_METAL_RUNTIME_ROOT"] = str(runtime_root)
    else:
        environment.pop("TT_METAL_RUNTIME_ROOT", None)
        environment.pop("TT_METAL_HOME", None)
        environment["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "bin" / "tt-triage"), "argument"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "loaded argument\n"


def test_launcher_uses_isolated_container_python(tmp_path: Path) -> None:
    toolchain_directory = tmp_path / "toolchain"
    triage_environment = toolchain_directory / "tt-triage-venv"
    triage_python = triage_environment / "bin" / "python"
    triage_python.parent.mkdir(parents=True)
    triage_python.write_text("#!/bin/sh\n" 'printf "%s %s\\n" "$VIRTUAL_ENV" "$2"\n')
    triage_python.chmod(0o755)

    environment = os.environ.copy()
    environment["TTLANG_TOOLCHAIN_DIR"] = str(toolchain_directory)
    environment.pop("TT_METAL_RUNTIME_ROOT", None)
    environment.pop("TT_METAL_HOME", None)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "bin" / "tt-triage"), "argument"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{triage_environment} argument\n"
