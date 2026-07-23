# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tt-triage launcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import venv

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


def test_launcher_runs_triage_from_container_toolchain(tmp_path: Path) -> None:
    toolchain_directory = tmp_path / "toolchain"
    triage_environment = toolchain_directory / "tt-triage-venv"
    venv.EnvBuilder(with_pip=False).create(triage_environment)

    triage_directory = toolchain_directory / "tt-metal" / "tools" / "triage"
    triage_directory.mkdir(parents=True)
    (triage_directory / "utils.py").write_text("VALUE = 'loaded'\n")
    (triage_directory / "triage.py").write_text(
        "import os\n"
        "import sys\n"
        "import utils\n"
        "print(utils.VALUE, os.environ['VIRTUAL_ENV'], sys.argv[1])\n"
    )

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
    assert result.stdout == f"loaded {triage_environment} argument\n"
