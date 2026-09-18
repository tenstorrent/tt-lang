# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for complete pytest-xdist execution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT


def _plugin_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_path = str(REPO_ROOT / "test")
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    return environment


def test_abnormal_xdist_worker_termination_fails_session(tmp_path: Path) -> None:
    crash_test = tmp_path / "test_000_crash.py"
    crash_test.write_text(
        "import os\n\n" "def test_worker_crash():\n" "    os._exit(17)\n"
    )
    pending_test = tmp_path / "test_pending.py"
    pending_test.write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('case', range(100))\n"
        "def test_pending(case):\n"
        "    assert case >= 0\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path),
            "-p",
            "hardware_pytest_plugin",
            "-n",
            "2",
            "--max-worker-restart=0",
            "--reruns=3",
        ],
        cwd=REPO_ROOT,
        env=_plugin_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, output
    assert "xdist workers terminated abnormally" in output


def test_compile_only_marker_is_registered_with_explicit_config(
    tmp_path: Path,
) -> None:
    pytest_config = tmp_path / "pytest.ini"
    pytest_config.write_text("[pytest]\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "hardware_pytest_plugin",
            "-c",
            str(pytest_config),
            "--markers",
        ],
        cwd=REPO_ROOT,
        env=_plugin_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.OK, output
    assert "@pytest.mark.compile_only:" in output
