# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIM_SCRIPT = REPO_ROOT / ".github/containers/trim-emule-venv.sh"


def run_trim(tmp_path, *, failure=""):
    interpreter = tmp_path / "venv python"
    log = tmp_path / "calls.jsonl"
    interpreter.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['TRIM_TEST_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('TRIM_TEST_FAILURE') in sys.argv[1:]:\n"
        "    sys.exit(1)\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    result = subprocess.run(
        ["bash", str(TRIM_SCRIPT), str(interpreter)],
        env={**os.environ, "TRIM_TEST_LOG": str(log), "TRIM_TEST_FAILURE": failure},
        capture_output=True,
        text=True,
        check=False,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    return result, calls


def test_trim_preserves_torch_and_test_dependencies(tmp_path):
    result, calls = run_trim(tmp_path)
    assert result.returncode == 0, result.stderr
    uninstall = calls[1]
    assert uninstall[:4] == ["-m", "pip", "uninstall", "--yes"]
    assert {"sphinx", "black", "pre-commit", "pyright", "tt-smi"}.issubset(
        uninstall[4:]
    )
    assert not any(
        name.startswith(("torch", "triton", "nvidia", "cuda", "pytest"))
        or name in {"numpy", "ml_dtypes", "nanobind", "pybind11", "lit", "packaging"}
        for name in uninstall[4:]
    )
    assert calls[2] == ["-m", "pip", "check"]
    assert "import torch" in calls[3][1]


@pytest.mark.parametrize("failure,expected_calls", [("-c", 1), ("check", 3)])
def test_trim_stops_on_environment_or_dependency_failure(
    tmp_path, failure, expected_calls
):
    result, calls = run_trim(tmp_path, failure=failure)
    assert result.returncode == 1
    assert len(calls) == expected_calls


def test_trim_requires_one_interpreter():
    result = subprocess.run(
        ["bash", str(TRIM_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_trim_rejects_system_python_with_optimization_enabled(tmp_path):
    pip_marker = tmp_path / "pip-invoked"
    (tmp_path / "pip.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(pip_marker)!r}).touch()\n"
        "raise SystemExit('pip must not be invoked')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(TRIM_SCRIPT), sys._base_executable],
        env={**os.environ, "PYTHONOPTIMIZE": "1", "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "expected a virtual environment" in result.stderr
    assert not pip_marker.exists()


def test_development_dependencies_remain_enabled_by_default():
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert (
        "option(TTLANG_INSTALL_DEV_REQUIREMENTS\n"
        '  "Install Python development requirements during configuration" ON)' in cmake
    )
    assert (
        "if(TTLANG_INSTALL_DEV_REQUIREMENTS)\n"
        '  ttlang_pip_install_requirements("${Python3_EXECUTABLE}" '
        '"${CMAKE_CURRENT_SOURCE_DIR}/dev-requirements.txt")\n'
        "endif()" in cmake
    )
