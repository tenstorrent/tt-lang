#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# TODO: This could probably be done better with lit tests
"""CLI tests that invoke ttlsim for simulator examples.

Runs the ttlsim launcher against each script under examples/ and verifies
that the output indicates success.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Paths
THIS_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    """Find the repository root by searching upward from the starting path.

    Args:
        start: Directory to begin searching from

    Returns:
        Path to the repository root directory

    The function searches upward through parent directories looking for
    characteristic markers (examples/ and python/sim/). If not found,
    falls back to the parent of the starting directory.
    """
    for p in [start] + list(start.parents):
        if (p / "examples").exists() and (p / "python" / "sim").exists():
            return p
    # Fallback: assume repo root is the parent of tests
    return start.parent


REPO_ROOT = find_repo_root(THIS_DIR)
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLES_METAL_DIR = REPO_ROOT / "examples" / "metal_examples"

# Use the current Python interpreter to run the launcher module reliably
PYTHON = sys.executable
LAUNCHER_MODULE = [PYTHON, "-m", "sim.ttlsim"]


def run_ttlsim_and_capture(script_path: Path) -> tuple[int, str]:
    """Run ttlsim against the provided example script and return (code, output)."""
    proc = subprocess.run(
        LAUNCHER_MODULE + [str(script_path)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def assert_success_output(code: int, out: str) -> None:
    """Assert that ttlsim ran successfully and produced success output."""
    assert code == 0, f"ttlsim exited with code {code}. Output:\n{out}"
    assert "PASSED" in out, f"Expected 'PASSED' in output. Got:\n{out}"


@pytest.mark.parametrize(
    "script_name",
    [
        "eltwise_add.py",
        "eltwise_pipe.py",
        "eltwise_pipe_core3.py",
        "singlecore_matmul.py",
        "multicore_matmul.py",
    ],
)
def test_example_cli(script_name: str) -> None:
    """Test simulator examples run successfully via ttlsim CLI."""
    code, out = run_ttlsim_and_capture(EXAMPLES_DIR / script_name)
    assert_success_output(code, out)


@pytest.mark.parametrize(
    "example_path",
    [
        "singlecore_matmul/ttlang/singlecore_matmul.py",
        "multicore_matmul/ttlang/multicore_matmul.py",
    ],
)
def test_metal_example_cli(example_path: str) -> None:
    """Test metal examples run successfully via ttlsim CLI."""
    code, out = run_ttlsim_and_capture(EXAMPLES_METAL_DIR / example_path)
    assert_success_output(code, out)
