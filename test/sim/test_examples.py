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

# Paths
THIS_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
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


class TestExamplesCLI:
    def assert_success_output(self, code: int, out: str) -> None:
        assert code == 0, f"ttlsim exited with code {code}. Output:\n{out}"
        assert "success" in out.lower(), f"Expected 'success' in output. Got:\n{out}"

    def test_eltwise_add_cli(self) -> None:
        code, out = run_ttlsim_and_capture(EXAMPLES_DIR / "eltwise_add.py")
        self.assert_success_output(code, out)

    def test_eltwise_pipe_cli(self) -> None:
        code, out = run_ttlsim_and_capture(EXAMPLES_DIR / "eltwise_pipe.py")
        self.assert_success_output(code, out)

    def test_eltwise_pipe_core3_cli(self) -> None:
        code, out = run_ttlsim_and_capture(EXAMPLES_DIR / "eltwise_pipe_core3.py")
        self.assert_success_output(code, out)

    def test_singlecore_matmul_cli(self) -> None:
        code, out = run_ttlsim_and_capture(EXAMPLES_DIR / "singlecore_matmul.py")
        self.assert_success_output(code, out)

    def test_multicore_matmul_cli(self) -> None:
        code, out = run_ttlsim_and_capture(EXAMPLES_DIR / "multicore_matmul.py")
        self.assert_success_output(code, out)


class TestMetalExamplesCLI:
    def assert_success_output(self, code: int, out: str) -> None:
        assert code == 0, f"ttlsim exited with code {code}. Output:\n{out}"
        out_lower = out.lower()
        assert (
            "success" in out_lower or "passed" in out_lower
        ), f"Expected 'success' or 'passed' in output. Got:\n{out}"

    def test_singlecore_matmul_metal_cli(self) -> None:
        code, out = run_ttlsim_and_capture(
            EXAMPLES_METAL_DIR / "singlecore_matmul" / "ttlang" / "singlecore_matmul.py"
        )
        self.assert_success_output(code, out)

    def test_multicore_matmul_metal_cli(self) -> None:
        code, out = run_ttlsim_and_capture(
            EXAMPLES_METAL_DIR / "multicore_matmul" / "ttlang" / "multicore_matmul.py"
        )
        self.assert_success_output(code, out)
