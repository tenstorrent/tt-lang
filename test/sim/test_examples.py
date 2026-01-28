#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# TODO: This could probably be done better with lit tests
"""CLI tests that invoke ttlang-sim for simulator examples.

Runs the ttlang-sim launcher against each script under examples/ and verifies
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
LAUNCHER_MODULE = [PYTHON, "-m", "sim.ttlang_sim"]


def run_ttlang_sim_and_capture(script_path: Path) -> tuple[int, str]:
    """Run ttlang-sim against the provided example script and return (code, output)."""
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
    """Assert that ttlang-sim ran successfully and produced success output."""
    assert code == 0, f"ttlang-sim exited with code {code}. Output:\n{out}"


@pytest.mark.parametrize(
    "script_name",
    [
        "broadcast.py",
        "broadcast_demo.py",
        "general_broadcast.py",
        "eltwise_add.py",
        "eltwise_pipe.py",  # Now supported: reserve() DM blocks can do multiple copies
        "eltwise_pipe_core3.py",  # Now supported: reserve() DM blocks can do multiple copies
        "singlecore_matmul.py",
        "multicore_matmul.py",
        "demo_one.py",
    ],
)
def test_example_cli(script_name: str) -> None:
    """Test simulator examples run successfully via ttlang-sim CLI."""
    code, out = run_ttlang_sim_and_capture(EXAMPLES_DIR / script_name)
    assert_success_output(code, out)


@pytest.mark.parametrize(
    "example_path",
    [
        "singlecore_matmul/ttlang/singlecore_matmul.py",
        "multicore_matmul/ttlang/multicore_matmul.py",
    ],
)
def test_metal_example_cli(example_path: str) -> None:
    """Test metal examples run successfully via ttlang-sim CLI."""
    code, out = run_ttlang_sim_and_capture(EXAMPLES_METAL_DIR / example_path)
    assert_success_output(code, out)


@pytest.mark.skip(reason="multicore reuse matmul not yet supported in simulator")
def test_multicore_reuse_matmul() -> None:
    """Test multicore reuse matmul example (skipped until matmul support is ready)."""
    code, out = run_ttlang_sim_and_capture(
        EXAMPLES_METAL_DIR / "multicore_reuse_matmul/ttlang/multicore_reuse_matmul.py"
    )
    assert_success_output(code, out)


def test_eltwise_add2_fails_with_expected_error() -> None:
    """Test that eltwise_add_error.py fails with the expected copy validation error.

    This example demonstrates a common mistake: copying a single tile into a
    block that expects multiple tiles. The error message should clearly indicate
    the mismatch and point to the exact line where the error occurs.
    """
    code, out = run_ttlang_sim_and_capture(EXAMPLES_DIR / "eltwise_add_error.py")
    assert (
        code != 0
    ), f"Expected eltwise_add_error.py to fail, but it exited with code 0"
    # Check for the core error message (shape mismatch)
    assert (
        "Tensor shape (32, 32) (=(1, 1) tiles) does not match Block shape (2, 2) tiles"
        in out
    ), f"Expected error message not found in output:\n{out}"
    # Verify source location is shown
    assert (
        "examples/eltwise_add_error.py:36" in out
    ), f"Expected source location not found in output:\n{out}"

    # Verify the reported line number is correct by checking the actual source
    source_file = EXAMPLES_DIR / "eltwise_add_error.py"
    with open(source_file) as f:
        lines = f.readlines()
        # Line 36 (1-indexed) should contain the problematic copy call
        error_line = lines[35].strip()  # 0-indexed
        assert "tx_a = copy(a[r, c], a_block)" in error_line, (
            f"Line 36 in eltwise_add_error.py does not contain expected copy call.\n"
            f"Expected: 'tx_a = copy(a[r, c], a_block)'\n"
            f"Got: {error_line}"
        )


def test_copy_lock_error_fails_with_expected_error() -> None:
    """Test that copy_lock_error.py fails with the expected copy locking error.

    This example demonstrates incorrect block access during copy operations:
    attempting to write to a block destination before wait() completes. The error
    message should clearly indicate the access violation.
    """
    code, out = run_ttlang_sim_and_capture(EXAMPLES_DIR / "copy_lock_error.py")
    assert code != 0, f"Expected copy_lock_error.py to fail, but it exited with code 0"
    # Check for the core error message (copy access violation)
    assert (
        "Cannot write to Block: Block has no access (NA state)" in out
    ), f"Expected error message not found in output:\n{out}"
    # Verify source location is shown (line 88 where we attempt to write to a_block)
    assert (
        "examples/copy_lock_error.py:88" in out
    ), f"Expected source location not found in output:\n{out}"

    # Verify the reported line number is correct by checking the actual source
    source_file = EXAMPLES_DIR / "copy_lock_error.py"
    with open(source_file) as f:
        lines = f.readlines()
        # Line 88 (1-indexed) should contain the problematic write
        error_line = lines[87].strip()  # 0-indexed
        assert "a_block.store" in error_line, (
            f"Line 88 in copy_lock_error.py does not contain expected write.\n"
            f"Expected: 'a_block.store'\n"
            f"Got: {error_line}"
        )
