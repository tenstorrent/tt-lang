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


def test_demo_one_deadlock_detection() -> None:
    """Test that demo_one.py with incorrect wait() instead of reserve() triggers deadlock detection.

    This test modifies demo_one.py to use wait() instead of reserve() for the output
    buffer, which causes a deadlock. The deadlock detection should clearly show:
    1. Which threads are blocked
    2. What operation they're blocked on
    3. Which CircularBuffer they're waiting for
    4. The source location where they're blocked (with accurate line numbers)
    """
    import tempfile
    import re

    # Read the original demo_one.py
    source_file = EXAMPLES_DIR / "demo_one.py"
    with open(source_file) as f:
        lines = f.readlines()
        content = "".join(lines)

    # Introduce the error: change y_cb.reserve() to y_cb.wait()
    # This creates a deadlock where compute waits for y_cb that it should be writing to
    modified_content = content.replace(
        "y_cb.reserve() as y_blk,", "y_cb.wait() as y_blk,"
    )

    # Verify we actually modified something
    assert modified_content != content, "Failed to modify demo_one.py content"

    # Find the line numbers where wait() and reserve() calls are made
    # We'll verify the deadlock message points to these exact lines
    compute_wait_line = None
    dm0_reserve_line = None
    dm1_wait_line = None

    for i, line in enumerate(lines, start=1):
        # In the compute function, after our modification, y_cb.wait() should be present
        if "y_cb.wait() as y_blk" in modified_content.split("\n")[i - 1]:
            # Find the first occurrence in compute function
            if (
                compute_wait_line is None and i > 50 and i < 70
            ):  # Rough range for compute function
                compute_wait_line = i
        # In dm0 function, a_cb.reserve() is the first reserve call
        if "a_cb.reserve() as a_blk" in line:
            if dm0_reserve_line is None and i > 70:  # After compute function
                dm0_reserve_line = i
        # In dm1 function, y_cb.wait() is present
        if "y_cb.wait() as y_blk" in line and i > 120:  # dm1 is later in file
            if dm1_wait_line is None:
                dm1_wait_line = i

    # Create a temporary file with the modified content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(modified_content)
        tmp_path = Path(tmp.name)

    try:
        # Run the modified script
        code, out = run_ttlang_sim_and_capture(tmp_path)

        # Should fail with non-zero exit code
        assert (
            code != 0
        ), f"Expected modified demo_one.py to fail, but it exited with code 0"

        # Check for deadlock detection message
        assert (
            "Deadlock detected: all generators blocked" in out
        ), f"Expected deadlock detection message not found in output:\n{out}"

        # Check that it shows which CB is blocked (y_cb)
        assert (
            "CircularBuffer(y_cb)" in out
        ), f"Expected to see y_cb in deadlock output:\n{out}"

        # Check that it shows the blocked operations
        assert (
            "blocked on wait()" in out
        ), f"Expected to see 'blocked on wait()' in deadlock output:\n{out}"
        assert (
            "blocked on reserve()" in out
        ), f"Expected to see 'blocked on reserve()' in deadlock output:\n{out}"

        # Check that source locations are included with line numbers
        assert (
            " at " in out and ".py:" in out
        ), f"Expected source location (file:line) in deadlock output:\n{out}"

        # Check for multiple cores being blocked (demo_one uses multiple cores)
        assert (
            "core0-compute:" in out
        ), f"Expected core0-compute in deadlock output:\n{out}"
        assert "core0-dm0:" in out, f"Expected core0-dm0 in deadlock output:\n{out}"
        assert "core0-dm1:" in out, f"Expected core0-dm1 in deadlock output:\n{out}"

        # Verify line numbers are accurate by checking they match actual wait()/reserve() calls
        # Extract line numbers from the deadlock output
        # Format: "coreX-Y: blocked on operation() on CircularBuffer(name) at file.py:LINE"
        line_number_pattern = r"core0-(\w+).*?at .*?:(\d+)"
        matches = re.findall(line_number_pattern, out)

        reported_lines = {}
        for thread_name, line_str in matches:
            reported_lines[thread_name] = int(line_str)

        # Note: The line numbers in the output will be for the temporary file,
        # but the structure should be the same as the original.
        # We verify the line numbers point to actual wait()/reserve() calls by
        # checking the temporary file content at those lines.
        with open(tmp_path) as f:
            tmp_lines = f.readlines()

        # Check compute thread line points to y_cb.wait()
        if "compute" in reported_lines:
            compute_line = reported_lines["compute"]
            # Line numbers are 1-indexed
            assert compute_line <= len(tmp_lines), f"Line {compute_line} out of range"
            line_content = tmp_lines[compute_line - 1]
            assert (
                "y_cb.wait()" in line_content
            ), f"Expected y_cb.wait() at line {compute_line} but got: {line_content.strip()}"

        # Check dm0 thread line points to reserve()
        if "dm0" in reported_lines:
            dm0_line = reported_lines["dm0"]
            assert dm0_line <= len(tmp_lines), f"Line {dm0_line} out of range"
            line_content = tmp_lines[dm0_line - 1]
            assert (
                "reserve()" in line_content
            ), f"Expected reserve() at line {dm0_line} but got: {line_content.strip()}"

        # Check dm1 thread line points to y_cb.wait()
        if "dm1" in reported_lines:
            dm1_line = reported_lines["dm1"]
            assert dm1_line <= len(tmp_lines), f"Line {dm1_line} out of range"
            line_content = tmp_lines[dm1_line - 1]
            assert (
                "y_cb.wait()" in line_content
            ), f"Expected y_cb.wait() at line {dm1_line} but got: {line_content.strip()}"

    finally:
        # Clean up temporary file
        tmp_path.unlink()
