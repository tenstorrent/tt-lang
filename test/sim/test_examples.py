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

# Check if ttnn is available
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

# Marker for tests that require ttnn
requires_ttnn = pytest.mark.skipif(
    not TTNN_AVAILABLE,
    reason="ttnn not available (required for tests using ttnn golden functions)",
)

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


def run_ttlang_sim_and_capture(
    script_path: Path, scheduler: str | None = None
) -> tuple[int, str]:
    """Run ttlang-sim against the provided example script and return (code, output).

    Args:
        script_path: Path to the script to run
        scheduler: Optional scheduler algorithm ('greedy' or 'fair')
    """
    cmd = LAUNCHER_MODULE + [str(script_path)]
    if scheduler:
        cmd += ["--scheduler", scheduler]
    proc = subprocess.run(
        cmd,
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
        pytest.param(
            "broadcast.py",
            marks=requires_ttnn,
        ),
        "broadcast_demo.py",
        pytest.param(
            "general_broadcast.py",
            marks=requires_ttnn,
        ),
        "eltwise_add.py",
        "eltwise_add_3d.py",
        "eltwise_pipe.py",
        "eltwise_pipe_core3.py",
        "matmul.py",
        "singlecore_matmul.py",
        "multicore_matmul.py",
        "matmul_1d.py",
        "matmul_1d_mcast.py",
        "eltwise_1d_broadcast.py",
        pytest.param(
            "tutorial/ttnn_base.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "tutorial/single_core_single_tile_block.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "tutorial/single_core_multitile_block.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "tutorial/multicore.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "tutorial/multicore_grid_auto.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "tutorial/single_core_broadcast_single_tile_block.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "tutorial/single_core_broadcast_multitile_blocks.py",
            marks=requires_ttnn,
        ),
    ],
)
@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_example_cli(script_name: str, scheduler: str) -> None:
    """Test simulator examples run successfully via ttlang-sim CLI with both schedulers."""
    # Skip matmul_1d_mcast.py with fair scheduler (times out due to pipe handling issue)
    if script_name == "matmul_1d_mcast.py" and scheduler == "fair":
        pytest.skip(
            "matmul_1d_mcast.py times out with fair scheduler (TODO: investigate)"
        )

    code, out = run_ttlang_sim_and_capture(EXAMPLES_DIR / script_name, scheduler)
    assert_success_output(code, out)


@pytest.mark.parametrize(
    "example_path",
    [
        "singlecore_matmul/ttlang/singlecore_matmul.py",
        "multicore_matmul/ttlang/multicore_matmul.py",
    ],
)
@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_metal_example_cli(example_path: str, scheduler: str) -> None:
    """Test metal examples run successfully via ttlang-sim CLI with both schedulers."""
    code, out = run_ttlang_sim_and_capture(EXAMPLES_METAL_DIR / example_path, scheduler)
    assert_success_output(code, out)


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_eltwise_add2_fails_with_expected_error(scheduler: str) -> None:
    """Test that eltwise_add_error.py fails with the expected copy validation error.

    This example demonstrates a common mistake: copying a single tile into a
    block that expects multiple tiles. The error message should clearly indicate
    the mismatch and point to the exact line where the error occurs.
    """
    code, out = run_ttlang_sim_and_capture(
        EXAMPLES_DIR / "eltwise_add_error.py", scheduler=scheduler
    )
    assert (
        code != 0
    ), f"Expected eltwise_add_error.py to fail, but it exited with code 0"
    # Check for the core error message (shape mismatch)
    assert (
        "Tensor shape (32, 32) does not match Block shape (2, 2) (tile counts: 1 vs 4)"
        in out
    ), f"Expected error message not found in output:\n{out}"

    # Find error line number
    import re

    error_line_number = int(
        re.findall(r"examples/eltwise_add_error.py:(\d+)", out)[0]
    )  # 1-indexed

    # Verify the reported line number is correct by checking the actual source
    source_file = EXAMPLES_DIR / "eltwise_add_error.py"
    with open(source_file) as f:
        lines = f.readlines()
        error_line = lines[error_line_number - 1].strip()  # 0-indexed
        expected_code = "tx_a = ttl.copy(a[r, c], a_block)"
        assert expected_code in error_line, (
            f"Expected line in eltwise_add_error.py does not contain expected copy call.\n"
            f"Expected: '{expected_code}'\n"
            f"Got: {error_line}"
        )


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_copy_lock_error_fails_with_expected_error(scheduler: str) -> None:
    """Test that copy_lock_error.py fails with the expected copy locking error.

    This example demonstrates incorrect block access during copy operations:
    attempting to write to a block destination before wait() completes. The error
    message should clearly indicate the access violation.
    """
    code, out = run_ttlang_sim_and_capture(
        EXAMPLES_DIR / "copy_lock_error.py", scheduler=scheduler
    )
    assert code != 0, f"Expected copy_lock_error.py to fail, but it exited with code 0"
    # Check for the core error message (copy access violation)
    assert (
        "Cannot write to Block: Block is locked as copy destination until tx.wait() completes (copy lock error)"
        in out
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


def test_eltwise_add_deadlock_detection() -> None:
    """Test deadlock detection in eltwise_add.py with reserve() changed to wait().

    Replacing a_dfb.reserve() with a_dfb.wait() in the read DM thread causes a
    deadlock: read blocks waiting for data that only it was supposed to produce,
    compute also blocks waiting on a_dfb, and write blocks waiting on out_dfb.
    """
    import re
    import tempfile

    source_file = EXAMPLES_DIR / "eltwise_add.py"
    with open(source_file) as f:
        content = f.read()

    original = "with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:"
    modified = "with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:"
    modified_content = content.replace(original, modified)

    assert (
        modified_content != content
    ), "Failed to modify eltwise_add.py: pattern not found"

    # Find line number of the modified line to verify it appears in the deadlock output
    modified_line_num = next(
        i
        for i, line in enumerate(modified_content.splitlines(), start=1)
        if modified in line
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(modified_content)
        tmp_path = Path(tmp.name)

    try:
        code, out = run_ttlang_sim_and_capture(tmp_path)

        assert (
            code != 0
        ), f"Expected modified eltwise_add.py to fail, but it exited with code 0"

        assert (
            "Deadlock detected: all generators blocked" in out
        ), f"Expected deadlock message:\n{out}"
        assert (
            "DataflowBuffer(a_dfb)" in out
        ), f"Expected to see a_dfb in deadlock output:\n{out}"
        assert (
            "blocked on wait()" in out
        ), f"Expected 'blocked on wait()' in deadlock output:\n{out}"

        # Check that reported source locations point to actual wait()/reserve() calls
        # and that the modified line is among them
        line_number_pattern = r"-->\s+.*?:(\d+):\d+"
        reported_line_numbers = {int(n) for n in re.findall(line_number_pattern, out)}
        assert reported_line_numbers, f"No source locations found in:\n{out}"
        assert modified_line_num in reported_line_numbers, (
            f"Expected line {modified_line_num} (the wait() call) in reported "
            f"locations {reported_line_numbers}.\nOutput:\n{out}"
        )

        tmp_lines = tmp_path.read_text().splitlines()
        for line_num in reported_line_numbers:
            assert line_num <= len(tmp_lines), f"Reported line {line_num} out of range"
            line_content = tmp_lines[line_num - 1]
            assert "wait()" in line_content or "reserve()" in line_content, (
                f"Line {line_num} does not contain wait() or reserve(): "
                f"{line_content.strip()}"
            )

    finally:
        tmp_path.unlink()


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_eltwise_1d_broadcast_warning(scheduler: str) -> None:
    """Test that eltwise_1d_broadcast.py displays 1D broadcast hardware warning.

    This example demonstrates broadcasting with 1D blocks. Since 1D broadcasts
    are not supported by current hardware, the simulator should emit warnings
    when ttl.math.broadcast() is called on 1D blocks, but the script should
    still execute successfully.
    """
    code, out = run_ttlang_sim_and_capture(
        EXAMPLES_DIR / "eltwise_1d_broadcast.py", scheduler=scheduler
    )

    # The example should run successfully (warnings don't fail execution)
    assert code == 0, (
        f"Expected eltwise_1d_broadcast.py to succeed, but it exited with code {code}\n"
        f"Output:\n{out}"
    )

    # Verify the 1D broadcast warning appears
    assert (
        "warning: 1D broadcast is not supported on current hardware" in out
    ), f"Expected 1D broadcast warning not found in output:\n{out}"

    # Verify source location is shown (the broadcast calls are in eltwise_compute function)
    assert (
        "examples/eltwise_1d_broadcast.py:" in out
    ), f"Expected source location not found in output:\n{out}"
