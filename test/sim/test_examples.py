#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for simulator examples.

Directly imports and runs examples to verify they work correctly with both
greedy and fair schedulers. This is much faster than spawning processes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sim.greenlet_scheduler import set_scheduler_algorithm

# Check if ttnn is available
_ttnn_available = False
try:
    import ttnn  # type: ignore[import-not-found]  # noqa: F401

    _ttnn_available = True
except ImportError:
    pass

requires_ttnn = pytest.mark.skipif(
    not _ttnn_available,
    reason="ttnn not available (run 'pip install ttnn' in .venv)",
)

# Paths
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLES_METAL_DIR = REPO_ROOT / "examples" / "metal_examples"


def run_example(script_path: Path) -> None:
    """Execute an example script as __main__, mirroring ttlang-sim behavior.

    Compiles and execs the script with __name__ == "__main__", so scripts using
    the standard ``if __name__ == "__main__":`` guard run their entry point, and
    scripts with unconditional top-level calls also execute normally.

    Args:
        script_path: Path to the Python script to run
    """
    # Shadow compiler imports with simulator implementations
    from sim import ttl, ttnn as sim_ttnn

    sys.modules["ttl"] = ttl  # type: ignore[assignment]
    sys.modules["ttnn"] = sim_ttnn  # type: ignore[assignment]

    script_dir = str(script_path.parent)
    sys.path.insert(0, script_dir)
    try:
        with open(script_path) as f:
            code = compile(f.read(), str(script_path), "exec")
        exec(
            code,
            {
                "__name__": "__main__",
                "__file__": str(script_path),
                "__builtins__": __builtins__,
            },
        )
    except SystemExit as e:
        # sys.exit(0) in a script is a clean exit; treat it as success.
        # Non-zero exits are real failures and should propagate.
        if e.code != 0:
            raise
    finally:
        sys.path.remove(script_dir)


@pytest.fixture(autouse=True)
def reset_scheduler():
    """Reset scheduler algorithm to fair after each test."""
    yield
    set_scheduler_algorithm("fair")


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
        pytest.param(
            "eltwise_pipe.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "eltwise_pipe_core3.py",
            marks=requires_ttnn,
        ),
        "singlecore_matmul.py",
        "multicore_matmul.py",
        "matmul_1d.py",
        "matmul_1d_mcast.py",
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
def test_example(script_name: str, scheduler: str) -> None:
    """Test simulator examples run successfully with both schedulers."""
    # Skip matmul_1d_mcast.py with fair scheduler (times out due to pipe handling issue)
    if script_name == "matmul_1d_mcast.py" and scheduler == "fair":
        pytest.skip(
            "matmul_1d_mcast.py times out with fair scheduler (TODO: investigate)"
        )

    set_scheduler_algorithm(scheduler)
    run_example(EXAMPLES_DIR / script_name)


@pytest.mark.parametrize(
    "example_path",
    [
        "singlecore_matmul/ttlang/singlecore_matmul.py",
        "multicore_matmul/ttlang/multicore_matmul.py",
    ],
)
@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_metal_example(example_path: str, scheduler: str) -> None:
    """Test metal examples run successfully with both schedulers."""
    set_scheduler_algorithm(scheduler)
    run_example(EXAMPLES_METAL_DIR / example_path)


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_eltwise_add2_fails_with_expected_error(scheduler: str) -> None:
    """Test that eltwise_add_error.py fails with the expected copy validation error."""
    set_scheduler_algorithm(scheduler)

    # Set up simulator imports (shadow compiler imports)
    from sim import ttl, ttnn as sim_ttnn

    sys.modules["ttl"] = ttl  # type: ignore[assignment]
    sys.modules["ttnn"] = sim_ttnn  # type: ignore[assignment]

    # Read and execute the example as __main__
    script_path = EXAMPLES_DIR / "eltwise_add_error.py"
    with open(script_path) as f:
        code = f.read()

    # Add examples directory to sys.path for utils imports
    sys.path.insert(0, str(script_path.parent))
    try:
        with pytest.raises(RuntimeError) as exc_info:
            exec(compile(code, str(script_path), "exec"), {"__name__": "__main__"})
    finally:
        sys.path.remove(str(script_path.parent))

    error_msg = str(exc_info.value)
    # Check for the core error message (shape mismatch)
    assert (
        "Tensor shape (32, 32) (=(1, 1) tiles) does not match Block shape (2, 2) tiles"
        in error_msg
    ), f"Expected error message not found in: {error_msg}"


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_copy_lock_error_fails_with_expected_error(scheduler: str) -> None:
    """Test that copy_lock_error.py fails with the expected copy locking error."""
    set_scheduler_algorithm(scheduler)

    # Set up simulator imports (shadow compiler imports)
    from sim import ttl, ttnn as sim_ttnn

    sys.modules["ttl"] = ttl  # type: ignore[assignment]
    sys.modules["ttnn"] = sim_ttnn  # type: ignore[assignment]

    # Read and execute the example as __main__
    script_path = EXAMPLES_DIR / "copy_lock_error.py"
    with open(script_path) as f:
        code = f.read()

    # Add examples directory to sys.path for utils imports
    sys.path.insert(0, str(script_path.parent))
    try:
        with pytest.raises(RuntimeError) as exc_info:
            exec(compile(code, str(script_path), "exec"), {"__name__": "__main__"})
    finally:
        sys.path.remove(str(script_path.parent))

    error_msg = str(exc_info.value)
    # Check for the core error message (copy access violation)
    assert (
        "Cannot write to Block: Block is locked as copy destination until tx.wait() completes (copy lock error)"
        in error_msg
    ), f"Expected error message not found in: {error_msg}"


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
        with pytest.raises(RuntimeError) as exc_info:
            run_example(tmp_path)

        error_msg = str(exc_info.value)

        assert (
            "Deadlock detected: all generators blocked" in error_msg
        ), f"Expected deadlock message:\n{error_msg}"
        assert (
            "CircularBuffer(a_dfb)" in error_msg
        ), f"Expected to see a_dfb in deadlock output:\n{error_msg}"
        assert (
            "blocked on wait()" in error_msg
        ), f"Expected 'blocked on wait()' in deadlock output:\n{error_msg}"

        # Check that reported source locations point to actual wait()/reserve() calls
        # and that the modified line is among them
        line_number_pattern = r"at .*?:(\d+)"
        reported_line_numbers = {
            int(n) for n in re.findall(line_number_pattern, error_msg)
        }
        assert reported_line_numbers, f"No source locations found in:\n{error_msg}"
        assert modified_line_num in reported_line_numbers, (
            f"Expected line {modified_line_num} (the wait() call) in reported "
            f"locations {reported_line_numbers}.\nError:\n{error_msg}"
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
