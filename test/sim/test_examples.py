#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""In-process tests for simulator examples.

Runs example scripts directly in the test process, using context reset
between tests for isolation. This is much faster than subprocess-based testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Check if ttnn is available BEFORE we shadow it with simulator version
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

# Import simulator modules
from python.sim.context import reset_context
from python.sim.greenlet_scheduler import set_scheduler_algorithm
from python.sim.program import set_max_l1_bytes
from python.sim.stats import reset_stats
from python.sim.ttlang_sim import execute_script_with_simulator
from python import sim

# Marker for tests that require ttnn
requires_ttnn = pytest.mark.skipif(
    not TTNN_AVAILABLE,
    reason="ttnn not available (required for tests using ttnn golden functions)",
)

# Paths
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
ERRORS_DIR = REPO_ROOT / "examples" / "errors"
EXAMPLES_METAL_DIR = REPO_ROOT / "examples" / "metal_examples"


@pytest.fixture(autouse=True)
def reset_simulator_context():
    """Reset simulator context before each test for isolation."""
    reset_context()
    reset_stats()
    yield


# Per-script L1 overrides for examples that legitimately exceed the default
# limit (e.g. due to large 3D block shapes). Values are in bytes.
_L1_OVERRIDES: dict[str, int] = {
    "eltwise_add_3d.py": 1_572_864,  # 3 x shape=(2,2,1) x bfloat16 CBs
}


def run_script_in_process(
    script_path: Path,
    scheduler: str = "fair",
    max_l1_bytes: int | None = None,
) -> tuple[int, str]:
    """Run a script in-process with simulator backend.

    Args:
        script_path: Path to the Python file to execute
        scheduler: Scheduler algorithm ('greedy' or 'fair')
        max_l1_bytes: Optional L1 memory limit override in bytes; uses the
            simulator default when None

    Returns:
        (exit_code, output) tuple where exit_code is 0 on success, 1 on error
    """
    set_scheduler_algorithm(scheduler)
    if max_l1_bytes is not None:
        set_max_l1_bytes(max_l1_bytes)

    # Shadow sys.modules locally (same as ttlang_sim.setup_simulator_imports())
    # Done here so it doesn't interfere with other tests in parallel execution
    original_modules = {"ttl": sys.modules.get("ttl"), "ttnn": sys.modules.get("ttnn")}
    sys.modules["ttl"] = sim.ttl  # type: ignore[assignment]
    sys.modules["ttnn"] = sim.ttnn  # type: ignore[assignment]

    try:
        # Use the shared execution logic from ttlang_sim
        return execute_script_with_simulator(script_path, capture_output=True)
    finally:
        # Restore original sys.modules to avoid interfering with other tests
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.mark.parametrize(
    "script_name",
    [
        pytest.param(
            "broadcast.py",
            marks=requires_ttnn,
        ),
        "broadcast_demo.py",
        "group_transfer_upsample.py",
        pytest.param(
            "general_broadcast.py",
            marks=requires_ttnn,
        ),
        "eltwise_add.py",
        "eltwise_add_3d.py",
        "eltwise_pipe.py",
        "eltwise_pipe_core3.py",
        pytest.param(
            "matmul.py",
            marks=pytest.mark.xfail(reason="Required broadcast not yet supported"),
        ),
        "matmul_acc.py",
        "single_node_matmul.py",
        "multinode_matmul.py",
        "matmul_1d.py",
        "matmul_1d_mcast.py",
        "eltwise_1d_broadcast.py",
        pytest.param(
            "elementwise-tutorial/step_0_ttnn_base.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "elementwise-tutorial/step_1_single_node_single_tile_block.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "elementwise-tutorial/step_2_single_node_multitile_block.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "elementwise-tutorial/step_3_multinode.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "elementwise-tutorial/step_4_multinode_grid_auto.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "tutorial/single_node_broadcast_single_tile_block.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "tutorial/single_node_broadcast_multitile_blocks.py",
            marks=requires_ttnn,
        ),
        pytest.param(
            "tt_upsample.py",
            marks=requires_ttnn,
        ),
    ],
)
@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_example_cli(script_name: str, scheduler: str) -> None:
    """Test simulator examples run successfully with both schedulers."""
    # Skip matmul_1d_mcast.py with fair scheduler (times out due to pipe handling issue)
    if script_name == "matmul_1d_mcast.py" and scheduler == "fair":
        pytest.skip(
            "matmul_1d_mcast.py times out with fair scheduler (TODO: investigate)"
        )

    code, out = run_script_in_process(
        EXAMPLES_DIR / script_name,
        scheduler,
        max_l1_bytes=_L1_OVERRIDES.get(script_name),
    )
    assert code == 0, f"Script failed with code {code}. Output:\n{out}"


@pytest.mark.parametrize(
    "example_path",
    [
        "single_node_matmul/ttlang/single_node_matmul.py",
        "multinode_matmul/ttlang/multinode_matmul.py",
    ],
)
@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_metal_example_cli(example_path: str, scheduler: str) -> None:
    """Test metal examples run successfully with both schedulers."""
    code, out = run_script_in_process(EXAMPLES_METAL_DIR / example_path, scheduler)
    assert code == 0, f"Script failed with code {code}. Output:\n{out}"


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_eltwise_add2_fails_with_expected_error(scheduler: str) -> None:
    """Test that eltwise_add_error.py fails with the expected copy validation error.

    This example demonstrates a common mistake: copying a single tile into a
    block that expects multiple tiles. The error message should clearly indicate
    the mismatch and point to the exact line where the error occurs.
    """
    code, out = run_script_in_process(ERRORS_DIR / "eltwise_add_error.py", scheduler)
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
        re.findall(r"examples/errors/eltwise_add_error.py:(\d+)", out)[0]
    )  # 1-indexed

    # Verify the reported line number is correct by checking the actual source
    source_file = ERRORS_DIR / "eltwise_add_error.py"
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
    code, out = run_script_in_process(ERRORS_DIR / "copy_lock_error.py", scheduler)
    assert code != 0, f"Expected copy_lock_error.py to fail, but it exited with code 0"
    # Check for the core error message (copy access violation)
    assert (
        "Cannot write to Block: Block is locked as copy destination until tx.wait() completes (copy lock error)"
        in out
    ), f"Expected error message not found in output:\n{out}"
    # Verify source location is shown (line 90 where we attempt to write to a_block)
    assert (
        "examples/errors/copy_lock_error.py:90" in out
    ), f"Expected source location not found in output:\n{out}"

    # Verify the reported line number is correct by checking the actual source
    source_file = ERRORS_DIR / "copy_lock_error.py"
    with open(source_file) as f:
        lines = f.readlines()
        # Line 90 (1-indexed) should contain the problematic write
        error_line = lines[89].strip()  # 0-indexed
        assert "a_block.store" in error_line, (
            f"Line 90 in copy_lock_error.py does not contain expected write.\n"
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
        code, out = run_script_in_process(tmp_path)

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
def test_max_dfbs_warning_warns_at_limit(scheduler: str) -> None:
    """Test that max_dfbs_warning.py emits a DFB limit warning but still succeeds.

    This example allocates 36 DataflowBuffers, exceeding the default limit of 32.
    The warning is issued at kernel definition time before any thread execution.
    """
    with pytest.warns(UserWarning, match="hardware limit is 32"):
        code, out = run_script_in_process(ERRORS_DIR / "max_dfbs_warning.py", scheduler)
    assert (
        code == 0
    ), f"Expected max_dfbs_warning.py to succeed, but it exited with code {code}:\n{out}"


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_eltwise_1d_broadcast_warning(scheduler: str) -> None:
    """Test that eltwise_1d_broadcast.py displays 1D broadcast hardware warning.

    This example demonstrates broadcasting with 1D blocks. Since 1D broadcasts
    are not supported by current hardware, the simulator should emit warnings
    when ttl.math.broadcast() is called on 1D blocks, but the script should
    still execute successfully.
    """
    code, out = run_script_in_process(
        EXAMPLES_DIR / "eltwise_1d_broadcast.py", scheduler
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
