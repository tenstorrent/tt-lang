#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""In-process tests for simulator examples.

Runs example scripts directly in the test process, using context reset
between tests for isolation. This is much faster than subprocess-based testing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Check if ttnn is available BEFORE we shadow it with simulator version
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

# Import simulator modules
from sim.context import reset_context

from test_helpers.sim_runner import run_script_in_process

_requires_ttnn_skip = pytest.mark.skipif(
    not TTNN_AVAILABLE,
    reason="ttnn not available (required for tests using ttnn golden functions)",
)
requires_ttnn_marks = (pytest.mark.requires_ttnn, _requires_ttnn_skip)

# Paths
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
ERRORS_DIR = REPO_ROOT / "examples" / "errors"
EXAMPLES_METAL_DIR = REPO_ROOT / "examples" / "metal_examples"
MATMUL_TUTORIAL_DIR = EXAMPLES_DIR / "matmul-tutorial"


@pytest.fixture(autouse=True)
def reset_simulator_context():
    """Reset simulator context before each test for isolation."""
    reset_context()
    yield


# Per-script L1 overrides for examples that legitimately exceed the default
# limit (e.g. due to large 3D block shapes). Values are in bytes.
_L1_OVERRIDES: dict[str, int] = {
    "eltwise_add_3d.py": 1_572_864,  # 3 x shape=(2,2,1) x bfloat16 CBs
}

# Scripts whose correctness checks are calibrated for their declared dtypes
# (e.g. bfloat16 ULP tolerances).  Run with float32 promotion disabled so
# they execute with the dtypes written in the source file.
_NO_PROMOTION_SCRIPTS: frozenset[str] = frozenset(
    [
        "matmul_1d.py",
        "matmul_1d_mcast.py",
    ]
)

# --- Spec-example coverage (single source of truth) ---------------------------
#
# Every file under examples/spec/**/*.py must be listed here (or in
# _SPEC_EXAMPLES_EXPECT_FAILURE). test_all_spec_examples_are_covered enforces
# this: a spec example added without a simulator test fails the suite. This is
# how we force each new spec example to ship with a test.
#
# _SPEC_EXAMPLES_PASSING: run successfully on the simulator (golden- or
# structurally-checked); they are parametrized into test_example_cli below.
_SPEC_EXAMPLES_PASSING = [
    # Wrapped in @ttl.operation, with a torch golden asserted outside the
    # spec:begin/end markers, so what the example computes is checked and not
    # only run.
    "spec/block/batched_matmul_bias.py",
    "spec/block/elementwise_broadcast_reduce.py",
    "spec/copy/group_transfer.py",
    "spec/operation_function/operation_function.py",
    "spec/tensor_slice/tensor_slice.py",
    # Pipe data movement (unicast / multicast / loopback) with goldens.
    "spec/pipe/scatter.py",
    "spec/pipe/scatter_gather.py",
    "spec/pipe/forward_neighbor.py",
    "spec/pipe/gather.py",
    # Shape introspection: these assert the shapes they demonstrate.
    "spec/dataflow_buffer/tiled_tensor_shape.py",
    "spec/dataflow_buffer/row_major_tensor_shape.py",
    "spec/dataflow_buffer/dataflow_buffer.py",
    "spec/operation_function/multi_kernel_operation.py",
    # Debugging snippets wrapped in @ttl.operation; debug_printing.py also
    # asserts the text its kernel prints.
    "spec/performance_and_debugging/debug_printing.py",
    "spec/performance_and_debugging/signpost.py",
    # Grid/node introspection: node-dependent setup runs per node, asserted.
    "spec/grid/grid_size.py",
    "spec/grid/node.py",
]

# _SPEC_EXAMPLES_EXPECT_FAILURE: exercise an interface that is not implemented
# in the simulator yet; each is asserted to fail *at that interface* by
# test_semaphore_examples_fail_at_unimplemented_interface. When the feature
# lands, that test flips red and the example should move to _SPEC_EXAMPLES_PASSING.
_SPEC_EXAMPLES_EXPECT_FAILURE = [
    "spec/semaphore/many_to_one_barrier.py",
    "spec/semaphore/one_to_many_barrier.py",
]


@pytest.mark.parametrize(
    "script_name",
    [
        pytest.param(
            "broadcast.py",
            marks=requires_ttnn_marks,
        ),
        "group_transfer_upsample.py",
        "height_shard_gather.py",
        pytest.param(
            "general_broadcast.py",
            marks=requires_ttnn_marks,
        ),
        "eltwise_add.py",
        "eltwise_add_3d.py",
        "eltwise_pipe.py",
        "eltwise_pipe_node3.py",
        # Runnable spec examples (single source of truth: _SPEC_EXAMPLES_PASSING).
        *_SPEC_EXAMPLES_PASSING,
        pytest.param(
            "matmul.py",
            marks=pytest.mark.xfail(reason="Required broadcast not yet supported"),
        ),
        "matmul_acc.py",
        "single_node_matmul.py",
        "multinode_matmul.py",
        "matmul_1d.py",
        "matmul_1d_mcast.py",
        pytest.param(
            "elementwise-tutorial/step_0_ttnn_base.py",
            marks=requires_ttnn_marks,
        ),
        pytest.param(
            "elementwise-tutorial/step_1_single_node_single_tile_block.py",
            marks=requires_ttnn_marks,
        ),
        pytest.param(
            "elementwise-tutorial/step_2_single_node_multitile_block.py",
            marks=requires_ttnn_marks,
        ),
        pytest.param(
            "elementwise-tutorial/step_3_multinode.py",
            marks=requires_ttnn_marks,
        ),
        pytest.param(
            "elementwise-tutorial/step_4_multinode_grid_full.py",
            marks=requires_ttnn_marks,
        ),
        pytest.param(
            "tutorial/single_node_broadcast_single_tile_block.py",
            marks=requires_ttnn_marks,
        ),
        pytest.param(
            "tutorial/single_node_broadcast_multitile_blocks.py",
            marks=requires_ttnn_marks,
        ),
        pytest.param(
            "tt_upsample.py",
            marks=requires_ttnn_marks,
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
        no_float32_promotion=script_name in _NO_PROMOTION_SCRIPTS,
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
    code, out = run_script_in_process(
        EXAMPLES_METAL_DIR / example_path,
        scheduler,
        no_float32_promotion=True,
    )
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
        "Cannot write to this buffer block" in out
        and "copy lock error" in out
        and "in-flight" in out
    ), f"Expected error message not found in output:\n{out}"
    # Verify source location is shown (line 90 where we attempt to write to a_block)
    assert (
        "examples/errors/copy_lock_error.py:90" in out
    ), f"Expected source location not found in output:\n{out}"
    assert (
        "Where: copy into this block was requested at" in out
        and "copy_lock_error.py:87" in out
    ), f"Expected pending-copy callsite (ttl.copy line 87) in output:\n{out}"

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


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_copy_source_lock_error_fails_with_expected_error(scheduler: str) -> None:
    """copy_source_lock_error.py fails with ROR copy-source lock and pending-copy Where line."""
    code, out = run_script_in_process(
        ERRORS_DIR / "copy_source_lock_error.py", scheduler
    )
    assert code != 0, f"Expected copy_source_lock_error.py to fail, got exit {code}"
    assert (
        "Cannot write to this buffer block" in out
        and "ROR" in out
        and "in-flight" in out.lower()
    ), f"Expected ROR copy-source lock message in output:\n{out}"
    assert (
        "examples/errors/copy_source_lock_error.py:87" in out
    ), f"Expected diagnostic line for bad store in output:\n{out}"
    assert (
        "Where: copy from this block was requested at" in out
        and "copy_source_lock_error.py:85" in out
    ), f"Expected pending-copy callsite (ttl.copy from block, line 85) in output:\n{out}"

    source_file = ERRORS_DIR / "copy_source_lock_error.py"
    lines = source_file.read_text().splitlines()
    assert "tx_src = ttl.copy(a_block, out[row_slice, col_slice])" in lines[84].strip()
    assert "a_block.store(a_block)" in lines[86].strip()


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
@pytest.mark.parametrize("script_name", _SPEC_EXAMPLES_EXPECT_FAILURE)
def test_semaphore_examples_fail_at_unimplemented_interface(
    script_name: str, scheduler: str
) -> None:
    """The ttl.Semaphore barrier API used by these spec examples is not
    implemented in the simulator (or the compiler) yet: see
    https://github.com/tenstorrent/tt-lang/issues/176 (simulator),
    https://github.com/tenstorrent/tt-lang/issues/182 (compiler) and
    https://github.com/tenstorrent/tt-lang/issues/177 (multi-chip).

    Each example is wrapped so its node-dependent setup runs, but it must fail
    *specifically* at the ttl.Semaphore() call. A success -- or a failure for
    any other reason -- means the situation changed (most likely semaphores
    were implemented) and the example should be promoted to a real,
    golden-checked test rather than an expect-failure one.
    """
    code, out = run_script_in_process(EXAMPLES_DIR / script_name, scheduler)
    assert code != 0, (
        f"{script_name} unexpectedly succeeded. The ttl.Semaphore barrier API "
        f"appears to be implemented now -- promote this to a real golden test.\n"
        f"Output:\n{out}"
    )
    assert "no attribute 'Semaphore'" in out, (
        f"{script_name} failed, but not at the unimplemented ttl.Semaphore "
        f"interface. The failure mode changed; investigate and update the "
        f"example/test.\nOutput:\n{out}"
    )


def test_all_spec_examples_are_covered() -> None:
    """Enforce that every spec example ships with a simulator test.

    Discovers every ``examples/spec/**/*.py`` on disk and requires each to be
    registered as a simulator test -- either in ``_SPEC_EXAMPLES_PASSING`` (run
    + golden/structural check via ``test_example_cli``) or, if it exercises an
    interface the simulator does not implement yet, in
    ``_SPEC_EXAMPLES_EXPECT_FAILURE`` (asserted to fail at that interface).

    A new spec example therefore cannot land without a test: this guard fails
    until the author adds it to one of those lists. It also flags registered
    entries whose files were removed or renamed.
    """
    spec_root = EXAMPLES_DIR / "spec"
    on_disk = {
        p.relative_to(EXAMPLES_DIR).as_posix()
        for p in spec_root.rglob("*.py")
        if p.name != "__init__.py"
    }
    registered = set(_SPEC_EXAMPLES_PASSING) | set(_SPEC_EXAMPLES_EXPECT_FAILURE)

    unregistered = sorted(on_disk - registered)
    assert not unregistered, (
        "These spec examples have no simulator test. Every example under "
        "examples/spec/ must ship with one: add each path to "
        "_SPEC_EXAMPLES_PASSING (if it runs on the simulator) or to "
        "_SPEC_EXAMPLES_EXPECT_FAILURE (if it must fail at an unimplemented "
        "interface) in test_examples.py:\n"
        + "\n".join(f"  - {s}" for s in unregistered)
    )

    stale = sorted(registered - on_disk)
    assert not stale, (
        "These spec examples are registered in test_examples.py but no longer "
        "exist on disk (removed or renamed?). Update the lists:\n"
        + "\n".join(f"  - {s}" for s in stale)
    )


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
@pytest.mark.parametrize(
    "error_script", ["copy_lock_error.py", "copy_source_lock_error.py"]
)
def test_operation_kernel_errors_do_not_print_simulator_stack_frames(
    scheduler: str,
    error_script: str,
) -> None:
    """Failures inside kernels driven by ``ttl.operation`` must not dump Python tracebacks through simulator internals.

    The cooperative scheduler prints ``print_diagnostic_error`` (user file:line + snippet). ``ttlang_sim`` must not
    call ``traceback.print_exception`` for those wrapped errors, which would list frames such as
    ``greenlet_scheduler.py``, ``program.py``, or ``dfb.py``.
    """
    code, out = run_script_in_process(ERRORS_DIR / error_script, scheduler)
    assert code != 0
    assert f"examples/errors/{error_script}" in out

    assert (
        "Traceback (most recent call last):" not in out
    ), f"Expected no Python traceback header in captured output:\n{out}"

    for path_fragment in (
        "greenlet_scheduler.py",
        "python/sim/program.py",
        "python/sim/dfb.py",
        "python/sim/greenlet_scheduler.py",
    ):
        assert (
            path_fragment not in out
        ), f"Unexpected simulator frame reference {path_fragment!r} in output:\n{out}"


def test_eltwise_add_deadlock_detection() -> None:
    """Deadlock example: read uses wait() on a_dfb/b_dfb instead of reserve().

    See ``examples/errors/eltwise_add_deadlock.py``. Read blocks waiting for data
    that only it was supposed to produce; compute also blocks on a_dfb/b_dfb; write
    blocks on out_dfb.
    """
    import re

    script = ERRORS_DIR / "eltwise_add_deadlock.py"
    deadlock_line_mark = "with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:"
    deadlock_lines = script.read_text().splitlines()
    deadlock_line_num = next(
        i
        for i, line in enumerate(deadlock_lines, start=1)
        if deadlock_line_mark in line
    )

    code, out = run_script_in_process(script)

    assert code != 0, f"Expected {script.name} to fail, but it exited with code 0"

    assert (
        "Deadlock detected: all generators blocked" in out
    ), f"Expected deadlock message:\n{out}"
    assert (
        "DataflowBuffer(a_dfb)" in out
    ), f"Expected to see a_dfb in deadlock output:\n{out}"
    assert (
        "blocked on wait()" in out
    ), f"Expected 'blocked on wait()' in deadlock output:\n{out}"

    line_number_pattern = r"-->\s+.*?:(\d+):\d+"
    reported_line_numbers = {int(n) for n in re.findall(line_number_pattern, out)}
    assert reported_line_numbers, f"No source locations found in:\n{out}"
    assert deadlock_line_num in reported_line_numbers, (
        f"Expected line {deadlock_line_num} (read path wait() on a_dfb/b_dfb) in "
        f"reported locations {reported_line_numbers}.\nOutput:\n{out}"
    )

    for line_num in reported_line_numbers:
        assert line_num <= len(deadlock_lines), f"Reported line {line_num} out of range"
        line_content = deadlock_lines[line_num - 1]
        assert "wait()" in line_content or "reserve()" in line_content, (
            f"Line {line_num} does not contain wait() or reserve(): "
            f"{line_content.strip()}"
        )


@pytest.mark.parametrize("scheduler", ["greedy", "fair"])
def test_max_dfbs_warning_warns_at_limit(scheduler: str) -> None:
    """Test that max_dfbs_warning.py emits a DFB limit warning but still succeeds.

    This example allocates 36 DataflowBuffers, exceeding the default limit of 32.
    The warning is issued at kernel definition time before any kernel execution.
    """
    with pytest.warns(UserWarning, match="configured limit is 32"):
        code, out = run_script_in_process(ERRORS_DIR / "max_dfbs_warning.py", scheduler)
    assert (
        code == 0
    ), f"Expected max_dfbs_warning.py to succeed, but it exited with code {code}:\n{out}"


# ---- Matmul tutorial -------------------------------------------------------


@pytest.mark.parametrize(
    "script_name",
    [
        # step_0 is a plain ttnn program; all ttnn surface calls (add, matmul, relu,
        # from_torch, to_torch, open_device) are implemented in the simulator.
        "step_0_ttnn_base.py",
        # steps 2–7 define custom ttl kernels; verification uses torch.relu(a @ b + c).
        # All ttnn surface calls (from_torch, to_torch, open_mesh_device,
        # GetNumAvailableDevices, ShardTensorToMesh, set_fabric_config, all_reduce,
        # relu) are natively implemented in the simulator.
        # step_1 is excluded: single-tile-block granularity produces too many simulator
        # coroutine steps at M=K=N=8192 to be practical.
        "step_2_single_node_multitile_block.py",
        "step_3_multinode.py",
        "step_4_multinode_grid_full.py",
        "step_5_multidevice_shard_m.py",
        "step_6_multidevice_shard_k.py",
        "step_7_multidevice_shard_k_all_reduce.py",
    ],
)
@pytest.mark.matmul_tutorial
def test_matmul_tutorial(script_name: str) -> None:
    """Test matmul-tutorial steps 0 and 2-7 on the simulator (pass --run-matmul-tutorial-dry to enable).

    Runs in dry-run mode: math and data operations are skipped, but all
    structural checks (deadlock detection, DFB state machine, copy-wait
    injection, L1 allocation) are exercised.

    step_1 is excluded because single-tile-block granularity at M=K=N=8192
    produces too many simulator coroutine steps to be practical.
    """
    code, out = run_script_in_process(
        MATMUL_TUTORIAL_DIR / script_name, scheduler="fair", dry_run=True
    )
    assert code == 0, f"Script failed with code {code}. Output:\n{out}"
