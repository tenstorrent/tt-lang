#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# type: ignore
"""
Tests for ttlang_sim.py module (simulator launcher).
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

import pytest

from test_utils import make_zeros_tensor

from python.sim import ttl, ttnn
from python.sim.kernel import get_default_grid, set_default_grid
from python.sim.typedefs import Shape


class TestDefaultGrid:
    """Test default grid configuration for grid='auto'."""

    def test_get_default_grid_initial_value(self):
        """Test that get_default_grid returns initial default of (8, 8)."""
        # Save original to restore later
        original = get_default_grid()
        try:
            # Reset to default
            set_default_grid((8, 8))
            assert get_default_grid() == (8, 8)
        finally:
            set_default_grid(original)

    def test_set_default_grid(self):
        """Test that set_default_grid changes the default grid."""
        # Save original to restore later
        original = get_default_grid()
        try:
            set_default_grid((4, 4))
            assert get_default_grid() == (4, 4)

            set_default_grid((2, 3))
            assert get_default_grid() == (2, 3)
        finally:
            set_default_grid(original)

    def test_kernel_auto_grid_uses_default(self):
        """Test that kernel with grid='auto' uses the configured default grid."""
        original = get_default_grid()
        try:
            # Set custom default
            set_default_grid((3, 5))

            @ttl.kernel(grid="auto")
            def test_kernel(a: ttnn.Tensor, b: ttnn.Tensor):
                assert a is not None and b is not None

                @ttl.compute()
                def compute():
                    grid_h, grid_w = cast(Shape, ttl.grid_size(dims=2))
                    assert grid_h == 3, f"Expected grid_h=3, got {grid_h}"
                    assert grid_w == 5, f"Expected grid_w=5, got {grid_w}"

                @ttl.datamovement()
                def dm0():
                    pass

                @ttl.datamovement()
                def dm1():
                    pass

            # Create dummy tensors
            a = make_zeros_tensor(32, 32)
            b = make_zeros_tensor(32, 32)

            # Should use the custom default grid (3, 5)
            test_kernel(a, b)
        finally:
            set_default_grid(original)

    def test_kernel_explicit_grid_ignores_default(self):
        """Test that kernel with explicit grid ignores the default."""
        original = get_default_grid()
        try:
            # Set custom default
            set_default_grid((4, 4))

            @ttl.kernel(grid=(2, 2))
            def test_kernel(a: ttnn.Tensor, b: ttnn.Tensor):
                assert a is not None and b is not None

                @ttl.compute()
                def compute():
                    grid_h, grid_w = cast(Shape, ttl.grid_size(dims=2))
                    # Should use explicit grid (2, 2), not default (4, 4)
                    assert grid_h == 2, f"Expected grid_h=2, got {grid_h}"
                    assert grid_w == 2, f"Expected grid_w=2, got {grid_w}"

                @ttl.datamovement()
                def dm0():
                    pass

                @ttl.datamovement()
                def dm1():
                    pass

            # Create dummy tensors
            a = make_zeros_tensor(32, 32)
            b = make_zeros_tensor(32, 32)

            test_kernel(a, b)
        finally:
            set_default_grid(original)


class TestGridCommandLineOption:
    """Test --grid command-line option in ttlang-sim."""

    @staticmethod
    def create_test_script(grid_check: tuple[int, int]) -> Path:
        """Create a temporary test script that checks grid size."""
        content = f"""
import ttl
import ttnn
import torch

@ttl.kernel(grid='auto')
def test_kernel(a: ttnn.Tensor):
    @ttl.compute()
    def compute():
        grid_h, grid_w = ttl.grid_size(dims=2)
        if grid_h != {grid_check[0]} or grid_w != {grid_check[1]}:
            raise ValueError(f"Expected grid {{({grid_check[0]}, {grid_check[1]})}}, got {{(grid_h, grid_w)}}")

    @ttl.datamovement()
    def dm0():
        pass

    @ttl.datamovement()
    def dm1():
        pass

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    a = torch.zeros(32, 32)
    a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    test_kernel(a_tt)
    ttnn.close_device(device)
    print("SUCCESS")
"""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_grid_option_custom_grid(self):
        """Test that --grid option sets custom grid."""
        script = self.create_test_script((4, 6))
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--grid",
                    "4,6",
                    str(script),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            script.unlink()

    def test_grid_option_default_grid(self):
        """Test that default grid is (8, 8) when --grid not specified."""
        script = self.create_test_script((8, 8))
        try:
            result = subprocess.run(
                [sys.executable, "-m", "sim.ttlang_sim", str(script)],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            script.unlink()

    def test_grid_option_invalid_format(self):
        """Test that invalid --grid format produces error."""
        script = self.create_test_script((8, 8))
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--grid",
                    "invalid",
                    str(script),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            assert "Invalid grid specification" in result.stderr
            assert "Grid must be specified as ROWS,COLS" in result.stderr
        finally:
            script.unlink()

    def test_grid_option_zero_dimension(self):
        """Test that zero grid dimension produces error."""
        script = self.create_test_script((8, 8))
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--grid",
                    "0,4",
                    str(script),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            assert "Invalid grid specification" in result.stderr
            assert "Grid dimensions must be positive" in result.stderr
        finally:
            script.unlink()

    def test_grid_option_non_numeric(self):
        """Test that non-numeric grid values produce error."""
        script = self.create_test_script((8, 8))
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--grid",
                    "a,b",
                    str(script),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            assert "Invalid grid specification" in result.stderr
        finally:
            script.unlink()

    def test_grid_option_single_value(self):
        """Test that single value for grid produces error."""
        script = self.create_test_script((8, 8))
        try:
            result = subprocess.run(
                [sys.executable, "-m", "sim.ttlang_sim", "--grid", "4", str(script)],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            assert "Invalid grid specification" in result.stderr
            assert "Grid must be specified as ROWS,COLS" in result.stderr
        finally:
            script.unlink()

    def test_grid_option_with_spaces(self):
        """Test that --grid with spaces around values works."""
        script = self.create_test_script((5, 7))
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--grid",
                    " 5 , 7 ",
                    str(script),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            script.unlink()


class TestMaxDfbsCommandLineOption:
    """Test --max-dfbs command-line option in ttlang-sim."""

    @staticmethod
    def create_test_script(num_cbs: int) -> Path:
        """Create a temporary test script that uses a specific number of CBs."""
        # Generate CB declarations
        cb_declarations = "\n    ".join(
            f"cb{i} = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)"
            for i in range(num_cbs)
        )

        # Use cb0 for input, cb1 for middle, and last CB for output
        last_cb = f"cb{num_cbs - 1}"
        middle_cb = "cb1" if num_cbs > 1 else "cb0"

        content = f"""
import ttl
import ttnn
import torch

@ttl.kernel(grid=(1, 1))
def test_kernel(a: ttnn.Tensor):
    {cb_declarations}

    @ttl.compute()
    def compute():
        with cb0.reserve() as blk:
            blk.store(ttl.math.fill(blk, 1.0))
        with cb0.wait() as a, {middle_cb}.reserve() as o:
            o.store(a)
        with {middle_cb}.wait() as a, {last_cb}.reserve() as o:
            o.store(a)

    @ttl.datamovement()
    def dm0():
        pass

    @ttl.datamovement()
    def dm1():
        pass

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    a = torch.zeros(32, 32)
    a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    test_kernel(a_tt)
    ttnn.close_device(device)
    print("SUCCESS")
"""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_max_dfbs_option_below_limit(self):
        """Test that --max-dfbs below actual usage emits a warning but still succeeds."""
        script = self.create_test_script(3)  # Script uses 3 CBs
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--max-dfbs",
                    "2",
                    str(script),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert (
                result.returncode == 0
            ), f"Expected success with --max-dfbs 2, got stderr: {result.stderr}"
            assert (
                "hardware limit is 2" in result.stderr
            ), f"Expected DFB limit warning in stderr, got: {result.stderr}"
        finally:
            script.unlink()

    def test_max_dfbs_option_at_limit(self):
        """Test that --max-dfbs at exact usage succeeds."""
        script = self.create_test_script(3)  # Script uses 3 CBs
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--max-dfbs",
                    "3",
                    str(script),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert (
                result.returncode == 0
            ), f"Expected success with --max-dfbs 3, got stderr: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            script.unlink()

    def test_max_dfbs_option_above_limit(self):
        """Test that --max-dfbs above actual usage succeeds."""
        script = self.create_test_script(3)  # Script uses 3 CBs
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--max-dfbs",
                    "10",
                    str(script),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert (
                result.returncode == 0
            ), f"Expected success with --max-dfbs 10, got stderr: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            script.unlink()

    def test_max_dfbs_option_default(self):
        """Test that default max_dfbs (32) works."""
        script = self.create_test_script(3)  # Script uses 3 CBs
        try:
            result = subprocess.run(
                [sys.executable, "-m", "sim.ttlang_sim", str(script)],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert (
                result.returncode == 0
            ), f"Expected success with default limit, got stderr: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            script.unlink()

    def test_max_dfbs_option_negative(self):
        """Test that negative --max-dfbs produces error."""
        script = self.create_test_script(3)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--max-dfbs",
                    "-5",
                    str(script),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0, "Expected failure with negative --max-dfbs"
            assert (
                "must be non-negative" in result.stderr
            ), f"Expected validation error in stderr, got: {result.stderr}"
        finally:
            script.unlink()

    def test_max_dfbs_option_zero(self):
        """Test that --max-dfbs 0 emits a warning but still succeeds when CBs are used."""
        script = self.create_test_script(1)  # Script uses 1 CB
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--max-dfbs",
                    "0",
                    str(script),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert (
                result.returncode == 0
            ), f"Expected success with --max-dfbs 0, got stderr: {result.stderr}"
            assert "hardware limit is 0" in result.stderr
        finally:
            script.unlink()


class TestMaxL1CommandLineOption:
    """Test --max-l1 command-line option in ttlang-sim.

    Each CB uses shape=(1,1), buffer_factor=2, bfloat16:
      capacity_bytes = 2 (slots) * 32*32 (elements/slot) * 2 (bytes/element) = 4096
    Three CBs total: 3 * 4096 = 12288 bytes.

    Exceeding the limit issues a warning but does not abort execution.
    """

    # Bytes used by the three CBs in create_test_script(3).
    _TOTAL_BYTES = 12288

    @staticmethod
    def create_test_script() -> Path:
        """Create a temporary test script that uses 3 CBs of known size."""
        content = """
import ttl
import ttnn
import torch

@ttl.kernel(grid=(1, 1))
def test_kernel(a: ttnn.Tensor):
    cb0 = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    cb1 = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    cb2 = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with cb0.reserve() as blk:
            blk.store(ttl.math.fill(blk, 1.0))
        with cb0.wait() as inp, cb1.reserve() as o:
            o.store(inp)
        with cb1.wait() as inp, cb2.reserve() as o:
            o.store(inp)

    @ttl.datamovement()
    def dm0():
        pass

    @ttl.datamovement()
    def dm1():
        pass

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    a = torch.zeros(32, 32)
    a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    test_kernel(a_tt)
    ttnn.close_device(device)
    print("SUCCESS")
"""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def _run(self, *extra_args: str) -> subprocess.CompletedProcess[str]:
        script = self.create_test_script()
        try:
            return subprocess.run(
                [sys.executable, "-m", "sim.ttlang_sim", *extra_args, str(script)],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
        finally:
            script.unlink()

    def test_max_l1_below_limit_warns_and_continues(self):
        """--max-l1 one byte below total CB capacity should warn but still succeed."""
        result = self._run("--max-l1", str(self._TOTAL_BYTES - 1))
        assert (
            result.returncode == 0
        ), f"Expected success (warning only), got stderr: {result.stderr}"
        assert "SUCCESS" in result.stdout
        assert (
            "exceeds the L1 memory limit" in result.stderr
        ), f"Expected L1 warning in stderr, got: {result.stderr}"

    def test_max_l1_at_limit_succeeds(self):
        """--max-l1 equal to total CB capacity should succeed without warning."""
        result = self._run("--max-l1", str(self._TOTAL_BYTES))
        assert (
            result.returncode == 0
        ), f"Expected success at exact limit, got stderr: {result.stderr}"
        assert "SUCCESS" in result.stdout
        assert "exceeds the L1 memory limit" not in result.stderr

    def test_max_l1_above_limit_succeeds(self):
        """--max-l1 above total CB capacity should succeed without warning."""
        result = self._run("--max-l1", str(self._TOTAL_BYTES + 1))
        assert (
            result.returncode == 0
        ), f"Expected success above limit, got stderr: {result.stderr}"
        assert "SUCCESS" in result.stdout
        assert "exceeds the L1 memory limit" not in result.stderr

    def test_max_l1_zero_is_rejected(self):
        """--max-l1 0 should be rejected as invalid."""
        result = self._run("--max-l1", "0")
        assert result.returncode != 0, "Expected failure with --max-l1 0"
        assert (
            "must be positive" in result.stderr
        ), f"Expected validation error in stderr, got: {result.stderr}"

    def test_max_l1_no_flag_uses_default_limit(self):
        """Without --max-l1, the default limit (1336 KiB) applies; small CBs should not warn."""
        result = self._run()
        assert (
            result.returncode == 0
        ), f"Expected success under default L1 limit, got stderr: {result.stderr}"
        assert "SUCCESS" in result.stdout
        assert "exceeds the L1 memory limit" not in result.stderr


class TestTensorStatsOption:
    """Test --show-stats command-line option."""

    def test_tensor_stats_flag_basic(self):
        """Test that --show-stats prints statistics."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sim.ttlang_sim",
                "--show-stats",
                "examples/single_node_matmul.py",
            ],
            cwd=Path(__file__).parent.parent.parent,
            env={"PYTHONPATH": "python"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Tensor Access Statistics" in result.stdout
        assert "Tensor" in result.stdout
        assert "Reads" in result.stdout
        assert "Writes" in result.stdout
        assert "TOTAL" in result.stdout

    def test_tensor_stats_shows_tensor_names(self):
        """Test that tensor statistics show parameter names."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sim.ttlang_sim",
                "--show-stats",
                "examples/single_node_matmul.py",
            ],
            cwd=Path(__file__).parent.parent.parent,
            env={"PYTHONPATH": "python"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        # Check that statistics contain tensor names
        assert "a" in result.stdout
        assert "b" in result.stdout
        assert "out" in result.stdout

    def test_tensor_stats_without_flag(self):
        """Test that statistics are not printed without --show-stats flag."""
        result = subprocess.run(
            [sys.executable, "-m", "sim.ttlang_sim", "examples/single_node_matmul.py"],
            cwd=Path(__file__).parent.parent.parent,
            env={"PYTHONPATH": "python"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Tensor Access Statistics" not in result.stdout
        assert "TOTAL" not in result.stdout

    def test_tensor_stats_no_data(self):
        """Test that --show-stats handles programs with no tensor operations gracefully."""
        script = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        script.write(
            """
# Simple program with no kernel calls
print("No kernels here!")
"""
        )
        script.close()
        script_path = Path(script.name)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sim.ttlang_sim",
                    "--show-stats",
                    str(script_path),
                ],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "No statistics collected" in result.stdout
        finally:
            script_path.unlink()


class TestSchedulerAlgorithmOption:
    """Test --scheduler command-line option."""

    def test_schedalg_option_greedy(self):
        """Test that --scheduler greedy is accepted."""
        examples_dir = Path(__file__).parent.parent.parent / "examples"
        script_path = examples_dir / "broadcast_demo.py"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sim.ttlang_sim",
                str(script_path),
                "--scheduler",
                "greedy",
            ],
            cwd=Path(__file__).parent.parent.parent,
            env={"PYTHONPATH": "python"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

    def test_schedalg_option_fair(self):
        """Test that --scheduler fair is accepted."""
        examples_dir = Path(__file__).parent.parent.parent / "examples"
        script_path = examples_dir / "broadcast_demo.py"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sim.ttlang_sim",
                str(script_path),
                "--scheduler",
                "fair",
            ],
            cwd=Path(__file__).parent.parent.parent,
            env={"PYTHONPATH": "python"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

    def test_schedalg_option_invalid(self):
        """Test that invalid --scheduler value is rejected."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sim.ttlang_sim",
                "--scheduler",
                "invalid",
                "nonexistent.py",
            ],
            cwd=Path(__file__).parent.parent.parent,
            env={"PYTHONPATH": "python"},
            capture_output=True,
            text=True,
        )
        # Should fail due to invalid choice
        assert result.returncode != 0
        assert "invalid choice" in result.stderr

    def test_schedalg_default_is_fair(self):
        """Test that default scheduling algorithm is fair when not specified."""
        # Create a temporary script that checks the algorithm
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            script_path = Path(f.name)
            f.write(
                """
from python.sim.greenlet_scheduler import get_scheduler_algorithm
print(f"Algorithm: {get_scheduler_algorithm()}")
"""
            )

        try:
            result = subprocess.run(
                [sys.executable, "-m", "sim.ttlang_sim", str(script_path)],
                cwd=Path(__file__).parent.parent.parent,
                env={"PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "Algorithm: fair" in result.stdout
        finally:
            script_path.unlink()


class TestSignpost:
    """Test ttl.signpost stub for simulator."""

    def test_signpost_does_nothing(self):
        """Test that ttl.signpost accepts arguments and does nothing."""
        # Should not raise any exceptions
        ttl.signpost("test_message")
        ttl.signpost("test", "with", "multiple", "args")
        ttl.signpost(label="test", value=42)
        ttl.signpost("mixed", args=True, keyword=False)

    def test_signpost_as_context_manager(self):
        """Test that ttl.signpost works as a context manager."""
        # Should not raise any exceptions
        with ttl.signpost("test_context"):
            pass

        with ttl.signpost("context", "with", "args"):
            # Do some work inside
            x = 1 + 1
            assert x == 2

        with ttl.signpost(label="test", phase="start"):
            pass

    def test_signpost_context_manager_returns_value(self):
        """Test that signpost context manager returns itself."""
        with ttl.signpost("test") as ctx:
            assert ctx is not None

    def test_signpost_context_manager_with_exception(self):
        """Test that signpost context manager doesn't suppress exceptions."""
        try:
            with ttl.signpost("test"):
                raise ValueError("test exception")
        except ValueError as e:
            assert str(e) == "test exception"
        else:
            pytest.fail("Exception should have been raised")

    def test_signpost_in_kernel(self):
        """Test that ttl.signpost can be called inside kernel code."""

        @ttl.kernel(grid=(1, 1))
        def test_kernel(a: ttnn.Tensor):
            assert a is not None

            @ttl.compute()
            def compute():
                with ttl.signpost("compute_start"):
                    # Do nothing
                    pass
                ttl.signpost("compute_end")

            @ttl.datamovement()
            def dm0():
                with ttl.signpost("dm0_start", phase="datamovement"):
                    x = 1 + 1

            @ttl.datamovement()
            def dm1():
                pass

        # Create dummy tensor
        a = make_zeros_tensor(32, 32)

        # Should not raise - signpost is a no-op
        test_kernel(a)

    def test_signpost_returns_context_manager(self):
        """Test that ttl.signpost returns a context manager."""
        result = ttl.signpost("test")
        assert result is not None
        # Should have __enter__ and __exit__ methods
        assert hasattr(result, "__enter__")
        assert hasattr(result, "__exit__")
