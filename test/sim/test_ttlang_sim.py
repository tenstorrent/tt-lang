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
                "examples/singlecore_matmul.py",
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
                "examples/singlecore_matmul.py",
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
            [sys.executable, "-m", "sim.ttlang_sim", "examples/singlecore_matmul.py"],
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
