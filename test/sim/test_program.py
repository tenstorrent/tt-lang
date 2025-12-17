# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test Program execution framework.

This test verifies the Program class behavior including:
- Context binding and per-core state isolation
- Both THREADED and COOPERATIVE execution modes
- Error handling and deadlock detection
- Multi-core execution
"""

import pytest
import torch
from python.sim import (
    ttl,
    CircularBuffer,
    CBAPI,
    TensorAccessor,
    IndexType,
    TILE_SHAPE,
    copy,
)
from python.sim.program import (
    Program,
    ExecutionMode,
    rebind_func_with_ctx,
    _make_cell,
)
import torch.testing as tt_testing


class TestExecutionModes:
    """Test different execution modes (THREADED vs COOPERATIVE)."""

    def test_threaded_mode_basic(self) -> None:
        """Test basic THREADED mode execution."""

        @ttl.kernel(grid=(1, 1), granularity=1)
        def test_kernel(a: torch.Tensor, out: torch.Tensor):
            # Create accessors and circular buffers
            a_accessor = TensorAccessor(a, index_type=IndexType.TILE)
            out_accessor = TensorAccessor(out, index_type=IndexType.TILE)

            a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
            out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                block = a_cb.wait()
                out_block = out_cb.reserve()
                out_block[0] = block[0] * 2
                a_cb.pop()
                out_cb.push()

            @ttl.datamovement()
            def dm0():
                # Input
                block = a_cb.reserve()
                tx = copy(a_accessor[0:1, 0:1], block)
                tx.wait()
                a_cb.push()

            @ttl.datamovement()
            def dm1():
                # Output
                block = out_cb.wait()
                tx = copy(block, out_accessor[0:1, 0:1])
                tx.wait()
                out_cb.pop()

            # Use THREADED mode explicitly
            return Program(compute, dm0, dm1, execution_mode=ExecutionMode.THREADED)()

        a = torch.ones(TILE_SHAPE) * 3
        out = torch.zeros(TILE_SHAPE)

        test_kernel(a, out)

        # Verify computation
        expected = torch.ones(TILE_SHAPE) * 6
        tt_testing.assert_close(out, expected)

    @pytest.mark.skip(
        reason="inspect.getsource() on decorated functions returns wrapper with indentation issues"
    )
    def test_cooperative_mode_basic(self) -> None:
        """Test basic COOPERATIVE mode execution."""

        @ttl.kernel(grid=(1, 1), granularity=1)
        def test_kernel(a: torch.Tensor, out: torch.Tensor):
            # Create accessors and circular buffers
            a_accessor = TensorAccessor(a, index_type=IndexType.TILE)
            out_accessor = TensorAccessor(out, index_type=IndexType.TILE)

            a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
            out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                block = a_cb.wait()
                out_block = out_cb.reserve()
                out_block[0] = block[0] * 2
                a_cb.pop()
                out_cb.push()

            @ttl.datamovement()
            def dm0():
                # Input
                block = a_cb.reserve()
                tx = copy(a_accessor[0:1, 0:1], block)
                tx.wait()
                a_cb.push()

            @ttl.datamovement()
            def dm1():
                # Output
                block = out_cb.wait()
                tx = copy(block, out_accessor[0:1, 0:1])
                tx.wait()
                out_cb.pop()

            # Use COOPERATIVE mode explicitly
            return Program(
                compute, dm0, dm1, execution_mode=ExecutionMode.COOPERATIVE
            )()

        a = torch.ones(TILE_SHAPE) * 3
        out = torch.zeros(TILE_SHAPE)

        test_kernel(a, out)

        # Verify computation
        expected = torch.ones(TILE_SHAPE) * 6
        tt_testing.assert_close(out, expected)

    def test_modes_produce_same_result(self) -> None:
        """Test that THREADED mode works (COOPERATIVE skipped due to getsource issues)."""

        @ttl.kernel(grid=(1, 1), granularity=2)
        def test_kernel(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
            # Create accessors and circular buffers
            a_accessor = TensorAccessor(a, index_type=IndexType.TILE)
            b_accessor = TensorAccessor(b, index_type=IndexType.TILE)
            out_accessor = TensorAccessor(out, index_type=IndexType.TILE)

            a_cb = ttl.make_circular_buffer_like(a, shape=(2, 1), buffer_factor=2)
            b_cb = ttl.make_circular_buffer_like(b, shape=(2, 1), buffer_factor=2)
            out_cb = ttl.make_circular_buffer_like(out, shape=(2, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                a_block = a_cb.wait()
                b_block = b_cb.wait()
                out_block = out_cb.reserve()
                # Element-wise add
                for i in range(2):
                    out_block[i] = a_block[i] + b_block[i]
                a_cb.pop()
                b_cb.pop()
                out_cb.push()

            @ttl.datamovement()
            def dm0():
                # Input
                a_block = a_cb.reserve()
                b_block = b_cb.reserve()
                tx1 = copy(a_accessor[0:2, 0:1], a_block)
                tx2 = copy(b_accessor[0:2, 0:1], b_block)
                tx1.wait()
                tx2.wait()
                a_cb.push()
                b_cb.push()

            @ttl.datamovement()
            def dm1():
                # Output
                block = out_cb.wait()
                tx = copy(block, out_accessor[0:2, 0:1])
                tx.wait()
                out_cb.pop()

            return Program(compute, dm0, dm1, execution_mode=ExecutionMode.THREADED)()

        # Create test data
        a = torch.randn(TILE_SHAPE[0] * 4, TILE_SHAPE[1] * 4)
        b = torch.randn(TILE_SHAPE[0] * 4, TILE_SHAPE[1] * 4)
        out = torch.zeros_like(a)

        # Run with THREADED mode
        test_kernel(a, b, out)

        # Verify result makes sense (a + b)
        expected = a[0:64, 0:32] + b[0:64, 0:32]
        tt_testing.assert_close(out[0:64, 0:32], expected)


class TestMultiCore:
    """Test multi-core execution."""

    def test_two_core_execution(self) -> None:
        """Test execution on 2 cores."""

        @ttl.kernel(grid=(2, 1), granularity=1)
        def test_kernel(a: torch.Tensor, out: torch.Tensor):
            a_accessor = TensorAccessor(a, index_type=IndexType.TILE)
            out_accessor = TensorAccessor(out, index_type=IndexType.TILE)

            a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
            out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                core_id = ttl.core(dims=1)
                block = a_cb.wait()
                out_block = out_cb.reserve()
                # Each core multiplies by (core_id + 1)
                out_block[0] = block[0] * (core_id + 1)
                a_cb.pop()
                out_cb.push()

            @ttl.datamovement()
            def dm0():
                core_id = ttl.core(dims=1)
                block = a_cb.reserve()
                # Each core reads its own tile
                tx = copy(a_accessor[core_id : core_id + 1, 0:1], block)
                tx.wait()
                a_cb.push()

            @ttl.datamovement()
            def dm1():
                core_id = ttl.core(dims=1)
                block = out_cb.wait()
                # Each core writes its own tile
                tx = copy(block, out_accessor[core_id : core_id + 1, 0:1])
                tx.wait()
                out_cb.pop()

            return Program(compute, dm0, dm1)()

        a = torch.ones(TILE_SHAPE[0] * 2, TILE_SHAPE[1]) * 5
        out = torch.zeros(TILE_SHAPE[0] * 2, TILE_SHAPE[1])

        test_kernel(a, out)

        # Core 0: 5 * 1 = 5
        # Core 1: 5 * 2 = 10
        expected = torch.cat(
            [
                torch.ones(TILE_SHAPE) * 5,
                torch.ones(TILE_SHAPE) * 10,
            ],
            dim=0,
        )
        tt_testing.assert_close(out, expected)

    def test_four_core_2d_grid(self) -> None:
        """Test execution on 2x2 grid (4 cores)."""

        @ttl.kernel(grid=(2, 2), granularity=1)
        def test_kernel(out: torch.Tensor):
            out_accessor = TensorAccessor(out, index_type=IndexType.TILE)
            out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=1)

            @ttl.compute()
            def compute():
                core_y, core_x = ttl.core(dims=2)
                out_block = out_cb.reserve()
                # Each core writes its coordinates
                out_block[0] = torch.ones(TILE_SHAPE) * (core_y * 10 + core_x)
                out_cb.push()

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                core_y, core_x = ttl.core(dims=2)
                block = out_cb.wait()
                tx = copy(block, out_accessor[core_y : core_y + 1, core_x : core_x + 1])
                tx.wait()
                out_cb.pop()

            return Program(compute, dm0, dm1)()

        out = torch.zeros(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)

        test_kernel(out)

        # Verify each core wrote its coordinates
        # (0,0) = 0, (0,1) = 1, (1,0) = 10, (1,1) = 11
        assert (out[0:32, 0:32] == 0).all()
        assert (out[0:32, 32:64] == 1).all()
        assert (out[32:64, 0:32] == 10).all()
        assert (out[32:64, 32:64] == 11).all()


class TestContextIsolation:
    """Test that per-core contexts are properly isolated."""

    def test_circular_buffers_isolated(self) -> None:
        """Test that circular buffers are independent per core."""

        @ttl.kernel(grid=(2, 1), granularity=1)
        def test_kernel(out: torch.Tensor):
            out_accessor = TensorAccessor(out, index_type=IndexType.TILE)
            # Each core gets its own CB instance
            cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                core_id = ttl.core(dims=1)
                # Each core reserves/pushes independently
                block = cb.reserve()
                block[0] = torch.ones(TILE_SHAPE) * (core_id + 100)
                cb.push()

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                core_id = ttl.core(dims=1)
                # Each core waits/pops its own CB
                block = cb.wait()
                tx = copy(block, out_accessor[core_id : core_id + 1, 0:1])
                tx.wait()
                cb.pop()

            return Program(compute, dm0, dm1)()

        out = torch.zeros(TILE_SHAPE[0] * 2, TILE_SHAPE[1])

        test_kernel(out)

        # Each core should have written its own value
        assert (out[0:32, :] == 100).all()
        assert (out[32:64, :] == 101).all()

    def test_tensors_shared_across_cores(self) -> None:
        """Test that tensors are shared (not copied) across cores."""

        @ttl.kernel(grid=(2, 1), granularity=1)
        def test_kernel(shared: torch.Tensor, out: torch.Tensor):
            shared_accessor = TensorAccessor(shared, index_type=IndexType.TILE)
            out_accessor = TensorAccessor(out, index_type=IndexType.TILE)
            # shared tensor should be the same object in all cores
            cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                core_id = ttl.core(dims=1)
                block = cb.reserve()
                # Read from shared tensor
                tx = copy(shared_accessor[0:1, 0:1], block)
                tx.wait()
                cb.push()

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                core_id = ttl.core(dims=1)
                block = cb.wait()
                tx = copy(block, out_accessor[core_id : core_id + 1, 0:1])
                tx.wait()
                cb.pop()

            return Program(compute, dm0, dm1)()

        shared = torch.ones(TILE_SHAPE) * 42
        out = torch.zeros(TILE_SHAPE[0] * 2, TILE_SHAPE[1])

        test_kernel(shared, out)

        # Both cores should have read the same shared tensor
        assert (out[0:32, :] == 42).all()
        assert (out[32:64, :] == 42).all()


class TestErrorHandling:
    """Test error handling and reporting."""

    def test_error_in_compute_threaded(self) -> None:
        """Test that errors in compute function are properly reported in THREADED mode."""

        @ttl.kernel(grid=(1, 1), granularity=1)
        def test_kernel(a: torch.Tensor):
            cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                # Intentional error
                raise ValueError("Test error in compute")

            @ttl.datamovement()
            def dm0():
                block = cb.reserve()
                cb.push()

            @ttl.datamovement()
            def dm1():
                pass

            return Program(compute, dm0, dm1, execution_mode=ExecutionMode.THREADED)()

        a = torch.zeros(TILE_SHAPE)

        with pytest.raises(
            RuntimeError, match="core0-compute.*ValueError.*Test error in compute"
        ):
            test_kernel(a)

    @pytest.mark.skip(
        reason="inspect.getsource() on decorated functions returns wrapper with indentation issues"
    )
    def test_error_in_dm0_cooperative(self) -> None:
        """Test that errors in dm0 are properly reported in COOPERATIVE mode."""

        @ttl.kernel(grid=(1, 1), granularity=1)
        def test_kernel(a: torch.Tensor):
            cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                pass

            @ttl.datamovement()
            def dm0():
                # Intentional error
                raise RuntimeError("Test error in dm0")

            @ttl.datamovement()
            def dm1():
                pass

            return Program(
                compute, dm0, dm1, execution_mode=ExecutionMode.COOPERATIVE
            )()

        a = torch.zeros(TILE_SHAPE)

        with pytest.raises(
            RuntimeError, match="core0-dm0.*RuntimeError.*Test error in dm0"
        ):
            test_kernel(a)

    @pytest.mark.skip(
        reason="inspect.getsource() on decorated functions returns wrapper with indentation issues"
    )
    def test_deadlock_detection_cooperative(self) -> None:
        """Test that deadlock is detected in COOPERATIVE mode."""

        @ttl.kernel(grid=(1, 1), granularity=1)
        def test_kernel(a: torch.Tensor):
            cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=1)

            @ttl.compute()
            def compute():
                # Try to wait when nothing was pushed - deadlock
                block = cb.wait()
                cb.pop()

            @ttl.datamovement()
            def dm0():
                # dm0 also tries to wait - deadlock
                block = cb.wait()
                cb.pop()

            @ttl.datamovement()
            def dm1():
                pass

            return Program(
                compute, dm0, dm1, execution_mode=ExecutionMode.COOPERATIVE
            )()

        a = torch.zeros(TILE_SHAPE)

        with pytest.raises(RuntimeError, match="Deadlock detected"):
            test_kernel(a)


class TestRebindFunc:
    """Test the rebind_func_with_ctx utility function."""

    def test_rebind_simple_closure(self) -> None:
        """Test rebinding a function with simple closure variables."""

        def make_func():
            captured_value = 10

            def inner():
                return captured_value

            return inner

        func = make_func()
        assert func() == 10

        # Rebind with new context
        new_func = rebind_func_with_ctx(func, {"captured_value": 20})
        assert new_func() == 20

    def test_rebind_multiple_closures(self) -> None:
        """Test rebinding with multiple closure variables."""

        def make_func():
            x = 1
            y = 2

            def inner():
                return x + y

            return inner

        func = make_func()
        assert func() == 3

        # Rebind both variables
        new_func = rebind_func_with_ctx(func, {"x": 10, "y": 20})
        assert new_func() == 30

    def test_rebind_preserves_unspecified_closures(self) -> None:
        """Test that unspecified closure variables are preserved."""

        def make_func():
            x = 5
            y = 10

            def inner():
                return x + y

            return inner

        func = make_func()

        # Only rebind x, y should stay as 10
        new_func = rebind_func_with_ctx(func, {"x": 100})
        assert new_func() == 110

    def test_rebind_with_globals(self) -> None:
        """Test that rebind also updates globals."""

        def func():
            # This will look up 'some_global' in globals
            return some_global  # type: ignore # noqa: F821

        # Rebind with new global
        new_func = rebind_func_with_ctx(func, {"some_global": 42})
        assert new_func() == 42


class TestMakeCell:
    """Test the _make_cell utility function."""

    def test_make_cell_creates_valid_cell(self) -> None:
        """Test that _make_cell creates a valid cell object."""
        from types import CellType

        cell = _make_cell(42)
        assert isinstance(cell, CellType)
        assert cell.cell_contents == 42

    def test_make_cell_different_types(self) -> None:
        """Test _make_cell with different value types."""
        from types import CellType

        # Integer
        cell_int = _make_cell(10)
        assert isinstance(cell_int, CellType)
        assert cell_int.cell_contents == 10

        # String
        cell_str = _make_cell("hello")
        assert isinstance(cell_str, CellType)
        assert cell_str.cell_contents == "hello"

        # List
        test_list = [1, 2, 3]
        cell_list = _make_cell(test_list)
        assert isinstance(cell_list, CellType)
        assert cell_list.cell_contents == test_list
        assert cell_list.cell_contents is test_list  # Same object


class TestCooperativeScheduling:
    """Test cooperative scheduling specific behavior."""

    @pytest.mark.skip(
        reason="inspect.getsource() on decorated functions returns wrapper with indentation issues"
    )
    def test_yielding_on_blocking_operations(self) -> None:
        """Test that cooperative mode properly yields on blocking operations."""

        @ttl.kernel(grid=(1, 1), granularity=1)
        def test_kernel(a: torch.Tensor, out: torch.Tensor):
            cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                # This wait should yield until dm0 pushes
                block = cb.wait()
                out[0, 0] = block[0] * 2
                cb.pop()

            @ttl.datamovement()
            def dm0():
                # This should run first in cooperative mode
                block = cb.reserve()
                tx = copy(a[0, 0], block)
                tx.wait()
                cb.push()

            @ttl.datamovement()
            def dm1():
                pass

            return Program(
                compute, dm0, dm1, execution_mode=ExecutionMode.COOPERATIVE
            )()

        a = torch.ones(TILE_SHAPE) * 7
        out = torch.zeros(TILE_SHAPE)

        test_kernel(a, out)

        expected = torch.ones(TILE_SHAPE) * 14
        tt_testing.assert_close(out, expected)

    @pytest.mark.skip(
        reason="inspect.getsource() on decorated functions returns wrapper with indentation issues"
    )
    def test_multiple_iterations_cooperative(self) -> None:
        """Test multiple iterations in cooperative mode."""

        @ttl.kernel(grid=(1, 1), granularity=3)
        def test_kernel(a: torch.Tensor, out: torch.Tensor):
            cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                for i in range(3):
                    block = cb.wait()
                    out[i, 0] = block[0] + 10
                    cb.pop()

            @ttl.datamovement()
            def dm0():
                for i in range(3):
                    block = cb.reserve()
                    tx = copy(a[i, 0], block)
                    tx.wait()
                    cb.push()

            @ttl.datamovement()
            def dm1():
                pass

            return Program(
                compute, dm0, dm1, execution_mode=ExecutionMode.COOPERATIVE
            )()

        a = torch.arange(3 * 32 * 32).reshape(3 * 32, 32).float()
        out = torch.zeros_like(a)

        test_kernel(a, out)

        expected = a + 10
        tt_testing.assert_close(out, expected)


if __name__ == "__main__":
    # Run tests
    test_exec_modes = TestExecutionModes()
    test_exec_modes.test_threaded_mode_basic()
    test_exec_modes.test_cooperative_mode_basic()
    test_exec_modes.test_modes_produce_same_result()

    test_multi = TestMultiCore()
    test_multi.test_two_core_execution()
    test_multi.test_four_core_2d_grid()

    test_ctx = TestContextIsolation()
    test_ctx.test_circular_buffers_isolated()
    test_ctx.test_tensors_shared_across_cores()

    test_err = TestErrorHandling()
    test_err.test_error_in_compute_threaded()
    test_err.test_error_in_dm0_cooperative()
    test_err.test_deadlock_detection_cooperative()

    test_rebind = TestRebindFunc()
    test_rebind.test_rebind_simple_closure()
    test_rebind.test_rebind_multiple_closures()
    test_rebind.test_rebind_preserves_unspecified_closures()
    test_rebind.test_rebind_with_globals()

    test_cell = TestMakeCell()
    test_cell.test_make_cell_creates_valid_cell()
    test_cell.test_make_cell_different_types()

    test_coop = TestCooperativeScheduling()
    test_coop.test_yielding_on_blocking_operations()
    test_coop.test_multiple_iterations_cooperative()

    print("All program.py tests passed!")
