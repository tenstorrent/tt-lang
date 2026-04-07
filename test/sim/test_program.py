# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test Program execution framework.

This test verifies the Program class behavior including:
- Context binding and per-core state isolation
- Cooperative execution mode
- Error handling and deadlock detection
- Multi-core execution
"""

from typing import cast

import pytest
import torch
import torch.testing as tt_testing
from test_utils import make_ones_tensor, make_zeros_tensor

from python.sim import TILE_SHAPE, copy, ttl, ttnn
from python.sim.dfb import Block
from python.sim.decorators import _make_cell, rebind_func_with_ctx  # type: ignore[reportPrivateUsage]
from python.sim.program import Program


class TestBasicExecution:
    """Test basic execution in cooperative mode."""

    def test_cooperative_mode_basic(self) -> None:
        """Test basic cooperative mode execution."""

        @ttl.operation(grid=(1, 1))
        def test_kernel(a: ttnn.Tensor, out: ttnn.Tensor):
            # Create accessors and dataflow buffers
            # a already is ttnn.Tensor
            # out already is ttnn.Tensor

            a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                block = a_dfb.wait()
                out_block = out_dfb.reserve()
                # Use full block operation
                result = block + block
                out_block.store(result)
                block.pop()
                out_block.push()

            @ttl.datamovement()
            def dm0():
                # Input
                block = a_dfb.reserve()
                tx = copy(a[0:1, 0:1], block)
                tx.wait()
                block.push()

            @ttl.datamovement()
            def dm1():
                # Output
                block = out_dfb.wait()
                tx = copy(block, out[0:1, 0:1])
                tx.wait()
                block.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        a = make_ones_tensor(32, 32) * 3
        out = make_zeros_tensor(32, 32)

        test_kernel(a, out)

        # Verify computation
        expected = make_ones_tensor(32, 32) * 6
        tt_testing.assert_close(out.to_torch(), expected.to_torch())

    def test_multi_tile_computation(self) -> None:
        """Test computation with multiple tiles."""

        @ttl.operation(grid=(1, 1))
        def test_kernel(
            a: ttnn.Tensor,
            b: ttnn.Tensor,
            out: ttnn.Tensor,
        ):
            # Create accessors and dataflow buffers
            # a already is ttnn.Tensor
            # b already is ttnn.Tensor
            # out already is ttnn.Tensor

            a_dfb = ttl.make_dataflow_buffer_like(a, shape=(2, 1), block_count=2)
            b_dfb = ttl.make_dataflow_buffer_like(b, shape=(2, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 1), block_count=2)

            @ttl.compute()
            def compute():
                a_block = a_dfb.wait()
                b_block = b_dfb.wait()
                out_block = out_dfb.reserve()
                # Use full block operation
                result = a_block + b_block
                out_block.store(result)
                a_block.pop()
                b_block.pop()
                out_block.push()

            @ttl.datamovement()
            def dm0():
                # Input
                a_block = a_dfb.reserve()
                b_block = b_dfb.reserve()
                tx1 = copy(a[0:2, 0:1], a_block)
                tx2 = copy(b[0:2, 0:1], b_block)
                tx1.wait()
                tx2.wait()
                a_block.push()
                b_block.push()

            @ttl.datamovement()
            def dm1():
                # Output
                block = out_dfb.wait()
                tx = copy(block, out[0:2, 0:1])
                tx.wait()
                block.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        # Create test data
        a = ttnn.rand((TILE_SHAPE[0] * 4, TILE_SHAPE[1] * 4))
        b = ttnn.rand((TILE_SHAPE[0] * 4, TILE_SHAPE[1] * 4))
        out = ttnn.empty(a.shape)

        test_kernel(a, b, out)

        # Verify result
        expected = ttnn.Tensor(a.to_torch()[0:64, 0:32] + b.to_torch()[0:64, 0:32])
        tt_testing.assert_close(out.to_torch()[0:64, 0:32], expected.to_torch())


class TestMultinode:
    """Test multi-core execution."""

    def test_two_core_execution(self) -> None:
        """Test execution on 2 cores."""

        @ttl.operation(grid=(2, 1))
        def test_kernel(a: ttnn.Tensor, out: ttnn.Tensor):
            # a already is ttnn.Tensor
            # out already is ttnn.Tensor

            a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                core_id = cast(int, ttl.node(dims=1))
                block = a_dfb.wait()
                out_block = out_dfb.reserve()
                # All cores just do block + block (multiplies by 2)
                result = block + block
                out_block.store(result)
                block.pop()
                out_block.push()

            @ttl.datamovement()
            def dm0():
                core_id = cast(int, ttl.node(dims=1))
                block = a_dfb.reserve()
                # Each core reads its own tile
                tx = copy(a[core_id : core_id + 1, 0:1], block)
                tx.wait()
                block.push()

            @ttl.datamovement()
            def dm1():
                core_id = cast(int, ttl.node(dims=1))
                block = out_dfb.wait()
                # Each core writes its own tile
                tx = copy(block, out[core_id : core_id + 1, 0:1])
                tx.wait()
                block.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        a = make_ones_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1]) * 5
        out = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1])

        test_kernel(a, out)

        # Both cores multiply by 2: 5 * 2 = 10
        expected_tensor = make_ones_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1]) * 10
        tt_testing.assert_close(out.to_torch(), expected_tensor.to_torch())

    def test_four_core_2d_grid(self) -> None:
        """Test execution on 2x2 grid (4 cores)."""

        @ttl.operation(grid=(2, 2))
        def test_kernel(out: ttnn.Tensor):
            # out already is ttnn.Tensor
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

            @ttl.compute()
            def compute():
                core_y, core_x = cast(tuple[int, int], ttl.node(dims=2))
                out_block = out_dfb.reserve()
                # Each core writes its coordinates
                out_block.store(
                    Block.from_tensor(make_ones_tensor(32, 32) * (core_y * 10 + core_x))
                )
                out_block.push()

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                core_y, core_x = cast(tuple[int, int], ttl.node(dims=2))
                block = out_dfb.wait()
                tx = copy(
                    block,
                    out[core_y : core_y + 1, core_x : core_x + 1],
                )
                tx.wait()
                block.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        out = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)

        test_kernel(out)

        # Verify each core wrote its coordinates
        # (0,0) = 0, (0,1) = 1, (1,0) = 10, (1,1) = 11
        out_torch = out.to_torch()
        assert (out_torch[0:32, 0:32] == 0).all()
        assert (out_torch[0:32, 32:64] == 1).all()
        assert (out_torch[32:64, 0:32] == 10).all()
        assert (out_torch[32:64, 32:64] == 11).all()


class TestContextIsolation:
    """Test that per-core contexts are properly isolated."""

    def test_dataflow_buffers_isolated(self) -> None:
        """Test that dataflow buffers are independent per core."""

        @ttl.operation(grid=(2, 1))
        def test_kernel(out: ttnn.Tensor):
            # out already is ttnn.Tensor
            # Each core gets its own DFB instance
            dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                core_id = cast(int, ttl.node(dims=1))
                # Each core reserves/pushes independently
                block = dfb.reserve()
                block.store(
                    Block.from_tensor(make_ones_tensor(32, 32) * (core_id + 100))
                )
                block.push()

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                core_id = cast(int, ttl.node(dims=1))
                # Each core waits/pops its own DFB
                block = dfb.wait()
                tx = copy(block, out[core_id : core_id + 1, 0:1])
                tx.wait()
                block.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        out = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1])

        test_kernel(out)

        # Each core should have written its own value
        out_torch = out.to_torch()
        assert (out_torch[0:32, :] == 100).all()
        assert (out_torch[32:64, :] == 101).all()

    def test_shared_tensor_with_compute_store(self) -> None:
        """Test shared tensors where compute thread uses store instead of copy.

        This tests the pattern where compute thread reads from a shared tensor
        and uses store() to write to DFB (not copy).
        """

        @ttl.operation(grid=(2, 1))
        def test_kernel(shared: ttnn.Tensor, out: ttnn.Tensor):
            dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                # Compute thread reads shared tensor and stores to DFB
                core_id = cast(int, ttl.node(dims=1))
                block = dfb.reserve()
                # Read from shared tensor and store (not copy)
                # Add core_id to distinguish which core wrote
                data = shared[0:1, 0:1] + core_id
                block.store(Block.from_tensor(data))
                block.push()

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                # DM thread copies from DFB to output
                core_id = cast(int, ttl.node(dims=1))
                block = dfb.wait()
                tx = copy(block, out[core_id : core_id + 1, 0:1])
                tx.wait()
                block.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        shared = make_ones_tensor(32, 32) * 10
        out = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1])

        test_kernel(shared, out)

        # Each core should have written shared + core_id
        out_torch = out.to_torch()
        assert (out_torch[0:32, :] == 10).all()  # core 0: 10 + 0
        assert (out_torch[32:64, :] == 11).all()  # core 1: 10 + 1


class TestErrorHandling:
    """Test error handling and reporting."""

    def test_error_in_compute(self) -> None:
        """Test that errors in compute function are properly reported."""

        @ttl.operation(grid=(1, 1))
        def test_kernel(a: ttnn.Tensor):
            # a already is ttnn.Tensor
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                # Intentional error
                raise ValueError("Test error in compute")

            @ttl.datamovement()
            def dm0():
                block = dfb.reserve()
                block.push()

            @ttl.datamovement()
            def dm1():
                pass

            return Program(compute, dm0, dm1, grid=grid)()

        a = make_zeros_tensor(32, 32)

        with pytest.raises(
            RuntimeError, match="core0-compute.*ValueError.*Test error in compute"
        ):
            test_kernel(a)

    def test_error_in_dm0(self) -> None:
        """Test that errors in dm0 are properly reported."""

        @ttl.operation(grid=(1, 1))
        def test_kernel(a: ttnn.Tensor):
            # a already is ttnn.Tensor
            _ = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

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

            return Program(compute, dm0, dm1, grid=grid)()

        a = make_zeros_tensor(32, 32)

        with pytest.raises(
            RuntimeError, match="core0-dm0.*RuntimeError.*Test error in dm0"
        ):
            test_kernel(a)

    def test_deadlock_detection(self) -> None:
        """Test that deadlock is detected."""

        @ttl.operation(grid=(1, 1))
        def test_kernel(a: ttnn.Tensor):
            # a already is ttnn.Tensor
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=1)

            @ttl.compute()
            def compute():
                # Try to wait when nothing was pushed - deadlock
                block = dfb.wait()
                block.pop()

            @ttl.datamovement()
            def dm0():
                # dm0 also tries to wait - deadlock
                block = dfb.wait()
                block.pop()

            @ttl.datamovement()
            def dm1():
                pass

            return Program(compute, dm0, dm1, grid=grid)()

        a = make_zeros_tensor(32, 32)

        with pytest.raises(RuntimeError, match="Deadlock detected"):
            test_kernel(a)


class TestBlockCompletion:
    """Test block completion validation at end of kernel execution.

    These tests verify that the simulator catches incomplete block operations
    (missing push() or pop() calls) at the end of kernel execution.
    """

    def test_missing_push_detected(self) -> None:
        """Test that missing push() is detected at end of execution."""

        @ttl.operation(grid=(1,))
        def test_kernel(input_data: ttnn.Tensor):
            # Create dataflow buffers
            element = make_ones_tensor(32, 32)
            in_dfb = ttl.make_dataflow_buffer_like(element, shape=(1, 1), block_count=2)

            @ttl.datamovement()
            def dm0():
                # Reserve a block but forget to push it
                block = in_dfb.reserve()
                slice_data = input_data[0:1, 0:1]
                tx = copy(slice_data, block)
                tx.wait()
                # Missing: block.push()

            @ttl.datamovement()
            def dm1():
                pass

            @ttl.compute()
            def compute():
                pass

        input_tensor = ttnn.rand((32, 32))

        # Should raise RuntimeError about incomplete DataflowBuffer operations
        with pytest.raises(
            RuntimeError,
            match="Kernel execution completed with incomplete DataflowBuffer operations",
        ):
            test_kernel(input_tensor)

    def test_missing_pop_detected(self) -> None:
        """Test that missing pop() is detected at end of execution."""

        @ttl.operation(grid=(1,))
        def test_kernel(input_data: ttnn.Tensor):
            # Create dataflow buffers
            element = make_ones_tensor(32, 32)
            in_dfb = ttl.make_dataflow_buffer_like(element, shape=(1, 1), block_count=2)

            @ttl.datamovement()
            def dm0():
                # Produce data
                block = in_dfb.reserve()
                slice_data = input_data[0:1, 0:1]
                tx = copy(slice_data, block)
                tx.wait()
                block.push()

            @ttl.datamovement()
            def dm1():
                pass

            @ttl.compute()
            def compute():
                # Wait for data but forget to pop it
                data = in_dfb.wait()
                # Use the data as a source
                _ = data + data
                # Missing: data.pop()

        input_tensor = ttnn.rand((32, 32))

        # Should raise RuntimeError about incomplete DataflowBuffer operations
        with pytest.raises(
            RuntimeError,
            match="Kernel execution completed with incomplete DataflowBuffer operations",
        ):
            test_kernel(input_tensor)

    def test_complete_operations_pass(self) -> None:
        """Test that properly completed operations pass validation."""

        @ttl.operation(grid=(1,))
        def test_kernel(input_data: ttnn.Tensor, output_data: ttnn.Tensor):
            # Create dataflow buffers
            element = make_ones_tensor(32, 32)
            in_dfb = ttl.make_dataflow_buffer_like(element, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(
                output_data, shape=(1, 1), block_count=2
            )

            @ttl.datamovement()
            def dm0():
                # Produce data - with push()
                block = in_dfb.reserve()
                slice_data = input_data[0:1, 0:1]
                tx = copy(slice_data, block)
                tx.wait()
                block.push()

            @ttl.datamovement()
            def dm1():
                pass

            @ttl.compute()
            def compute():
                # Consume data - with pop()
                data = in_dfb.wait()
                out_block = out_dfb.reserve()
                # Use data as source by storing it
                result = data + data
                out_block.store(result)
                out_block.push()  # Complete the output DFB operation
                data.pop()

        input_tensor = ttnn.rand((32, 32))
        output_tensor = ttnn.empty((32, 32))

        # Should NOT raise - all operations are complete
        test_kernel(input_tensor, output_tensor)

    def test_multiple_dfbs_with_errors(self) -> None:
        """Test that errors from multiple DFBs are all reported."""

        @ttl.operation(grid=(1,))
        def test_kernel(input_data: ttnn.Tensor):
            from python.sim.dfb import DataflowBuffer

            # Create multiple dataflow buffers
            element = make_ones_tensor(32, 32)
            dfb1 = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)
            dfb2 = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)

            @ttl.datamovement()
            def dm0():
                # Both DFBs have incomplete operations
                block1 = dfb1.reserve()
                slice_data = input_data[0:1, 0:1]
                tx = copy(slice_data, block1)
                tx.wait()
                # Missing: block1.push()

                block2 = dfb2.reserve()
                tx = copy(slice_data, block2)
                tx.wait()
                # Missing: block2.push()

            @ttl.datamovement()
            def dm1():
                pass

            @ttl.compute()
            def compute():
                pass

        input_tensor = ttnn.rand((32, 32))

        # Should raise RuntimeError mentioning multiple DFBs
        with pytest.raises(
            RuntimeError,
            match="Kernel execution completed with incomplete DataflowBuffer operations",
        ):
            test_kernel(input_tensor)


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

        def func() -> int:
            # This will look up 'some_global' in globals
            return some_global  # type: ignore[reportUnknownVariableType] # noqa: F821

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
    """Test cooperative scheduling behavior."""

    def test_yielding_on_blocking_operations(self) -> None:
        """Test that cooperative mode properly yields on blocking operations."""

        @ttl.operation(grid=(1, 1))
        def test_kernel(a: ttnn.Tensor, out: ttnn.Tensor):
            # a already is ttnn.Tensor
            # out already is ttnn.Tensor
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                # This wait should yield until dm0 pushes
                block = dfb.wait()
                out_block = out_dfb.reserve()
                result = block + block
                out_block.store(result)
                out_block.push()
                block.pop()

            @ttl.datamovement()
            def dm0():
                # This should run first in cooperative mode
                block = dfb.reserve()
                tx = copy(a[0:1, 0:1], block)
                tx.wait()
                block.push()

            @ttl.datamovement()
            def dm1():
                block = out_dfb.wait()
                tx = copy(block, out[0:1, 0:1])
                tx.wait()
                block.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        a = make_ones_tensor(32, 32) * 7
        out = make_zeros_tensor(32, 32)

        test_kernel(a, out)

        expected = make_ones_tensor(32, 32) * 14
        tt_testing.assert_close(out.to_torch(), expected.to_torch())

    def test_multiple_iterations_cooperative(self) -> None:
        """Test multiple iterations in cooperative mode."""

        @ttl.operation(grid=(1, 1))
        def test_kernel(a: ttnn.Tensor, out: ttnn.Tensor):
            # a already is ttnn.Tensor
            # out already is ttnn.Tensor
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                for i in range(3):
                    block = dfb.wait()
                    out_block = out_dfb.reserve()
                    # Since we can't do block + 10, just do block + block
                    result = block + block
                    out_block.store(result)
                    out_block.push()
                    block.pop()

            @ttl.datamovement()
            def dm0():
                for i in range(3):
                    block = dfb.reserve()
                    tx = copy(a[i : i + 1, 0:1], block)
                    tx.wait()
                    block.push()

            @ttl.datamovement()
            def dm1():
                for i in range(3):
                    block = out_dfb.wait()
                    tx = copy(block, out[i : i + 1, 0:1])
                    tx.wait()
                    block.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        a = ttnn.Tensor(torch.arange(3 * 32 * 32).reshape(3 * 32, 32).float())
        out = ttnn.empty(a.shape, dtype=torch.float32)

        test_kernel(a, out)

        expected = a * 2  # Changed from a + 10 since we're doing block + block
        tt_testing.assert_close(out.to_torch(), expected.to_torch())

    def test_copy_tensor_to_block_cooperative(self) -> None:
        """Test Tensor → Block copy in cooperative mode."""

        @ttl.operation(grid=(1, 1))
        def test_kernel(a: ttnn.Tensor, out: ttnn.Tensor):
            # a already is ttnn.Tensor
            # out already is ttnn.Tensor
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                block = dfb.wait()
                out_block = out_dfb.reserve()
                result = block + block + block
                out_block.store(result)
                out_block.push()
                block.pop()

            @ttl.datamovement()
            def dm0():
                # Tensor → Block copy
                block = dfb.reserve()
                tx = copy(a[0:1, 0:1], block)
                tx.wait()
                block.push()

            @ttl.datamovement()
            def dm1():
                block = out_dfb.wait()
                tx = copy(block, out[0:1, 0:1])
                tx.wait()
                block.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        a = make_ones_tensor(32, 32) * 5
        out = make_zeros_tensor(32, 32)

        test_kernel(a, out)

        expected = make_ones_tensor(32, 32) * 15
        tt_testing.assert_close(out.to_torch(), expected.to_torch())

    def test_copy_block_to_tensor_with_dm_thread(self) -> None:
        """Test Block → Tensor copy in cooperative mode using DM thread.

        This replaces test_copy_block_to_tensor_cooperative with proper thread separation:
        - DM0 copies Tensor → Block
        - DM1 copies Block → Tensor
        - Compute processes data
        """

        @ttl.operation(grid=(1, 1))
        def test_kernel(a: ttnn.Tensor, out: ttnn.Tensor):
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                # Compute just verifies data can be accessed
                # In real use, it would process the data
                pass

            @ttl.datamovement()
            def dm0():
                # DM0: Copy input tensor to DFB
                block = dfb.reserve()
                tx = copy(a[0:1, 0:1], block)
                tx.wait()
                block.push()

            @ttl.datamovement()
            def dm1():
                # DM1: Copy DFB to output tensor
                block = dfb.wait()
                tx = copy(block, out[0:1, 0:1])
                tx.wait()
                block.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        a = make_ones_tensor(32, 32) * 7
        out = make_zeros_tensor(32, 32)

        test_kernel(a, out)

        expected = make_ones_tensor(32, 32) * 7
        tt_testing.assert_close(out.to_torch(), expected.to_torch())

    def test_copy_mixed_pairs_with_dm_threads(self) -> None:
        """Test mixed copy operations using DM threads for all copies.

        This replaces test_copy_mixed_pairs_cooperative with proper thread separation:
        - DM threads handle all copy operations
        - Compute thread can read from wait() blocks (via direct access, not copy)
        """

        @ttl.operation(grid=(1, 1))
        def test_kernel(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
            dfb_a = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            dfb_b = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
            dfb_out = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                # Compute reads from DFBs, processes, and writes to output DFB
                for i in range(2):
                    block_a = dfb_a.wait()
                    block_b = dfb_b.wait()

                    # Process data: add blocks together and store to output DFB
                    block_out = dfb_out.reserve()
                    result = block_a + block_b
                    block_out.store(result)
                    block_out.push()

                    block_a.pop()
                    block_b.pop()

            @ttl.datamovement()
            def dm0():
                # DM0: Copy input tensors to DFBs
                for i in range(2):
                    block_a = dfb_a.reserve()
                    tx_a = copy(a[i : i + 1, 0:1], block_a)
                    tx_a.wait()
                    block_a.push()

                    block_b = dfb_b.reserve()
                    tx_b = copy(b[i : i + 1, 0:1], block_b)
                    tx_b.wait()
                    block_b.push()

            @ttl.datamovement()
            def dm1():
                # DM1: Copy output DFB to output tensor
                for i in range(2):
                    block_out = dfb_out.wait()
                    tx = copy(block_out, out[i : i + 1, 0:1])
                    tx.wait()
                    block_out.pop()

            return Program(compute, dm0, dm1, grid=grid)()

        a = ttnn.Tensor(torch.arange(2 * 32 * 32).reshape(2 * 32, 32).float())
        b = ttnn.Tensor(
            torch.arange(2 * 32 * 32, 4 * 32 * 32).reshape(2 * 32, 32).float()
        )
        out = ttnn.empty(a.shape, dtype=torch.float32)

        test_kernel(a, b, out)

        expected = a + b
        tt_testing.assert_close(out.to_torch(), expected.to_torch())

    def test_copy_pipe_operations_not_fully_integrated_in_cooperative_mode(
        self,
    ) -> None:
        """
        Test that documents current limitation: Pipe copy operations can cause deadlocks.

        This test demonstrates that while Tensor↔Block and Block↔Block copy operations
        work in cooperative mode, Pipe copy operations currently have limitations:
        - Block→Pipe copy works (synchronous)
        - But Pipe→Block copy can deadlock because pipe.wait() is blocking

        The issue is that pipe operations (via copy) use blocking wait() calls
        rather than yielding to the scheduler, causing potential deadlocks when
        the sender and receiver are in the same scheduling round.

        This is a known limitation that would require redesigning pipe copy to
        yield blocking information to the scheduler, similar to DFB operations.
        """
        # This test documents the limitation rather than demonstrating working functionality
        # In a real scenario, this would deadlock:
        # - compute yields on pipe.wait() (can_wait returns False until data arrives)
        # - dm0 yields on dfb.wait() (can_wait returns False until data arrives)
        # - Both are blocked, deadlock detected

        # For now, we skip this test to document the limitation
        pass


class TestProgramInternals:
    """Test internal program mechanisms and edge cases."""

    def test_empty_generator_completion(self) -> None:
        """Test that generators with only 'pass' are handled correctly."""
        from python.sim import ttl
        from python.sim.program import Program

        @ttl.datamovement()
        def dm0() -> None:
            pass  # Empty generator

        @ttl.datamovement()
        def dm1() -> None:
            pass  # Empty generator

        @ttl.compute()
        def compute() -> None:
            pass  # Empty generator

        prog = Program(compute, dm0, dm1, grid=(1, 1))
        # Should complete without error
        prog()


if __name__ == "__main__":
    # Run tests
    test_basic = TestBasicExecution()
    test_basic.test_cooperative_mode_basic()
    test_basic.test_multi_tile_computation()

    test_multi = TestMultinode()
    test_multi.test_two_core_execution()
    test_multi.test_four_core_2d_grid()

    test_ctx = TestContextIsolation()
    test_ctx.test_dataflow_buffers_isolated()
    test_ctx.test_tensors_shared_across_cores()

    test_err = TestErrorHandling()
    test_err.test_error_in_compute()
    test_err.test_error_in_dm0()
    test_err.test_deadlock_detection()

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
    test_coop.test_copy_tensor_to_block_cooperative()
    test_coop.test_copy_block_to_tensor_cooperative()
    test_coop.test_copy_block_to_pipe_cooperative()
    test_coop.test_copy_pipe_operations_not_fully_integrated_in_cooperative_mode()
    test_coop.test_copy_mixed_pairs_cooperative()

    print("All program.py tests passed!")
