# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test the operation execution framework.

This test verifies how ``@ttl.operation`` bodies are run, including:
- Context binding and per-node state isolation
- Cooperative execution mode
- Error handling and deadlock detection
- Multi-node execution
"""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.testing as tt_testing
from test_utils import make_ones_tensor, make_zeros_tensor

from sim import TILE_SHAPE, copy, ttl, ttnn
from sim.kernel import KernelKind
from sim.dfb import Block
from sim.decorators import _make_cell, rebind_func_with_ctx  # type: ignore[reportPrivateUsage]
from sim.program import _order_kernels  # type: ignore[reportPrivateUsage]


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

        # Create test data
        a = ttnn.rand((TILE_SHAPE[0] * 4, TILE_SHAPE[1] * 4))
        b = ttnn.rand((TILE_SHAPE[0] * 4, TILE_SHAPE[1] * 4))
        out = ttnn.empty(a.shape)

        test_kernel(a, b, out)

        # Verify result
        expected = ttnn.Tensor(a.to_torch()[0:64, 0:32] + b.to_torch()[0:64, 0:32])
        tt_testing.assert_close(out.to_torch()[0:64, 0:32], expected.to_torch())


class TestMultinode:
    """Test multi-node execution."""

    def test_two_node_execution(self) -> None:
        """Test execution on 2 nodes."""

        @ttl.operation(grid=(2, 1))
        def test_kernel(a: ttnn.Tensor, out: ttnn.Tensor):
            # a already is ttnn.Tensor
            # out already is ttnn.Tensor

            a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                node_id = cast(int, ttl.node(dims=1))
                block = a_dfb.wait()
                out_block = out_dfb.reserve()
                # All nodes just do block + block (multiplies by 2)
                result = block + block
                out_block.store(result)
                block.pop()
                out_block.push()

            @ttl.datamovement()
            def dm0():
                node_id = cast(int, ttl.node(dims=1))
                block = a_dfb.reserve()
                # Each node reads its own tile
                tx = copy(a[node_id : node_id + 1, 0:1], block)
                tx.wait()
                block.push()

            @ttl.datamovement()
            def dm1():
                node_id = cast(int, ttl.node(dims=1))
                block = out_dfb.wait()
                # Each node writes its own tile
                tx = copy(block, out[node_id : node_id + 1, 0:1])
                tx.wait()
                block.pop()

        a = make_ones_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1]) * 5
        out = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1])

        test_kernel(a, out)

        # Both nodes multiply by 2: 5 * 2 = 10
        expected_tensor = make_ones_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1]) * 10
        tt_testing.assert_close(out.to_torch(), expected_tensor.to_torch())

    def test_four_node_2d_grid(self) -> None:
        """Test execution on 2x2 grid (4 nodes)."""

        @ttl.operation(grid=(2, 2))
        def test_kernel(out: ttnn.Tensor):
            # out already is ttnn.Tensor
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

            @ttl.compute()
            def compute():
                node_y, node_x = cast(tuple[int, int], ttl.node(dims=2))
                out_block = out_dfb.reserve()
                # Each node writes its coordinates
                out_block.store(
                    Block.from_tensor(make_ones_tensor(32, 32) * (node_y * 10 + node_x))
                )
                out_block.push()

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                node_y, node_x = cast(tuple[int, int], ttl.node(dims=2))
                block = out_dfb.wait()
                tx = copy(
                    block,
                    out[node_y : node_y + 1, node_x : node_x + 1],
                )
                tx.wait()
                block.pop()

        out = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)

        test_kernel(out)

        # Verify each node wrote its coordinates
        # (0,0) = 0, (0,1) = 1, (1,0) = 10, (1,1) = 11
        out_torch = out.to_torch()
        assert (out_torch[0:32, 0:32] == 0).all()
        assert (out_torch[0:32, 32:64] == 1).all()
        assert (out_torch[32:64, 0:32] == 10).all()
        assert (out_torch[32:64, 32:64] == 11).all()


class TestContextIsolation:
    """Test that per-node contexts are properly isolated."""

    def test_dataflow_buffers_isolated(self) -> None:
        """Test that dataflow buffers are independent per node."""

        @ttl.operation(grid=(2, 1))
        def test_kernel(out: ttnn.Tensor):
            # out already is ttnn.Tensor
            # Each node gets its own DFB instance
            dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                node_id = cast(int, ttl.node(dims=1))
                # Each node reserves/pushes independently
                block = dfb.reserve()
                block.store(
                    Block.from_tensor(make_ones_tensor(32, 32) * (node_id + 100))
                )
                block.push()

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                node_id = cast(int, ttl.node(dims=1))
                # Each node waits/pops its own DFB
                block = dfb.wait()
                tx = copy(block, out[node_id : node_id + 1, 0:1])
                tx.wait()
                block.pop()

        out = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1])

        test_kernel(out)

        # Each node should have written its own value
        out_torch = out.to_torch()
        assert (out_torch[0:32, :] == 100).all()
        assert (out_torch[32:64, :] == 101).all()

    def test_shared_tensor_with_compute_store(self) -> None:
        """Test shared tensors where compute kernel uses store instead of copy.

        This tests the pattern where compute kernel reads from a shared tensor
        and uses store() to write to DFB (not copy).
        """

        @ttl.operation(grid=(2, 1))
        def test_kernel(shared: ttnn.Tensor, out: ttnn.Tensor):
            dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                # Compute kernel reads shared tensor and stores to DFB
                node_id = cast(int, ttl.node(dims=1))
                block = dfb.reserve()
                # Read from shared tensor and store (not copy)
                # Add node_id to distinguish which node wrote
                data = shared[0:1, 0:1] + node_id
                block.store(Block.from_tensor(data))
                block.push()

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                # DM kernel copies from DFB to output
                node_id = cast(int, ttl.node(dims=1))
                block = dfb.wait()
                tx = copy(block, out[node_id : node_id + 1, 0:1])
                tx.wait()
                block.pop()

        shared = make_ones_tensor(32, 32) * 10
        out = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1])

        test_kernel(shared, out)

        # Each node should have written shared + node_id
        out_torch = out.to_torch()
        assert (out_torch[0:32, :] == 10).all()  # node 0: 10 + 0
        assert (out_torch[32:64, :] == 11).all()  # node 1: 10 + 1


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

        a = make_zeros_tensor(32, 32)

        with pytest.raises(
            RuntimeError, match="node0-compute.*ValueError.*Test error in compute"
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

        a = make_zeros_tensor(32, 32)

        with pytest.raises(
            RuntimeError, match="node0-dm0.*RuntimeError.*Test error in dm0"
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

        a = make_zeros_tensor(32, 32)

        with pytest.raises(RuntimeError, match="Deadlock detected"):
            test_kernel(a)


class TestKernelSetShape:
    """An operation runs one compute and two data-movement kernels, and no other set.

    Three threads is what the hardware node offers, so the count is not a
    simulator convenience: a body that writes two compute kernels or forgets one
    has no execution to model. These are the diagnostics a user meets while
    writing a multi-kernel body, and they name what is wrong with the set rather
    than failing later inside the scheduler.
    """

    def test_a_body_with_too_few_kernels_is_refused(self) -> None:
        """Two kernels leave a thread with nothing to run."""

        @ttl.operation(grid=(1, 1))
        def op(a: ttnn.Tensor):
            @ttl.compute()
            def compute():
                pass

            @ttl.datamovement()
            def dm0():
                pass

        with pytest.raises(ValueError, match="exactly 3 kernels.*got 2"):
            op(make_zeros_tensor(32, 32))

    def test_a_body_with_two_compute_kernels_is_refused(self) -> None:
        """A node has one compute thread, so the second has nowhere to run.

        The count is right here and the roles are not, which is why it is worth
        its own message: the kernels look like a valid set until they are sorted.
        """

        @ttl.operation(grid=(1, 1))
        def op(a: ttnn.Tensor):
            @ttl.compute()
            def compute():
                pass

            @ttl.compute()
            def also_compute():
                pass

            @ttl.datamovement()
            def dm0():
                pass

        with pytest.raises(ValueError, match="exactly 1 compute kernel, got 2"):
            op(make_zeros_tensor(32, 32))

    def test_a_kernel_that_is_neither_role_is_not_counted_as_data_movement(
        self,
    ) -> None:
        """The two data-movement kernels are the ones that say they are.

        No decorator produces a third role today, so this is checked against the
        ordering directly rather than through a body. It is the reason the count is
        a count and not "the two that are left": something that arrives without a
        role must not be handed to a data-movement thread, where it would run as
        one.
        """
        compute = SimpleNamespace(kernel_type=KernelKind.COMPUTE, __name__="compute")
        dm = SimpleNamespace(kernel_type=KernelKind.DATA_MOVEMENT, __name__="dm0")
        roleless = SimpleNamespace(kernel_type=None, __name__="stranger")

        with pytest.raises(ValueError, match="exactly 2 datamovement kernels, got 1"):
            _order_kernels(cast(list, [compute, dm, roleless]))


class TestBlockCompletion:
    """Test block completion validation at end of kernel execution.

    These tests verify that the simulator catches incomplete block operations
    (missing push() or pop() calls) at the end of kernel execution.
    """

    def test_missing_push_detected(self) -> None:
        """Test that a missing push() is handled automatically by auto push/pop.

        With the simulator's AST-based auto push/pop insertion, a kernel that
        omits block.push() after a reserve() no longer produces an error.  The
        push is inserted automatically before the next reserve on the same DFB
        or at function return.
        """

        @ttl.operation(grid=(1,))
        def test_kernel(input_data: ttnn.Tensor):
            # Create dataflow buffers
            element = make_ones_tensor(32, 32)
            in_dfb = ttl.make_dataflow_buffer_like(element, shape=(1, 1), block_count=2)

            @ttl.datamovement()
            def dm0():
                # Reserve a block without explicit push — auto push/pop handles it.
                block = in_dfb.reserve()
                slice_data = input_data[0:1, 0:1]
                tx = copy(slice_data, block)
                tx.wait()
                # Omitted: block.push()  -> inserted automatically on return

            @ttl.datamovement()
            def dm1():
                pass

            @ttl.compute()
            def compute():
                pass

        input_tensor = ttnn.rand((32, 32))

        # Should NOT raise: auto push/pop inserts push() on function return.
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

    def test_the_failure_names_the_node_it_happened_on(self) -> None:
        """A buffer left pending is reported against its own node.

        A pipe net leaves nodes out, so the nodes that ran are not 0..n-1: here
        only nodes 1 and 3 of a 2x2 grid participate, and the one that forgets to
        pop is node 3.  Counting the contexts instead of carrying the node names
        it "node1", which is a node that ran and did nothing wrong.
        """
        net = ttl.PipeNet([ttl.Pipe(src=(0, 1), dst=(1, 1))])

        @ttl.operation(grid=(2, 2))
        def test_kernel(input_data: ttnn.Tensor):
            element = make_ones_tensor(32, 32)
            in_dfb = ttl.make_dataflow_buffer_like(element, shape=(1, 1), block_count=2)

            @ttl.datamovement()
            def dm0():
                block = in_dfb.reserve()
                copy(input_data[0:1, 0:1], block).wait()
                block.push()

            @ttl.datamovement()
            def dm1():
                pass

            @ttl.compute()
            def compute():
                if net.is_dst():
                    # Node 3 alone leaves the block it waited for pending.
                    data = in_dfb.wait()
                    _ = data + data

        with pytest.raises(RuntimeError, match=r"node3\.in_dfb") as failure:
            test_kernel(ttnn.rand((32, 32)))

        assert "node1." not in str(failure.value), str(failure.value)

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
        """Test that multiple DFBs with missing pushes are all handled by auto push/pop."""

        @ttl.operation(grid=(1,))
        def test_kernel(input_data: ttnn.Tensor):
            from sim.dfb import DataflowBuffer

            # Create multiple dataflow buffers
            element = make_ones_tensor(32, 32)
            dfb1 = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)
            dfb2 = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)

            @ttl.datamovement()
            def dm0():
                # Both DFBs omit push — auto push/pop inserts them both.
                block1 = dfb1.reserve()
                slice_data = input_data[0:1, 0:1]
                tx = copy(slice_data, block1)
                tx.wait()
                # Omitted: block1.push()

                block2 = dfb2.reserve()
                tx = copy(slice_data, block2)
                tx.wait()
                # Omitted: block2.push()

            @ttl.datamovement()
            def dm1():
                pass

            @ttl.compute()
            def compute():
                pass

        input_tensor = ttnn.rand((32, 32))

        # Should NOT raise: auto push/pop inserts push() for both blocks.
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

        a = make_ones_tensor(32, 32) * 5
        out = make_zeros_tensor(32, 32)

        test_kernel(a, out)

        expected = make_ones_tensor(32, 32) * 15
        tt_testing.assert_close(out.to_torch(), expected.to_torch())

    def test_copy_block_to_tensor_with_dm_kernel(self) -> None:
        """Test Block → Tensor copy in cooperative mode using DM kernel.

        This replaces test_copy_block_to_tensor_cooperative with proper kernel separation:
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

        a = make_ones_tensor(32, 32) * 7
        out = make_zeros_tensor(32, 32)

        test_kernel(a, out)

        expected = make_ones_tensor(32, 32) * 7
        tt_testing.assert_close(out.to_torch(), expected.to_torch())

    def test_copy_mixed_pairs_with_dm_kernels(self) -> None:
        """Test mixed copy operations using DM kernels for all copies.

        This replaces test_copy_mixed_pairs_cooperative with proper kernel separation:
        - DM kernels handle all copy operations
        - Compute kernel can read from wait() blocks (via direct access, not copy)
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

        @ttl.operation(grid=(1, 1))
        def test_kernel() -> None:
            @ttl.datamovement()
            def dm0() -> None:
                pass  # Empty generator

            @ttl.datamovement()
            def dm1() -> None:
                pass  # Empty generator

            @ttl.compute()
            def compute() -> None:
                pass  # Empty generator

        # Should complete without error
        test_kernel()

    def test_a_kernel_may_read_a_name_the_body_never_assigned(self) -> None:
        """A name the body leaves unset is skipped, not an error.

        The per-node context is built by reading every name a kernel closes over,
        looking for the buffers and tensors it must be able to reach. A name the
        body binds only on a path it did not take is closed over all the same and
        has no value yet, and that is the user's business: the kernels here never
        read it, so naming it must not fail the run before they start.
        """

        @ttl.operation(grid=(1, 1))
        def op(a: ttnn.Tensor, out: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            if a is None:
                only_on_the_path_not_taken = 1

            @ttl.compute()
            def compute() -> None:
                if a is None:
                    print(only_on_the_path_not_taken)

            @ttl.datamovement()
            def dm0() -> None:
                block = dfb.reserve()
                copy(a[0:1, 0:1], block).wait()
                block.push()

            @ttl.datamovement()
            def dm1() -> None:
                block = dfb.wait()
                copy(block, out[0:1, 0:1]).wait()
                block.pop()

        op(make_ones_tensor(32, 32), make_zeros_tensor(32, 32))


class TestPerNodeKernelAnalysis:
    """Kernel analysis covers the kernels of every node, not just node 0's.

    The operation body is re-run per node, so a body that chooses its kernels by
    ``ttl.node()`` hands different nodes different code. Analysing one node's
    kernels would leave the rest without the copy-wait injection points their
    code needs, and would not see the patterns their code is refused for.
    """

    def test_a_pattern_only_a_later_node_writes_is_reported(self) -> None:
        """A refused ``ttl.copy()`` position on node 1 fails the run."""

        @ttl.operation(grid=(1, 2))
        def op(a: ttnn.Tensor, out: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute() -> None:
                pass

            @ttl.datamovement()
            def dm1() -> None:
                pass

            if ttl.node(dims=1) == 0:

                @ttl.datamovement()
                def dm0() -> None:
                    block = dfb.reserve()
                    ttl.copy(a[0:1, 0:1], block).wait()
                    block.push()

            else:

                @ttl.datamovement()
                def dm0_other_node() -> None:
                    block = dfb.reserve()
                    # A copy inside a container is a position the simulator
                    # refuses; only node 1 reaches this definition.
                    pending = [ttl.copy(a[0:1, 0:1], block)]
                    pending[0].wait()
                    block.push()

        with pytest.raises(RuntimeError, match="unsupported pattern"):
            op(make_ones_tensor(32, 32), make_zeros_tensor(32, 32))

    def test_shared_kernel_code_is_reported_once(self) -> None:
        """Nodes running the same code do not each report the same violation."""

        @ttl.operation(grid=(1, 4))
        def op(a: ttnn.Tensor, out: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute() -> None:
                pass

            @ttl.datamovement()
            def dm1() -> None:
                pass

            @ttl.datamovement()
            def dm0() -> None:
                block = dfb.reserve()
                pending = [ttl.copy(a[0:1, 0:1], block)]
                pending[0].wait()
                block.push()

        with pytest.raises(RuntimeError) as excinfo:
            op(make_ones_tensor(32, 32), make_zeros_tensor(32, 32))

        assert "Found 1 unsupported pattern" in str(excinfo.value), str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__])
