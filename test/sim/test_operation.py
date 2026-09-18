# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for operation.py module (operation decorator, grid_size, etc.).
"""

import torch
from typing import cast

import pytest
from test_utils import make_zeros_tensor

from sim import ttl, ttnn
from sim.nodecontext import flatten_node_index
from sim.typedefs import Shape


class TestGridSize:
    """Test grid_size() function."""

    def test_grid_size_in_operation_2d(self):
        """Test grid_size returns correct dimensions in 2D grid."""

        @ttl.operation(grid=(4, 8))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute():
                grid_h, grid_w = cast(Shape, ttl.grid_size(dims=2))
                assert grid_h == 4
                assert grid_w == 8

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        # Create dummy tensors
        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)

        # Should not raise
        test_operation(a, b)

    def test_grid_size_in_operation_full(self):
        """Test grid_size with auto grid (defaults to 8x8)."""

        @ttl.operation(grid="full")
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute():
                grid_h, grid_w = cast(Shape, ttl.grid_size(dims=2))
                assert grid_h == 8
                assert grid_w == 8

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        # Create dummy tensors
        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)

        # Should not raise
        test_operation(a, b)

    def test_grid_size_in_operation_1d(self):
        """Test grid_size with 1D grid."""

        @ttl.operation(grid=(16,))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute():
                grid_size_val = ttl.grid_size(dims=1)
                assert grid_size_val == 16

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        # Create dummy tensors
        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)

        # Should not raise
        test_operation(a, b)

    def test_grid_size_in_operation_3d(self):
        """Test grid_size with 3D grid."""

        @ttl.operation(grid=(2, 3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute():
                grid_d1, grid_d2, grid_d3 = cast(Shape, ttl.grid_size(dims=3))
                assert grid_d1 == 2
                assert grid_d2 == 3
                assert grid_d3 == 4

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        # Create dummy tensors
        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)

        # Should not raise
        test_operation(a, b)

    def test_grid_size_outside_operation_raises(self):
        """Test that grid_size raises error when called outside operation context."""
        with pytest.raises(RuntimeError, match="grid not available"):
            ttl.grid_size()

    def test_grid_size_in_compute_function(self):
        """Test grid_size can be called from within compute/datamovement functions."""

        @ttl.operation(grid=(3, 5))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                grid_h, grid_w = cast(Shape, ttl.grid_size(dims=2))
                assert grid_h == 3
                assert grid_w == 5

            @ttl.datamovement()
            def dm0():
                grid_h, grid_w = cast(Shape, ttl.grid_size(dims=2))
                assert grid_h == 3
                assert grid_w == 5

            @ttl.datamovement()
            def dm1():
                pass

        # Create dummy tensors
        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)

        # Should not raise
        test_operation(a, b)

    def test_grid_size_unpacking(self):
        """Test various ways to unpack grid_size result."""

        @ttl.operation(grid=(2, 3))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute():
                # Unpack to individual variables
                h, w = cast(Shape, ttl.grid_size(dims=2))
                assert h == 2
                assert w == 3

                # Get as tuple
                grid_dims = cast(Shape, ttl.grid_size(dims=2))
                assert grid_dims == (2, 3)
                assert len(grid_dims) == 2

                # Access by index
                assert cast(Shape, ttl.grid_size(dims=2))[0] == 2
                assert cast(Shape, ttl.grid_size(dims=2))[1] == 3

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        # Create dummy tensors
        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)

        # Should not raise
        test_operation(a, b)

    def test_grid_size_in_nested_functions(self):
        """Test grid_size works in nested function calls within operation."""

        @ttl.operation(grid=(6, 7))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute():
                def helper_function():
                    return cast(Shape, ttl.grid_size(dims=2))

                def another_helper():
                    h, w = helper_function()
                    return h * w

                result = another_helper()
                assert result == 42  # 6 * 7

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        # Create dummy tensors
        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)

        # Should not raise
        test_operation(a, b)

    def test_grid_size_consistent_across_calls(self):
        """Test that grid_size returns consistent values across multiple calls."""

        @ttl.operation(grid=(5, 9))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute():
                grid1 = cast(Shape, ttl.grid_size(dims=2))
                grid2 = cast(Shape, ttl.grid_size(dims=2))
                grid3 = cast(Shape, ttl.grid_size(dims=2))

                assert grid1 == grid2 == grid3
                assert grid1 == (5, 9)

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        # Create dummy tensors
        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)

        # Should not raise
        test_operation(a, b)


class TestNode:
    """Test node() function."""

    def test_node_1d_grid_dims_1(self):
        """Test node() returns single Index for 1D grid with dims=1."""

        @ttl.operation(grid=(8,))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node_id = ttl.node(dims=1)
                # Should be an int, not a tuple
                assert isinstance(node_id, int)
                assert 0 <= node_id < 8

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_2d_grid_dims_1(self):
        """Test node() with dims=1 on 2D grid returns flattened index."""

        @ttl.operation(grid=(2, 3))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node_id = ttl.node(dims=1)
                # Should be a single int from 0 to 5 (2*3 - 1)
                assert isinstance(node_id, int)
                assert 0 <= node_id < 6

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_2d_grid_dims_2(self):
        """Test node() returns 2D coordinates for 2D grid with dims=2."""

        @ttl.operation(grid=(3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node_coord = ttl.node(dims=2)
                # Should be a tuple of 2 ints
                assert isinstance(node_coord, tuple)
                assert len(node_coord) == 2
                assert 0 <= node_coord[0] < 3
                assert 0 <= node_coord[1] < 4

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_3d_grid_dims_1(self):
        """Test node() with dims=1 on 3D grid returns fully flattened index."""

        @ttl.operation(grid=(2, 3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node_id = ttl.node(dims=1)
                # Should be a single int from 0 to 23 (2*3*4 - 1)
                assert isinstance(node_id, int)
                assert 0 <= node_id < 24

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_3d_grid_dims_2_flattens_first_dimension(self):
        """Test node() with dims=2 on 3D grid flattens first two dimensions."""

        @ttl.operation(grid=(2, 3, 5))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node_coord = ttl.node(dims=2)
                # Should be a tuple of 2 ints
                assert isinstance(node_coord, tuple)
                assert len(node_coord) == 2
                # First dimension: flattened [0,1] x [0,1,2] -> [0,5]
                assert 0 <= node_coord[0] < 6  # 2 * 3
                # Second dimension: unchanged
                assert 0 <= node_coord[1] < 5

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_3d_grid_dims_3(self):
        """Test node() returns 3D coordinates for 3D grid with dims=3."""

        @ttl.operation(grid=(2, 3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node_coord = ttl.node(dims=3)
                # Should be a tuple of 3 ints
                assert isinstance(node_coord, tuple)
                assert len(node_coord) == 3
                assert 0 <= node_coord[0] < 2
                assert 0 <= node_coord[1] < 3
                assert 0 <= node_coord[2] < 4

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_2d_grid_dims_3_pads_with_zeros(self):
        """Test node() pads with zeros when dims > grid dimensions."""

        @ttl.operation(grid=(2, 3))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node_coord = ttl.node(dims=3)
                # Should be a tuple of 3 ints, third one padded with 0
                assert isinstance(node_coord, tuple)
                assert len(node_coord) == 3
                assert 0 <= node_coord[0] < 2
                assert 0 <= node_coord[1] < 3
                assert node_coord[2] == 0  # Padded

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_default_dims_is_2(self):
        """Test that node() defaults to dims=2."""

        @ttl.operation(grid=(4, 5))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node_default = ttl.node()
                node_explicit = ttl.node(dims=2)
                # Should be the same
                assert node_default == node_explicit
                assert isinstance(node_default, tuple)
                assert len(node_default) == 2

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_outside_program_raises(self):
        """Test that node() raises error when called outside Program context."""
        with pytest.raises(RuntimeError, match="node not available"):
            ttl.node()

    def test_node_in_nested_functions(self):
        """Test node() works in nested function calls."""

        @ttl.operation(grid=(3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                def helper_function():
                    return ttl.node(dims=2)

                def another_helper():
                    coord = helper_function()
                    # Verify it's a valid 2D coordinate
                    assert isinstance(coord, tuple)
                    assert len(coord) == 2
                    assert 0 <= coord[0] < 3
                    assert 0 <= coord[1] < 4

                another_helper()

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_in_datamovement_functions(self):
        """Test node() can be called from datamovement functions."""

        @ttl.operation(grid=(2, 3))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                pass

            @ttl.datamovement()
            def dm0():
                node_coord = ttl.node(dims=2)
                assert isinstance(node_coord, tuple)
                assert len(node_coord) == 2
                assert 0 <= node_coord[0] < 2
                assert 0 <= node_coord[1] < 3

            @ttl.datamovement()
            def dm1():
                node_coord = ttl.node(dims=2)
                assert isinstance(node_coord, tuple)
                assert len(node_coord) == 2
                assert 0 <= node_coord[0] < 2
                assert 0 <= node_coord[1] < 3

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_consistent_across_calls(self):
        """Test that node() returns consistent values across multiple calls."""

        @ttl.operation(grid=(3, 5))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node1 = ttl.node(dims=2)
                node2 = ttl.node(dims=2)
                node3 = ttl.node(dims=2)

                # All calls should return the same value
                assert node1 == node2 == node3

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_node_different_dims_same_node(self):
        """Test that different dims values on same node produce correct transformations."""

        @ttl.operation(grid=(2, 3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node1d = ttl.node(dims=1)
                node2d = ttl.node(dims=2)
                node3d = ttl.node(dims=3)

                # Verify consistency: all should be valid
                assert isinstance(node1d, int)
                assert isinstance(node2d, tuple) and len(node2d) == 2
                assert isinstance(node3d, tuple) and len(node3d) == 3

                # Verify ranges
                assert 0 <= node1d < 24
                assert 0 <= node2d[0] < 6 and 0 <= node2d[1] < 4
                assert 0 <= node3d[0] < 2 and 0 <= node3d[1] < 3 and 0 <= node3d[2] < 4

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)


class TestPerNodeBodyExecution:
    """The operation body is evaluated once per node in the grid.

    Every node performs the work the body describes, so the simulator evaluates
    the body once for each of them, with that node's context injected.  The
    compiler evaluates it once instead and resolves ttl.node() on the device, so
    a body that mutates state of the enclosing scope behaves differently on the
    two; the specification does not currently say which is right.
    """

    def test_body_runs_once_per_node_on_every_call(self) -> None:
        """Each node evaluates the body, sees its own index, and does so per call.

        The count follows the grid rather than the call, so a second call is a
        second round of evaluations and not a cached one.
        """
        nodes: list[int] = []

        @ttl.operation(grid=(2, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor) -> None:
            nodes.append(cast(int, ttl.node(dims=1)))

            @ttl.compute()
            def compute():
                pass

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)

        test_operation(a, b)
        assert sorted(nodes) == [0, 1, 2, 3, 4, 5, 6, 7]

        test_operation(a, b)
        assert sorted(nodes) == sorted([0, 1, 2, 3, 4, 5, 6, 7] * 2)

    def test_nodes_a_pipe_net_leaves_out_still_run_their_setup(self) -> None:
        """Inactive nodes skip their kernels but not the body that built them.

        Which nodes a pipe net covers is only known once every body has run and
        its pipes have been collected, so the body runs everywhere first.  On
        hardware an inactive node allocates nothing, so its dataflow buffers
        exist here and not there.
        """
        setup_nodes: list[int] = []
        kernel_nodes: list[int] = []

        @ttl.operation(grid=(2, 2))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor) -> None:
            setup_nodes.append(cast(int, ttl.node(dims=1)))
            # A single pipe, so two of the four nodes take no part in the net.
            net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])

            @ttl.compute()
            def compute():
                kernel_nodes.append(cast(int, ttl.node(dims=1)))

            @ttl.datamovement()
            def dm0():
                net.if_src(lambda pipe: None)

            @ttl.datamovement()
            def dm1():
                pass

        test_operation(make_zeros_tensor(32, 32), make_zeros_tensor(32, 32))

        assert sorted(setup_nodes) == [0, 1, 2, 3]
        assert sorted(kernel_nodes) == [0, 2]

    def test_a_unified_statement_with_no_thread_runs_on_every_selected_kernel(
        self,
    ) -> None:
        """In a unified body, only setup and pinned statements run once per node.

        Thread assignment pins a statement through the TT-Lang call it makes, so a
        statement that makes none belongs to no thread and is replicated onto every
        kernel the operation selects -- where a side effect happens once per such
        kernel per node, while the hoisted construction happens once.  Documented in
        docs/sphinx/simulator.md, and worth pinning because it is the difference
        between a unified body and the same body written as explicit kernels.
        """
        buffers_seen: list[int] = []

        @ttl.operation(grid=(2, 1))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            buffers_seen.append(id(dfb))
            blk = dfb.reserve()
            ttl.copy(a[0:1, 0:1], blk).wait()
            blk.push()
            out_blk = dfb.wait()
            ttl.copy(out_blk, b[0:1, 0:1]).wait()
            out_blk.pop()

        test_operation(make_zeros_tensor(32, 32), make_zeros_tensor(32, 32))

        # The body only moves data, so it selects one data movement kernel and no
        # compute kernel. Two nodes: the statement that pins no thread runs on that
        # one kernel, and each node's kernels see the one buffer its lifted
        # construction built.
        assert len(buffers_seen) == 2
        assert len(set(buffers_seen)) == 2


class TestFlattenNodeCoord:
    """Test flatten_node_index() function."""

    def test_flatten_already_linear_coord(self):
        """Test flattening an already linear coordinate returns it unchanged."""

        @ttl.operation(grid=(8, 8))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Linear coordinate should be returned unchanged
                result = flatten_node_index(5)
                assert result == 5
                assert isinstance(result, int)

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_flatten_2d_node_coord(self):
        """Test flattening a 2D node coordinate to linear."""

        @ttl.operation(grid=(4, 8))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # (0, 0) -> 0
                assert flatten_node_index((0, 0)) == 0
                # (0, 1) -> 1
                assert flatten_node_index((0, 1)) == 1
                # (0, 7) -> 7
                assert flatten_node_index((0, 7)) == 7
                # (1, 0) -> 8 (1 * 8 + 0)
                assert flatten_node_index((1, 0)) == 8
                # (1, 1) -> 9 (1 * 8 + 1)
                assert flatten_node_index((1, 1)) == 9
                # (2, 3) -> 19 (2 * 8 + 3)
                assert flatten_node_index((2, 3)) == 19
                # (3, 7) -> 31 (3 * 8 + 7)
                assert flatten_node_index((3, 7)) == 31

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_flatten_3d_node_coord(self):
        """Test flattening a 3D node coordinate to linear."""

        @ttl.operation(grid=(2, 3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # (0, 0, 0) -> 0
                assert flatten_node_index((0, 0, 0)) == 0
                # (0, 0, 1) -> 1
                assert flatten_node_index((0, 0, 1)) == 1
                # (0, 1, 0) -> 4 (0 * 3 * 4 + 1 * 4 + 0)
                assert flatten_node_index((0, 1, 0)) == 4
                # (0, 2, 3) -> 11 (0 * 3 * 4 + 2 * 4 + 3)
                assert flatten_node_index((0, 2, 3)) == 11
                # (1, 0, 0) -> 12 (1 * 3 * 4 + 0 * 4 + 0)
                assert flatten_node_index((1, 0, 0)) == 12
                # (1, 2, 3) -> 23 (1 * 3 * 4 + 2 * 4 + 3)
                assert flatten_node_index((1, 2, 3)) == 23

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_flatten_with_node_function(self):
        """Test flattening the result of node() function."""

        @ttl.operation(grid=(3, 5))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Get 2D node coordinates
                node_2d = ttl.node(dims=2)
                # Get 1D node index
                node_1d = ttl.node(dims=1)

                # Flattening the 2D coordinates should equal the 1D index
                flattened = flatten_node_index(node_2d)
                assert flattened == node_1d

                # Flattening the already-linear index should return itself
                flattened_linear = flatten_node_index(node_1d)
                assert flattened_linear == node_1d

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_flatten_idempotent(self):
        """Test that flattening twice gives the same result."""

        @ttl.operation(grid=(2, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                node_2d = (1, 2)
                flat1 = flatten_node_index(node_2d)
                flat2 = flatten_node_index(flat1)

                # Should be the same (idempotent)
                assert flat1 == flat2
                assert flat1 == 6  # 1 * 4 + 2

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_flatten_different_grid_sizes(self):
        """Test flattening works correctly with different grid dimensions."""

        @ttl.operation(grid=(10, 5))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Test with 10x5 grid
                # (0, 0) -> 0
                assert flatten_node_index((0, 0)) == 0
                # (1, 0) -> 5 (1 * 5 + 0)
                assert flatten_node_index((1, 0)) == 5
                # (5, 3) -> 28 (5 * 5 + 3)
                assert flatten_node_index((5, 3)) == 28
                # (9, 4) -> 49 (9 * 5 + 4)
                assert flatten_node_index((9, 4)) == 49

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_flatten_returns_int_type(self):
        """Test that flatten_node_index always returns an int."""

        @ttl.operation(grid=(2, 2))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Test with linear coordinate
                result1 = flatten_node_index(3)
                assert isinstance(result1, int)

                # Test with 2D tuple
                result2 = flatten_node_index((1, 1))
                assert isinstance(result2, int)

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)


class TestKernelOrderIndependence:
    """Test that kernel definition order doesn't matter in operations."""

    def test_kernel_order_dm_compute_dm(self):
        """Test operation with order: DM, compute, DM (like broadcast_demo.py)."""

        @ttl.operation(grid=(1, 1))
        def operation_dm_compute_dm(
            A: ttnn.Tensor, B: ttnn.Tensor, Y: ttnn.Tensor
        ) -> None:
            a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1))
            b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1))
            y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1))

            @ttl.datamovement()
            def dm_read():
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    a_xf = ttl.copy(A[0, 0], a_blk)
                    b_xf = ttl.copy(B[0, 0], b_blk)
                    a_xf.wait()
                    b_xf.wait()

            @ttl.compute()
            def compute():
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    result = a_blk + b_blk
                    y_blk.store(result)

            @ttl.datamovement()
            def dm_write():
                with y_dfb.wait() as y_blk:
                    y_xf = ttl.copy(y_blk, Y[0, 0])
                    y_xf.wait()

        # Create test tensors
        A = ttnn.from_torch(torch.ones((32, 32), dtype=torch.float32))
        B = ttnn.from_torch(torch.ones((32, 32), dtype=torch.float32) * 2)
        Y = ttnn.empty((32, 32), dtype=torch.float32)

        # Run operation
        operation_dm_compute_dm(A, B, Y)

        # Verify result
        Y_torch = Y.to_torch()
        expected = torch.ones((32, 32), dtype=torch.float32) * 3
        assert torch.allclose(Y_torch, expected)

    def test_kernel_order_compute_dm_dm(self):
        """Test operation with order: compute, DM, DM (traditional order)."""

        @ttl.operation(grid=(1, 1))
        def operation_compute_dm_dm(
            A: ttnn.Tensor, B: ttnn.Tensor, Y: ttnn.Tensor
        ) -> None:
            a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1))
            b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1))
            y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1))

            @ttl.compute()
            def compute():
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    result = a_blk + b_blk
                    y_blk.store(result)

            @ttl.datamovement()
            def dm_read():
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    a_xf = ttl.copy(A[0, 0], a_blk)
                    b_xf = ttl.copy(B[0, 0], b_blk)
                    a_xf.wait()
                    b_xf.wait()

            @ttl.datamovement()
            def dm_write():
                with y_dfb.wait() as y_blk:
                    y_xf = ttl.copy(y_blk, Y[0, 0])
                    y_xf.wait()

        # Create test tensors
        A = ttnn.from_torch(torch.ones((32, 32), dtype=torch.float32))
        B = ttnn.from_torch(torch.ones((32, 32), dtype=torch.float32) * 2)
        Y = ttnn.empty((32, 32), dtype=torch.float32)

        # Run operation
        operation_compute_dm_dm(A, B, Y)

        # Verify result
        Y_torch = Y.to_torch()
        expected = torch.ones((32, 32), dtype=torch.float32) * 3
        assert torch.allclose(Y_torch, expected)

    def test_kernel_order_dm_dm_compute(self):
        """Test operation with order: DM, DM, compute."""

        @ttl.operation(grid=(1, 1))
        def operation_dm_dm_compute(
            A: ttnn.Tensor, B: ttnn.Tensor, Y: ttnn.Tensor
        ) -> None:
            a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1))
            b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1))
            y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1))

            @ttl.datamovement()
            def dm_read():
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    a_xf = ttl.copy(A[0, 0], a_blk)
                    b_xf = ttl.copy(B[0, 0], b_blk)
                    a_xf.wait()
                    b_xf.wait()

            @ttl.datamovement()
            def dm_write():
                with y_dfb.wait() as y_blk:
                    y_xf = ttl.copy(y_blk, Y[0, 0])
                    y_xf.wait()

            @ttl.compute()
            def compute():
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    result = a_blk + b_blk
                    y_blk.store(result)

        # Create test tensors
        A = ttnn.from_torch(torch.ones((32, 32), dtype=torch.float32))
        B = ttnn.from_torch(torch.ones((32, 32), dtype=torch.float32) * 2)
        Y = ttnn.empty((32, 32), dtype=torch.float32)

        # Run operation
        operation_dm_dm_compute(A, B, Y)

        # Verify result
        Y_torch = Y.to_torch()
        expected = torch.ones((32, 32), dtype=torch.float32) * 3
        assert torch.allclose(Y_torch, expected)


class TestRowMajoroperation:
    """End-to-end tests for row-major layout through the full DM->compute->DM pipeline.

    These tests verify that row-major tensors and DFBs work correctly across
    the full operation execution flow: copy into DFB, compute on blocks, copy out.
    """

    def test_row_major_double_rows(self):
        """Single-node operation doubles each row of a row-major tensor via DFB.

        DM reader copies one row at a time into the input DFB.
        Compute doubles each row via block addition (in_blk + in_blk).
        DM writer copies each result row back to the output tensor.
        Verifies that layout is preserved end-to-end.
        """
        from sim.ttnnsim import ROW_MAJOR_LAYOUT, Tensor as SimTensor

        N, C = 4, 8
        input_data = torch.arange(N * C, dtype=torch.float32).reshape(N, C)
        output_data = torch.zeros(N, C, dtype=torch.float32)

        input_tensor = SimTensor(input_data.clone(), ROW_MAJOR_LAYOUT)
        output_tensor = SimTensor(output_data, ROW_MAJOR_LAYOUT)

        likeness = SimTensor(torch.zeros(1, C, dtype=torch.float32), ROW_MAJOR_LAYOUT)

        @ttl.operation(grid=(1, 1))
        def double_rows(
            inp: ttnn.Tensor,
            out: ttnn.Tensor,
        ) -> None:
            in_dfb = ttl.make_dataflow_buffer_like(likeness, shape=(1, C))
            out_dfb = ttl.make_dataflow_buffer_like(likeness, shape=(1, C))

            @ttl.compute()
            def compute() -> None:
                for _ in range(N):
                    with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
                        result = in_blk + in_blk
                        out_blk.store(result)

            @ttl.datamovement()
            def dm_read() -> None:
                for i in range(N):
                    with in_dfb.reserve() as in_blk:
                        ttl.copy(input_tensor[i, :], in_blk).wait()

            @ttl.datamovement()
            def dm_write() -> None:
                for i in range(N):
                    with out_dfb.wait() as out_blk:
                        ttl.copy(out_blk, output_tensor[i, :]).wait()

        double_rows(input_tensor, output_tensor)

        assert torch.allclose(
            output_data, input_data * 2
        ), f"Expected input*2, got {output_data}"

    def test_row_major_single_row_passthrough(self):
        """Single-row operation: copy in, double via addition, copy out.

        Verifies a minimal one-row DM->compute->DM pipeline with row-major layout.
        Distinct from test_row_major_double_rows by using a non-tile-aligned
        column count (C=6) and a single row.
        """
        from sim.ttnnsim import ROW_MAJOR_LAYOUT, Tensor as SimTensor

        C = 6
        input_data = torch.ones(1, C, dtype=torch.float32) * 3.0
        output_data = torch.zeros(1, C, dtype=torch.float32)

        inp_t = SimTensor(input_data.clone(), ROW_MAJOR_LAYOUT)
        out_t = SimTensor(output_data, ROW_MAJOR_LAYOUT)
        likeness = SimTensor(torch.zeros(1, C, dtype=torch.float32), ROW_MAJOR_LAYOUT)

        @ttl.operation(grid=(1, 1))
        def passthrough(inp: ttnn.Tensor, out: ttnn.Tensor) -> None:
            in_dfb = ttl.make_dataflow_buffer_like(likeness, shape=(1, C))
            out_dfb = ttl.make_dataflow_buffer_like(likeness, shape=(1, C))

            @ttl.compute()
            def compute() -> None:
                with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
                    result = in_blk + in_blk
                    out_blk.store(result)

            @ttl.datamovement()
            def dm_read() -> None:
                with in_dfb.reserve() as in_blk:
                    ttl.copy(inp_t[0, :], in_blk).wait()

            @ttl.datamovement()
            def dm_write() -> None:
                with out_dfb.wait() as out_blk:
                    ttl.copy(out_blk, out_t[0, :]).wait()

        passthrough(inp_t, out_t)

        assert torch.allclose(
            output_data, input_data * 2
        ), f"Expected {input_data * 2}, got {output_data}"

    def test_row_major_multirow_unary(self):
        """Row-major operation using a unary math op (exp) preserves layout and values."""
        from sim.ttnnsim import ROW_MAJOR_LAYOUT, Tensor as SimTensor

        N, C = 3, 5
        input_data = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]] * N, dtype=torch.float32)
        output_data = torch.zeros(N, C, dtype=torch.float32)

        inp_t = SimTensor(input_data.clone(), ROW_MAJOR_LAYOUT)
        out_t = SimTensor(output_data, ROW_MAJOR_LAYOUT)
        likeness = SimTensor(torch.zeros(1, C, dtype=torch.float32), ROW_MAJOR_LAYOUT)

        @ttl.operation(grid=(1, 1))
        def exp_rows(inp: ttnn.Tensor, out: ttnn.Tensor) -> None:
            in_dfb = ttl.make_dataflow_buffer_like(likeness, shape=(1, C))
            out_dfb = ttl.make_dataflow_buffer_like(likeness, shape=(1, C))

            @ttl.compute()
            def compute() -> None:
                for _ in range(N):
                    with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
                        result = ttl.math.exp(in_blk)
                        out_blk.store(result)

            @ttl.datamovement()
            def dm_read() -> None:
                for i in range(N):
                    with in_dfb.reserve() as in_blk:
                        ttl.copy(inp_t[i, :], in_blk).wait()

            @ttl.datamovement()
            def dm_write() -> None:
                for i in range(N):
                    with out_dfb.wait() as out_blk:
                        ttl.copy(out_blk, out_t[i, :]).wait()

        exp_rows(inp_t, out_t)

        expected = torch.exp(input_data)
        assert torch.allclose(
            output_data, expected, atol=1e-5
        ), f"Expected exp(input), got {output_data}"


def _make_passthrough_kernel(decorator):
    """Build a simple copy kernel using the given @ttl.operation decorator."""

    @decorator
    def kernel(x, y):
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1))
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1))

        @ttl.compute()
        def compute():
            with x_dfb.wait() as x_blk, y_dfb.reserve() as y_blk:
                y_blk.store(x_blk)

        @ttl.datamovement()
        def reader():
            with x_dfb.reserve() as blk:
                ttl.copy(x[0, 0], blk).wait()

        @ttl.datamovement()
        def writer():
            with y_dfb.wait() as blk:
                ttl.copy(blk, y[0, 0]).wait()

    return kernel


class TestHardwareKeywordsIgnored:
    """Compiler-specific keyword arguments are silently ignored by the simulator."""

    def test_fp32_dest_acc_en_accepted(self) -> None:
        """fp32_dest_acc_en=True does not raise."""
        a = ttnn.from_torch(torch.zeros(32, 32))
        b = ttnn.from_torch(torch.zeros(32, 32))
        kernel = _make_passthrough_kernel(
            ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
        )
        kernel(a, b)

    def test_dst_full_sync_en_accepted(self) -> None:
        """dst_full_sync_en=False does not raise."""
        a = ttnn.from_torch(torch.zeros(32, 32))
        b = ttnn.from_torch(torch.zeros(32, 32))
        kernel = _make_passthrough_kernel(
            ttl.operation(grid=(1, 1), dst_full_sync_en=False)
        )
        kernel(a, b)

    @pytest.mark.parametrize("math_fidelity", ["LoFi", "HiFi2", "HiFi3", "HiFi4"])
    def test_math_fidelity_accepted(self, math_fidelity: str) -> None:
        """Supported math fidelities do not raise."""
        a = ttnn.from_torch(torch.zeros(32, 32))
        b = ttnn.from_torch(torch.zeros(32, 32))
        kernel = _make_passthrough_kernel(
            ttl.operation(grid=(1, 1), math_fidelity=math_fidelity)
        )
        kernel(a, b)

    def test_invalid_math_fidelity_rejected(self) -> None:
        """Unsupported math fidelity raises before execution."""
        with pytest.raises(ValueError, match="math_fidelity must be one of"):
            ttl.operation(grid=(1, 1), math_fidelity="HiFi5")

    def test_multiple_hardware_kwargs_accepted(self) -> None:
        """Multiple hardware kwargs together do not raise."""
        a = ttnn.from_torch(torch.zeros(32, 32))
        b = ttnn.from_torch(torch.zeros(32, 32))
        kernel = _make_passthrough_kernel(
            ttl.operation(
                grid=(1, 1),
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
                math_fidelity="HiFi4",
            )
        )
        kernel(a, b)

    def test_unknown_kwarg_raises(self) -> None:
        """An unrecognised keyword argument raises TypeError."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            ttl.operation(grid=(1, 1), totally_unknown_option=42)


class TestGridValidation:
    """A grid that names no node is rejected where it is written.

    The node count is a product over the dimensions, so a zero dimension leaves
    the run with nothing to schedule and a negative one is counted as its
    absolute contribution.  Both are reported against the grid rather than as a
    later failure to find a node's state.
    """

    @pytest.mark.parametrize("grid", [(0, 2), (2, 0), (0, 0), (1, 1, 0)])
    def test_zero_dimension_rejected(self, grid: Shape) -> None:
        """A dimension of zero names no node."""
        with pytest.raises(ValueError, match="names no node"):
            _make_passthrough_kernel(ttl.operation(grid=grid))

    @pytest.mark.parametrize("grid", [(-1, 2), (-1, -2)])
    def test_negative_dimension_rejected(self, grid: Shape) -> None:
        """A negative dimension is rejected rather than multiplied out."""
        with pytest.raises(ValueError, match="names no node"):
            _make_passthrough_kernel(ttl.operation(grid=grid))

    def test_empty_grid_rejected(self) -> None:
        """A grid with no dimensions has no node to run on."""
        with pytest.raises(ValueError, match="at least one dimension"):
            _make_passthrough_kernel(ttl.operation(grid=()))

    def test_message_names_the_offending_dimensions(self) -> None:
        """The message points at which dimensions are wrong, and at the grid."""
        with pytest.raises(ValueError) as excinfo:
            _make_passthrough_kernel(ttl.operation(grid=(0, 2)))

        reason = str(excinfo.value)
        assert "(0, 2)" in reason, reason
        assert "dimension 0 is 0" in reason, reason


class TestOperationInterface:
    """The signature and body rules an operation must satisfy, with the compiler.

    The specification states them under "Operation function": an operation takes
    only tensors, parameters have no defaults and the signature no ``*args`` /
    ``**kwargs``, and the function returns nothing. Everything else it needs is a
    compile-time argument captured from the enclosing scope.

    The wording asserted here is the compiler's, pinned on that side by
    test/python/atom/operation_boundaries_invalid.py, because both frontends now
    raise it from one place (atom_rules.validate_operation_interface). A program
    the simulator runs and the compiler refuses is the failure mode worth a test:
    it is found after the simulator has said the program is fine.
    """

    def test_a_parameter_with_a_default_is_refused(self) -> None:
        """A default value would be a compile-time argument wearing runtime clothes."""
        with pytest.raises(ValueError, match=r"cannot have default values.*'b'"):

            @ttl.operation(grid=(1, 1))
            def op(a: ttnn.Tensor, b: object = None) -> None:
                pass

    def test_a_variadic_signature_is_refused(self) -> None:
        """The tensor parameters are the interface, so it cannot be open-ended."""
        with pytest.raises(ValueError, match=r"\*args or \*\*kwargs.*'rest'"):

            @ttl.operation(grid=(1, 1))
            def positional(a: ttnn.Tensor, *rest: ttnn.Tensor) -> None:
                pass

        with pytest.raises(ValueError, match=r"\*args or \*\*kwargs.*'rest'"):

            @ttl.operation(grid=(1, 1))
            def keyword(a: ttnn.Tensor, **rest: ttnn.Tensor) -> None:
                pass

    def test_a_body_that_returns_is_refused(self) -> None:
        """An operation writes its results into its output tensors, and returns none."""
        with pytest.raises(ValueError, match="cannot return a value"):

            @ttl.operation(grid=(1, 1))
            def op(a: ttnn.Tensor) -> ttnn.Tensor:
                return a

    def test_a_kernel_inside_the_body_may_still_return(self) -> None:
        """The rule is the operation's, not its kernels'.

        A kernel is a function of its own; the walk stops at nested definitions, so
        a body that writes one is not refused for what that kernel does.
        """
        a = make_zeros_tensor(32, 32)
        out = make_zeros_tensor(32, 32)

        @ttl.operation(grid=(1, 1))
        def op(src: ttnn.Tensor, dst: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(src, shape=(1, 1), block_count=2)

            @ttl.datamovement()
            def reader() -> None:
                block = dfb.reserve()
                ttl.copy(src[0:1, 0:1], block).wait()
                block.push()

            @ttl.compute()
            def nothing() -> None:
                pass

            @ttl.datamovement()
            def writer() -> int:
                block = dfb.wait()
                ttl.copy(block, dst[0:1, 0:1]).wait()
                block.pop()
                return 0

        op(a, out)

    def test_a_body_whose_source_cannot_be_read_is_left_alone(self) -> None:
        """Only the return rule needs the source, and an unreadable body is not wrong.

        A function defined by ``exec`` (a REPL line, a generated body) has no
        source to parse. Refusing it would refuse a program that is otherwise
        fine, so the signature rules still apply and the return rule stands down.
        """
        namespace: dict[str, object] = {"ttl": ttl}
        exec(
            "@ttl.operation(grid=(1, 1))\n" "def op(a):\n" "    pass\n",
            namespace,
        )
        assert callable(namespace["op"])

        with pytest.raises(ValueError, match="cannot have default values"):
            exec(
                "@ttl.operation(grid=(1, 1))\n" "def bad(a=None):\n" "    pass\n",
                {"ttl": ttl},
            )
