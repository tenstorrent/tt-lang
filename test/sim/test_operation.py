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

from python.sim import ttl, ttnn
from python.sim.corecontext import flatten_core_index
from python.sim.typedefs import Shape


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

    def test_grid_size_in_operation_auto(self):
        """Test grid_size with auto grid (defaults to 8x8)."""

        @ttl.operation(grid="auto")
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


class TestCore:
    """Test core() function."""

    def test_core_1d_grid_dims_1(self):
        """Test core() returns single Index for 1D grid with dims=1."""

        @ttl.operation(grid=(8,))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_id = ttl.node(dims=1)
                # Should be an int, not a tuple
                assert isinstance(core_id, int)
                assert 0 <= core_id < 8

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_core_2d_grid_dims_1(self):
        """Test core() with dims=1 on 2D grid returns flattened index."""

        @ttl.operation(grid=(2, 3))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_id = ttl.node(dims=1)
                # Should be a single int from 0 to 5 (2*3 - 1)
                assert isinstance(core_id, int)
                assert 0 <= core_id < 6

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_core_2d_grid_dims_2(self):
        """Test core() returns 2D coordinates for 2D grid with dims=2."""

        @ttl.operation(grid=(3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_coord = ttl.node(dims=2)
                # Should be a tuple of 2 ints
                assert isinstance(core_coord, tuple)
                assert len(core_coord) == 2
                assert 0 <= core_coord[0] < 3
                assert 0 <= core_coord[1] < 4

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_core_3d_grid_dims_1(self):
        """Test core() with dims=1 on 3D grid returns fully flattened index."""

        @ttl.operation(grid=(2, 3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_id = ttl.node(dims=1)
                # Should be a single int from 0 to 23 (2*3*4 - 1)
                assert isinstance(core_id, int)
                assert 0 <= core_id < 24

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_core_3d_grid_dims_2_flattens_first_dimension(self):
        """Test core() with dims=2 on 3D grid flattens first two dimensions."""

        @ttl.operation(grid=(2, 3, 5))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_coord = ttl.node(dims=2)
                # Should be a tuple of 2 ints
                assert isinstance(core_coord, tuple)
                assert len(core_coord) == 2
                # First dimension: flattened [0,1] x [0,1,2] -> [0,5]
                assert 0 <= core_coord[0] < 6  # 2 * 3
                # Second dimension: unchanged
                assert 0 <= core_coord[1] < 5

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_core_3d_grid_dims_3(self):
        """Test core() returns 3D coordinates for 3D grid with dims=3."""

        @ttl.operation(grid=(2, 3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_coord = ttl.node(dims=3)
                # Should be a tuple of 3 ints
                assert isinstance(core_coord, tuple)
                assert len(core_coord) == 3
                assert 0 <= core_coord[0] < 2
                assert 0 <= core_coord[1] < 3
                assert 0 <= core_coord[2] < 4

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_core_2d_grid_dims_3_pads_with_zeros(self):
        """Test core() pads with zeros when dims > grid dimensions."""

        @ttl.operation(grid=(2, 3))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_coord = ttl.node(dims=3)
                # Should be a tuple of 3 ints, third one padded with 0
                assert isinstance(core_coord, tuple)
                assert len(core_coord) == 3
                assert 0 <= core_coord[0] < 2
                assert 0 <= core_coord[1] < 3
                assert core_coord[2] == 0  # Padded

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_core_default_dims_is_2(self):
        """Test that core() defaults to dims=2."""

        @ttl.operation(grid=(4, 5))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_default = ttl.node()
                core_explicit = ttl.node(dims=2)
                # Should be the same
                assert core_default == core_explicit
                assert isinstance(core_default, tuple)
                assert len(core_default) == 2

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_core_outside_program_raises(self):
        """Test that core() raises error when called outside Program context."""
        with pytest.raises(RuntimeError, match="core not available"):
            ttl.node()

    def test_core_in_nested_functions(self):
        """Test core() works in nested function calls."""

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

    def test_core_in_datamovement_functions(self):
        """Test core() can be called from datamovement functions."""

        @ttl.operation(grid=(2, 3))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                pass

            @ttl.datamovement()
            def dm0():
                core_coord = ttl.node(dims=2)
                assert isinstance(core_coord, tuple)
                assert len(core_coord) == 2
                assert 0 <= core_coord[0] < 2
                assert 0 <= core_coord[1] < 3

            @ttl.datamovement()
            def dm1():
                core_coord = ttl.node(dims=2)
                assert isinstance(core_coord, tuple)
                assert len(core_coord) == 2
                assert 0 <= core_coord[0] < 2
                assert 0 <= core_coord[1] < 3

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_core_consistent_across_calls(self):
        """Test that core() returns consistent values across multiple calls."""

        @ttl.operation(grid=(3, 5))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core1 = ttl.node(dims=2)
                core2 = ttl.node(dims=2)
                core3 = ttl.node(dims=2)

                # All calls should return the same value
                assert core1 == core2 == core3

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_core_different_dims_same_core(self):
        """Test that different dims values on same core produce correct transformations."""

        @ttl.operation(grid=(2, 3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core1d = ttl.node(dims=1)
                core2d = ttl.node(dims=2)
                core3d = ttl.node(dims=3)

                # Verify consistency: all should be valid
                assert isinstance(core1d, int)
                assert isinstance(core2d, tuple) and len(core2d) == 2
                assert isinstance(core3d, tuple) and len(core3d) == 3

                # Verify ranges
                assert 0 <= core1d < 24
                assert 0 <= core2d[0] < 6 and 0 <= core2d[1] < 4
                assert 0 <= core3d[0] < 2 and 0 <= core3d[1] < 3 and 0 <= core3d[2] < 4

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)


class TestFlattenCoreCoord:
    """Test flatten_core_index() function."""

    def test_flatten_already_linear_coord(self):
        """Test flattening an already linear coordinate returns it unchanged."""

        @ttl.operation(grid=(8, 8))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Linear coordinate should be returned unchanged
                result = flatten_core_index(5)
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

    def test_flatten_2d_core_coord(self):
        """Test flattening a 2D core coordinate to linear."""

        @ttl.operation(grid=(4, 8))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # (0, 0) -> 0
                assert flatten_core_index((0, 0)) == 0
                # (0, 1) -> 1
                assert flatten_core_index((0, 1)) == 1
                # (0, 7) -> 7
                assert flatten_core_index((0, 7)) == 7
                # (1, 0) -> 8 (1 * 8 + 0)
                assert flatten_core_index((1, 0)) == 8
                # (1, 1) -> 9 (1 * 8 + 1)
                assert flatten_core_index((1, 1)) == 9
                # (2, 3) -> 19 (2 * 8 + 3)
                assert flatten_core_index((2, 3)) == 19
                # (3, 7) -> 31 (3 * 8 + 7)
                assert flatten_core_index((3, 7)) == 31

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_flatten_3d_core_coord(self):
        """Test flattening a 3D core coordinate to linear."""

        @ttl.operation(grid=(2, 3, 4))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # (0, 0, 0) -> 0
                assert flatten_core_index((0, 0, 0)) == 0
                # (0, 0, 1) -> 1
                assert flatten_core_index((0, 0, 1)) == 1
                # (0, 1, 0) -> 4 (0 * 3 * 4 + 1 * 4 + 0)
                assert flatten_core_index((0, 1, 0)) == 4
                # (0, 2, 3) -> 11 (0 * 3 * 4 + 2 * 4 + 3)
                assert flatten_core_index((0, 2, 3)) == 11
                # (1, 0, 0) -> 12 (1 * 3 * 4 + 0 * 4 + 0)
                assert flatten_core_index((1, 0, 0)) == 12
                # (1, 2, 3) -> 23 (1 * 3 * 4 + 2 * 4 + 3)
                assert flatten_core_index((1, 2, 3)) == 23

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

        a = make_zeros_tensor(32, 32)
        b = make_zeros_tensor(32, 32)
        test_operation(a, b)

    def test_flatten_with_core_function(self):
        """Test flattening the result of core() function."""

        @ttl.operation(grid=(3, 5))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Get 2D core coordinates
                core_2d = ttl.node(dims=2)
                # Get 1D core index
                core_1d = ttl.node(dims=1)

                # Flattening the 2D coordinates should equal the 1D index
                flattened = flatten_core_index(core_2d)
                assert flattened == core_1d

                # Flattening the already-linear index should return itself
                flattened_linear = flatten_core_index(core_1d)
                assert flattened_linear == core_1d

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
                core_2d = (1, 2)
                flat1 = flatten_core_index(core_2d)
                flat2 = flatten_core_index(flat1)

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
                assert flatten_core_index((0, 0)) == 0
                # (1, 0) -> 5 (1 * 5 + 0)
                assert flatten_core_index((1, 0)) == 5
                # (5, 3) -> 28 (5 * 5 + 3)
                assert flatten_core_index((5, 3)) == 28
                # (9, 4) -> 49 (9 * 5 + 4)
                assert flatten_core_index((9, 4)) == 49

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
        """Test that flatten_core_index always returns an int."""

        @ttl.operation(grid=(2, 2))
        def test_operation(a: ttnn.Tensor, b: ttnn.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Test with linear coordinate
                result1 = flatten_core_index(3)
                assert isinstance(result1, int)

                # Test with 2D tuple
                result2 = flatten_core_index((1, 1))
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


class TestThreadOrderIndependence:
    """Test that thread definition order doesn't matter in operations."""

    def test_thread_order_dm_compute_dm(self):
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

    def test_thread_order_compute_dm_dm(self):
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

    def test_thread_order_dm_dm_compute(self):
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
        """Single-core operation doubles each row of a row-major tensor via DFB.

        DM reader copies one row at a time into the input DFB.
        Compute doubles each row via block addition (in_blk + in_blk).
        DM writer copies each result row back to the output tensor.
        Verifies that layout is preserved end-to-end.
        """
        from python.sim.ttnnsim import ROW_MAJOR_LAYOUT, Tensor as SimTensor

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
        from python.sim.ttnnsim import ROW_MAJOR_LAYOUT, Tensor as SimTensor

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
        from python.sim.ttnnsim import ROW_MAJOR_LAYOUT, Tensor as SimTensor

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
