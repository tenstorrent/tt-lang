# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for kernel.py module (kernel decorator, grid_size, etc.).
"""

import pytest
import torch
from python.sim import ttl


class TestGridSize:
    """Test grid_size() function."""

    def test_grid_size_in_kernel_2d(self):
        """Test grid_size returns correct dimensions in 2D grid."""

        @ttl.kernel(grid=(4, 8), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None
            grid_h, grid_w = ttl.grid_size()
            assert grid_h == 4
            assert grid_w == 8

        # Create dummy tensors
        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)

        # Should not raise
        test_kernel(a, b)

    def test_grid_size_in_kernel_auto(self):
        """Test grid_size with auto grid (defaults to 8x8)."""

        @ttl.kernel(grid="auto", granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None
            grid_h, grid_w = ttl.grid_size()
            assert grid_h == 8
            assert grid_w == 8

        # Create dummy tensors
        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)

        # Should not raise
        test_kernel(a, b)

    def test_grid_size_in_kernel_1d(self):
        """Test grid_size with 1D grid."""

        @ttl.kernel(grid=(16,), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None
            (grid_size_val,) = ttl.grid_size()
            assert grid_size_val == 16

        # Create dummy tensors
        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)

        # Should not raise
        test_kernel(a, b)

    def test_grid_size_in_kernel_3d(self):
        """Test grid_size with 3D grid."""

        @ttl.kernel(grid=(2, 3, 4), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None
            grid_d1, grid_d2, grid_d3 = ttl.grid_size()
            assert grid_d1 == 2
            assert grid_d2 == 3
            assert grid_d3 == 4

        # Create dummy tensors
        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)

        # Should not raise
        test_kernel(a, b)

    def test_grid_size_outside_kernel_raises(self):
        """Test that grid_size raises error when called outside kernel context."""
        with pytest.raises(RuntimeError, match="grid not available"):
            ttl.grid_size()

    def test_grid_size_in_compute_function(self):
        """Test grid_size can be called from within compute/datamovement functions."""

        @ttl.kernel(grid=(3, 5), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                grid_h, grid_w = ttl.grid_size()
                assert grid_h == 3
                assert grid_w == 5

            @ttl.datamovement()
            def dm0():
                grid_h, grid_w = ttl.grid_size()
                assert grid_h == 3
                assert grid_w == 5

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        # Create dummy tensors
        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)

        # Should not raise
        test_kernel(a, b)

    def test_grid_size_unpacking(self):
        """Test various ways to unpack grid_size result."""

        @ttl.kernel(grid=(2, 3), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None
            # Unpack to individual variables
            h, w = ttl.grid_size()
            assert h == 2
            assert w == 3

            # Get as tuple
            grid_dims = ttl.grid_size()
            assert grid_dims == (2, 3)
            assert len(grid_dims) == 2

            # Access by index
            assert ttl.grid_size()[0] == 2
            assert ttl.grid_size()[1] == 3

        # Create dummy tensors
        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)

        # Should not raise
        test_kernel(a, b)

    def test_grid_size_in_nested_functions(self):
        """Test grid_size works in nested function calls within kernel."""

        @ttl.kernel(grid=(6, 7), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            def helper_function():
                return ttl.grid_size()

            def another_helper():
                h, w = helper_function()
                return h * w

            result = another_helper()
            assert result == 42  # 6 * 7

        # Create dummy tensors
        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)

        # Should not raise
        test_kernel(a, b)

    def test_grid_size_consistent_across_calls(self):
        """Test that grid_size returns consistent values across multiple calls."""

        @ttl.kernel(grid=(5, 9), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None
            grid1 = ttl.grid_size()
            grid2 = ttl.grid_size()
            grid3 = ttl.grid_size()

            assert grid1 == grid2 == grid3
            assert grid1 == (5, 9)

        # Create dummy tensors
        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)

        # Should not raise
        test_kernel(a, b)


class TestCore:
    """Test core() function."""

    def test_core_1d_grid_dims_1(self):
        """Test core() returns single Index for 1D grid with dims=1."""

        @ttl.kernel(grid=(8,), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_id = ttl.core(dims=1)
                # Should be an int, not a tuple
                assert isinstance(core_id, int)
                assert 0 <= core_id < 8

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_2d_grid_dims_1(self):
        """Test core() with dims=1 on 2D grid returns flattened index."""

        @ttl.kernel(grid=(2, 3), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_id = ttl.core(dims=1)
                # Should be a single int from 0 to 5 (2*3 - 1)
                assert isinstance(core_id, int)
                assert 0 <= core_id < 6

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_2d_grid_dims_2(self):
        """Test core() returns 2D coordinates for 2D grid with dims=2."""

        @ttl.kernel(grid=(3, 4), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_coord = ttl.core(dims=2)
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

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_3d_grid_dims_1(self):
        """Test core() with dims=1 on 3D grid returns fully flattened index."""

        @ttl.kernel(grid=(2, 3, 4), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_id = ttl.core(dims=1)
                # Should be a single int from 0 to 23 (2*3*4 - 1)
                assert isinstance(core_id, int)
                assert 0 <= core_id < 24

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_3d_grid_dims_2_flattens_first_dimension(self):
        """Test core() with dims=2 on 3D grid flattens first two dimensions."""

        @ttl.kernel(grid=(2, 3, 5), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_coord = ttl.core(dims=2)
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

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_3d_grid_dims_3(self):
        """Test core() returns 3D coordinates for 3D grid with dims=3."""

        @ttl.kernel(grid=(2, 3, 4), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_coord = ttl.core(dims=3)
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

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_2d_grid_dims_3_pads_with_zeros(self):
        """Test core() pads with zeros when dims > grid dimensions."""

        @ttl.kernel(grid=(2, 3), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_coord = ttl.core(dims=3)
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

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_default_dims_is_2(self):
        """Test that core() defaults to dims=2."""

        @ttl.kernel(grid=(4, 5), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_default = ttl.core()
                core_explicit = ttl.core(dims=2)
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

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_outside_program_raises(self):
        """Test that core() raises error when called outside Program context."""
        with pytest.raises(RuntimeError, match="core not available"):
            ttl.core()

    def test_core_in_nested_functions(self):
        """Test core() works in nested function calls."""

        @ttl.kernel(grid=(3, 4), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                def helper_function():
                    return ttl.core(dims=2)

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

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_in_datamovement_functions(self):
        """Test core() can be called from datamovement functions."""

        @ttl.kernel(grid=(2, 3), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                pass

            @ttl.datamovement()
            def dm0():
                core_coord = ttl.core(dims=2)
                assert isinstance(core_coord, tuple)
                assert len(core_coord) == 2
                assert 0 <= core_coord[0] < 2
                assert 0 <= core_coord[1] < 3

            @ttl.datamovement()
            def dm1():
                core_coord = ttl.core(dims=2)
                assert isinstance(core_coord, tuple)
                assert len(core_coord) == 2
                assert 0 <= core_coord[0] < 2
                assert 0 <= core_coord[1] < 3

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_consistent_across_calls(self):
        """Test that core() returns consistent values across multiple calls."""

        @ttl.kernel(grid=(3, 5), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core1 = ttl.core(dims=2)
                core2 = ttl.core(dims=2)
                core3 = ttl.core(dims=2)

                # All calls should return the same value
                assert core1 == core2 == core3

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_core_different_dims_same_core(self):
        """Test that different dims values on same core produce correct transformations."""

        @ttl.kernel(grid=(2, 3, 4), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core1d = ttl.core(dims=1)
                core2d = ttl.core(dims=2)
                core3d = ttl.core(dims=3)

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

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)


class TestFlattenCoreIndex:
    """Test flatten_core_index() function."""

    def test_flatten_already_linear_index(self):
        """Test flattening an already linear index returns it unchanged."""

        @ttl.kernel(grid=(8, 8), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Linear index should be returned unchanged
                result = ttl.flatten_core_index(5)
                assert result == 5
                assert isinstance(result, int)

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_flatten_2d_core_index(self):
        """Test flattening a 2D core index to linear."""

        @ttl.kernel(grid=(4, 8), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # (0, 0) -> 0
                assert ttl.flatten_core_index((0, 0)) == 0
                # (0, 1) -> 1
                assert ttl.flatten_core_index((0, 1)) == 1
                # (0, 7) -> 7
                assert ttl.flatten_core_index((0, 7)) == 7
                # (1, 0) -> 8 (1 * 8 + 0)
                assert ttl.flatten_core_index((1, 0)) == 8
                # (1, 1) -> 9 (1 * 8 + 1)
                assert ttl.flatten_core_index((1, 1)) == 9
                # (2, 3) -> 19 (2 * 8 + 3)
                assert ttl.flatten_core_index((2, 3)) == 19
                # (3, 7) -> 31 (3 * 8 + 7)
                assert ttl.flatten_core_index((3, 7)) == 31

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_flatten_3d_core_index(self):
        """Test flattening a 3D core index to linear."""

        @ttl.kernel(grid=(2, 3, 4), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # (0, 0, 0) -> 0
                assert ttl.flatten_core_index((0, 0, 0)) == 0
                # (0, 0, 1) -> 1
                assert ttl.flatten_core_index((0, 0, 1)) == 1
                # (0, 1, 0) -> 4 (0 * 3 * 4 + 1 * 4 + 0)
                assert ttl.flatten_core_index((0, 1, 0)) == 4
                # (0, 2, 3) -> 11 (0 * 3 * 4 + 2 * 4 + 3)
                assert ttl.flatten_core_index((0, 2, 3)) == 11
                # (1, 0, 0) -> 12 (1 * 3 * 4 + 0 * 4 + 0)
                assert ttl.flatten_core_index((1, 0, 0)) == 12
                # (1, 2, 3) -> 23 (1 * 3 * 4 + 2 * 4 + 3)
                assert ttl.flatten_core_index((1, 2, 3)) == 23

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_flatten_with_core_function(self):
        """Test flattening the result of core() function."""

        @ttl.kernel(grid=(3, 5), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Get 2D core coordinates
                core_2d = ttl.core(dims=2)
                # Get 1D core index
                core_1d = ttl.core(dims=1)

                # Flattening the 2D coordinates should equal the 1D index
                flattened = ttl.flatten_core_index(core_2d)
                assert flattened == core_1d

                # Flattening the already-linear index should return itself
                flattened_linear = ttl.flatten_core_index(core_1d)
                assert flattened_linear == core_1d

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_flatten_idempotent(self):
        """Test that flattening twice gives the same result."""

        @ttl.kernel(grid=(2, 4), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                core_2d = (1, 2)
                flat1 = ttl.flatten_core_index(core_2d)
                flat2 = ttl.flatten_core_index(flat1)

                # Should be the same (idempotent)
                assert flat1 == flat2
                assert flat1 == 6  # 1 * 4 + 2

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_flatten_different_grid_sizes(self):
        """Test flattening works correctly with different grid dimensions."""

        @ttl.kernel(grid=(10, 5), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Test with 10x5 grid
                # (0, 0) -> 0
                assert ttl.flatten_core_index((0, 0)) == 0
                # (1, 0) -> 5 (1 * 5 + 0)
                assert ttl.flatten_core_index((1, 0)) == 5
                # (5, 3) -> 28 (5 * 5 + 3)
                assert ttl.flatten_core_index((5, 3)) == 28
                # (9, 4) -> 49 (9 * 5 + 4)
                assert ttl.flatten_core_index((9, 4)) == 49

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)

    def test_flatten_returns_int_type(self):
        """Test that flatten_core_index always returns an int."""

        @ttl.kernel(grid=(2, 2), granularity=1)
        def test_kernel(a: torch.Tensor, b: torch.Tensor):
            assert a is not None and b is not None

            @ttl.compute()
            def compute_func():
                # Test with linear index
                result1 = ttl.flatten_core_index(3)
                assert isinstance(result1, int)

                # Test with 2D tuple
                result2 = ttl.flatten_core_index((1, 1))
                assert isinstance(result2, int)

            @ttl.datamovement()
            def dm0():
                pass

            @ttl.datamovement()
            def dm1():
                pass

            from python.sim import Program

            return Program(compute_func, dm0, dm1)()

        a = torch.zeros(ttl.TILE_SHAPE)
        b = torch.zeros(ttl.TILE_SHAPE)
        test_kernel(a, b)
