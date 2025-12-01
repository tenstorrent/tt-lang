# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for kernel.py module (kernel decorator, grid_size, etc.).
"""

import pytest
import torch
from python.sim import ttl, TILE_SHAPE


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
        a = torch.zeros(TILE_SHAPE)
        b = torch.zeros(TILE_SHAPE)

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
        a = torch.zeros(TILE_SHAPE)
        b = torch.zeros(TILE_SHAPE)

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
        a = torch.zeros(TILE_SHAPE)
        b = torch.zeros(TILE_SHAPE)

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
        a = torch.zeros(TILE_SHAPE)
        b = torch.zeros(TILE_SHAPE)

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
        a = torch.zeros(TILE_SHAPE)
        b = torch.zeros(TILE_SHAPE)

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
        a = torch.zeros(TILE_SHAPE)
        b = torch.zeros(TILE_SHAPE)

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
        a = torch.zeros(TILE_SHAPE)
        b = torch.zeros(TILE_SHAPE)

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
        a = torch.zeros(TILE_SHAPE)
        b = torch.zeros(TILE_SHAPE)

        # Should not raise
        test_kernel(a, b)
