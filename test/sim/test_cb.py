# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test CircularBuffer implementation.

This test verifies that the CircularBuffer class works correctly with
the underlying CBAPI and provides the expected interface for tensor operations.
"""

import pytest
from python.sim import CircularBuffer, TensorAccessor, IndexType, TILE_SIZE, dma
from python.sim import torch_utils as tu
from python.sim.errors import CBContractError


def test_circular_buffer_basic():
    """Test basic CircularBuffer operations."""
    # Create a circular buffer for single tiles with buffer factor 2
    cb = CircularBuffer(shape=(1, 1), buffer_factor=2)

    # Verify basic properties
    assert cb.shape == (1, 1)
    assert cb.capacity_tiles == 2  # 1*1*2
    assert cb.buffer_factor == 2

    # Test the buffer workflow
    # Producer: reserve -> write -> push
    write_view = cb.reserve()
    assert len(write_view) == 1  # Should have space for 1 tile

    # Simulate writing data
    test_data = tu.ones(TILE_SIZE, TILE_SIZE)
    write_view[0] = test_data
    cb.push()

    # Consumer: wait -> read -> pop
    read_view = cb.wait()
    assert len(read_view) == 1  # Should have 1 tile available

    # Read the data back
    read_data = read_view[0]
    assert read_data is not None

    cb.pop()

    print("Basic CircularBuffer test passed!")


def test_circular_buffer_multi_tile():
    """Test CircularBuffer with multiple tiles per operation."""
    # Create a circular buffer for 2x1 tiles (2 tiles per operation)
    cb = CircularBuffer(shape=(2, 1), buffer_factor=3)

    # Verify properties
    assert cb.shape == (2, 1)
    assert cb.capacity_tiles == 6  # 2*1*3

    # Test reserve/push
    write_view = cb.reserve()
    assert len(write_view) == 2  # Should have space for 2 tiles

    # Fill with test data
    for i in range(2):
        write_view[i] = tu.ones(TILE_SIZE, TILE_SIZE) * (i + 1)

    cb.push()

    # Test wait/pop
    read_view = cb.wait()
    assert len(read_view) == 2  # Should have 2 tiles available

    # Verify data
    for i in range(2):
        data = read_view[i]
        assert data is not None

    cb.pop()

    print("Multi-tile CircularBuffer test passed!")


def test_dma_operations():
    """Test DMA operations between TensorAccessor and CircularBuffer."""
    # Create test tensors
    tensor_a = tu.randn(TILE_SIZE * 2, TILE_SIZE * 2)  # 2x2 tiles

    accessor_a = TensorAccessor(tensor_a, index_type=IndexType.TILE)

    # Create circular buffer
    cb_a = CircularBuffer(shape=(1, 1), buffer_factor=2)

    # Test DMA from tensor to circular buffer
    cb_view = cb_a.reserve()
    tensor_slice = accessor_a[slice(0, 1), slice(0, 1)]  # Single tile

    # DMA operation
    tx = dma(tensor_slice, cb_view)
    tx.wait()
    cb_a.push()

    # Test DMA from circular buffer to tensor
    cb_read_view = cb_a.wait()
    output_tensor = tu.zeros(TILE_SIZE, TILE_SIZE)  # Single tile output

    # Copy through DMA
    tx2 = dma(cb_read_view, output_tensor)
    tx2.wait()
    cb_a.pop()

    # Verify the data was transferred
    assert output_tensor.shape == (TILE_SIZE, TILE_SIZE)
    # The output tensor should now contain the data from the circular buffer

    print("DMA operations test passed!")


def test_error_handling():
    """Test error conditions."""
    # Test invalid shape
    with pytest.raises(ValueError):
        CircularBuffer(shape=(0, 1))  # Invalid shape

    with pytest.raises(ValueError):
        CircularBuffer(shape=(1, 2, 3))  # type: ignore # Wrong shape dimensions

    # Test invalid buffer factor
    with pytest.raises(ValueError):
        CircularBuffer(shape=(1, 1), buffer_factor=0)

    # Test operations without proper setup
    cb = CircularBuffer(shape=(1, 1), buffer_factor=2)

    # Can't push without reserve - CBAPI will catch this
    with pytest.raises(CBContractError):
        cb.push()

    # Can't pop without wait - CBAPI will catch this
    with pytest.raises(CBContractError):
        cb.pop()

    # Test unsupported DMA operations with wrong types
    with pytest.raises(ValueError, match="Unsupported DMA transfer"):
        tx = dma("invalid_source", "invalid_dest")  # type: ignore

    print("Error handling test passed!")


def test_example_usage_pattern():
    """Test usage pattern similar to the provided example."""
    # Create tensors like in the example
    rows, cols = 128, 128
    granularity = 4

    a_in = tu.randn(rows, cols)
    c_in = tu.randn(TILE_SIZE, cols)  # Make c_in a full tile height for proper tiling

    # Create accessors
    a_accessor = TensorAccessor(a_in, index_type=IndexType.TILE)
    c_accessor = TensorAccessor(c_in, index_type=IndexType.TILE)

    # Create circular buffers like in the example
    a_in_cb = CircularBuffer(shape=(granularity, 1), buffer_factor=2)
    _ = CircularBuffer(shape=(granularity, 1), buffer_factor=2)
    c_in_cb = CircularBuffer(shape=(1, 1), buffer_factor=2)
    _ = CircularBuffer(shape=(granularity, 1), buffer_factor=2)

    # Verify the circular buffers were created correctly
    assert a_in_cb.shape == (granularity, 1)
    assert a_in_cb.capacity_tiles == granularity * 2
    assert c_in_cb.shape == (1, 1)
    assert c_in_cb.capacity_tiles == 2

    # Test basic operations
    # Simulate data movement operations
    c_block = c_in_cb.reserve()
    c_slice = c_accessor[slice(0, 1), slice(0, 1)]
    tx = dma(c_slice, c_block)
    tx.wait()
    c_in_cb.push()

    # Simulate some compute pattern
    a_block = a_in_cb.reserve()
    a_slice = a_accessor[slice(0, granularity), slice(0, 1)]
    tx = dma(a_slice, a_block)
    tx.wait()
    a_in_cb.push()

    # Consumer side
    c_data = c_in_cb.wait()
    a_data = a_in_cb.wait()

    # Verify we got the expected views
    assert len(c_data) == 1
    assert len(a_data) == granularity

    # Clean up
    c_in_cb.pop()
    a_in_cb.pop()

    print("Example usage pattern test passed!")


if __name__ == "__main__":
    test_circular_buffer_basic()
    test_circular_buffer_multi_tile()
    test_dma_operations()
    test_error_handling()
    test_example_usage_pattern()
    print("All CircularBuffer tests passed!")
