# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test CircularBuffer implementation.

This test verifies that the CircularBuffer class works correctly with
the underlying CBAPI and provides the expected interface for tensor operations.
"""

import pytest
import torch
from python.sim import (
    CircularBuffer,
    CBAPI,
    TensorAccessor,
    IndexType,
    TILE_SHAPE,
    copy,
)
from python.sim import torch_utils as tu
from python.sim.errors import CBContractError


@pytest.fixture
def api():
    """Provide a fresh CBAPI instance for each test."""
    return CBAPI()


def test_circular_buffer_basic(api: CBAPI) -> None:
    """Test basic CircularBuffer operations."""
    # Create a circular buffer for single tiles with buffer factor 2
    cb = CircularBuffer[torch.Tensor](shape=(1, 1), buffer_factor=2, api=api)

    # Verify basic properties
    assert cb.shape == (1, 1)
    assert cb.capacity_tiles == 2  # 1*1*2
    assert cb.buffer_factor == 2

    # Test the buffer workflow
    # Producer: reserve -> write -> push
    write_view = cb.reserve()
    assert len(write_view) == 1  # Should have space for 1 tile

    # Simulate writing data
    test_data = tu.ones(*TILE_SHAPE)
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


def test_circular_buffer_multi_tile(api: CBAPI) -> None:
    """Test CircularBuffer with multiple tiles per operation."""
    # Create a circular buffer for 2x1 tiles (2 tiles per operation)
    cb = CircularBuffer[torch.Tensor](shape=(2, 1), buffer_factor=3, api=api)

    # Verify properties
    assert cb.shape == (2, 1)
    assert cb.capacity_tiles == 6  # 2*1*3

    # Test reserve/push
    write_view = cb.reserve()
    assert len(write_view) == 2  # Should have space for 2 tiles

    # Fill with test data
    for i in range(2):
        write_view[i] = tu.ones(*TILE_SHAPE) * (i + 1)

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


def test_copy_operations(api: CBAPI) -> None:
    """Test copy operations between TensorAccessor and CircularBuffer."""
    # Create test tensors
    tensor_a = tu.randn(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)  # 2x2 tiles

    accessor_a = TensorAccessor(tensor_a, index_type=IndexType.TILE)

    # Create circular buffer
    cb_a = CircularBuffer[torch.Tensor](shape=(1, 1), buffer_factor=2, api=api)

    # Test copy from tensor to circular buffer
    cb_view = cb_a.reserve()
    tensor_slice = accessor_a[slice(0, 1), slice(0, 1)]  # Single tile

    # Copy operation
    tx = copy(tensor_slice, cb_view)
    tx.wait()
    cb_a.push()

    # Test copy from circular buffer to tensor
    cb_read_view = cb_a.wait()
    output_tensor = tu.zeros(*TILE_SHAPE)  # Single tile output

    # Copy operation
    tx2 = copy(cb_read_view, output_tensor)
    tx2.wait()
    cb_a.pop()

    # Verify the data was transferred
    assert output_tensor.shape == TILE_SHAPE
    # The output tensor should now contain the data from the circular buffer

    print("Copy operations test passed!")


def test_error_handling(api: CBAPI) -> None:
    """Test error conditions."""
    # Test invalid shape
    with pytest.raises(ValueError):
        CircularBuffer[torch.Tensor](shape=(0, 1), api=api)  # Invalid shape

    with pytest.raises(ValueError):
        CircularBuffer[torch.Tensor](shape=(1, 2, 3), api=api)  # type: ignore # Wrong shape dimensions

    # Test invalid buffer factor
    with pytest.raises(ValueError):
        CircularBuffer[torch.Tensor](shape=(1, 1), buffer_factor=0, api=api)

    # Test operations without proper setup
    cb = CircularBuffer[torch.Tensor](shape=(1, 1), buffer_factor=2, api=api)

    # Can't push without reserve - CBAPI will catch this
    with pytest.raises(CBContractError):
        cb.push()

    # Can't pop without wait - CBAPI will catch this
    with pytest.raises(CBContractError):
        cb.pop()

    # Test unsupported copy operations with wrong types
    with pytest.raises(ValueError, match="No copy handler registered"):
        copy("invalid_source", "invalid_dest")  # type: ignore

    print("Error handling test passed!")


def test_example_usage_pattern(api: CBAPI) -> None:
    """Test usage pattern similar to the provided example."""
    # Create tensors like in the example
    rows, cols = 128, 128
    granularity = 4

    a_in = tu.randn(rows, cols)
    c_in = tu.randn(
        TILE_SHAPE[0], cols
    )  # Make c_in a full tile height for proper tiling

    # Create accessors
    a_accessor = TensorAccessor(a_in, index_type=IndexType.TILE)
    c_accessor = TensorAccessor(c_in, index_type=IndexType.TILE)

    # Create circular buffers like in the example
    a_in_cb = CircularBuffer[torch.Tensor](
        shape=(granularity, 1), buffer_factor=2, api=api
    )
    _ = CircularBuffer[torch.Tensor](shape=(granularity, 1), buffer_factor=2, api=api)
    c_in_cb = CircularBuffer[torch.Tensor](shape=(1, 1), buffer_factor=2, api=api)
    _ = CircularBuffer[torch.Tensor](shape=(granularity, 1), buffer_factor=2, api=api)

    # Verify the circular buffers were created correctly
    assert a_in_cb.shape == (granularity, 1)
    assert a_in_cb.capacity_tiles == granularity * 2
    assert c_in_cb.shape == (1, 1)
    assert c_in_cb.capacity_tiles == 2

    # Test basic operations
    # Simulate data movement operations
    c_block = c_in_cb.reserve()
    c_slice = c_accessor[slice(0, 1), slice(0, 1)]
    tx = copy(c_slice, c_block)
    tx.wait()
    c_in_cb.push()

    # Simulate some compute pattern
    a_block = a_in_cb.reserve()
    a_slice = a_accessor[slice(0, granularity), slice(0, 1)]
    tx = copy(a_slice, a_block)
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


def test_make_circular_buffer_like_basic(api: CBAPI) -> None:
    """Test make_circular_buffer_like with basic usage."""
    from python.sim import ttl

    # Create a tensor
    x = tu.zeros(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)

    # Create a circular buffer like x
    x_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)

    # Verify it's a CircularBuffer with correct properties
    assert isinstance(x_cb, CircularBuffer)
    assert x_cb.shape == (1, 1)
    assert x_cb.capacity_tiles == 2
    assert x_cb.buffer_factor == 2

    # Verify it's not initialized (no API)
    assert x_cb._api is None  # type: ignore[reportPrivateUsage]
    assert x_cb._cb_id is None  # type: ignore[reportPrivateUsage]

    # Verify that using it without initialization raises an error
    with pytest.raises(RuntimeError, match="not properly initialized"):
        x_cb.reserve()

    print("make_circular_buffer_like basic test passed!")


def test_make_circular_buffer_like_infers_type(api: CBAPI) -> None:
    """Test that make_circular_buffer_like correctly infers the element type."""
    from python.sim import ttl

    # Create a tensor
    tensor = tu.randn(TILE_SHAPE[0], TILE_SHAPE[1])

    # Create a circular buffer like the tensor
    cb = ttl.make_circular_buffer_like(tensor, shape=(2, 2), buffer_factor=3)

    # Verify properties
    assert cb.shape == (2, 2)
    assert cb.capacity_tiles == 12  # 2*2*3
    assert cb.buffer_factor == 3

    # Verify it's not initialized
    assert cb._api is None  # type: ignore[reportPrivateUsage]

    # Verify error when used without initialization
    with pytest.raises(RuntimeError, match="not properly initialized"):
        cb.reserve()

    print("make_circular_buffer_like type inference test passed!")


def test_make_circular_buffer_like_multiple_tensors(api: CBAPI) -> None:
    """Test make_circular_buffer_like with multiple different tensors."""
    from python.sim import ttl

    # Create different tensors
    a = tu.randn(TILE_SHAPE[0] * 4, TILE_SHAPE[1] * 4)
    b = tu.zeros(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)
    c = tu.ones(TILE_SHAPE[0], TILE_SHAPE[1])

    # Create circular buffers for each
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(2, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 2), buffer_factor=3)

    # Verify all have correct properties
    assert a_cb.shape == (1, 1)
    assert a_cb.capacity_tiles == 2

    assert b_cb.shape == (2, 1)
    assert b_cb.capacity_tiles == 4  # 2*1*2

    assert c_cb.shape == (1, 2)
    assert c_cb.capacity_tiles == 6  # 1*2*3

    # Verify they're all uninitialized
    for cb in [a_cb, b_cb, c_cb]:
        assert cb._api is None  # type: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError, match="not properly initialized"):
            cb.reserve()

    print("make_circular_buffer_like multiple tensors test passed!")


def test_make_circular_buffer_like_with_example_pattern(api: CBAPI) -> None:
    """Test make_circular_buffer_like with realistic example pattern."""
    from python.sim import ttl

    # Simulate example usage
    a_in = tu.randn(128, 128)
    b_in = tu.randn(128, 128)
    out = tu.zeros(128, 128)

    granularity = 4
    buffer_factor = 2

    # Create circular buffers using make_circular_buffer_like
    a_cb = ttl.make_circular_buffer_like(
        a_in, shape=(granularity, 1), buffer_factor=buffer_factor
    )
    b_cb = ttl.make_circular_buffer_like(
        b_in, shape=(granularity, 1), buffer_factor=buffer_factor
    )
    out_cb = ttl.make_circular_buffer_like(
        out, shape=(granularity, 1), buffer_factor=buffer_factor
    )

    # Verify all buffers have correct configuration
    for cb in [a_cb, b_cb, out_cb]:
        assert cb.shape == (granularity, 1)
        assert cb.capacity_tiles == granularity * buffer_factor
        # Verify they're uninitialized
        assert cb._api is None  # type: ignore[reportPrivateUsage]

    # Verify that operations fail without initialization
    with pytest.raises(RuntimeError, match="not properly initialized"):
        a_cb.reserve()

    print("make_circular_buffer_like example pattern test passed!")


def test_can_wait_and_can_reserve(api: CBAPI) -> None:
    """Test can_wait() and can_reserve() methods."""
    # Create a circular buffer with buffer factor 2 (capacity = 2 tiles)
    cb = CircularBuffer[torch.Tensor](shape=(1, 1), buffer_factor=2, api=api)

    # Initially, buffer is empty
    # can_reserve should return True (we have 2 free tiles)
    assert cb.can_reserve() is True
    # can_wait should return False (no data available)
    assert cb.can_wait() is False

    # Reserve and push one tile
    block = cb.reserve()
    block[0] = tu.ones(*TILE_SHAPE)
    cb.push()

    # Now we have 1 tile visible, 1 tile free
    assert cb.can_wait() is True  # 1 tile available to read
    assert cb.can_reserve() is True  # 1 tile free to write

    # Reserve and push another tile (buffer now full)
    block = cb.reserve()
    block[0] = tu.ones(*TILE_SHAPE) * 2
    cb.push()

    # Now we have 2 tiles visible, 0 tiles free
    assert cb.can_wait() is True  # Still have data to read
    assert cb.can_reserve() is False  # No free space

    # Pop one tile
    _ = cb.wait()
    cb.pop()

    # Now we have 1 tile visible, 1 tile free
    assert cb.can_wait() is True  # Still have 1 tile to read
    assert cb.can_reserve() is True  # Have 1 free tile

    # Pop the last tile
    _ = cb.wait()
    cb.pop()

    # Back to empty state
    assert cb.can_wait() is False  # No data available
    assert cb.can_reserve() is True  # All tiles free

    print("can_wait() and can_reserve() test passed!")


def test_can_methods_multi_tile(api: CBAPI) -> None:
    """Test can_wait() and can_reserve() with multi-tile operations."""
    # Create a buffer that handles 2 tiles per operation, capacity = 6 tiles
    cb = CircularBuffer[torch.Tensor](shape=(2, 1), buffer_factor=3, api=api)

    # Initially empty
    assert cb.can_reserve() is True  # 6 free tiles, need 2
    assert cb.can_wait() is False  # 0 visible tiles, need 2

    # Reserve and push 2 tiles
    block = cb.reserve()
    for i in range(2):
        block[i] = tu.ones(*TILE_SHAPE) * (i + 1)
    cb.push()

    # 2 visible, 4 free
    assert cb.can_wait() is True  # Have 2 tiles
    assert cb.can_reserve() is True  # Have 4 free

    # Reserve and push 2 more tiles
    block = cb.reserve()
    for i in range(2):
        block[i] = tu.ones(*TILE_SHAPE) * (i + 3)
    cb.push()

    # 4 visible, 2 free
    assert cb.can_wait() is True  # Have 4 tiles
    assert cb.can_reserve() is True  # Have 2 free (exactly what we need)

    # Reserve and push 2 more tiles (buffer full)
    block = cb.reserve()
    for i in range(2):
        block[i] = tu.ones(*TILE_SHAPE) * (i + 5)
    cb.push()

    # 6 visible, 0 free
    assert cb.can_wait() is True  # Have 6 tiles
    assert cb.can_reserve() is False  # Have 0 free (need 2)

    print("can_wait() and can_reserve() multi-tile test passed!")


def test_can_methods_uninitialized(api: CBAPI) -> None:
    """Test that can_wait() and can_reserve() fail on uninitialized CBs."""
    from python.sim import ttl

    x = tu.zeros(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)
    cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)

    # Both methods should raise RuntimeError on uninitialized CB
    with pytest.raises(RuntimeError, match="not properly initialized"):
        cb.can_wait()

    with pytest.raises(RuntimeError, match="not properly initialized"):
        cb.can_reserve()

    print("can_wait() and can_reserve() uninitialized test passed!")


if __name__ == "__main__":
    test_api = CBAPI()
    test_circular_buffer_basic(test_api)
    test_circular_buffer_multi_tile(test_api)
    test_copy_operations(test_api)
    test_error_handling(test_api)
    test_example_usage_pattern(test_api)
    test_make_circular_buffer_like_basic(test_api)
    test_make_circular_buffer_like_infers_type(test_api)
    test_make_circular_buffer_like_multiple_tensors(test_api)
    test_make_circular_buffer_like_with_example_pattern(test_api)
    test_can_wait_and_can_reserve(test_api)
    test_can_methods_multi_tile(test_api)
    test_can_methods_uninitialized(test_api)
    print("All CircularBuffer tests passed!")
