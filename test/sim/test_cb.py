# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test CircularBuffer implementation.

This test verifies that the CircularBuffer class works correctly with
the underlying CBAPI and provides the expected interface for tensor operations.
"""

import pytest
from test_utils import (
    make_ones_tensor,
    make_ones_tile,
    make_rand_tensor,
    make_zeros_tensor,
    make_zeros_tile,
    tensors_equal,
)

from python.sim import TILE_SHAPE, copy, ttnn
from python.sim.block import ThreadType, _set_current_thread_type
from python.sim.cb import CircularBuffer
from python.sim.cbapi import CBAPI
from python.sim.errors import CBContractError


@pytest.fixture(autouse=True)
def setup_thread_context():
    """Automatically set thread context to COMPUTE for all CB tests.

    Note: These tests primarily exercise COMPUTE thread patterns (using store()).
    DM thread patterns (using copy operations) are tested separately in copy/pipe tests.
    The state machine enforces different expected operations for DM vs COMPUTE threads,
    so parametrizing these tests would require substantial test logic changes.
    """
    _set_current_thread_type(ThreadType.COMPUTE)
    yield
    _set_current_thread_type(None)  # Clean up


@pytest.fixture
def api():
    """Provide a fresh CBAPI instance for each test."""
    return CBAPI()


def test_circular_buffer_basic(api: CBAPI) -> None:
    """Test basic CircularBuffer operations."""
    # Create a circular buffer for single tiles with buffer factor 2
    element = make_ones_tile()
    cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

    # Verify basic properties
    assert cb.shape == (1, 1)
    assert cb.capacity_tiles == 2  # 1*1*2
    assert cb.buffer_factor == 2

    # Test the buffer workflow
    # Producer: reserve -> write -> push
    write_view = cb.reserve()
    assert len(write_view) == 1  # Should have space for 1 tile

    # Simulate writing data
    test_data = make_ones_tile()
    write_view.store([test_data])
    cb.push()

    # Consumer: wait -> read -> pop
    read_view = cb.wait()
    assert len(read_view) == 1  # Should have 1 tile available

    # Use waited block as source (STORE_SRC) before pop
    out_cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    out_block = out_cb.reserve()
    out_block.store(read_view)
    out_cb.push()

    # Verify data was transferred correctly
    read_data = read_view.to_list()
    assert read_data[0] is not None
    assert tensors_equal(read_data[0], test_data)

    cb.pop()

    print("Basic CircularBuffer test passed!")


def test_circular_buffer_multi_tile(api: CBAPI) -> None:
    """Test CircularBuffer with multiple tiles per operation."""
    # Create a circular buffer for 2x1 tiles (2 tiles per operation)
    element = make_ones_tile()
    cb = CircularBuffer(element=element, shape=(2, 1), buffer_factor=3, api=api)

    # Verify properties
    assert cb.shape == (2, 1)
    assert cb.capacity_tiles == 6  # 2*1*3

    # Test reserve/push
    write_view = cb.reserve()
    assert len(write_view) == 2  # Should have space for 2 tiles

    # Fill with test data
    tiles = []
    for i in range(2):
        tile = ttnn.rand(TILE_SHAPE)
        tile.to_torch().fill_(float(i + 1))
        tiles.append(tile)
    write_view.store(tiles)

    cb.push()

    # Test wait/pop
    read_view = cb.wait()
    assert len(read_view) == 2  # Should have 2 tiles available

    # Use waited block as source (STORE_SRC) before pop
    out_cb = CircularBuffer(element=element, shape=(2, 1), buffer_factor=2, api=api)
    out_block = out_cb.reserve()
    out_block.store(read_view)
    out_cb.push()

    # Verify data was transferred correctly
    read_data = read_view.to_list()
    for i in range(2):
        assert read_data[i] is not None
        expected_value = float(i + 1)
        actual_value = read_data[i].to_torch()[0, 0].item()
        assert (
            abs(actual_value - expected_value) < 1e-5
        ), f"Tile {i}: expected {expected_value}, got {actual_value}"

    cb.pop()

    print("Multi-tile CircularBuffer test passed!")


def test_copy_operations_with_dm_context(api: CBAPI) -> None:
    """Test copy operations between tensor and CircularBuffer with proper DM thread context.

    This replaces the old test_copy_operations that was disabled due to lack of thread context.
    """
    from python.sim.block import (
        _set_current_thread_type,
        _clear_current_thread_type,
        ThreadType,
    )

    # Set DM thread context (required for copy operations)
    _set_current_thread_type(ThreadType.DM)

    try:
        # Create test tensors
        tensor_a = make_rand_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)  # 2x2 tiles

        # Create circular buffer
        element = make_ones_tile()
        cb_a = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

        # Test copy from tensor to circular buffer (DM thread can do this)
        cb_view = cb_a.reserve()
        tensor_slice = tensor_a[0:1, 0:1]  # Single tile

        # Copy operation
        tx = copy(tensor_slice, cb_view)
        tx.wait()
        cb_a.push()

        # Test copy from circular buffer back to tensor
        cb_read_view = cb_a.wait()
        output_tensor = make_zeros_tile()  # Single tile output

        # Copy operation
        tx2 = copy(cb_read_view, output_tensor)
        tx2.wait()
        cb_a.pop()

        # Verify the data was transferred
        assert output_tensor.shape == TILE_SHAPE
        # The output tensor should now contain the data from the circular buffer
        # Verify at least some data was copied (non-zero)
        import torch

        assert output_tensor.to_torch().sum() != 0

    finally:
        # Clean up thread context
        _clear_current_thread_type()

    print("Copy operations with DM context test passed!")


def test_error_handling(api: CBAPI) -> None:
    """Test error conditions."""
    # Test invalid shape
    element = make_ones_tile()
    with pytest.raises(ValueError):
        CircularBuffer(element=element, shape=(0, 1), api=api)  # Invalid shape

    with pytest.raises(ValueError):
        CircularBuffer(element=element, shape=(1, 2, 3), api=api)  # type: ignore # Wrong shape dimensions

    # Test invalid buffer factor
    with pytest.raises(ValueError):
        CircularBuffer(element=element, shape=(1, 1), buffer_factor=0, api=api)

    # Test operations without proper setup
    cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

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


def test_copy_in_dm_thread_context(api: CBAPI) -> None:
    """Test copy operations with proper DM thread context.

    This test demonstrates the full workflow:
    - DM thread: copy data into CBs (reserve + copy + push)
    - Switch to COMPUTE thread for consumption (wait + read + pop)
    """
    from python.sim.block import (
        _set_current_thread_type,
        _clear_current_thread_type,
        ThreadType,
    )

    try:
        # Create tensors
        rows, cols = 128, 128
        granularity = 4

        a_in = make_rand_tensor(rows, cols)
        c_in = make_rand_tensor(TILE_SHAPE[0], cols)

        # Create circular buffers
        element = make_ones_tile()
        a_in_cb = CircularBuffer(
            element=element, shape=(granularity, 1), buffer_factor=2, api=api
        )
        c_in_cb = CircularBuffer(
            element=element, shape=(1, 1), buffer_factor=2, api=api
        )

        # Verify the circular buffers were created correctly
        assert a_in_cb.shape == (granularity, 1)
        assert a_in_cb.capacity_tiles == granularity * 2
        assert c_in_cb.shape == (1, 1)
        assert c_in_cb.capacity_tiles == 2

        # DM thread: Producer side - copy data into CBs
        _set_current_thread_type(ThreadType.DM)

        # Copy c_in data
        c_block = c_in_cb.reserve()
        c_slice = c_in[0:1, 0:1]
        tx = copy(c_slice, c_block)
        tx.wait()
        c_in_cb.push()

        # Copy a_in data
        a_block = a_in_cb.reserve()
        a_slice = a_in[0:granularity, 0:1]
        tx = copy(a_slice, a_block)
        tx.wait()
        a_in_cb.push()

        # Switch to COMPUTE thread: Consumer side - read data back
        _set_current_thread_type(ThreadType.COMPUTE)

        c_data = c_in_cb.wait()
        a_data = a_in_cb.wait()

        # Verify we got the expected views
        assert len(c_data) == 1
        assert len(a_data) == granularity

        # Verify data was copied correctly
        c_list = c_data.to_list()
        a_list = a_data.to_list()
        assert c_list[0] is not None
        assert a_list[0] is not None

        # In COMPUTE thread, wait() blocks must be used as STORE_SRC before pop
        out_cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
        out_block = out_cb.reserve()
        out_block.store(c_data)
        out_cb.push()
        c_in_cb.pop()

        out_cb2 = CircularBuffer(
            element=element, shape=(granularity, 1), buffer_factor=2, api=api
        )
        out_block2 = out_cb2.reserve()
        out_block2.store(a_data)
        out_cb2.push()
        a_in_cb.pop()

    finally:
        # Clean up thread context
        _clear_current_thread_type()

    print("Copy in DM thread context test passed!")


def test_single_pending_reserve_constraint(api: CBAPI) -> None:
    """Test that only one reserve() is allowed before push()."""
    from python.sim.block import _set_current_thread_type, ThreadType
    from python.sim.copy import copy

    _set_current_thread_type(ThreadType.DM)

    try:
        element = make_ones_tile()
        cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

        # Create a source tensor for copy operations
        src_tensor = make_ones_tile()

        # First reserve() should succeed
        block1 = cb.reserve()
        assert block1 is not None

        # Second reserve() before push() should fail
        with pytest.raises(
            RuntimeError, match="Cannot call reserve\\(\\) again before push\\(\\)"
        ):
            cb.reserve()

        # Complete the copy operation and push to get to PUSH state
        tx = copy(src_tensor, block1)
        tx.wait()

        # After push(), should be able to reserve() again
        cb.push()
        block2 = cb.reserve()
        assert block2 is not None

        # Complete second block's operations
        tx = copy(src_tensor, block2)
        tx.wait()
        cb.push()
    finally:
        from python.sim.block import _clear_current_thread_type

        _clear_current_thread_type()


def test_single_pending_wait_constraint(api: CBAPI) -> None:
    """Test that only one wait() is allowed before pop()."""
    from python.sim.block import _set_current_thread_type, ThreadType
    from python.sim.copy import copy

    _set_current_thread_type(ThreadType.COMPUTE)

    try:
        element = make_ones_tile()
        cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

        # First populate the CB with data (using DM thread)
        _set_current_thread_type(ThreadType.DM)
        block = cb.reserve()
        test_data = make_rand_tensor(TILE_SHAPE[0], TILE_SHAPE[1])
        test_slice = test_data[0:1, 0:1]
        tx = copy(test_slice, block)
        tx.wait()
        cb.push()

        # Switch to COMPUTE thread for consumption
        _set_current_thread_type(ThreadType.COMPUTE)

        # First wait() should succeed
        data1 = cb.wait()
        assert data1 is not None

        # Second wait() before pop() should fail
        with pytest.raises(
            RuntimeError, match="Cannot call wait\\(\\) again before pop\\(\\)"
        ):
            cb.wait()

        # After pop(), should be able to wait() again (if there's more data)
        # Use waited block as STORE_SRC before pop
        out_cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
        out_block = out_cb.reserve()
        out_block.store(data1)
        out_cb.push()
        cb.pop()

        # Add more data (using DM thread)
        _set_current_thread_type(ThreadType.DM)
        block = cb.reserve()
        tx = copy(test_slice, block)
        tx.wait()
        cb.push()

        _set_current_thread_type(ThreadType.COMPUTE)
        data2 = cb.wait()
        assert data2 is not None
        # Use second waited block as STORE_SRC before pop
        out_block2 = out_cb.reserve()
        out_block2.store(data2)
        out_cb.push()
        cb.pop()
    finally:
        from python.sim.block import _clear_current_thread_type

        _clear_current_thread_type()


def test_reserve_store_push_pop_workflow(api: CBAPI) -> None:
    """Test the complete reserve->store->push->wait->pop workflow.

    This tests the primary usage pattern for compute operations without
    using copy (which requires DM thread context).
    """
    import torch
    from python.sim import ttnn, TILE_SHAPE

    # Create circular buffer
    element = make_zeros_tile()
    cb = CircularBuffer(element=element, shape=(2, 1), buffer_factor=2, api=api)

    # Producer: reserve -> store -> push
    with cb.reserve() as write_block:
        # Create test data
        data = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 5.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 10.0)),
        ]
        write_block.store(data)

    # Consumer: wait -> read -> pop
    out_cb = CircularBuffer(element=element, shape=(2, 1), buffer_factor=4, api=api)
    with cb.wait() as read_block:
        # Use waited block as STORE_SRC before context exit
        out_block = out_cb.reserve()
        out_block.store(read_block)
        out_cb.push()

    # Test multiple iterations
    for i in range(3):
        with cb.reserve() as write_block:
            data = [
                ttnn.Tensor(torch.full(TILE_SHAPE, float(i * 2))),
                ttnn.Tensor(torch.full(TILE_SHAPE, float(i * 2 + 1))),
            ]
            write_block.store(data)

        with cb.wait() as read_block:
            # Use waited block as STORE_SRC before context exit
            out_block = out_cb.reserve()
            out_block.store(read_block)
            out_cb.push()

            # Verify data correctness for this iteration
            read_data = read_block.to_list()
            assert read_data[0].to_torch()[0, 0].item() == float(i * 2)
            assert read_data[1].to_torch()[0, 0].item() == float(i * 2 + 1)

    print("Reserve-store-push-pop workflow test passed!")


def test_make_circular_buffer_like_basic(api: CBAPI) -> None:
    """Test make_circular_buffer_like with basic usage."""
    from python.sim import ttl

    # Create a tensor
    x = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)

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
    tensor = make_rand_tensor(TILE_SHAPE[0], TILE_SHAPE[1])

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
    a = make_rand_tensor(TILE_SHAPE[0] * 4, TILE_SHAPE[1] * 4)
    b = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)
    c = make_ones_tensor(TILE_SHAPE[0], TILE_SHAPE[1])

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
    a_in = make_rand_tensor(128, 128)
    b_in = make_rand_tensor(128, 128)
    out = make_zeros_tensor(128, 128)

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
    element = make_ones_tile()
    cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

    # Initially, buffer is empty
    # can_reserve should return True (we have 2 free tiles)
    assert cb.can_reserve() is True
    # can_wait should return False (no data available)
    assert cb.can_wait() is False

    # Reserve and push one tile
    block = cb.reserve()
    block.store([make_ones_tile()])
    cb.push()

    # Now we have 1 tile visible, 1 tile free
    assert cb.can_wait() is True  # 1 tile available to read
    assert cb.can_reserve() is True  # 1 tile free to write

    # Reserve and push another tile (buffer now full)
    block = cb.reserve()
    tile = ttnn.rand(TILE_SHAPE)
    tile.to_torch().fill_(2.0)
    block.store([tile])
    cb.push()

    # Now we have 2 tiles visible, 0 tiles free
    assert cb.can_wait() is True  # Still have data to read
    assert cb.can_reserve() is False  # No free space

    # Wait for the first tile
    out_cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    read1 = cb.wait()

    # After wait(), we have 1 tile read-locked, 1 tile still visible, 0 tiles free
    assert cb.can_wait() is True  # Can still wait for the second visible tile
    assert cb.can_reserve() is False  # No free tiles (both occupied)

    # Pop the first tile - use waited block as STORE_SRC first
    out_block = out_cb.reserve()
    out_block.store(read1)
    out_cb.push()
    cb.pop()

    # Now we have 1 tile visible, 1 tile free
    assert cb.can_wait() is True  # Still have 1 tile to read
    assert cb.can_reserve() is True  # Have 1 free tile

    # Pop the last tile - use waited block as STORE_SRC first
    read2 = cb.wait()
    out_block2 = out_cb.reserve()
    out_block2.store(read2)
    out_cb.push()
    cb.pop()

    # Back to empty state
    assert cb.can_wait() is False  # No data available
    assert cb.can_reserve() is True  # All tiles free

    print("can_wait() and can_reserve() test passed!")


def test_can_methods_multi_tile(api: CBAPI) -> None:
    """Test can_wait() and can_reserve() with multi-tile operations."""
    # Create a buffer that handles 2 tiles per operation, capacity = 6 tiles
    element = make_ones_tile()
    cb = CircularBuffer(element=element, shape=(2, 1), buffer_factor=3, api=api)

    # Initially empty
    assert cb.can_reserve() is True  # 6 free tiles, need 2
    assert cb.can_wait() is False  # 0 visible tiles, need 2

    # Reserve and push 2 tiles
    block = cb.reserve()
    tiles = []
    for i in range(2):
        tile = ttnn.rand(TILE_SHAPE)
        tile.to_torch().fill_(float(i + 1))
        tiles.append(tile)
    block.store(tiles)
    cb.push()

    # 2 visible, 4 free
    assert cb.can_wait() is True  # Have 2 tiles
    assert cb.can_reserve() is True  # Have 4 free

    # Reserve and push 2 more tiles
    block = cb.reserve()
    tiles = []
    for i in range(2):
        tile = ttnn.rand(TILE_SHAPE)
        tile.to_torch().fill_(float(i + 3))
        tiles.append(tile)
    block.store(tiles)
    cb.push()

    # 4 visible, 2 free
    assert cb.can_wait() is True  # Have 4 tiles
    assert cb.can_reserve() is True  # Have 2 free (exactly what we need)

    # Reserve and push 2 more tiles (buffer full)
    block = cb.reserve()
    tiles = []
    for i in range(2):
        tile = ttnn.rand(TILE_SHAPE)
        tile.to_torch().fill_(float(i + 5))
        tiles.append(tile)
    block.store(tiles)
    cb.push()

    # 6 visible, 0 free
    assert cb.can_wait() is True  # Have 6 tiles
    assert cb.can_reserve() is False  # Have 0 free (need 2)

    print("can_wait() and can_reserve() multi-tile test passed!")


def test_can_methods_uninitialized(api: CBAPI) -> None:
    """Test that can_wait() and can_reserve() fail on uninitialized CBs."""
    from python.sim import ttl

    x = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)
    cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)

    # Both methods should raise RuntimeError on uninitialized CB
    with pytest.raises(RuntimeError, match="not properly initialized"):
        cb.can_wait()

    with pytest.raises(RuntimeError, match="not properly initialized"):
        cb.can_reserve()

    print("can_wait() and can_reserve() uninitialized test passed!")


def test_context_manager_syntax(api: CBAPI) -> None:
    """Test the context manager (with statement) syntax for reserve and wait."""
    # Create a circular buffer
    element = make_ones_tile()
    cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

    # Test reserve with context manager
    test_data = make_ones_tile()
    with cb.reserve() as write_view:
        write_view.store([test_data])
        # push() is automatically called on exit

    # Test wait with context manager
    out_cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    with cb.wait() as read_view:
        # Use waited block as STORE_SRC before pop() is automatically called on exit
        out_block = out_cb.reserve()
        out_block.store(read_view)
        out_cb.push()
        # pop() is automatically called on exit

    # Verify that we can still use the old style (backward compatibility)
    write_view2 = cb.reserve()
    write_view2.store([make_zeros_tile()])
    cb.push()

    read_view2 = cb.wait()
    # Use waited block as STORE_SRC before pop
    out_block2 = out_cb.reserve()
    out_block2.store(read_view2)
    out_cb.push()
    cb.pop()

    # Test with multiple context managers on same line
    cb.reset()  # Reset to clean state

    # Write data first
    with cb.reserve() as w1:
        w1.store([make_ones_tile()])

    # Create another CB for multi-context test
    cb2 = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    with cb2.reserve() as w2:
        w2.store([make_zeros_tile()])

    # Test multiple wait contexts (simulating the matmul pattern)
    out_cb3 = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    out_cb4 = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    with cb.wait() as r1, cb2.wait() as r2:
        # Use waited blocks as STORE_SRC before context managers exit and call pop
        out_block3 = out_cb3.reserve()
        out_block3.store(r1)
        out_cb3.push()
        out_block4 = out_cb4.reserve()
        out_block4.store(r2)
        out_cb4.push()

        # Verify data correctness
        d1 = r1.to_list()[0]
        d2 = r2.to_list()[0]
        assert d1 is not None
        assert d2 is not None
        # Verify shape and type
        assert d1.to_torch().shape == (32, 32)
        assert d2.to_torch().shape == (32, 32)

    print("Context manager syntax test passed!")


def test_store_accumulate_first_assigns(api: CBAPI) -> None:
    """Test that the first store(acc=True) assigns instead of accumulates."""
    element = make_zeros_tile()
    cb = CircularBuffer(element=element, shape=(3, 1), buffer_factor=2, api=api)

    with cb.reserve() as block:
        # Create test values
        import torch
        from python.sim import ttnn, TILE_SHAPE

        values1 = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 5.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 10.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 15.0)),
        ]

        # First store(acc=True) - should assign (y = x), not accumulate (y += x)
        block.store(values1, acc=True)

        # Second store(acc=True) - should accumulate (y += x)
        values2 = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 3.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 6.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 9.0)),
        ]
        block.store(values2, acc=True)

        # Verify results using to_list()
        result = block.to_list()
        assert result[0].to_torch()[0, 0].item() == 8.0  # 5 + 3
        assert result[1].to_torch()[0, 0].item() == 16.0  # 10 + 6
        assert result[2].to_torch()[0, 0].item() == 24.0  # 15 + 9

    print("Store accumulate first assigns test passed!")


def test_store_accumulate_vs_regular_store(api: CBAPI) -> None:
    """Test that regular store() and store(acc=True) have different paths."""
    element = make_zeros_tile()
    cb = CircularBuffer(element=element, shape=(2, 1), buffer_factor=2, api=api)

    import torch
    from python.sim import ttnn, TILE_SHAPE

    # Test 1: Regular store() followed by push (cannot use store(acc=True) after)
    with cb.reserve() as block1:
        values = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 7.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 14.0)),
        ]
        block1.store(values)  # Regular store

    # Verify we can read it back
    out_cb = CircularBuffer(element=element, shape=(2, 1), buffer_factor=2, api=api)
    with cb.wait() as block_read:
        # Use waited block as STORE_SRC before context exit
        out_block = out_cb.reserve()
        out_block.store(block_read)
        out_cb.push()

    # Test 2: store(acc=True) path - can be called multiple times
    with cb.reserve() as block2:
        values1 = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 2.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 4.0)),
        ]
        block2.store(values1, acc=True)  # First: assigns

        values2 = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 3.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 6.0)),
        ]
        block2.store(values2, acc=True)  # Second: accumulates

        # Verify accumulation worked: 2+3=5, 4+6=10
        result = block2.to_list()
        assert result[0].to_torch()[0, 0].item() == 5.0
        assert result[1].to_torch()[0, 0].item() == 10.0

    print("Store accumulate vs regular store test passed!")


def test_block_state_machine_restrictions(api: CBAPI) -> None:
    """Test that block state machine enforces access restrictions."""
    element = make_zeros_tile()
    cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

    import torch
    from python.sim import ttnn, TILE_SHAPE

    # Test: Cannot index blocks - block indexing is not allowed
    block = cb.reserve()

    # Attempting to index block should fail
    with pytest.raises(RuntimeError, match="Block indexing.*not allowed"):
        _ = block[0]

    # Store makes it RO (for regular store) or RW (for acc store)
    values = [ttnn.Tensor(torch.full(TILE_SHAPE, 5.0))]
    block.store(values, acc=True)

    cb.push()

    # Test: Cannot write to RO (Read-Only) state after wait()
    read_block = cb.wait()

    # Cannot write - wait() blocks expect STORE_SRC, not STORE
    with pytest.raises(RuntimeError, match="Impossible.*Invalid state for store"):
        read_block.store([ttnn.Tensor(torch.full(TILE_SHAPE, 10.0))])

    # Use waited block as STORE_SRC before pop
    out_cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    out_block = out_cb.reserve()
    out_block.store(read_block)
    out_cb.push()
    cb.pop()

    print("Block state machine restrictions test passed!")


def test_copy_sets_block_to_na_state(api: CBAPI) -> None:
    """Test that copy operations set blocks to NA (No Access) state."""
    from python.sim.block import (
        Block,
        BlockAcquisition,
        ThreadType,
        _set_current_thread_type,
    )
    from python.sim.typedefs import Span
    import torch
    from python.sim import ttnn

    # Set thread type to DM (required for copy operations)
    _set_current_thread_type(ThreadType.DM)

    try:
        # Create a block manually in DM thread context
        buf = [None, None]
        block = Block(
            buf, 2, Span(0, 2), (2, 1), BlockAcquisition.RESERVE, ThreadType.DM
        )

        # Create source tensor
        source_tensor = ttnn.Tensor(torch.ones((64, 32)))

        # Start copy - block should transition to NA state
        tx = copy(source_tensor, block)

        # Cannot read or write while copy is in progress (NAW state)
        # But also, block indexing is not allowed regardless of state
        with pytest.raises(RuntimeError, match="Block indexing.*not allowed"):
            _ = block[0]

        with pytest.raises(
            RuntimeError, match="Cannot write to Block.*copy lock error.*NAW"
        ):
            from python.sim import TILE_SHAPE

            # Need 2 items for block with span length 2
            block.store(
                [
                    ttnn.Tensor(torch.full(TILE_SHAPE, 5.0)),
                    ttnn.Tensor(torch.full(TILE_SHAPE, 6.0)),
                ]
            )

        # After tx.wait(), block becomes RW (can do more operations)
        tx.wait()

        # Block indexing is not allowed regardless of state
    finally:
        # Clean up thread context
        from python.sim.block import _clear_current_thread_type

        _clear_current_thread_type()

    print("Copy sets block to NA state test passed!")


def test_push_validates_expected_state(api: CBAPI) -> None:
    """Test that push() validates the block is in a valid state before completing.

    This test verifies that push() can only be called on reserve() blocks
    (not wait() blocks) and only when PUSH is in the expected operations.
    """
    from python.sim.block import (
        Block,
        BlockAcquisition,
        ThreadType,
        ExpectedOp,
        _set_current_thread_type,
    )

    _set_current_thread_type(ThreadType.COMPUTE)

    try:
        element = make_ones_tile()
        cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

        # Create a block in WAIT state (POP expected)
        # First, populate the CB
        _set_current_thread_type(ThreadType.DM)
        from python.sim.copy import copy

        src = make_ones_tile()
        blk = cb.reserve()
        tx = copy(src, blk)
        tx.wait()
        cb.push()

        # Now wait for it in COMPUTE thread
        _set_current_thread_type(ThreadType.COMPUTE)
        waited_context = cb.wait()
        waited_block = waited_context.block()

        # Try to call push() on a wait() block - should fail
        # because waited_block is WAIT acquisition, not RESERVE
        with pytest.raises(
            RuntimeError,
            match="Cannot perform push\\(\\): Expected RESERVE acquisition, got WAIT",
        ):
            # Manually try to mark push (bypassing CB's push method which checks pending_reserved_block)
            waited_block.mark_push_complete()

        # Clean up properly - use waited block as STORE_SRC before pop
        out_cb = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
        out_block = out_cb.reserve()
        out_block.store(waited_context)
        out_cb.push()
        cb.pop()

        print("Push validates expected state test passed!")
    finally:
        from python.sim.block import _clear_current_thread_type

        _clear_current_thread_type()


if __name__ == "__main__":
    test_api = CBAPI()
    test_circular_buffer_basic(test_api)
    test_circular_buffer_multi_tile(test_api)
    test_copy_operations_with_dm_context(test_api)
    test_error_handling(test_api)
    test_copy_in_dm_thread_context(test_api)
    test_reserve_store_push_pop_workflow(test_api)
    test_make_circular_buffer_like_basic(test_api)
    test_make_circular_buffer_like_infers_type(test_api)
    test_make_circular_buffer_like_multiple_tensors(test_api)
    test_make_circular_buffer_like_with_example_pattern(test_api)
    test_can_wait_and_can_reserve(test_api)
    test_can_methods_multi_tile(test_api)
    test_can_methods_uninitialized(test_api)
    test_context_manager_syntax(test_api)
    test_store_accumulate_first_assigns(test_api)
    test_store_accumulate_vs_regular_store(test_api)
    test_block_state_machine_restrictions(test_api)
    test_copy_sets_block_to_na_state(test_api)
    print("All CircularBuffer tests passed!")
