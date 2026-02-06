# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from typing import List, Tuple

import pytest
from test_utils import make_full_tensor, tensors_exact_equal

from python.sim.block import ThreadType, _set_current_thread_type
from python.sim.cb import CircularBuffer
from python.sim.cbapi import CBAPI
from python.sim.cbstate import CBSlot
from python.sim.errors import CBContractError, CBTimeoutError
from python.sim.typedefs import CBID


# Pytest fixtures to reduce redundant setup code
@pytest.fixture(autouse=True)
def setup_thread_context():
    """Automatically set thread context for all CBAPI tests.

    CBAPI is thread-agnostic - it just needs some thread context set to create blocks.
    We use COMPUTE here, but the actual thread type doesn't affect CBAPI behavior.
    State machine validation happens at the Block level, which is tested separately.
    """
    _set_current_thread_type(ThreadType.COMPUTE)
    yield
    _set_current_thread_type(None)  # Clean up


@pytest.fixture
def api() -> CBAPI:
    """Create a fresh CBAPI instance for each test."""
    return CBAPI()


@pytest.fixture
def configured_cb(api: CBAPI) -> Tuple[CBAPI, CBID]:
    """Create a configured CB with capacity 4."""
    cb_id = 0
    api.host_configure_cb(cb_id, 4, shape=(1, 1))
    return api, cb_id


@pytest.fixture
def configured_cb8(api: CBAPI) -> Tuple[CBAPI, CBID]:
    """Create a configured CB with capacity 8."""
    cb_id = 0
    api.host_configure_cb(cb_id, 8, shape=(1, 1))
    return api, cb_id


@pytest.fixture
def timeout_api() -> CBAPI:
    """Create a CBAPI instance with short timeout for timeout tests."""
    return CBAPI(timeout=0.1)


def test_circular_buffer_basic_flow(configured_cb8: Tuple[CBAPI, CBID]):
    api, cb0 = configured_cb8
    stats = api.cb_stats(cb0)
    assert stats.capacity == 8
    assert stats.visible == 0

    # Reserve and write 4 tiles
    api.cb_reserve_back(cb0, 4)
    ptr = api.get_write_ptr(cb0)
    test_tensors = [make_full_tensor(32, 32, i + 1.0) for i in range(4)]
    ptr.store(test_tensors)
    api.cb_push_back(cb0, 4)
    stats = api.cb_stats(cb0)
    assert stats.visible == 4
    assert stats.free == 4

    # Wait and read
    api.cb_wait_front(cb0, 4)
    read_values = api.get_read_ptr(cb0).to_list()
    assert len(read_values) == 4
    for i in range(4):
        val = read_values[i]
        assert val is not None
        assert tensors_exact_equal(val, test_tensors[i])
    api.cb_pop_front(cb0, 4)
    stats = api.cb_stats(cb0)
    assert stats.visible == 0

    # Reserve full capacity and write
    api.cb_reserve_back(cb0, 8)
    ptr = api.get_write_ptr(cb0)
    test_tensors = [make_full_tensor(32, 32, float(i)) for i in range(8)]
    ptr.store(test_tensors)
    api.cb_push_back(cb0, 8)
    stats = api.cb_stats(cb0)
    assert stats.visible == 8

    # Cumulative wait and read
    api.cb_wait_front(cb0, 4)
    api.cb_wait_front(cb0, 8)
    read_values = api.get_read_ptr(cb0).to_list()
    assert len(read_values) == 8
    for i in range(8):
        val = read_values[i]
        assert val is not None
        assert tensors_exact_equal(val, test_tensors[i])
    api.cb_pop_front(cb0, 8)
    stats = api.cb_stats(cb0)
    assert stats.visible == 0


def test_per_instance_timeout_effect():
    # consumer should timeout based on instance timeout
    api = CBAPI(timeout=0.2)
    cb = 3
    api.host_configure_cb(cb, 4, shape=(1, 1))
    start = time.perf_counter()
    with pytest.raises(CBTimeoutError, match="timed out after 0.2s"):
        api.cb_wait_front(cb, 1)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.4


def test_threaded_produce_consume(configured_cb: Tuple[CBAPI, CBID]):
    api, cb0 = configured_cb
    result: List[List[CBSlot]] = []

    def consumer():
        api.cb_wait_front(cb0, 4)
        result.append(api.get_read_ptr(cb0).to_list())
        api.cb_pop_front(cb0, 4)

    t = threading.Thread(target=consumer)
    t.start()

    # Give consumer thread time to block in wait_front
    time.sleep(0.5)

    # Producer reserves and writes
    api.cb_reserve_back(cb0, 4)
    ptr = api.get_write_ptr(cb0)
    test_tensors = [make_full_tensor(32, 32, 100.0 + i) for i in range(4)]
    ptr.store(test_tensors)
    api.cb_push_back(cb0, 4)
    t.join(timeout=1)

    # Verify results
    assert len(result) == 1
    assert len(result[0]) == 4
    for i in range(4):
        val = result[0][i]
        assert val is not None  # Type narrowing: CBSlot -> Tensor
        assert tensors_exact_equal(val, test_tensors[i])


def test_cb_pages_nonblocking(configured_cb8: Tuple[CBAPI, CBID]):
    api, cb2 = configured_cb8

    # No pages initially; test non-error behavior
    assert not api.cb_pages_available_at_front(cb2, 1)
    assert api.cb_pages_reservable_at_back(cb2, 8)

    # After partial reserve, free decreases
    api.cb_reserve_back(cb2, 4)
    # free = 4, so reservable for 4 but not for 5
    assert api.cb_pages_reservable_at_back(cb2, 4)

    # After initial reserve of 4, push data to make pages available
    ptr = api.get_write_ptr(cb2)
    test_tensors = [make_full_tensor(32, 32, i + 1.0) for i in range(4)]
    ptr.store(test_tensors)
    api.cb_push_back(cb2, 4)
    # Divisible sizes: 4 and 2 are both valid
    assert api.cb_pages_available_at_front(cb2, 4)
    assert api.cb_pages_available_at_front(cb2, 2)

    # After pop, availability resets
    api.cb_wait_front(cb2, 4)
    api.cb_pop_front(cb2, 4)
    assert not api.cb_pages_available_at_front(cb2, 1)


# Focused error tests for page operations
def test_cb_pages_available_out_of_range_error(configured_cb: Tuple[CBAPI, CBID]):
    api, cb = configured_cb
    with pytest.raises(CBContractError, match="num_tiles must be <= capacity"):
        api.cb_pages_available_at_front(cb, 5)


def test_cb_pages_reservable_out_of_range_error(configured_cb: Tuple[CBAPI, CBID]):
    api, cb = configured_cb
    with pytest.raises(CBContractError, match="num_tiles must be <= capacity"):
        api.cb_pages_reservable_at_back(cb, 5)


def test_cb_pages_reservable_divisibility_error(
    configured_cb8: Tuple[CBAPI, CBID],
):
    api, cb = configured_cb8
    with pytest.raises(
        CBContractError, match="First num_tiles=5 must evenly divide capacity=8"
    ):
        api.cb_pages_reservable_at_back(cb, 5)


def test_cb_pages_available_divisibility_error(configured_cb8: Tuple[CBAPI, CBID]):
    api, cb = configured_cb8
    api.cb_reserve_back(cb, 4)
    ptr = api.get_write_ptr(cb)
    test_tensors = [make_full_tensor(32, 32, i + 1.0) for i in range(4)]
    ptr.store(test_tensors)
    api.cb_push_back(cb, 4)
    with pytest.raises(
        CBContractError, match="First num_tiles=3 must evenly divide capacity=8"
    ):
        api.cb_pages_available_at_front(cb, 3)


# Pointer requirement error tests
def test_get_read_ptr_requires_wait(configured_cb: Tuple[CBAPI, CBID]):
    api, cb = configured_cb
    with pytest.raises(
        CBContractError, match="get_read_ptr requires prior cb_wait_front"
    ):
        api.get_read_ptr(cb)


def test_get_write_ptr_requires_reserve(configured_cb: Tuple[CBAPI, CBID]):
    api, cb = configured_cb
    with pytest.raises(
        CBContractError, match="get_write_ptr requires prior cb_reserve_back"
    ):
        api.get_write_ptr(cb)


def test_multiple_consumers_error(timeout_api: CBAPI):
    api = timeout_api
    cb = 0
    api.host_configure_cb(cb, 4, shape=(1, 1))
    errors: List[str] = []

    def consumer():
        try:
            api.cb_wait_front(cb, 4)
        except (CBContractError, CBTimeoutError) as e:
            errors.append(str(e))

    t1 = threading.Thread(target=consumer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert any(
        "Only one consumer thread may wait on a CB at a time" in msg for msg in errors
    )


def test_multiple_producers_error(timeout_api: CBAPI):
    api = timeout_api
    cb = 0
    api.host_configure_cb(cb, 4, shape=(1, 1))
    errors: List[str] = []

    def producer():
        try:
            api.cb_reserve_back(cb, 4)
        except (CBContractError, CBTimeoutError) as e:
            errors.append(str(e))

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=producer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert any(
        "Only one producer thread may reserve on a CB at a time" in msg
        for msg in errors
    )


def test_allocate_cb_id(api: CBAPI):
    """Test that allocate_cb_id allocates sequential IDs."""
    cb_id0 = api.allocate_cb_id()
    cb_id1 = api.allocate_cb_id()
    cb_id2 = api.allocate_cb_id()

    assert cb_id0 == 0
    assert cb_id1 == 1
    assert cb_id2 == 2


def test_allocate_cb_id_thread_safe(api: CBAPI):
    """Test that allocate_cb_id is thread-safe."""
    allocated_ids: List[CBID] = []
    lock = threading.Lock()

    def allocate():
        cb_id = api.allocate_cb_id()
        with lock:
            allocated_ids.append(cb_id)

    threads = [threading.Thread(target=allocate) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All IDs should be unique
    assert len(allocated_ids) == 10
    assert len(set(allocated_ids)) == 10
    # IDs should be sequential (in some order)
    assert sorted(allocated_ids) == list(range(10))


def test_allocate_cb_id_exceeds_max():
    """Test that allocating more than MAX_CBS raises RuntimeError."""
    from python.sim.constants import MAX_CBS

    api = CBAPI()
    # Allocate up to MAX_CBS
    for _ in range(MAX_CBS):
        api.allocate_cb_id()

    # Next allocation should fail
    with pytest.raises(
        RuntimeError, match=f"Maximum number of circular buffers exceeded: {MAX_CBS}"
    ):
        api.allocate_cb_id()


def test_heterogeneous_cbs_in_same_api():
    """Test that a single CBAPI instance can handle multiple circular buffers."""
    from python.sim.block import ThreadType, _set_current_thread_type

    # Set COMPUTE thread context
    _set_current_thread_type(ThreadType.COMPUTE)

    try:
        # Create a shared CBAPI instance
        api = CBAPI()
        element = make_full_tensor(32, 32, 1.0)

        # Create circular buffers for different use cases
        cb1 = CircularBuffer(element=element, shape=(2, 2), buffer_factor=2, api=api)
        cb2 = CircularBuffer(element=element, shape=(2, 2), buffer_factor=2, api=api)

        # Test first circular buffer: write, read back using as source
        write1 = cb1.reserve()
        test_tensors_1 = [make_full_tensor(32, 32, i + 1.0) for i in range(len(write1))]
        write1.store(test_tensors_1)
        cb1.push()

        # Read and use as source
        read1 = cb1.wait()
        write1_2 = cb1.reserve()
        write1_2.store(read1)  # This marks read1 as used (STORE_SRC)
        cb1.pop()  # Pop read1 (now that it's been used as source)
        cb1.push()  # Push write1_2

        # Test second circular buffer
        write2 = cb2.reserve()
        test_tensors_2 = [
            make_full_tensor(32, 32, i + 10.0) for i in range(len(write2))
        ]
        write2.store(test_tensors_2)
        cb2.push()

        # Read and use as source
        read2 = cb2.wait()
        write2_2 = cb2.reserve()
        write2_2.store(read2)  # This marks read2 as used (STORE_SRC)
        cb2.pop()  # Pop read2 (now that it's been used as source)
        cb2.push()  # Push write2_2

        # Verify both CBs used the same API instance
        assert cb1._api is api  # type: ignore
        assert cb2._api is api  # type: ignore
    finally:
        # Clear thread context
        _set_current_thread_type(None)


def test_default_api_heterogeneous():
    """Test that an explicit API can handle multiple circular buffers."""
    from python.sim.block import ThreadType, _set_current_thread_type

    # Set COMPUTE thread context
    _set_current_thread_type(ThreadType.COMPUTE)

    try:
        # Create an explicit API instance
        api = CBAPI()
        element = make_full_tensor(32, 32, 1.0)

        # Create circular buffers using explicit API
        cb1 = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
        cb2 = CircularBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

        # Both should use the same API instance
        assert cb1._api is cb2._api  # type: ignore
        assert cb1._api is api  # type: ignore

        # Test that both work correctly
        write1 = cb1.reserve()
        tensor1 = make_full_tensor(32, 32, 42.0)
        write1.store([tensor1])
        cb1.push()

        write2 = cb2.reserve()
        tensor2 = make_full_tensor(32, 32, 0.0)
        write2.store([tensor2])
        cb2.push()

        # Read and use as source
        read1 = cb1.wait()
        write1_2 = cb1.reserve()
        write1_2.store(read1)  # Use as source (STORE_SRC)
        cb1.pop()  # Pop read1 (now that it's been used as source)
        cb1.push()  # Push write1_2

        read2 = cb2.wait()
        write2_2 = cb2.reserve()
        write2_2.store(read2)  # Use as source (STORE_SRC)
        cb2.pop()  # Pop read2 (now that it's been used as source)
        cb2.push()  # Push write2_2
    finally:
        # Clear thread context
        _set_current_thread_type(None)


if __name__ == "__main__":
    pytest.main([__file__])
