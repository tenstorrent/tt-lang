# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import threading
import time
from cbsim.errors import CBContractError

from cbsim.api import CBAPI
from cbsim.errors import CBContractError, CBTimeoutError


# Pytest fixtures to reduce redundant setup code
@pytest.fixture
def api():
    """Create a fresh CBAPI instance for each test."""
    return CBAPI()


@pytest.fixture
def configured_cb(api):
    """Provide a configured CB with capacity 4 on CB ID 0."""
    cb_id = 0
    api.host_configure_cb(cb_id, 4)
    return api, cb_id


@pytest.fixture
def configured_cb8(api):
    """Provide a configured CB with capacity 8 on CB ID 0."""
    cb_id = 0
    api.host_configure_cb(cb_id, 8)
    return api, cb_id


@pytest.fixture
def timeout_api():
    """Create a CBAPI instance with short timeout for testing timeouts."""
    return CBAPI(timeout=0.1)


def test_circular_buffer_basic_flow(configured_cb8):
    api, cb0 = configured_cb8
    stats = api.cb_stats(cb0)
    assert stats.capacity == 8
    assert stats.visible == 0

    # Reserve and write 4 tiles
    api.cb_reserve_back(cb0, 4)
    ptr = api.get_write_ptr(cb0)
    ptr.fill([1, 2, 3, 4])
    api.cb_push_back(cb0, 4)
    stats = api.cb_stats(cb0)
    assert stats.visible == 4
    assert stats.free == 4

    # Wait and read
    api.cb_wait_front(cb0, 4)
    read_values = api.get_read_ptr(cb0).to_list()
    assert read_values == [1, 2, 3, 4]
    api.cb_pop_front(cb0, 4)
    stats = api.cb_stats(cb0)
    assert stats.visible == 0

    # Reserve full capacity and write
    api.cb_reserve_back(cb0, 8)
    ptr = api.get_write_ptr(cb0)
    ptr.fill(list(range(8)))
    api.cb_push_back(cb0, 8)
    stats = api.cb_stats(cb0)
    assert stats.visible == 8

    # Cumulative wait and read
    api.cb_wait_front(cb0, 4)
    api.cb_wait_front(cb0, 8)
    read_values = api.get_read_ptr(cb0).to_list()
    assert read_values == list(range(8))
    api.cb_pop_front(cb0, 8)
    stats = api.cb_stats(cb0)
    assert stats.visible == 0


def test_per_instance_timeout_effect():
    # consumer should timeout based on instance timeout
    api = CBAPI(timeout=0.01)
    cb = 3
    api.host_configure_cb(cb, 4)
    start = time.time()
    with pytest.raises(CBTimeoutError, match="timed out after 0.01s"):
        api.cb_wait_front(cb, 1)
    elapsed = time.time() - start
    assert elapsed < 0.1


def test_threaded_produce_consume(configured_cb):
    api, cb0 = configured_cb
    result = []

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
    ptr.fill([100, 200, 300, 400])
    api.cb_push_back(cb0, 4)
    t.join(timeout=1)
    assert result == [[100, 200, 300, 400]]


def test_cb_pages_nonblocking(configured_cb8):
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
    ptr.fill([1, 2, 3, 4])
    api.cb_push_back(cb2, 4)
    # Divisible sizes: 4 and 2 are both valid
    assert api.cb_pages_available_at_front(cb2, 4)
    assert api.cb_pages_available_at_front(cb2, 2)

    # After pop, availability resets
    api.cb_wait_front(cb2, 4)
    api.cb_pop_front(cb2, 4)
    assert not api.cb_pages_available_at_front(cb2, 1)


# Focused error tests for page operations
def test_cb_pages_available_out_of_range_error(configured_cb):
    api, cb = configured_cb
    with pytest.raises(CBContractError, match="num_tiles must be <= capacity"):
        api.cb_pages_available_at_front(cb, 5)


def test_cb_pages_reservable_out_of_range_error(configured_cb):
    api, cb = configured_cb
    with pytest.raises(CBContractError, match="num_tiles must be <= capacity"):
        api.cb_pages_reservable_at_back(cb, 5)


def test_cb_pages_reservable_divisibility_error(configured_cb8):
    api, cb = configured_cb8
    with pytest.raises(
        CBContractError, match="First num_tiles=5 must evenly divide capacity=8"
    ):
        api.cb_pages_reservable_at_back(cb, 5)


def test_cb_pages_available_divisibility_error(configured_cb8):
    api, cb = configured_cb8
    api.cb_reserve_back(cb, 4)
    ptr = api.get_write_ptr(cb)
    ptr.fill([1, 2, 3, 4])
    api.cb_push_back(cb, 4)
    with pytest.raises(
        CBContractError, match="First num_tiles=3 must evenly divide capacity=8"
    ):
        api.cb_pages_available_at_front(cb, 3)


# Pointer requirement error tests
def test_get_read_ptr_requires_wait(configured_cb):
    api, cb = configured_cb
    with pytest.raises(
        CBContractError, match="get_read_ptr requires prior cb_wait_front"
    ):
        api.get_read_ptr(cb)


def test_get_write_ptr_requires_reserve(configured_cb):
    api, cb = configured_cb
    with pytest.raises(
        CBContractError, match="get_write_ptr requires prior cb_reserve_back"
    ):
        api.get_write_ptr(cb)


def test_multiple_consumers_error(timeout_api):
    api = timeout_api
    cb = 0
    api.host_configure_cb(cb, 4)
    errors = []

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


def test_multiple_producers_error(timeout_api):
    api = timeout_api
    cb = 0
    api.host_configure_cb(cb, 4)
    errors = []

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
