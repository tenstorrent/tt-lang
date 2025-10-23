import pytest
import threading
import time
from cbsim.errors import CBContractError

from cbsim.api import CBAPI
from cbsim.errors import CBContractError, CBTimeoutError

# Create an instance of the API for each test
api = CBAPI()


def test_circular_buffer_basic_flow():
    cb0 = 0
    # Use class-based API instance
    # api = CBAPI()

    # Configure CB with capacity 8
    api.host_configure_cb(cb0, 8)
    stats = api.cb_stats(cb0)
    assert stats['capacity'] == 8
    assert stats['visible'] == 0

    # Reserve and write 4 tiles
    api.cb_reserve_back(cb0, 4)
    ptr = api.get_write_ptr(cb0)
    ptr.fill([1, 2, 3, 4])
    api.cb_push_back(cb0, 4)
    stats = api.cb_stats(cb0)
    assert stats['visible'] == 4
    assert stats['free'] == 4

    # Wait and read
    api.cb_wait_front(cb0, 4)
    read_values = api.get_read_ptr(cb0).to_list()
    assert read_values == [1, 2, 3, 4]
    api.cb_pop_front(cb0, 4)
    stats = api.cb_stats(cb0)
    assert stats['visible'] == 0

    # Reserve full capacity and write
    api.cb_reserve_back(cb0, 8)
    ptr = api.get_write_ptr(cb0)
    ptr.fill(list(range(8)))
    api.cb_push_back(cb0, 8)
    stats = api.cb_stats(cb0)
    assert stats['visible'] == 8

    # Cumulative wait and read
    api.cb_wait_front(cb0, 4)
    api.cb_wait_front(cb0, 8)
    read_values = api.get_read_ptr(cb0).to_list()
    assert read_values == list(range(8))
    api.cb_pop_front(cb0, 8)
    stats = api.cb_stats(cb0)
    assert stats['visible'] == 0

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

def test_threaded_produce_consume():
    cb0 = 1

    api.host_configure_cb(cb0, 4)
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

def test_cb_pages_nonblocking():
    cb2 = 2
    api.host_configure_cb(cb2, 8)

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
def test_cb_pages_available_out_of_range_error():
    api = CBAPI()
    cb = 0
    api.host_configure_cb(cb, 4)
    with pytest.raises(CBContractError, match="num_tiles must be <= capacity"):
        api.cb_pages_available_at_front(cb, 5)

def test_cb_pages_reservable_out_of_range_error():
    api = CBAPI()
    cb = 0
    api.host_configure_cb(cb, 4)
    with pytest.raises(CBContractError, match="num_tiles must be <= capacity"):
        api.cb_pages_reservable_at_back(cb, 5)

def test_cb_pages_reservable_divisibility_error():
    api = CBAPI()
    cb = 0
    api.host_configure_cb(cb, 8)
    with pytest.raises(CBContractError, match="First num_tiles=5 must evenly divide capacity=8"):
        api.cb_pages_reservable_at_back(cb, 5)

def test_cb_pages_available_divisibility_error():
    api = CBAPI()
    cb = 0
    api.host_configure_cb(cb, 8)
    api.cb_reserve_back(cb, 4)
    ptr = api.get_write_ptr(cb)
    ptr.fill([1, 2, 3, 4])
    api.cb_push_back(cb, 4)
    with pytest.raises(CBContractError, match="First num_tiles=3 must evenly divide capacity=8"):
        api.cb_pages_available_at_front(cb, 3)

# Pointer requirement error tests
def test_get_read_ptr_requires_wait():
    api = CBAPI()
    cb = 0
    api.host_configure_cb(cb, 4)
    with pytest.raises(CBContractError, match="get_read_ptr requires prior cb_wait_front"):
        api.get_read_ptr(cb)

def test_get_write_ptr_requires_reserve():
    api = CBAPI()
    cb = 0
    api.host_configure_cb(cb, 4)
    with pytest.raises(CBContractError, match="get_write_ptr requires prior cb_reserve_back"):
        api.get_write_ptr(cb)
