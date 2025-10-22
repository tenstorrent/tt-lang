import pytest
import time

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

if __name__ == '__main__':
    pytest.main()