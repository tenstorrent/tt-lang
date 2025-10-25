import os
import pytest

from cbsim import (
    host_configure_cb,
    host_reset_cb,
    cb_stats,
    cb_pages_available_at_front,
    cb_pages_reservable_at_back,
    cb_wait_front,
    cb_reserve_back,
    cb_push_back,
    cb_pop_front,
    get_read_ptr,
    get_write_ptr,
    set_global_timeout,
    get_global_timeout,
)


def test_circular_buffer_basic_flow():
    cb0 = 0
    # Configure CB with capacity 8
    host_configure_cb(cb0, 8)
    stats = cb_stats(cb0)
    assert stats['capacity'] == 8
    assert stats['visible'] == 0

    # Reserve and write 4 tiles
    cb_reserve_back(cb0, 4)
    ptr = get_write_ptr(cb0)
    ptr.fill([1, 2, 3, 4])
    cb_push_back(cb0, 4)
    stats = cb_stats(cb0)
    assert stats['visible'] == 4
    assert stats['free'] == 4

    # Wait and read
    cb_wait_front(cb0, 4)
    read_values = get_read_ptr(cb0).to_list()
    assert read_values == [1, 2, 3, 4]
    cb_pop_front(cb0, 4)
    stats = cb_stats(cb0)
    assert stats['visible'] == 0

    # Reserve full capacity and write
    cb_reserve_back(cb0, 8)
    ptr = get_write_ptr(cb0)
    ptr.fill(list(range(8)))
    cb_push_back(cb0, 8)
    stats = cb_stats(cb0)
    assert stats['visible'] == 8

    # Cumulative wait and read
    cb_wait_front(cb0, 4)
    cb_wait_front(cb0, 8)
    read_values = get_read_ptr(cb0).to_list()
    assert read_values == list(range(8))
    cb_pop_front(cb0, 8)
    stats = cb_stats(cb0)
    assert stats['visible'] == 0

if __name__ == '__main__':
    pytest.main()