"""
A python simulator for the TT-Metal kernel circular buffer (CB) APIs,
described here:

https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.html


Semantics enforced:
- The first num_tiles used in wait sets the **step size**; it must
  evenly divide the capacity. Later ops must be multiples of that step size.
- Repeated cb_wait_front calls are **cumulative** until a cb_pop_front occurs.
- Reserve → write (via get_write_ptr) → push; wait → read (via get_read_ptr) → pop.
- Thread-safe producer/consumer behavior (RLock + Condition vars).

"""

from __future__ import annotations
import sys
from pathlib import Path

# Allow running as script:
if __name__ == "__main__" and __package__ is None:
    pkg_root = Path(__file__).parent.parent
    sys.path[0] = str(pkg_root)
    __package__ = "cbsim"

from typing import List, Optional, TypeVar
from pydantic import validate_call
from .errors import CBContractError, CBTimeoutError
from .typedefs import Size, CBID, MAX_CBS
from .ringview import _RingView
from .cbstate import _CBState
from .timeout import set_global_timeout, get_global_timeout

T = TypeVar("T")

from .api import (
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
)

if __name__ == "__main__":
  cb0 = 0
  host_configure_cb(cb0, 8)

  # Producer reserves 4 tiles and writes
  cb_reserve_back(cb0, 4)
  write_ptr = get_write_ptr(cb0)
  write_ptr.fill([10, 11, 12, 13])
  cb_push_back(cb0, 4)
  print("stats:", cb_stats(cb0))

  # Consumer waits and reads
  cb_wait_front(cb0, 4)
  print("Front1:", get_read_ptr(cb0).to_list())
  cb_pop_front(cb0, 4)
  print("stats:", cb_stats(cb0))

  # Producer reserves another 8 tiles and writes
  cb_reserve_back(cb0, 8)
  write_ptr = get_write_ptr(cb0)
  write_ptr.fill([14, 15, 16, 17, 18, 19, 20, 21])
  cb_push_back(cb0, 8)
  print("stats:", cb_stats(cb0))

  # Consumer waits cumulatively and reads
  cb_wait_front(cb0, 4)
  cb_wait_front(cb0, 8)
  print("Front2:", get_read_ptr(cb0).to_list())
  cb_pop_front(cb0, 8)
  print("stats:", cb_stats(cb0))


