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

# ---------------- Global static pool ----------------
_pool: List[_CBState] = [_CBState() for _ in range(MAX_CBS)]

# ---------------- Host-side helpers ----------------

@validate_call
def host_configure_cb(cb_id: CBID, capacity_tiles: Size) -> None:
    s = _pool[int(cb_id)]
    with s.lock:
        s.cap = capacity_tiles
        s.buf = [None] * capacity_tiles
        s.head = 0
        s.visible = 0
        s.reserved = 0
        s.step = None
        s.last_wait_target = 0
        s.last_reserve_target = 0
        s.configured = True
        with s.can_consume:
            s.can_consume.notify_all()
        with s.can_produce:
            s.can_produce.notify_all()


@validate_call
def host_reset_cb(cb_id: CBID) -> None:
    s = _pool[int(cb_id)]
    with s.lock:
        if not s.configured:
            return
        s.buf[:] = [None] * s.cap
        s.head = 0
        s.visible = 0
        s.reserved = 0
        s.step = None
        s.last_wait_target = 0
        s.last_reserve_target = 0
        with s.can_consume:
            s.can_consume.notify_all()
        with s.can_produce:
            s.can_produce.notify_all()


@validate_call
def cb_stats(cb_id: CBID) -> dict:
    s = _pool[int(cb_id)]
    with s.lock:
        s._require_configured()
        return {
            "capacity": s.cap,
            "visible": s.visible,
            "reserved": s.reserved,
            "free": s._free(),
            "step": s.step,
            "head": s.head,
            "list": s.buf,
        }

# ---------------- Non-blocking queries ----------------

@validate_call
def cb_pages_available_at_front(cb_id: CBID, num_tiles: Size) -> bool:
    s = _pool[int(cb_id)]
    with s.lock:
        s._require_configured()
        s._check_num_tiles(num_tiles)
        return s.visible >= num_tiles


@validate_call
def cb_pages_reservable_at_back(cb_id: CBID, num_tiles: Size) -> bool:
    s = _pool[int(cb_id)]
    with s.lock:
        s._require_configured()
        s._check_num_tiles(num_tiles)
        return s._free() >= num_tiles

# ---------------- Blocking calls ----------------

@validate_call
def cb_wait_front(cb_id: CBID, num_tiles: Size) -> None:
    """Block until num_tiles are visible; enforce cumulative waiting.
    Raises CBTimeoutError if the global timeout expires.
    """
    s = _pool[int(cb_id)]
    with s.can_consume:
        s._require_configured()
        s._check_num_tiles(num_tiles)
        if s.step is None:
            s.step = num_tiles
        else:
            if num_tiles != s.last_wait_target + s.step:
                raise CBContractError(
                    "cb_wait_front must be cumulative with an increment of the initial number of tiles"
                    " requested until a pop occurs")
        ok = s.can_consume.wait_for(lambda: s.visible >= num_tiles, timeout=get_global_timeout())
        if not ok:
            raise CBTimeoutError(f"cb_wait_front timed out after {get_global_timeout()}s")
        s.last_wait_target = num_tiles

@validate_call
def cb_reserve_back(cb_id: CBID, num_tiles: Size) -> None:
    """Block until num_tiles can be reserved; then reserve them.
    Raises CBTimeoutError if the global timeout expires.
    """
    s = _pool[int(cb_id)]
    with s.can_produce:
        s._require_configured()
        s._check_num_tiles(num_tiles)
        target = s.reserved + num_tiles
        if target < s.last_reserve_target:
            raise CBContractError("reserve target cannot regress within epoch")
        ok = s.can_produce.wait_for(lambda: s._free() >= num_tiles, timeout=get_global_timeout())
        if not ok:
            raise CBTimeoutError(f"cb_reserve_back timed out after {get_global_timeout()}s")
        s.reserved += num_tiles
        s.last_reserve_target = target

# ---------------- State-mutating ops ----------------

@validate_call
def cb_push_back(cb_id: CBID, num_tiles: Size) -> None:
    s = _pool[int(cb_id)]
    with s.lock:
        s._require_configured()
        s._check_num_tiles(num_tiles)
        if num_tiles > s.reserved:
            raise CBContractError(
                f"cb_push_back({num_tiles}) exceeds reserved={s.reserved}"
            )
        # Pointer-style semantics: data must have already been written via get_write_ptr()
        s.reserved -= num_tiles
        s.visible += num_tiles
        with s.can_consume:
            s.can_consume.notify_all()

@validate_call
def cb_pop_front(cb_id: CBID, num_tiles: Size) -> None:
    s = _pool[int(cb_id)]
    with s.lock:
        s._require_configured()
        s._check_num_tiles(num_tiles)
        if num_tiles > s.visible:
            raise CBContractError(
                f"cb_pop_front({num_tiles}) exceeds visible={s.visible}")
        span = s._front_span(num_tiles)
        view = _RingView(s.buf, s.cap, span)
        for i in range(len(view)):
            view[i] = None
        s.head = (s.head + num_tiles) % s.cap
        s.visible -= num_tiles
        s.last_wait_target = 0  # reset cumulative wait epoch
        with s.can_produce:
            s.can_produce.notify_all()

# ---------------- Pointer-style helpers ----------------

@validate_call
def get_read_ptr(cb_id: CBID) -> _RingView[Optional[T]]:
    s = _pool[int(cb_id)]
    with s.lock:
        s._require_configured()
        if s.last_wait_target <= 0:
            raise CBContractError("get_read_ptr requires prior cb_wait_front")
        if s.visible < s.last_wait_target:
            raise CBContractError("read window invalidated; call cb_wait_front again")
        span = s._front_span(s.last_wait_target)
        return _RingView(s.buf, s.cap, span)


@validate_call
def get_write_ptr(cb_id: CBID, length: Optional[Size] = None) -> _RingView[Optional[T]]:
    s = _pool[int(cb_id)]
    with s.lock:
        s._require_configured()
        if s.reserved <= 0:
            raise CBContractError("get_write_ptr requires prior cb_reserve_back")
        L = s.reserved if length is None else length
        if not (0 < L <= s.reserved):
            raise ValueError("length must be in 1..reserved inclusive")
        span = s._front_span(L)
        return _RingView(s.buf, s.cap, span)

# ---------------- Minimal demo ----------------
if __name__ == "__main__":
    cb0 = 0
    host_configure_cb(cb0, 8)    
    
    # Producer reserves 4 tiles and writes
    cb_reserve_back(cb0, 4)
    get_write_ptr(cb0).fill([10, 11, 12, 13])
    cb_push_back(cb0, 4)
    print("stats:", cb_stats(cb0))

    # Consumer waits and reads
    cb_wait_front(cb0, 4)
    print("Front1:", get_read_ptr(cb0).to_list())
    cb_pop_front(cb0, 4)
    print("stats:", cb_stats(cb0))

    # Producer reserves another 8 tiles and writes
    cb_reserve_back(cb0, 8)
    get_write_ptr(cb0).fill([14, 15, 16, 17, 18, 19, 20, 21])
    cb_push_back(cb0, 8)
    print("stats:", cb_stats(cb0))

    # Consumer waits cumulatively and reads
    cb_wait_front(cb0, 4)
    cb_wait_front(cb0, 8)
    print("Front2:", get_read_ptr(cb0).to_list())
    cb_pop_front(cb0, 8)
    print("stats:", cb_stats(cb0))

