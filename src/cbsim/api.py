"""
Public API for cbsim: a class-based interface with a singleton default.
"""
import threading
from typing import List, Optional, TypeVar, Dict, Annotated
from pydantic import validate_call, Field
from .errors import CBContractError, CBTimeoutError
from .constants import MAX_CBS
from .typedefs import Size, CBID
from .ringview import _RingView
from .cbstate import _CBState

T = TypeVar("T")

class CBAPI:
    """Circular buffer simulator API interface with its own state pool.
       The simulator is based on the following API:
       https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.html
    """
    def __init__(self, timeout: Optional[float] = 5.0):
        """Initialize simulator with optional per-instance timeout (seconds)."""

        self._pool: List[_CBState] = [_CBState() for _ in range(MAX_CBS)]
        self._timeout: Optional[float] = timeout

    @validate_call
    def host_configure_cb(self, cb_id: CBID, capacity_tiles: Size) -> None:
        s = self._pool[int(cb_id)]
        with s.lock:
            s.cap = capacity_tiles
            s._reset()

    @validate_call
    def host_reset_cb(self, cb_id: CBID) -> None:
        s = self._pool[int(cb_id)]
        with s.lock:
            if not s.configured:
                raise CBContractError("CB not configured; cannot reset")
            s._reset()


    @validate_call
    def cb_stats(self, cb_id: CBID) -> Dict[str, int]:
        s = self._pool[int(cb_id)]
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

    @validate_call
    def cb_pages_available_at_front(self, cb_id: CBID, num_tiles: Size) -> bool:
        s = self._pool[int(cb_id)]
        with s.lock:
            s._require_configured()
            s._check_num_tiles(num_tiles)
            return s.visible >= num_tiles

    @validate_call
    def cb_pages_reservable_at_back(self, cb_id: CBID, num_tiles: Size) -> bool:
        s = self._pool[int(cb_id)]
        with s.lock:
            s._require_configured()
            s._check_num_tiles(num_tiles)
            return s._free() >= num_tiles

    @validate_call
    def cb_wait_front(self, cb_id: CBID, num_tiles: Size) -> None:
        s = self._pool[int(cb_id)]
        with s.can_consume:
            s._require_configured()
            s._check_num_tiles(num_tiles)
            thread = threading.current_thread()
            if (s.consumer_waiting is not None) and (s.consumer_waiting != thread):
                raise CBContractError("Only one consumer thread may wait on a CB at a time")
            s.consumer_waiting = thread
            if s.step is None:
                s.step = num_tiles
            else:
                if num_tiles != s.last_wait_target + s.step:
                    raise CBContractError(
                        "cb_wait_front must be cumulative with an increment of the initial number of tiles"
                        " requested until a pop occurs"
                    )
            ok = s.can_consume.wait_for(
                lambda: s.visible >= num_tiles, timeout=self._timeout
            )
            if not ok:
                raise CBTimeoutError(f"cb_wait_front timed out after {self._timeout}s")
            s.last_wait_target = num_tiles
                
    @validate_call
    def cb_reserve_back(self, cb_id: CBID, num_tiles: Size) -> None:
        s = self._pool[int(cb_id)]
        with s.can_produce:
            s._require_configured()
            s._check_num_tiles(num_tiles)
            thread = threading.current_thread()
            if (s.producer_reserving is not None) and (s.producer_reserving != thread):
                raise CBContractError("Only one producer thread may reserve on a CB at a time")
            s.producer_reserving = thread
            if num_tiles < s.reserved:
                raise CBContractError("reserve target cannot regress within epoch")
            ok = s.can_produce.wait_for(
                lambda: s._free() >= num_tiles, timeout=self._timeout
            )
            if not ok:
                raise CBTimeoutError(f"cb_reserve_back timed out after {self._timeout}s")
            s.reserved = num_tiles

    @validate_call
    def cb_push_back(self, cb_id: CBID, num_tiles: Size) -> None:
        s = self._pool[int(cb_id)]
        with s.lock:
            s._require_configured()
            s._check_num_tiles(num_tiles)
            if num_tiles > s.reserved:
                raise CBContractError(
                    f"cb_push_back({num_tiles}) exceeds reserved={s.reserved}"
                )
            s.reserved -= num_tiles
            s.visible += num_tiles
            if s.reserved == 0:
                s.producer_reserving = None
            with s.can_consume:
                s.can_consume.notify_all()

    @validate_call
    def cb_pop_front(self, cb_id: CBID, num_tiles: Size) -> None:
        s = self._pool[int(cb_id)]
        with s.lock:
            s._require_configured()
            s._check_num_tiles(num_tiles)
            if num_tiles > s.visible:
                raise CBContractError(
                    f"cb_pop_front({num_tiles}) exceeds visible={s.visible}"
                )
            span = s._front_span(num_tiles)
            view = _RingView(s.buf, s.cap, span)
            for i in range(len(view)):
                view[i] = None
            s.head = (s.head + num_tiles) % s.cap
            s.visible -= num_tiles
            s.last_wait_target = 0
            if s.visible == 0:
                s.consumer_waiting = None
            with s.can_produce:
                s.can_produce.notify_all()

    @validate_call
    def get_read_ptr(self, cb_id: CBID) -> _RingView[Optional[T]]:
        s = self._pool[int(cb_id)]
        with s.lock:
            s._require_configured()
            if s.last_wait_target <= 0:
                raise CBContractError("get_read_ptr requires prior cb_wait_front")
            if s.visible < s.last_wait_target:
                raise CBContractError("read window invalidated; call cb_wait_front again")
            span = s._front_span(s.last_wait_target)
            return _RingView(s.buf, s.cap, span)

    @validate_call
    def get_write_ptr(self, cb_id: CBID, length: Optional[Size] = None) -> _RingView[Optional[T]]:
        s = self._pool[int(cb_id)]
        with s.lock:
            s._require_configured()
            if s.reserved <= 0:
                raise CBContractError("get_write_ptr requires prior cb_reserve_back")
            L = s.reserved if length is None else length
            if not (0 < L <= s.reserved):
                raise ValueError("length must be in 1..reserved inclusive")
            span = s._front_span(L)
            return _RingView(s.buf, s.cap, span)

    @validate_call
    def set_timeout(self, seconds: Optional[Annotated[float, Field(gt=0)]]) -> None:
        """Set this simulator instance's timeout."""
        self._timeout = seconds

    def get_timeout(self) -> Optional[float]:
        """Return this simulator instance's timeout."""
        return self._timeout

# Default global API instance and module-level aliases
_default_api = CBAPI()

host_configure_cb = _default_api.host_configure_cb
host_reset_cb = _default_api.host_reset_cb
cb_stats = _default_api.cb_stats
cb_pages_available_at_front = _default_api.cb_pages_available_at_front
cb_pages_reservable_at_back = _default_api.cb_pages_reservable_at_back
cb_wait_front = _default_api.cb_wait_front
cb_reserve_back = _default_api.cb_reserve_back
cb_push_back = _default_api.cb_push_back
cb_pop_front = _default_api.cb_pop_front
get_read_ptr = _default_api.get_read_ptr
get_write_ptr = _default_api.get_write_ptr
