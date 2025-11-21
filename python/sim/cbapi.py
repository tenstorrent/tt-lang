# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Public API for cbsim: a class-based interface with a singleton default.
"""

import threading
from typing import List, Optional, Annotated, NamedTuple, Generic, Any
from pydantic import validate_call, Field
from .errors import CBContractError, CBTimeoutError
from .constants import MAX_CBS
from .typedefs import Size, CBID, CBElemType
from .ringview import RingView
from .cbstate import CBState


class CBStats(NamedTuple):
    """Statistics for a circular buffer."""

    capacity: int
    visible: int
    reserved: int
    free: int
    step: Optional[int]
    head: int
    list: List[Optional[object]]


class CBAPI(Generic[CBElemType]):
    """Circular buffer simulator API interface with its own state pool.
    The simulator is based on the following API:
    https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.html
    """

    def __init__(self, timeout: Optional[float] = 1.0):
        """Initialize simulator with optional per-instance timeout (seconds)."""

        self._pool: List[CBState[CBElemType]] = [
            CBState[CBElemType]() for _ in range(MAX_CBS)
        ]
        self._timeout: Optional[float] = timeout
        self._next_cb_id: CBID = 0
        self._cb_allocator_lock = threading.Lock()

    def allocate_cb_id(self) -> CBID:
        """Allocate a unique CB ID from this API instance. Thread-safe."""
        with self._cb_allocator_lock:
            cb_id = self._next_cb_id
            self._next_cb_id += 1
            if self._next_cb_id > MAX_CBS:
                raise RuntimeError(
                    f"Maximum number of circular buffers exceeded: {MAX_CBS}"
                )
            return cb_id

    @validate_call
    def host_configure_cb(self, cb_id: CBID, capacity_tiles: Size) -> None:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.lock:
            cb_state.cap = capacity_tiles
            cb_state.reset()

    @validate_call
    def host_reset_cb(self, cb_id: CBID) -> None:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.lock:
            if not cb_state.configured:
                raise CBContractError("CB not configured; cannot reset")
            cb_state.reset()

    @validate_call
    def cb_stats(self, cb_id: CBID) -> CBStats:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.lock:
            cb_state.require_configured()
            return CBStats(
                capacity=cb_state.cap,
                visible=cb_state.visible,
                reserved=cb_state.reserved,
                free=cb_state.free(),
                step=cb_state.step,
                head=cb_state.head,
                list=list(cb_state.buf),  # Convert to avoid type variance issues
            )

    @validate_call
    def cb_pages_available_at_front(self, cb_id: CBID, num_tiles: Size) -> bool:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.lock:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            return cb_state.visible >= num_tiles

    @validate_call
    def cb_pages_reservable_at_back(self, cb_id: CBID, num_tiles: Size) -> bool:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.lock:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            return cb_state.free() >= num_tiles

    @validate_call
    def cb_wait_front(self, cb_id: CBID, num_tiles: Size) -> None:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.can_consume:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            thread = threading.current_thread()
            if (cb_state.consumer_waiting is not None) and (
                cb_state.consumer_waiting != thread
            ):
                raise CBContractError(
                    "Only one consumer thread may wait on a CB at a time"
                )
            cb_state.consumer_waiting = thread
            if cb_state.step is None:
                cb_state.step = num_tiles
            else:
                if num_tiles != cb_state.last_wait_target + cb_state.step:
                    raise CBContractError(
                        "cb_wait_front must be cumulative with an increment of the initial number of tiles"
                        " requested until a pop occurs"
                    )
            ok = cb_state.can_consume.wait_for(
                lambda: cb_state.visible >= num_tiles, timeout=self._timeout
            )
            if not ok:
                raise CBTimeoutError(f"cb_wait_front timed out after {self._timeout}s")
            cb_state.last_wait_target = num_tiles

    @validate_call
    def cb_reserve_back(self, cb_id: CBID, num_tiles: Size) -> None:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.can_produce:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            thread = threading.current_thread()
            if (cb_state.producer_reserving is not None) and (
                cb_state.producer_reserving != thread
            ):
                raise CBContractError(
                    "Only one producer thread may reserve on a CB at a time"
                )
            cb_state.producer_reserving = thread
            if num_tiles < cb_state.reserved:
                raise CBContractError("reserve target cannot regress within epoch")
            ok = cb_state.can_produce.wait_for(
                lambda: cb_state.free() >= num_tiles, timeout=self._timeout
            )
            if not ok:
                raise CBTimeoutError(
                    f"cb_reserve_back timed out after {self._timeout}s"
                )
            cb_state.reserved = num_tiles
            cb_state.last_reserve_target = num_tiles

    @validate_call
    def cb_push_back(self, cb_id: CBID, num_tiles: Size) -> None:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.lock:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            if num_tiles > cb_state.reserved:
                raise CBContractError(
                    f"cb_push_back({num_tiles}) exceeds reserved={cb_state.reserved}"
                )
            cb_state.reserved -= num_tiles
            cb_state.visible += num_tiles
            if cb_state.reserved == 0:
                cb_state.producer_reserving = None
            with cb_state.can_consume:
                cb_state.can_consume.notify_all()

    @validate_call
    def cb_pop_front(self, cb_id: CBID, num_tiles: Size) -> None:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.lock:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            if num_tiles > cb_state.visible:
                raise CBContractError(
                    f"cb_pop_front({num_tiles}) exceeds visible={cb_state.visible}"
                )
            span = cb_state.front_span(num_tiles)
            view = RingView[CBElemType](cb_state.buf, cb_state.cap, span)
            for i in range(len(view)):
                view.pop(i)
            cb_state.head = (cb_state.head + num_tiles) % cb_state.cap
            cb_state.visible -= num_tiles
            cb_state.last_wait_target = 0
            if cb_state.visible == 0:
                cb_state.consumer_waiting = None
            with cb_state.can_produce:
                cb_state.can_produce.notify_all()

    @validate_call
    def get_read_ptr(self, cb_id: CBID) -> RingView[CBElemType]:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.lock:
            cb_state.require_configured()
            if cb_state.last_wait_target <= 0:
                raise CBContractError("get_read_ptr requires prior cb_wait_front")
            if cb_state.visible < cb_state.last_wait_target:
                raise CBContractError(
                    "read window invalidated; call cb_wait_front again"
                )
            span = cb_state.front_span(cb_state.last_wait_target)
            return RingView[CBElemType](cb_state.buf, cb_state.cap, span)

    @validate_call
    def get_write_ptr(self, cb_id: CBID) -> RingView[CBElemType]:
        cb_state: CBState[CBElemType] = self._pool[int(cb_id)]
        with cb_state.lock:
            cb_state.require_configured()
            if cb_state.last_reserve_target <= 0:
                raise CBContractError("get_write_ptr requires prior cb_reserve_back")
            if cb_state.reserved < cb_state.last_reserve_target:
                raise CBContractError("write window invalidated; call cb_reserve again")
            span = cb_state.back_span(cb_state.last_reserve_target)
            return RingView[CBElemType](cb_state.buf, cb_state.cap, span)

    @validate_call
    def set_timeout(self, seconds: Optional[Annotated[float, Field(gt=0)]]) -> None:
        """Set this simulator instance's timeout."""
        self._timeout = seconds

    def get_timeout(self) -> Optional[float]:
        """Return this simulator instance's timeout."""
        return self._timeout


# Default global API instance and module-level aliases
_default_api = CBAPI[Any]()

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
