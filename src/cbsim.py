"""
A python simulator for the TT-Metal kernel circular buffer (CB) APIs,
described here:

https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.html


Semantics enforced:
- The first num_tiles used in wait/reserve sets the **step size**; it must
  evenly divide the capacity. Later ops must be multiples of that step size.
- Repeated cb_wait_front calls are **cumulative** until a cb_pop_front occurs.
- Reserve → write (via get_write_ptr) → push; wait → read (via get_read_ptr) → pop.
- Thread-safe producer/consumer behavior (RLock + Condition vars).

"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, RLock
from typing import Generic, List, Optional, Sequence, TypeVar

T = TypeVar("T")

# Runtime-enforced types
class PositiveInt(int):
    def __new__(cls, value: int):
        if value <= 0:
            raise ValueError("value must be a positive integer (> 0)")
        return super().__new__(cls, value)

class NaturalInt(int):
    def __new__(cls, value: int):
        if value < 0:
            raise ValueError("value must be a positive integer or 0 (>= 0)")
        return super().__new__(cls, value)

class Size(PositiveInt):
    pass

class Index(NaturalInt):
    pass

class Count(NaturalInt):
    pass

class CBID(NaturalInt):
    def __new__(cls, value: NaturalInt):
        if value >= MAX_CBS:
            raise ValueError(f"id must be in range 0..{MAX_CBS-1}")
        return super().__new__(cls, value)
    pass

# ---------------- Constants ----------------
MAX_CBS = 32  # Fixed pool of circular buffers
# Global timeout (seconds) used internally by blocking waits; set to None to
# block indefinitely
# This is useful in the context of a simulator in that it avoid infinite loops
# and fails the simulation when the timeout is reached. The danger here is that
# if the timeout is too tight, we might hit cases where the simulation fails for
# legitimate input that is just causing slow execution. But the option is there
# to deactivate the timeout. 
GLOBAL_WAIT_TIMEOUT: float | None = 5.0

def set_global_timeout(seconds: float | None) -> None:
    """Set the module-wide timeout for cb_wait_front / cb_reserve_back."""
    global GLOBAL_WAIT_TIMEOUT
    if seconds is not None and seconds <= 0:
        raise ValueError("timeout must be positive or None")
    GLOBAL_WAIT_TIMEOUT = seconds


def get_global_timeout() -> float | None:
    """Return the current module-wide timeout used for waits."""
    return GLOBAL_WAIT_TIMEOUT

# ---------------- Errors ----------------
class CBError(RuntimeError):
    pass

class CBContractError(CBError):
    pass

class CBNotConfigured(CBError):
    pass

class CBOutOfRange(CBError):
    pass

class CBTimeoutError(CBError):
    pass

# ---------------- Internal structures ----------------
@dataclass(frozen=True)
class _Span:
    start: Index  # inclusive index in underlying ring
    length: Size # number of tiles

# Notice that get_read_ptr and get_write_ptr return a C++ pointer which does not
# necessarily make sense in a python context. So we need something that can
# access the elements of the cb (as a pointer would) from the position the
# pointer points. To hide needless index arithmetic, we also add the ability to
# wrap around. Notice also that it handles a list and a capacity, instead of a
# _CBState, a delierate choice to make it closer in spirit to a pointer and
# minimizing the state that is exposed.
class _RingView(Generic[T]):
    """A logically contiguous window into the ring, possibly wrapping.
    Provides list-like access to elements while respecting wrap-around.
    """
    __slots__ = ("_buf", "_capacity", "_span")

    def __init__(self, buf: List[Optional[T]], capacity: Size, span: _Span):
        self._buf = buf
        self._capacity = capacity
        self._span = span

    def __len__(self) -> Size:
        return self._span.length

    def __getitem__(self, idx: Index) -> Optional[T]:
        if not (0 <= idx < self._span.length):
            raise IndexError(idx)
        return self._buf[(self._span.start + idx) % self._capacity]

    def __setitem__(self, idx: Index, value: Optional[T]) -> None:
        if not (0 <= idx < self._span.length):
            raise IndexError(idx)
        self._buf[(self._span.start + idx) % self._capacity] = value

    def to_list(self) -> List[Optional[T]]:
        return [self[i] for i in range(len(self))]

    def fill(self, items: Sequence[Optional[T]]) -> None:
        if len(items) != self._span.length:
            raise ValueError("Length mismatch in fill()")
        for i, v in enumerate(items):
            self[i] = v

# The C API is pointer-based and type-agnostic; we simulate this with a
# generic class.
class _CBState(Generic[T]):
    __slots__ = (
        "cap", "buf", "head", "visible", "reserved",
        "step", "last_wait_target", "last_reserve_target",
        "configured", "lock", "can_consume", "can_produce"
    )

    def __init__(self):
        # Not configured until host_configure_cb is called.
        self.cap = Size(1)
        self.buf: List[Optional[T]] = []
        self.head = Index(0)
        self.visible = Count(0)
        self.reserved = Count(0)
        self.step: Optional[Size] = None
        self.last_wait_target = Count(0)
        self.last_reserve_target = Count(0)
        self.configured = False
        self.lock = RLock()
        self.can_consume = Condition(self.lock)
        self.can_produce = Condition(self.lock)

    # helpers
    def _require_configured(self) -> None:
        if not self.configured:
            raise CBNotConfigured("CB not configured; call host_configure_cb")

    def _check_step(self, n: Size) -> None:
        if self.step is None:
            if self.cap % n != 0:
                raise CBContractError(
                    f"First num_tiles={n} must evenly divide capacity={self.cap}")
            self.step = n
        else:
            if n % self.step != 0:
                raise CBContractError(
                    f"num_tiles={n} must be a multiple of step size {self.step}")
        if n > self.cap:
            raise CBContractError("num_tiles must be <= capacity")

    def _free(self) -> Size:
        return self.cap - (self.visible + self.reserved)

    def _front_span(self, length: Size) -> _Span:
        return _Span((self.head) % self.cap, length)

    def _back_span(self, length: Size) -> _Span:
        start = (self.head + self.visible + self.reserved) % self.cap
        return _Span(start, length)

# ---------------- Global static pool ----------------
_pool: List[_CBState] = [_CBState() for _ in range(MAX_CBS)]

# ---------------- Host-side helpers ----------------

def host_configure_cb(cb_id: CBID, capacity_tiles: Size) -> None:
    s = _pool[cb_id]
    with s.lock:
        s.cap = capacity_tiles
        s.buf = [None] * capacity_tiles
        s.head = Index(0)
        s.visible = Count(0)
        s.reserved = Count(0)
        s.step = None
        s.last_wait_target = Count(0)
        s.last_reserve_target = Count(0)
        s.configured = True
        with s.can_consume:
            s.can_consume.notify_all()
        with s.can_produce:
            s.can_produce.notify_all()


def host_reset_cb(cb_id: CBID) -> None:
    s = _pool[cb_id]
    with s.lock:
        if not s.configured:
            return
        s.buf[:] = [None] * s.cap
        s.head = Index(0)
        s.visible = Count(0)
        s.reserved = Count(0)
        s.step = None
        s.last_wait_target = Count(0)
        s.last_reserve_target = Count(0)
        with s.can_consume:
            s.can_consume.notify_all()
        with s.can_produce:
            s.can_produce.notify_all()


def cb_stats(cb_id: CBID) -> dict:
    s = _pool[cb_id]
    with s.lock:
        s._require_configured()
        return {
            "capacity": s.cap,
            "visible": s.visible,
            "reserved": s.reserved,
            "free": s._free(),
            "step": s.step,
            "head": s.head,
        }

# ---------------- Non-blocking queries ----------------

def cb_pages_available_at_front(cb_id: CBID, num_tiles: Size) -> bool:
    s = _pool[cb_id]
    with s.lock:
        s._require_configured()
        s._check_step(num_tiles)
        return s.visible >= num_tiles


def cb_pages_reservable_at_back(cb_id: CBID, num_tiles: Size) -> bool:
    s = _pool[cb_id]
    with s.lock:
        s._require_configured()
        s._check_step(num_tiles)
        return s._free() >= num_tiles

# ---------------- Blocking calls ----------------

def cb_wait_front(cb_id: CBID, num_tiles: Size) -> None:
    """Block until num_tiles are visible; enforce cumulative waiting.
    Raises CBTimeoutError if the global timeout expires.
    """
    s = _pool[cb_id]
    with s.can_consume:
        s._require_configured()
        s._check_step(num_tiles)
        if num_tiles < s.last_wait_target:
            raise CBContractError(
                "cb_wait_front must be cumulative until a pop occurs")
        ok = s.can_consume.wait_for(lambda: s.visible >= num_tiles, timeout=GLOBAL_WAIT_TIMEOUT)
        if not ok:
            raise CBTimeoutError(f"cb_wait_front timed out after {GLOBAL_WAIT_TIMEOUT}s")
        s.last_wait_target = num_tiles


def cb_reserve_back(cb_id: CBID, num_tiles: Size) -> None:
    """Block until num_tiles can be reserved; then reserve them.
    Raises CBTimeoutError if the global timeout expires.
    """
    s = _pool[cb_id]
    with s.can_produce:
        s._require_configured()
        s._check_step(num_tiles)
        target = s.reserved + num_tiles
        if target < s.last_reserve_target:
            raise CBContractError("reserve target cannot regress within epoch")
        ok = s.can_produce.wait_for(lambda: s._free() >= num_tiles, timeout=GLOBAL_WAIT_TIMEOUT)
        if not ok:
            raise CBTimeoutError(f"cb_reserve_back timed out after {GLOBAL_WAIT_TIMEOUT}s")
        s.reserved += num_tiles
        s.last_reserve_target = target

# ---------------- State-mutating ops ----------------

def cb_push_back(cb_id: CBID, num_tiles: Size, data: Optional[Sequence[Optional[T]]] = None) -> None:
    s = _pool[cb_id]
    with s.lock:
        s._require_configured()
        s._check_step(num_tiles)
        if num_tiles > s.reserved:
            raise CBContractError(
                f"cb_push_back({num_tiles}) exceeds reserved={s.reserved}")
        span = s._back_span(num_tiles)
        view = _RingView(s.buf, s.cap, span)
        if data is not None:
            if len(data) != num_tiles:
                raise ValueError("data length must match num_tiles")
            view.fill(data)
        s.reserved -= num_tiles
        s.visible += num_tiles
        with s.can_consume:
            s.can_consume.notify_all()


def cb_pop_front(cb_id: CBID, num_tiles: Size) -> None:
    s = _pool[cb_id]
    with s.lock:
        s._require_configured()
        s._check_step(num_tiles)
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

def get_read_ptr(cb_id: CBID) -> _RingView[Optional[T]]:
    s = _pool[cb_id]
    with s.lock:
        s._require_configured()
        if s.last_wait_target <= 0:
            raise CBContractError("get_read_ptr requires prior cb_wait_front")
        if s.visible < s.last_wait_target:
            raise CBContractError("read window invalidated; call cb_wait_front again")
        span = s._front_span(s.last_wait_target)
        return _RingView(s.buf, s.cap, span)


def get_write_ptr(cb_id: CBID, length: Optional[Size] = None) -> _RingView[Optional[T]]:
    s = _pool[cb_id]
    with s.lock:
        s._require_configured()
        if s.reserved <= 0:
            raise CBContractError("get_write_ptr requires prior cb_reserve_back")
        L = s.reserved if length is None else length
        if not (0 < L <= s.reserved):
            raise ValueError("length must be in 1..reserved inclusive")
        span = s._back_span(L)
        return _RingView(s.buf, s.cap, span)

# ---------------- Minimal demo ----------------
if __name__ == "__main__":
    cb0 = 0
    host_configure_cb(cb0, 8)

    # Optional: change global timeout (seconds) or set to None to block forever
    # set_global_timeout(2.0)

    # Producer reserves 4 tiles and writes
    cb_reserve_back(cb0, 4)
    get_write_ptr(cb0).fill([10, 11, 12, 13])
    cb_push_back(cb0, 4)

    # Consumer waits cumulatively and reads
    cb_wait_front(cb0, 2)
    print("Front2:", get_read_ptr(cb0).to_list())
    cb_wait_front(cb0, 4)
    print("Front4:", get_read_ptr(cb0).to_list())
    cb_pop_front(cb0, 4)

    print("stats:", cb_stats(cb0))
