"""
_CState and related internal state management for cbsim.
"""
from threading import Condition, RLock
from typing import Generic, List, Optional, TypeVar
from .typedefs import Size, Index, Count
from .errors import CBContractError, CBNotConfigured
from .ringview import _Span

T = TypeVar("T")

class _CBState(Generic[T]):
    __slots__ = (
        "cap", "buf", "head", "visible", "reserved",
        "step", "last_wait_target", "last_reserve_target",
        "configured", "lock", "can_consume", "can_produce"
    )

    def __init__(self):
        self.cap: Size = 1
        self.buf: List[Optional[T]] = []
        self.head: Index = 0
        self.visible: Count = 0
        self.reserved: Count = 0
        self.step: Optional[Size] = None
        self.last_wait_target: Count = 0
        self.last_reserve_target: Count = 0
        self.configured = False
        self.lock = RLock()
        self.can_consume = Condition(self.lock)
        self.can_produce = Condition(self.lock)

    def _require_configured(self) -> None:
        if not self.configured:
            raise CBNotConfigured("CB not configured; call host_configure_cb")

    def _check_num_tiles(self, num_tiles: Size) -> None:
        if num_tiles > self.cap:
            raise CBContractError("num_tiles must be <= capacity")
        if self.cap % num_tiles != 0:
            raise CBContractError(
                f"First num_tiles={num_tiles} must evenly divide capacity={self.cap}")

    def _free(self) -> Size:
        return self.cap - (self.visible + self.reserved)

    def _front_span(self, length: Size) -> _Span:
        return _Span(self.head, length)
