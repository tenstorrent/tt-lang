# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
_CState and related internal state management for cbsim.
"""

from threading import Condition, RLock, Thread
from typing import Generic, List, Optional
from .typedefs import Size, Index, Count, CBElemType, CBSlotType
from .errors import CBContractError, CBNotConfigured
from .block import Span


# It is a deliberate design choice to use any generic type here to avoid dealing
# with byte arrays as would be the case in the C++ API.
class CBState(Generic[CBElemType]):
    __slots__ = (
        "cap",
        "buf",
        "head",
        "visible",
        "reserved",
        "step",
        "last_wait_target",
        "last_reserve_target",
        "configured",
        "lock",
        "can_consume",
        "can_produce",
        "consumer_waiting",
        "producer_reserving",
    )

    def __init__(self):
        self.cap: Size = 1
        self.buf: List[CBSlotType[CBElemType]] = []
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
        self.consumer_waiting: Optional[Thread] = None
        self.producer_reserving: Optional[Thread] = None

    def require_configured(self) -> None:
        if not self.configured:
            raise CBNotConfigured("CB not configured; call host_configure_cb")

    def check_num_tiles(self, num_tiles: Size) -> None:
        if num_tiles > self.cap:
            raise CBContractError("num_tiles must be <= capacity")
        if self.cap % num_tiles != 0:
            raise CBContractError(
                f"First num_tiles={num_tiles} must evenly divide capacity={self.cap}"
            )

    def free(self) -> Size:
        return self.cap - (self.visible + self.reserved)

    def front_span(self, length: Size) -> Span:
        return Span(self.head, length)

    def back_span(self, length: Size) -> Span:
        """Return span at the back of the buffer for writing."""
        back_start = (self.head + self.visible) % self.cap
        return Span(back_start, length)

    def reset(self) -> None:
        self.buf[:] = [None] * self.cap
        self.head = 0
        self.visible = 0
        self.reserved = 0
        self.step = None
        self.last_wait_target = 0
        self.consumer_waiting = None
        self.producer_reserving = None
        self.configured = True
        with self.can_consume:
            self.can_consume.notify_all()
        with self.can_produce:
            self.can_produce.notify_all()
