# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
_CState and related internal state management for dfbsim.
"""

from threading import Condition, RLock, Thread
from typing import List, Optional

from .errors import DFBContractError, DFBNotConfigured
from .ttnnsim import Tensor
from .typedefs import Count, Index, Shape, Size, Span

# Type alias for circular buffer slots
DFBSlot = Optional[Tensor]


# It is a deliberate design choice to use any generic type here to avoid dealing
# with byte arrays as would be the case in the C++ API.
class DFBState:
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
        "shape",
    )

    def __init__(self):
        self.cap: Size = 1
        self.buf: List[DFBSlot] = []
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
        self.shape: Shape  # Shape in tiles (rows, cols)

    def require_configured(self) -> None:
        if not self.configured:
            raise DFBNotConfigured("DFB not configured; call host_configure_dfb")

    def check_num_tiles(self, num_tiles: Size) -> None:
        if num_tiles > self.cap:
            raise DFBContractError("num_tiles must be <= capacity")
        if self.cap % num_tiles != 0:
            raise DFBContractError(
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
