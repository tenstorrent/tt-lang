# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DFBState: internal ring-buffer state for DataflowBuffer.

All counters (cap, head, visible, reserved) are in units of operations.  buf
is a list of cap slots; each slot is either None (empty) or a List[DFBSlot]
holding tiles_per_op tiles.  This means Block views are handed a direct
reference to a slot list -- no span arithmetic or ring-buffer wrapping needed
inside Block.
"""

from typing import List, Optional

from .errors import DFBNotConfigured
from .ttnnsim import Tensor
from .typedefs import Index, Shape, Size

# Type alias for a single tile slot
DFBSlot = Optional[Tensor]

# Type alias for one operation's tile list (one ring-buffer slot)
DFBOpSlot = Optional[List[DFBSlot]]


class DFBState:
    __slots__ = (
        "cap",  # capacity in operations (= buffer_factor)
        "tiles_per_op",  # tiles per operation (= math.prod(shape))
        "buf",  # ring buffer: List[DFBOpSlot], length = cap
        "head",  # current read slot index (in operations)
        "visible",  # number of complete operations ready to consume
        "reserved",  # number of complete operations reserved for writing
        "configured",
        "shape",  # tile-grid shape (for Block construction)
    )

    def __init__(self):
        self.cap: Size = 1
        self.tiles_per_op: Size = 1
        self.buf: List[DFBOpSlot] = []
        self.head: Index = 0
        self.visible: Size = 0
        self.reserved: Size = 0
        self.configured = False
        self.shape: Shape

    def require_configured(self) -> None:
        if not self.configured:
            raise DFBNotConfigured("DFB not configured; call host_configure_dfb")

    def free(self) -> Size:
        """Number of operation slots available for reservation."""
        return self.cap - self.visible - self.reserved

    def back_slot(self) -> Index:
        """Slot index where the next reservation will be placed."""
        return (self.head + self.visible) % self.cap

    def reset(self) -> None:
        self.buf[:] = [None] * self.cap
        self.head = 0
        self.visible = 0
        self.reserved = 0
        self.configured = True
