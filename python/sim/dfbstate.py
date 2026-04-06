# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DFBState: internal ring-buffer state for DataflowBuffer.

All counters (cap, head, visible, reserved) are in units of operations.  buf
is a list of cap slots; each slot is either None (empty) or a single Tensor
holding the entire operation's data in element space (shape determined by the
DataflowBuffer's tile-grid shape and TILE_SHAPE).  Block views hold a
reference to the same Tensor object so that in-place writes are immediately
visible in the ring buffer.
"""

from typing import List, Optional

from .ttnnsim import Tensor
from .typedefs import Index, Shape, Size


class DFBState:
    __slots__ = (
        "cap",  # capacity in operations (= block_count)
        "buf",  # ring buffer: List[Optional[Tensor]], length = cap
        "head",  # current read slot index (in operations)
        "visible",  # number of complete operations ready to consume
        "reserved",  # number of complete operations reserved for writing
        "shape",  # tile-grid shape (for Block construction)
    )

    def __init__(self):
        self.cap: Size = 1
        self.buf: List[Optional[Tensor]] = []
        self.head: Index = 0
        self.visible: Size = 0
        self.reserved: Size = 0
        self.shape: Shape

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
