# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
sim package: simulation components for TT-Lang including circular buffers, tensors, and DMA operations.
"""

from .cbapi import (
    CBAPI,
    CBStats,
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
from .tensoraccessor import TensorAccessor
from .index_type import IndexType
from .constants import TILE_SIZE
from .circular_buffer import CircularBuffer
from .dma import dma, DMATransaction

__all__ = [
    "CBAPI",
    "CBStats",
    "host_configure_cb",
    "host_reset_cb",
    "cb_stats",
    "cb_pages_available_at_front",
    "cb_pages_reservable_at_back",
    "cb_wait_front",
    "cb_reserve_back",
    "cb_push_back",
    "cb_pop_front",
    "get_read_ptr",
    "get_write_ptr",
    "TensorAccessor",
    "IndexType",
    "TILE_SIZE",
    "CircularBuffer",
    "dma",
    "DMATransaction",
]
