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
from .typedefs import IndexType, CoreIndex, Shape, MulticastAddress, MulticastType
from .constants import TILE_SHAPE, MAX_CBS
from .cb import CircularBuffer
from .dma import dma, DMATransaction
from .program import Program, BindableTemplate, core_index
from .decorators import compute, datamovement
from .kernel import kernel
from .testing import assert_pcc
from .torch_utils import is_tiled
from . import torch_utils

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
    "CoreIndex",
    "Shape",
    "MulticastAddress",
    "MulticastType",
    "TILE_SHAPE",
    "MAX_CBS",
    "CircularBuffer",
    "dma",
    "DMATransaction",
    "Program",
    "BindableTemplate",
    "core_index",
    "compute",
    "datamovement",
    "kernel",
    "assert_pcc",
    "is_tiled",
    "torch_utils",
]
