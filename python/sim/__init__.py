# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
sim package: simulation components for TT-Lang including circular buffers, tensors, and copy operations.
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
from .typedefs import IndexType, CoreIndex, Shape, MulticastAddress
from .constants import TILE_SHAPE, MAX_CBS
from .cb import CircularBuffer
from .copy import copy, CopyTransaction
from .program import Program, BindableTemplate, core
from .decorators import compute, datamovement
from .kernel import kernel
from .testing import assert_pcc
from .torch_utils import is_tiled
from . import torch_utils


# Create ttl namespace object
class _TTLNamespace:
    """TT-Lang namespace for DSL constructs."""

    def __init__(self):
        from .kernel import kernel, grid_size
        from .cb import CircularBuffer, make_circular_buffer_like
        from .decorators import compute, datamovement
        from .program import core
        from .copy import copy

        self.kernel = kernel
        self.grid_size = grid_size
        self.CircularBuffer = CircularBuffer
        self.make_circular_buffer_like = make_circular_buffer_like
        self.compute = compute
        self.datamovement = datamovement
        self.core = core
        self.copy = copy


ttl = _TTLNamespace()

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
    "TILE_SHAPE",
    "MAX_CBS",
    "CircularBuffer",
    "copy",
    "CopyTransaction",
    "Program",
    "BindableTemplate",
    "core",
    "compute",
    "datamovement",
    "kernel",
    "assert_pcc",
    "is_tiled",
    "torch_utils",
    "ttl",
]
