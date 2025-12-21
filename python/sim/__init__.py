# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
sim package: simulation components for TT-Lang including circular buffers, tensors, and copy operations.
"""

from .cbapi import CBAPI, CBStats
from .tensoraccessor import TensorAccessor
from .typedefs import IndexType, CoreIndex, Shape, Pipe
from .constants import TILE_SHAPE, MAX_CBS
from .cb import CircularBuffer
from .copy import copy, CopyTransaction
from .program import Program, BindableTemplate
from .kernel import core, flatten_core_index
from .decorators import compute, datamovement
from .kernel import kernel
from .testing import assert_pcc
from .torch_utils import is_tiled
from .pipe import if_pipe_src, if_pipe_dst, core_in_pipe
from . import torch_utils


# Create ttl namespace object
class _TTLNamespace:
    """TT-Lang namespace for DSL constructs."""

    def __init__(self):
        from .kernel import kernel, grid_size, core, flatten_core_index
        from .cb import CircularBuffer, make_circular_buffer_like
        from .decorators import compute, datamovement
        from .program import Program
        from .copy import copy
        from .typedefs import Pipe, IndexType
        from .torch_utils import is_tiled
        from .constants import TILE_SHAPE
        from .tensoraccessor import TensorAccessor
        from .pipe import if_pipe_src, if_pipe_dst, core_in_pipe

        self.kernel = kernel
        self.grid_size = grid_size
        self.CircularBuffer = CircularBuffer
        self.make_circular_buffer_like = make_circular_buffer_like
        self.compute = compute
        self.datamovement = datamovement
        self.core = core
        self.flatten_core_index = flatten_core_index
        self.copy = copy
        self.Pipe = Pipe
        self.is_tiled = is_tiled
        self.TILE_SHAPE = TILE_SHAPE
        self.TensorAccessor = TensorAccessor
        self.IndexType = IndexType
        self.Program = Program
        self.if_pipe_src = if_pipe_src
        self.if_pipe_dst = if_pipe_dst
        self.core_in_pipe = core_in_pipe


ttl = _TTLNamespace()

__all__ = [
    "CBAPI",
    "CBStats",
    "TensorAccessor",
    "IndexType",
    "CoreIndex",
    "Shape",
    "Pipe",
    "TILE_SHAPE",
    "MAX_CBS",
    "CircularBuffer",
    "copy",
    "CopyTransaction",
    "Program",
    "BindableTemplate",
    "core",
    "flatten_core_index",
    "compute",
    "datamovement",
    "kernel",
    "assert_pcc",
    "is_tiled",
    "if_pipe_src",
    "if_pipe_dst",
    "core_in_pipe",
    "torch_utils",
    "ttl",
]
