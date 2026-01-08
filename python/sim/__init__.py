# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
sim package: simulation components for TT-Lang including circular buffers, tensors, and copy operations.
"""

from . import ttnnsim as ttnn
from .cbapi import CBAPI, CBStats
from .constants import MAX_CBS, TILE_SHAPE
from .copy import CopyTransaction, copy
from .decorators import compute, datamovement
from .kernel import core, kernel
from .pipe import core_in_pipe, if_pipe_dst, if_pipe_src
from .program import Program
from .typedefs import CoreIndex, Pipe, Shape


# Create ttl namespace object
class _TTLNamespace:
    """TT-Lang namespace for DSL constructs."""

    def __init__(self):
        from .cb import make_circular_buffer_like
        from .constants import TILE_SHAPE
        from .copy import copy
        from .decorators import compute, datamovement
        from .kernel import core, grid_size, kernel
        from .pipe import core_in_pipe, if_pipe_dst, if_pipe_src
        from .program import Program
        from .typedefs import Pipe, Shape, Size

        self.kernel = kernel
        self.grid_size = grid_size
        self.make_circular_buffer_like = make_circular_buffer_like
        self.compute = compute
        self.datamovement = datamovement
        self.core = core
        self.copy = copy
        self.Pipe = Pipe
        self.Size = Size
        self.Shape = Shape
        self.TILE_SHAPE = TILE_SHAPE
        self.Program = Program
        self.if_pipe_src = if_pipe_src
        self.if_pipe_dst = if_pipe_dst
        self.core_in_pipe = core_in_pipe


ttl = _TTLNamespace()

__all__ = [
    "CBAPI",
    "CBStats",
    "CoreIndex",
    "Shape",
    "Pipe",
    "TILE_SHAPE",
    "MAX_CBS",
    "copy",
    "CopyTransaction",
    "Program",
    "core",
    "compute",
    "datamovement",
    "kernel",
    "if_pipe_src",
    "if_pipe_dst",
    "core_in_pipe",
    "ttl",
    "ttnn",
]
