# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
sim package: simulation components for TT-Lang including circular buffers, tensors, and copy operations.
"""

from .cbapi import CBAPI, CBStats
from .typedefs import CoreIndex, Shape, Pipe
from .constants import TILE_SHAPE, MAX_CBS
from .copy import copy, CopyTransaction
from .program import Program
from .kernel import core
from .decorators import compute, datamovement
from .kernel import kernel
from .pipe import if_pipe_src, if_pipe_dst, core_in_pipe
from . import ttnnsim as ttnn


# Create ttl namespace object
class _TTLNamespace:
    """TT-Lang namespace for DSL constructs."""

    def __init__(self):
        from .kernel import kernel, grid_size, core
        from .cb import make_circular_buffer_like
        from .decorators import compute, datamovement
        from .program import Program
        from .copy import copy
        from .typedefs import Pipe, Size, Shape
        from .constants import TILE_SHAPE
        from .pipe import if_pipe_src, if_pipe_dst, core_in_pipe

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
