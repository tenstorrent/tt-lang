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
from .pipe import DstPipeIdentity, PipeNet, SrcPipeIdentity
from .program import Program
from .ttnnsim import TTNN_AVAILABLE
from .typedefs import CoreCoord, CoreRange, DstT, Pipe, Shape


# Create ttl.math namespace object
class _TTLMathNamespace:
    """TT-Lang math namespace for block math functions.

    Auto-loads all functions from the math module, including auto-generated
    functions from the PyTorch mapping.
    """

    def __init__(self):
        from . import math as math_module

        # Manually add special functions that need custom logic
        self.broadcast = math_module.broadcast
        self.reduce_max = math_module.reduce_max
        self.reduce_sum = math_module.reduce_sum

        # Auto-load all other functions from the math module
        # This includes all auto-generated unary operations
        for name in dir(math_module):
            if not name.startswith("_") and not hasattr(self, name):
                attr = getattr(math_module, name)
                if callable(attr):
                    setattr(self, name, attr)


# Create ttl namespace object
class _TTLNamespace:
    """TT-Lang namespace for DSL constructs."""

    def __init__(self):
        from .cb import make_circular_buffer_like
        from .constants import TILE_SHAPE
        from .copy import copy
        from .decorators import compute, datamovement
        from .kernel import core, grid_size, kernel
        from . import math as math_module
        from .pipe import DstPipeIdentity, PipeNet, SrcPipeIdentity
        from .program import Program
        from .typedefs import CoreCoord, CoreRange, DstT, Pipe, Shape, Size

        self.kernel = kernel
        self.grid_size = grid_size
        self.make_circular_buffer_like = make_circular_buffer_like
        self.compute = compute
        self.datamovement = datamovement
        self.core = core
        self.copy = copy
        self.transpose = math_module.transpose
        self.Pipe = Pipe
        self.PipeNet = PipeNet
        self.SrcPipeIdentity = SrcPipeIdentity
        self.DstPipeIdentity = DstPipeIdentity
        self.CoreCoord = CoreCoord
        self.CoreRange = CoreRange
        self.DstT = DstT
        self.Size = Size
        self.Shape = Shape
        self.TILE_SHAPE = TILE_SHAPE
        self.Program = Program
        self.math = _TTLMathNamespace()


ttl = _TTLNamespace()

__all__ = [
    "CBAPI",
    "CBStats",
    "CoreCoord",
    "CoreRange",
    "DstT",
    "Shape",
    "Pipe",
    "PipeNet",
    "SrcPipeIdentity",
    "DstPipeIdentity",
    "TILE_SHAPE",
    "MAX_CBS",
    "copy",
    "CopyTransaction",
    "Program",
    "core",
    "compute",
    "datamovement",
    "kernel",
    "ttl",
    "ttnn",
    "TTNN_AVAILABLE",
]
