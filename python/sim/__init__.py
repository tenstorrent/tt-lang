# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
sim package: simulation components for TT-Lang including dataflow buffers, tensors, and copy operations.
"""

from typing import Any
import types
from . import ttnnsim as ttnn
from .dfb import DFBStats
from .constants import TILE_SHAPE
from .copy import CopyTransaction, GroupTransfer, copy
from .decorators import compute, datamovement
from .corecontext import node
from .operation import operation
from .pipe import DstPipeIdentity, DstT, Pipe, PipeNet, SrcPipeIdentity
from .program import Program
from .ttnnsim import TTNN_AVAILABLE, ROW_MAJOR_LAYOUT, TILE_LAYOUT
from .typedefs import CoreCoord, CoreRange, Shape


class _SignpostContextManager:
    """No-op context manager for ttl.signpost stub."""

    def __enter__(self) -> "_SignpostContextManager":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        return None


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
        from .dfb import make_dataflow_buffer_like
        from .constants import TILE_SHAPE
        from .copy import copy
        from .decorators import compute, datamovement
        from .corecontext import node, grid_size
        from .operation import operation
        from . import math as math_module
        from .pipe import DstPipeIdentity, DstT, Pipe, PipeNet, SrcPipeIdentity
        from .program import Program
        from .typedefs import CoreCoord, CoreRange, Shape, Size

        self.operation = operation
        self.grid_size = grid_size
        self.make_dataflow_buffer_like = make_dataflow_buffer_like
        self.compute = compute
        self.datamovement = datamovement
        self.node = node
        self.copy = copy
        self.GroupTransfer = GroupTransfer
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
        self.TILE_LAYOUT = TILE_LAYOUT
        self.ROW_MAJOR_LAYOUT = ROW_MAJOR_LAYOUT
        self.Program = Program
        self.math = _TTLMathNamespace()

    @staticmethod
    def signpost(*args: Any, **kwargs: Any) -> _SignpostContextManager:
        """Signpost stub for simulator. Returns a no-op context manager."""
        return _SignpostContextManager()


ttl = _TTLNamespace()

__all__ = [
    "DFBStats",
    "CoreCoord",
    "CoreRange",
    "DstT",
    "Shape",
    "Pipe",
    "PipeNet",
    "SrcPipeIdentity",
    "DstPipeIdentity",
    "TILE_SHAPE",
    "copy",
    "CopyTransaction",
    "GroupTransfer",
    "Program",
    "node",
    "compute",
    "datamovement",
    "operation",
    "ttl",
    "ttnn",
    "TTNN_AVAILABLE",
    "TILE_LAYOUT",
    "ROW_MAJOR_LAYOUT",
]
