# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTL DSL module providing the unified ttl.* API namespace.

Decorators:
    @ttl.kernel() - Define a kernel function
    @ttl.compute() - Define a compute thread
    @ttl.datamovement() - Define a data movement thread

Classes:
    ttl.Program - Kernel program executor
    ttl.Semaphore - Multi-core synchronization primitive

Functions:
    ttl.make_circular_buffer_like() - Create a circular buffer
    ttl.copy() - Asynchronous data transfer
    ttl.core_x() - Get current core's X coordinate (column)
    ttl.core_y() - Get current core's Y coordinate (row)

Math operations:
    ttl.math.sqrt(), ttl.math.exp(), etc.
"""

from .ttl_api import pykernel_gen as kernel, compute, datamovement, Program
from .circular_buffer import make_circular_buffer_like
from .operators import copy, core_x, core_y
from .semaphore import Semaphore

# Math operations namespace
from . import ttl_math as math

__all__ = [
    "kernel",
    "compute",
    "datamovement",
    "Program",
    "make_circular_buffer_like",
    "copy",
    "core_x",
    "core_y",
    "Semaphore",
    "math",
]
