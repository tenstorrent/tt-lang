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

Functions:
    ttl.make_circular_buffer_like() - Create a circular buffer
    ttl.copy() - Asynchronous data transfer
    ttl.core(dims=2) - Get current core's coordinates as (x, y) tuple
    ttl.grid_size(dims=2) - Get grid size as (x_size, y_size) tuple

Math operations:
    ttl.math.sqrt(), ttl.math.exp(), etc.
"""

from .ttl_api import pykernel_gen as kernel, compute, datamovement, Program
from .circular_buffer import make_circular_buffer_like
from .operators import copy, core, grid_size

# Math operations namespace
from . import ttl_math as math

__all__ = [
    "kernel",
    "compute",
    "datamovement",
    "Program",
    "make_circular_buffer_like",
    "copy",
    "core",
    "grid_size",
    "math",
]
