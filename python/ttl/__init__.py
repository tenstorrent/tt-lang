# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TT-Lang Python Package

__version__ = "0.1.0"

# Export TTL DSL API directly at package level so `import ttl; ttl.kernel` works
from ttl.ttl import (
    kernel,
    compute,
    datamovement,
    Program,
    make_circular_buffer_like,
    copy,
    core,
    grid_size,
    math,
)

# Export generated elementwise operators (auto-generated from TTLElementwiseOps.def)
from ttl._generated_elementwise import *  # noqa: F401,F403
from ttl._generated_elementwise import __all__ as _elementwise_all

# Export additional TTL DSL API classes
from ttl.ttl_api import (
    CircularBuffer,
    CopyTransferHandler,
    TensorBlock,
)
from ttlang.operators import matmul, power, reduce_max, reduce_sum, transpose, where

__all__ = [
    "kernel",
    "compute",
    "datamovement",
    "Program",
    "CircularBuffer",
    "TensorBlock",
    "CopyTransferHandler",
    "make_circular_buffer_like",
    "copy",
    "core",
    "grid_size",
    "math",
    "matmul",
    "power",
    "reduce_max",
    "reduce_sum",
    "transpose",
    "where",
    # Elementwise operators are automatically included from generated file
    *_elementwise_all,
]
