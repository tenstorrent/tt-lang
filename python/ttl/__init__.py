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
    make_dataflow_buffer_like,
    copy,
    core,
    grid_size,
    math,
)

# Export generated elementwise operators (auto-generated from TTLElementwiseOps.def)
from ttl._generated_elementwise import *  # noqa: F401,F403
from ttl._generated_elementwise import __all__ as _elementwise_all

# Export additional TTL DSL API classes
from ttl.operators import signpost
from ttl.compiler_options import CompilerOptions
from ttl.ttl_api import (
    CircularBuffer,
    CopyTransferHandler,
    TensorBlock,
)

__all__ = [
    "kernel",
    "compute",
    "datamovement",
    "Program",
    "CircularBuffer",
    "CompilerOptions",
    "TensorBlock",
    "CopyTransferHandler",
    "make_dataflow_buffer_like",
    "copy",
    "core",
    "grid_size",
    "math",
    "signpost",
    # Elementwise operators are automatically included from generated file
    *_elementwise_all,
]
