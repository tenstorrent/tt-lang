# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TT-Lang Python Package

from ttl.version import __version__

# Export TTL DSL API directly at package level so `import ttl; ttl.operation` works
from ttl.ttl import (
    operation,
    compute,
    datamovement,
    Program,
    make_dataflow_buffer_like,
    copy,
    node,
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
from ttl.pipe import Pipe, PipeNet

__all__ = [
    "operation",
    "compute",
    "datamovement",
    "Program",
    "CircularBuffer",
    "CompilerOptions",
    "TensorBlock",
    "CopyTransferHandler",
    "Pipe",
    "PipeNet",
    "make_dataflow_buffer_like",
    "copy",
    "node",
    "grid_size",
    "math",
    "signpost",
    # Elementwise operators are automatically included from generated file
    *_elementwise_all,
]
