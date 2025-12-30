# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TT-Lang Python Package

__version__ = "0.1.0"

# Export TTL DSL API module (decorators)
from ttlang import ttl

# Export TTL DSL API
from ttlang.ttl_api import (
    Program,
    CircularBuffer,
    TensorBlock,
    Semaphore,
    CopyTransferHandler,
)
from ttlang.circular_buffer import make_circular_buffer_like

# Export generated elementwise operators (auto-generated from TTLElementwiseOps.def)
from ttlang._generated_elementwise import *  # noqa: F401,F403
from ttlang._generated_elementwise import __all__ as _elementwise_all

__all__ = [
    "ttl",
    "Program",
    "CircularBuffer",
    "TensorBlock",
    "Semaphore",
    "CopyTransferHandler",
    "make_circular_buffer_like",
    # Elementwise operators are automatically included from generated file
    *_elementwise_all,
]
