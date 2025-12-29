# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TT-Lang Python Package

__version__ = "0.1.0"

# Export TTL DSL API
from ttlang.ttl_api import (
    pykernel_gen,
    Program,
    CircularBuffer,
    TensorAccessor,
    compute,
    datamovement,
    TensorBlock,
    Semaphore,
    CopyTransferHandler,
)

# Export generated elementwise operators (auto-generated from TTLElementwiseOps.def)
from ttlang._generated_elementwise import *  # noqa: F401,F403
from ttlang._generated_elementwise import __all__ as _elementwise_all

__all__ = [
    "pykernel_gen",
    "Program",
    "CircularBuffer",
    "TensorAccessor",
    "compute",
    "datamovement",
    "TensorBlock",
    "Semaphore",
    "CopyTransferHandler",
    # Elementwise operators are automatically included from generated file
    *_elementwise_all,
]
