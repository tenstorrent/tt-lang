# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TT-Lang Python Package

__version__ = "0.1.0"

# Export D2M DSL API
from ttlang.d2m_api import (
    pykernel_gen,
    Program,
    CircularBuffer,
    Stream,
    dma,
    compute,
    datamovement,
    TensorBlock,
    Semaphore,
    MemTx,
)

__all__ = [
    "pykernel_gen",
    "Program",
    "CircularBuffer",
    "Stream",
    "dma",
    "compute",
    "datamovement",
    "TensorBlock",
    "Semaphore",
    "MemTx",
]
