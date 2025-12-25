# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Builder module for constructing TTL MLIR via Python bindings.

This module provides utilities for building MLIR directly using Python bindings,
bypassing the TTL DSL front-end. This enables testing the compiler middle-end
in isolation.

Components:
- ttl_builder: Build TTL modules programmatically.
- dm_threads: Data movement thread templates (reader/writer).
- pipeline: Pass pipeline execution.
- kernels: Kernel execution utilities.
"""

from .ttl_builder import build_ttl_module
from .pipeline import compile_ttl_to_ttkernel

__all__ = [
    "build_ttl_module",
    "compile_ttl_to_ttkernel",
]
