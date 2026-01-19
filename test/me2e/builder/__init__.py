# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Builder module for constructing TTL MLIR via Python bindings.

This module provides utilities for building MLIR directly using Python bindings,
bypassing the TTL DSL front-end. This enables testing the compiler middle-end
in isolation.

Components:
- thread_builder: Base class with MLIR building blocks.
- dm_builder: Data movement thread builder (reader/writer).
- compute_builder: Compute thread builder.
- ttl_builder: Build TTL modules programmatically.
- pipeline: Pass pipeline execution.
- kernels: Kernel translation utilities.
- ttnn_runner: Device execution harness.
- dtype_utils: Shared dtype conversion utilities.
"""

from .thread_builder import (
    ThreadBuilder,
    StringBasedThreadBuilder,
    ThreadType,
    LoopContext,
    generate_layout_attrs,
)
from .dm_builder import DMThreadBuilder
from .compute_builder import ComputeThreadBuilder
from .ttl_builder import (
    build_ttl_module,
    build_e2e_module,
    build_e2e_module_mlir,
    build_e2e_module_mlir_custom,
    build_ttl_module_from_mlir,
)
from .pipeline import compile_ttl_to_ttkernel
from .kernels import translate_module_to_kernels, write_kernels, load_kernel_metadata
from .ttnn_runner import run_binary_op, run_unary_op
from .dtype_utils import torch_dtype_to_mlir_str, torch_dtype_to_ttcore_datatype
from .device_arch import get_mock_arch_from_device

__all__ = [
    # Thread builders
    "ThreadBuilder",
    "StringBasedThreadBuilder",
    "ThreadType",
    "LoopContext",
    "generate_layout_attrs",
    "DMThreadBuilder",
    "ComputeThreadBuilder",
    # Module builders
    "build_ttl_module",
    "build_e2e_module",
    "build_e2e_module_mlir",
    "build_e2e_module_mlir_custom",
    "build_ttl_module_from_mlir",
    # Pipeline and execution
    "compile_ttl_to_ttkernel",
    "translate_module_to_kernels",
    "write_kernels",
    "load_kernel_metadata",
    "run_binary_op",
    "run_unary_op",
    # Utilities
    "torch_dtype_to_mlir_str",
    "torch_dtype_to_ttcore_datatype",
    "get_mock_arch_from_device",
]
