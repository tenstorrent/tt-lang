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
- kernels: Kernel translation utilities.
- ttnn_runner: Device execution harness.
- dtype_utils: Shared dtype conversion utilities.
- system_desc: System descriptor utilities.
"""

from .ttl_builder import build_ttl_module, build_e2e_module, build_e2e_module_mlir
from .pipeline import compile_ttl_to_ttkernel
from .kernels import translate_module_to_kernels, write_kernels
from .ttnn_runner import run_binary_op, run_unary_op
from .dtype_utils import torch_dtype_to_mlir_str, torch_dtype_to_ttcore_datatype
from .system_desc import get_system_desc_path

__all__ = [
    "build_ttl_module",
    "build_e2e_module",
    "build_e2e_module_mlir",
    "compile_ttl_to_ttkernel",
    "translate_module_to_kernels",
    "write_kernels",
    "run_binary_op",
    "run_unary_op",
    "torch_dtype_to_mlir_str",
    "torch_dtype_to_ttcore_datatype",
    "get_system_desc_path",
]
