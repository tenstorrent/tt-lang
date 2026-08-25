# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTL DSL module providing the unified ttl.* API namespace.

Decorators:
    @ttl.operation() - Define a kernel function
    @ttl.compute() - Define a compute thread (auto-collected)
    @ttl.datamovement() - Define a data movement thread (auto-collected)

Functions:
    ttl.make_dataflow_buffer_like() - Create a dataflow buffer
    ttl.make_tensor_backed_dfb() - Bind a dataflow buffer to tensor L1 storage
    ttl.copy() - Asynchronous data transfer
    ttl.node(dims=2) - Get current core's coordinates as (x, y) tuple
    ttl.grid_size(dims=2) - Get grid size as (x_size, y_size) tuple

Math operations:
    ttl.math.sqrt(), ttl.math.exp(), etc.
"""

from .ttl_api import compute, datamovement
from .atom import operation, DFB
from .kernel import Kernel, KernelKind
from .runtime_resources import (
    CoreRuntimeArgs,
    KernelDefine,
    KernelRuntimeResources,
    ProgramRuntimeResources,
)
from .condition import DispatchCondition
from .dfb_reset import DFBReset
from .dfb_allocation_group import DFBAllocationGroup, make_dfb_allocation_group
from .scalar import ScalarType
from .dataflow_buffer import (
    make_dataflow_buffer_like,
    make_dfb,
    make_tensor_backed_dfb,
)
from .operators import (
    DFBAccess,
    DFBEffect,
    call_extern_func,
    copy,
    dfb_descriptor,
    get_dfb_id,
    grid_size,
    matmul,
    node,
    raw_addr,
    reset_all_dfbs,
    reset_dfbs,
)

# Math operations namespace
from . import ttl_math as math

__all__ = [
    "operation",
    "DFB",
    "Kernel",
    "KernelKind",
    "CoreRuntimeArgs",
    "KernelDefine",
    "KernelRuntimeResources",
    "ProgramRuntimeResources",
    "DispatchCondition",
    "DFBReset",
    "DFBAllocationGroup",
    "ScalarType",
    "compute",
    "datamovement",
    "make_dataflow_buffer_like",
    "make_dfb",
    "make_tensor_backed_dfb",
    "make_dfb_allocation_group",
    "copy",
    "node",
    "grid_size",
    "matmul",
    "call_extern_func",
    "DFBEffect",
    "DFBAccess",
    "dfb_descriptor",
    "get_dfb_id",
    "raw_addr",
    "reset_dfbs",
    "reset_all_dfbs",
    "math",
]
