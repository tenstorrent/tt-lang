# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared kernel execution logic for tt-lang.

Provides functions for building kernel descriptors, CB descriptors, and
executing kernels on device via ttnn.generic_op. Used by both the Python
DSL (CompiledTTNNKernel) and ME2E tests.

This module ensures a single source of truth for kernel argument building.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

try:
    import ttnn
except (ModuleNotFoundError, ImportError):
    ttnn = None

from .dtype_utils import tile_bytes_from_dtype, torch_dtype_to_ttnn_datatype


@dataclass
class KernelSpec:
    """Specification for a single kernel to execute.

    Attributes:
        path: Path to the kernel C++ source file.
        thread_type: Type of kernel ("compute", "noc", or "ethernet").
        tensor_indices: List of global tensor indices this kernel accesses.
            For DM kernels, these determine which buffer addresses go in
            common_runtime_args, in order.
        config: Kernel config descriptor (ComputeConfigDescriptor,
            ReaderConfigDescriptor, WriterConfigDescriptor, or EthernetConfigDescriptor).
    """

    path: str
    thread_type: str
    tensor_indices: List[int]
    config: Any


def build_tensor_accessor_args(tensors: List[Any]) -> List[int]:
    """
    Build compile-time args for tensor accessors.

    Args:
        tensors: List of ttnn.Tensor objects on device.

    Returns:
        List of compile-time args (flattened TensorAccessorArgs for all tensors).
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    args = []
    for tensor in tensors:
        tensor_args = ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
        args.extend(tensor_args)
    return args


def build_kernel_descriptors(
    kernel_specs: List[KernelSpec],
    tensors: List[Any],
    tensor_accessor_args: List[int],
    core_ranges: Any,
    grid_cols: int,
    grid_rows: int,
) -> List[Any]:
    """
    Build kernel descriptors for ttnn.generic_op.

    Args:
        kernel_specs: List of kernel specifications.
        tensors: List of ttnn.Tensor objects (in global order).
        tensor_accessor_args: Flattened compile-time args from all tensors.
        core_ranges: ttnn.CoreRangeSet for kernel execution.
        grid_cols: Number of grid columns (x dimension).
        grid_rows: Number of grid rows (y dimension).

    Returns:
        List of ttnn.KernelDescriptor objects.
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    kernel_descriptors = []

    # CB indices are 0, 1, 2, ... for each tensor.
    cb_indices = list(range(len(tensors)))

    for spec in kernel_specs:
        # runtime_args structure: [x][y][args_per_core].
        # Each core gets an empty arg list (we use my_x/my_y for indexing).
        runtime_args = [[[] for _ in range(grid_rows)] for _ in range(grid_cols)]

        # Build common_runtime_args using tensor_indices.
        # C++ indexes by function-local position, we provide addresses in that order.
        common_runtime_args = [
            tensors[idx].buffer_address() for idx in spec.tensor_indices
        ]

        # Compute kernels only need CB indices.
        # DM kernels need CB indices + TensorAccessorArgs config.
        if spec.thread_type == "compute":
            kernel_compile_time_args = cb_indices
        else:
            kernel_compile_time_args = cb_indices + list(tensor_accessor_args)

        kernel_desc = ttnn.KernelDescriptor(
            kernel_source=spec.path,
            core_ranges=core_ranges,
            compile_time_args=kernel_compile_time_args,
            runtime_args=runtime_args,
            common_runtime_args=common_runtime_args,
            config=spec.config,
        )
        kernel_descriptors.append(kernel_desc)

    return kernel_descriptors


def build_cb_descriptors(
    tensors: List[Any],
    cb_configs: List[Optional[Tuple[Tuple[int, int], int]]],
    core_ranges: Any,
) -> List[Any]:
    """
    Build circular buffer descriptors for ttnn.generic_op.

    Args:
        tensors: List of ttnn.Tensor objects.
        cb_configs: List of (shape, buffer_factor) tuples for each CB, indexed by cb_index.
            shape is (rows, cols) in tiles.
        core_ranges: ttnn.CoreRangeSet for CB allocation.

    Returns:
        List of ttnn.CBDescriptor objects.
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    cb_descriptors = []
    for i, tensor in enumerate(tensors):
        if hasattr(tensor, "dtype") and hasattr(tensor.dtype, "name"):
            data_format = tensor.dtype
        else:
            data_format = torch_dtype_to_ttnn_datatype(tensor.dtype)

        page_size = tile_bytes_from_dtype(data_format)

        if i >= len(cb_configs) or cb_configs[i] is None:
            raise ValueError(
                f"Missing CB config for tensor {i}. "
                f"Expected {len(tensors)} CB configs, got {len(cb_configs)}. "
                f"All tensors must have associated CircularBuffer configurations."
            )
        shape, buffer_factor = cb_configs[i]
        num_tiles = shape[0] * shape[1] * buffer_factor
        total_size = num_tiles * page_size

        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=i,
            data_format=data_format,
            page_size=page_size,
        )
        cb_desc = ttnn.CBDescriptor(
            total_size=total_size,
            core_ranges=core_ranges,
            format_descriptors=[cb_format],
        )
        cb_descriptors.append(cb_desc)

    return cb_descriptors


def run_kernel_on_device(
    kernel_specs: List[KernelSpec],
    tensors: List[Any],
    cb_configs: List[Optional[Tuple[Tuple[int, int], int]]],
    core_ranges: Any,
) -> Any:
    """
    Execute kernels on device using ttnn.generic_op.

    This is the main entry point for kernel execution. It builds all
    descriptors and runs the program.

    Args:
        kernel_specs: List of kernel specifications (path, thread_type, tensor_indices, config).
        tensors: List of ttnn.Tensor objects (in global order matching CB indices).
        cb_configs: List of (shape, buffer_factor) tuples for each CB.
        core_ranges: ttnn.CoreRangeSet for kernel execution.

    Returns:
        Result from ttnn.generic_op (typically None or output tensor).
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    # Build tensor accessor args.
    tensor_accessor_args = build_tensor_accessor_args(tensors)

    # Get grid dimensions from core_ranges.
    grid_size = core_ranges.bounding_box().grid_size()
    grid_cols = grid_size.x
    grid_rows = grid_size.y

    # Build kernel descriptors.
    kernel_descriptors = build_kernel_descriptors(
        kernel_specs=kernel_specs,
        tensors=tensors,
        tensor_accessor_args=tensor_accessor_args,
        core_ranges=core_ranges,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )

    # Build CB descriptors.
    cb_descriptors = build_cb_descriptors(
        tensors=tensors,
        cb_configs=cb_configs,
        core_ranges=core_ranges,
    )

    # Build and execute program.
    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=[],
    )

    return ttnn.generic_op(list(tensors), program)


__all__ = [
    "KernelSpec",
    "build_tensor_accessor_args",
    "build_kernel_descriptors",
    "build_cb_descriptors",
    "run_kernel_on_device",
]
