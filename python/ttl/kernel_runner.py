# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared kernel execution logic for tt-lang.

Provides functions for building kernel descriptors, CB descriptors, and
executing kernels on device via ttnn.generic_op. Used by both the Python
DSL (CompiledTTNNKernel) and ME2E tests.

This module provides a single reusable implementation of kernel argument
building and execution.
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
    num_cbs: int,
) -> List[Any]:
    """
    Build kernel descriptors for ttnn.generic_op.

    Args:
        kernel_specs: List of kernel specifications.
        tensors: List of ttnn.Tensor objects. Position in this list determines
            the global tensor index. Individual kernels access subsets via
            tensor_indices in each KernelSpec.
        tensor_accessor_args: Flattened compile-time args from all tensors.
        core_ranges: ttnn.CoreRangeSet for kernel execution.
        grid_cols: Number of grid columns (x dimension).
        grid_rows: Number of grid rows (y dimension).
        num_cbs: Total number of circular buffers (including intermediate CBs).

    Returns:
        List of ttnn.KernelDescriptor objects.
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    kernel_descriptors = []

    # CB indices are 0, 1, 2, ... for each CB (including intermediate CBs).
    cb_indices = list(range(num_cbs))

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
    cb_configs: List[Any],
    core_ranges: Any,
) -> List[Any]:
    """
    Build circular buffer descriptors for ttnn.generic_op.

    Args:
        tensors: List of ttnn.Tensor objects. Each tensor's position (0, 1, 2, ...)
            corresponds to its CB index. For intermediate CBs (not backed by
            input/output tensors), pass None in the corresponding position.
        cb_configs: List of CircularBuffer objects for each CB, indexed by CB index.
            Each CB has shape, buffer_factor, tensor (for dtype), and _cb_index attributes.
        core_ranges: ttnn.CoreRangeSet for CB allocation.

    Returns:
        List of ttnn.CBDescriptor objects.
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    cb_descriptors = []
    for i, cb in enumerate(cb_configs):
        if cb is None:
            raise ValueError(
                f"Missing CB config for index {i}. "
                f"All CB indices must have associated CircularBuffer configurations."
            )

        # Get dtype from CB's reference tensor.
        ref_tensor = cb.tensor
        if hasattr(ref_tensor, "dtype") and hasattr(ref_tensor.dtype, "name"):
            data_format = ref_tensor.dtype
        else:
            data_format = torch_dtype_to_ttnn_datatype(ref_tensor.dtype)

        page_size = tile_bytes_from_dtype(data_format)
        num_tiles = cb.shape[0] * cb.shape[1] * cb.buffer_factor
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
    cb_configs: List[Any],
    core_ranges: Any,
    program_hash: int = None,
) -> Any:
    """
    Execute kernels on device using ttnn.generic_op.

    This is the main entry point for kernel execution. It builds all
    descriptors and runs the program.

    Args:
        kernel_specs: List of kernel specifications (path, thread_type, tensor_indices, config).
        tensors: List of ttnn.Tensor objects. Position in this list determines the
            global tensor index. Individual kernels access subsets via tensor_indices
            in each KernelSpec.
        cb_configs: List of CircularBuffer objects for each CB, indexed by CB index.
            Includes both tensor-backed CBs and intermediate CBs. Each CB has shape,
            buffer_factor, tensor (for dtype), and _cb_index attributes.
        core_ranges: ttnn.CoreRangeSet for kernel execution.
        program_hash: Hash for tt-metal program cache (not yet used).

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
        num_cbs=len(cb_configs),
    )

    # Build CB descriptors.
    cb_descriptors = build_cb_descriptors(
        tensors=tensors,
        cb_configs=cb_configs,
        core_ranges=core_ranges,
    )

    # Build and execute program.
    # TODO: Enable custom_program_hash once tt-metal exposes it in Python bindings.
    # See tt-metal/ttnn/cpp/ttnn-nanobind/program_descriptors.cpp - needs to add
    # custom_program_hash parameter to ProgramDescriptor binding.
    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=[],
        # custom_program_hash=program_hash,
    )

    return ttnn.generic_op(list(tensors), program)


__all__ = [
    "KernelSpec",
    "build_tensor_accessor_args",
    "build_kernel_descriptors",
    "build_cb_descriptors",
    "run_kernel_on_device",
]
