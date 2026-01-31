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
import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import ttnn
except (ModuleNotFoundError, ImportError):
    ttnn = None

from .dtype_utils import tile_bytes_from_dtype, torch_dtype_to_ttnn_datatype


@dataclass
class PipeConnection:
    """Information about a pipe connection from the pipe graph.

    Attributes:
        srcX, srcY: Sender core coordinates.
        dstStartX, dstStartY: Receiver start coordinates.
        dstEndX, dstEndY: Receiver end coordinates.
        receiverCBIndex: CB index used by the receiver.
        runtimeArgSlot: Slot in runtime args for this pipe's receiver CB address.
    """

    srcX: int
    srcY: int
    dstStartX: int
    dstStartY: int
    dstEndX: int
    dstEndY: int
    receiverCBIndex: int
    runtimeArgSlot: int


def load_pipe_graph(json_path: Optional[str] = None) -> Optional[List[PipeConnection]]:
    """Load pipe graph from JSON file.

    Args:
        json_path: Path to the JSON file. If None, checks TTLANG_PIPE_GRAPH_JSON
            environment variable.

    Returns:
        List of PipeConnection objects, or None if no pipe graph available.
    """
    if json_path is None:
        json_path = os.environ.get("TTLANG_PIPE_GRAPH_JSON")

    if not json_path or not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        pipes = []
        for p in data.get("pipes", []):
            pipes.append(
                PipeConnection(
                    srcX=p["srcX"],
                    srcY=p["srcY"],
                    dstStartX=p["dstStartX"],
                    dstStartY=p["dstStartY"],
                    dstEndX=p["dstEndX"],
                    dstEndY=p["dstEndY"],
                    receiverCBIndex=p["receiverCBIndex"],
                    runtimeArgSlot=p["runtimeArgSlot"],
                )
            )
        return pipes
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Failed to load pipe graph from {json_path}: {e}")
        return None


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
    pipe_graph: Optional[List[PipeConnection]] = None,
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
        pipe_graph: Optional pipe graph for populating sender runtime args.

    Returns:
        List of ttnn.KernelDescriptor objects.
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    kernel_descriptors = []

    # CB indices are 0, 1, 2, ... for each CB (including intermediate CBs).
    cb_indices = list(range(num_cbs))

    # Debug: print pipe graph if available
    if pipe_graph and os.environ.get("TTLANG_DEBUG_PIPE_GRAPH"):
        print(f"=== Pipe Graph ({len(pipe_graph)} connections) ===")
        for p in pipe_graph:
            print(
                f"  Pipe: src=({p.srcX},{p.srcY}) -> dst=({p.dstStartX},{p.dstStartY})-({p.dstEndX},{p.dstEndY})"
            )
            print(f"    receiverCBIndex={p.receiverCBIndex}, runtimeArgSlot={p.runtimeArgSlot}")

    for spec in kernel_specs:
        # runtime_args structure: [x][y][args_per_core].
        # Each core gets an empty arg list (we use my_x/my_y for indexing).
        runtime_args = [[[] for _ in range(grid_rows)] for _ in range(grid_cols)]

        # Populate runtime args for sender cores based on pipe graph.
        # For gather patterns, senders need the receiver's CB info.
        if pipe_graph and spec.thread_type != "compute":
            for p in pipe_graph:
                if 0 <= p.srcX < grid_cols and 0 <= p.srcY < grid_rows:
                    # Add receiver CB index to sender's runtime args.
                    # The kernel will use this to look up the CB address.
                    # Ensure runtime_args[x][y] has enough slots
                    while len(runtime_args[p.srcX][p.srcY]) <= p.runtimeArgSlot:
                        runtime_args[p.srcX][p.srcY].append(0)
                    runtime_args[p.srcX][p.srcY][p.runtimeArgSlot] = p.receiverCBIndex

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

    # Load pipe graph for sender runtime args (gather patterns).
    pipe_graph = load_pipe_graph()

    # Build kernel descriptors.
    kernel_descriptors = build_kernel_descriptors(
        kernel_specs=kernel_specs,
        tensors=tensors,
        tensor_accessor_args=tensor_accessor_args,
        core_ranges=core_ranges,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        num_cbs=len(cb_configs),
        pipe_graph=pipe_graph,
    )

    # Build CB descriptors.
    cb_descriptors = build_cb_descriptors(
        tensors=tensors,
        cb_configs=cb_configs,
        core_ranges=core_ranges,
    )

    # Build semaphore descriptors for pipe synchronization.
    # Pipe semaphores are indexed by srcX coordinate (see getPipeSemaphoreIndex).
    # We need to allocate enough semaphores for all pipe sources.
    semaphore_descriptors = []
    if pipe_graph:
        max_sem_idx = max(p.srcX for p in pipe_graph)
        # Allocate semaphores 0 through max_sem_idx, initialized to 0
        for sem_id in range(max_sem_idx + 1):
            semaphore_descriptors.append(
                ttnn.SemaphoreDescriptor(sem_id, core_ranges=core_ranges, initial_value=0)
            )

    # Build and execute program.
    # TODO: Enable custom_program_hash once tt-metal exposes it in Python bindings.
    # See tt-metal/ttnn/cpp/ttnn-nanobind/program_descriptors.cpp - needs to add
    # custom_program_hash parameter to ProgramDescriptor binding.
    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=semaphore_descriptors,
        # custom_program_hash=program_hash,
    )

    return ttnn.generic_op(list(tensors), program)


__all__ = [
    "KernelSpec",
    "PipeConnection",
    "load_pipe_graph",
    "build_tensor_accessor_args",
    "build_kernel_descriptors",
    "build_cb_descriptors",
    "run_kernel_on_device",
]
