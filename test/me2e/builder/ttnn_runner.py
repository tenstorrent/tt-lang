# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN runner for ME2E tests.

Runs compiled kernels on Tenstorrent devices using ttnn.generic_op.
Uses the shared kernel_runner module to build kernel descriptors and execute kernels.
"""

from pathlib import Path
from typing import List, Any, Optional, Tuple
import sys

import torch
import ttnn

# Import test_helpers from test/python.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
from ttlang_test_utils import to_dram, to_l1, to_l1_sharded

# Import shared kernel runner from ttl package.
from ttl.kernel_runner import (
    KernelSpec as RunnerKernelSpec,
    run_kernel_on_device,
)
from ttl.dataflow_buffer import PhysicalDFBConfig

from .kernels import KernelSpec
from ..config import BufferType, E2EConfig, MemoryLayout

# Tile dimensions.
TILE_HEIGHT = 32
TILE_WIDTH = 32


def _data_format_name(dtype: torch.dtype) -> str:
    """Return the runtime DFB format; ME2E builders construct bf16/f32 only."""

    data_formats = {
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    try:
        return data_formats[dtype]
    except KeyError:
        raise ValueError(f"Unsupported me2e tensor dtype: {dtype}") from None


def run_binary_op(
    device: Any,
    noc_kernels: List[KernelSpec],
    compute_kernel: KernelSpec,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    kernel_dir: Path,
    config: E2EConfig | None = None,
) -> torch.Tensor:
    """
    Run a binary operation on device.

    Args:
        device: TTNN device.
        noc_kernels: List of NOC (reader/writer) kernel specs.
        compute_kernel: Compute kernel spec.
        input_a: First input tensor.
        input_b: Second input tensor.
        kernel_dir: Directory containing kernel C++ files.
        config: Tensor storage and distribution used for execution.
    Returns:
        Output tensor as torch tensor.
    """
    return _run_op(
        device=device,
        noc_kernels=noc_kernels,
        compute_kernel=compute_kernel,
        inputs=[input_a, input_b],
        kernel_dir=kernel_dir,
        config=config or E2EConfig(),
    )


def run_unary_op(
    device: Any,
    noc_kernels: List[KernelSpec],
    compute_kernel: KernelSpec,
    input_a: torch.Tensor,
    kernel_dir: Path,
    config: E2EConfig | None = None,
) -> torch.Tensor:
    """
    Run a unary operation on device.

    Args:
        device: TTNN device.
        noc_kernels: List of NOC (reader/writer) kernel specs.
        compute_kernel: Compute kernel spec.
        input_a: Input tensor.
        kernel_dir: Directory containing kernel C++ files.
        config: Tensor storage and distribution used for execution.
    Returns:
        Output tensor as torch tensor.
    """
    return _run_op(
        device=device,
        noc_kernels=noc_kernels,
        compute_kernel=compute_kernel,
        inputs=[input_a],
        kernel_dir=kernel_dir,
        config=config or E2EConfig(),
    )


def _get_compute_config(compute_kernel: KernelSpec):
    """Translate compiler-selected kernel configuration to TTNN."""
    config = ttnn.ComputeConfigDescriptor()
    config.fp32_dest_acc_en = compute_kernel.fp32_dest_acc_en
    config.dst_full_sync_en = compute_kernel.dst_full_sync_en
    if compute_kernel.unpack_to_dest_fp32:
        configured_indices = set(compute_kernel.unpack_to_dest_fp32)
        for dfb_index in range(64):
            config.unpack_to_dest_mode.append(
                ttnn.UnpackToDestMode.UnpackToDestFp32
                if dfb_index in configured_indices
                else ttnn.UnpackToDestMode.Default
            )
    return config


def _run_op(
    device: Any,
    noc_kernels: List[KernelSpec],
    compute_kernel: KernelSpec,
    inputs: List[torch.Tensor],
    kernel_dir: Path,
    config: E2EConfig,
) -> torch.Tensor:
    """
    Run an operation on device using shared kernel_runner infrastructure.

    Uses the same execution logic as CompiledTTNNKernel to ensure compiler-
    generated C++ runs unmodified (no shimming required).

    Args:
        device: TTNN device.
        noc_kernels: List of NOC (reader/writer) kernel specs.
        compute_kernel: Compute kernel spec.
        inputs: List of input tensors.
        kernel_dir: Directory containing kernel C++ files.
    Returns:
        Output tensor as torch tensor.
    """
    shape = list(inputs[0].shape)
    dtype = inputs[0].dtype
    if any(input_tensor.dtype != dtype for input_tensor in inputs[1:]):
        raise ValueError("ME2E runner requires all input tensors to have one dtype")

    def to_configured_tensor(tensor):
        if config.buffer_type == BufferType.DRAM:
            if config.memory_layout != MemoryLayout.INTERLEAVED:
                raise ValueError("ME2E DRAM tensors require interleaved memory")
            return to_dram(tensor, device)
        if config.memory_layout == MemoryLayout.INTERLEAVED:
            return to_l1(tensor, device)
        shard_layout = {
            MemoryLayout.HEIGHT_SHARDED: "height",
            MemoryLayout.WIDTH_SHARDED: "width",
            MemoryLayout.BLOCK_SHARDED: "block",
        }[config.memory_layout]
        return to_l1_sharded(tensor, device, layout=shard_layout)

    device_inputs = [to_configured_tensor(tensor) for tensor in inputs]

    output_torch = torch.zeros(shape, dtype=dtype)
    output_tensor = to_configured_tensor(output_torch)

    io_tensors = device_inputs + [output_tensor]

    # Configure core grid (single core for simplicity).
    max_core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    # Find reader and writer kernels.
    reader_kernel = None
    writer_kernel = None
    for kernel in noc_kernels:
        if "reader" in kernel.name.lower():
            reader_kernel = kernel
        elif "writer" in kernel.name.lower():
            writer_kernel = kernel

    if reader_kernel is None or writer_kernel is None:
        raise ValueError("Could not find reader and writer kernels")

    # Build kernel specs for kernel_runner.
    # Reader accesses input tensors (indices 0..num_inputs-1).
    # Writer accesses output tensor (index num_inputs).
    runner_specs = [
        RunnerKernelSpec(
            path=str(kernel_dir / f"{reader_kernel.name}.cpp"),
            thread_type="noc",
            tensor_indices=reader_kernel.tensor_indices,
            local_tensor_indices=reader_kernel.local_tensor_indices,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        RunnerKernelSpec(
            path=str(kernel_dir / f"{writer_kernel.name}.cpp"),
            thread_type="noc",
            tensor_indices=writer_kernel.tensor_indices,
            local_tensor_indices=writer_kernel.local_tensor_indices,
            config=ttnn.WriterConfigDescriptor(),
        ),
        RunnerKernelSpec(
            path=str(kernel_dir / f"{compute_kernel.name}.cpp"),
            thread_type="compute",
            tensor_indices=compute_kernel.tensor_indices,
            local_tensor_indices=compute_kernel.local_tensor_indices,
            config=_get_compute_config(compute_kernel),
        ),
    ]

    data_format = _data_format_name(dtype)
    dfb_configs = [
        PhysicalDFBConfig(
            dfb_index=dfb_index,
            num_tiles=1,
            data_format=data_format,
            block_count=1,
            page_size=ttnn.tile_size(io_tensor.dtype),
            tile=(TILE_HEIGHT, TILE_WIDTH),
        )
        for dfb_index, io_tensor in enumerate(io_tensors)
    ]

    # Execute using shared kernel runner.
    run_kernel_on_device(
        kernel_specs=runner_specs,
        tensors=io_tensors,
        cb_configs=dfb_configs,
        core_ranges=core_grid,
    )

    # Return result.
    result = ttnn.to_torch(output_tensor)
    return result
