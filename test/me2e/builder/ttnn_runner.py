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
from ttlang_test_utils import to_dram

# Import shared kernel runner from ttl package.
from ttl.kernel_runner import (
    KernelSpec as RunnerKernelSpec,
    run_kernel_on_device,
)
from ttl.circular_buffer import CircularBuffer

from .kernels import KernelSpec


# Tile dimensions.
TILE_HEIGHT = 32
TILE_WIDTH = 32


def run_binary_op(
    device: Any,
    noc_kernels: List[KernelSpec],
    compute_kernel: KernelSpec,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    kernel_dir: Path,
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

    Returns:
        Output tensor as torch tensor.
    """
    return _run_op(
        device=device,
        noc_kernels=noc_kernels,
        compute_kernel=compute_kernel,
        inputs=[input_a, input_b],
        kernel_dir=kernel_dir,
    )


def run_unary_op(
    device: Any,
    noc_kernels: List[KernelSpec],
    compute_kernel: KernelSpec,
    input_a: torch.Tensor,
    kernel_dir: Path,
) -> torch.Tensor:
    """
    Run a unary operation on device.

    Args:
        device: TTNN device.
        noc_kernels: List of NOC (reader/writer) kernel specs.
        compute_kernel: Compute kernel spec.
        input_a: Input tensor.
        kernel_dir: Directory containing kernel C++ files.

    Returns:
        Output tensor as torch tensor.
    """
    return _run_op(
        device=device,
        noc_kernels=noc_kernels,
        compute_kernel=compute_kernel,
        inputs=[input_a],
        kernel_dir=kernel_dir,
    )


def _run_op(
    device: Any,
    noc_kernels: List[KernelSpec],
    compute_kernel: KernelSpec,
    inputs: List[torch.Tensor],
    kernel_dir: Path,
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

    # Create device tensors using to_dram.
    device_inputs = []
    for inp in inputs:
        device_inp = to_dram(inp.to(torch.bfloat16), device)
        device_inputs.append(device_inp)

    # Create output tensor in DRAM.
    output_torch = torch.zeros(shape, dtype=torch.bfloat16)
    output_tensor = to_dram(output_torch, device)

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
    # Compute has no tensor indices (only uses CBs).
    runner_specs = [
        RunnerKernelSpec(
            path=str(kernel_dir / f"{reader_kernel.name}.cpp"),
            thread_type="noc",
            tensor_indices=reader_kernel.tensor_indices,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        RunnerKernelSpec(
            path=str(kernel_dir / f"{writer_kernel.name}.cpp"),
            thread_type="noc",
            tensor_indices=writer_kernel.tensor_indices,
            config=ttnn.WriterConfigDescriptor(),
        ),
        RunnerKernelSpec(
            path=str(kernel_dir / f"{compute_kernel.name}.cpp"),
            thread_type="compute",
            tensor_indices=[],  # Compute kernels don't access tensors directly.
            config=ttnn.ComputeConfigDescriptor(),
        ),
    ]

    # Build CB configs: CircularBuffer objects for each tensor.
    # Shape is (1, 1) for single tile, buffer_factor is 1 for single buffering.
    cb_configs: List[CircularBuffer] = [
        CircularBuffer(tensor=tensor, shape=(1, 1), buffer_factor=1)
        for tensor in io_tensors
    ]

    # Execute using shared kernel runner.
    run_kernel_on_device(
        kernel_specs=runner_specs,
        tensors=io_tensors,
        cb_configs=cb_configs,
        core_ranges=core_grid,
    )

    # Return result.
    result = ttnn.to_torch(output_tensor)
    return result
