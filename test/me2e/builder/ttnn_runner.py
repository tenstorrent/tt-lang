# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN runner for E2E tests.

Runs compiled kernels on Tenstorrent devices using ttnn.generic_op.
Provides shared execution logic for binary and unary operations, avoiding
code duplication across different op tests.
"""

from pathlib import Path
from typing import List, Any
import sys

import torch
import ttnn

# Import test_helpers from test/python
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
from test_helpers import to_dram

from .kernels import KernelSpec


# Constants for CB indices.
# NOTE: These must match the cb_index values in the generated MLIR.
# The TTL MLIR uses sequential indices: 0, 1, 2, ... not 0, 1, 16.
CB_IN0 = 0
CB_IN1 = 1
CB_OUT = 2  # Changed from 16 to match TTL MLIR output CB index

# Tile dimensions.
TILE_HEIGHT = 32
TILE_WIDTH = 32
BFLOAT16_SIZE = 2  # bytes per element


def _get_tile_size_bytes(dtype: ttnn.DataType) -> int:
    """Get tile size in bytes for a given data type."""
    if dtype == ttnn.bfloat16:
        return TILE_HEIGHT * TILE_WIDTH * BFLOAT16_SIZE
    elif dtype == ttnn.float32:
        return TILE_HEIGHT * TILE_WIDTH * 4
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _create_cb_descriptors(
    core_grid: ttnn.CoreRangeSet,
    num_inputs: int,
    tile_size_bytes: int,
    num_tiles: int,
) -> List[ttnn.CBDescriptor]:
    """
    Create circular buffer descriptors for input and output.

    Args:
        core_grid: Core range set for the operation.
        num_inputs: Number of input tensors (1 for unary, 2 for binary).
        tile_size_bytes: Size of each tile in bytes.
        num_tiles: Number of tiles per buffer.

    Returns:
        List of CB descriptors.
    """
    cb_total_size = tile_size_bytes * num_tiles

    descriptors = []

    # Input CBs.
    for i in range(num_inputs):
        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=i,
            data_format=ttnn.bfloat16,
            page_size=tile_size_bytes,
        )
        descriptors.append(
            ttnn.CBDescriptor(
                total_size=cb_total_size,
                core_ranges=core_grid,
                format_descriptors=[cb_format],
            )
        )

    # Output CB.
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_OUT,
        data_format=ttnn.bfloat16,
        page_size=tile_size_bytes,
    )
    descriptors.append(
        ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_grid,
            format_descriptors=[out_cb_format],
        )
    )

    return descriptors


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
    Run an operation on device.

    Args:
        device: TTNN device.
        noc_kernels: List of NOC (reader/writer) kernel specs.
        compute_kernel: Compute kernel spec.
        inputs: List of input tensors.
        kernel_dir: Directory containing kernel C++ files.

    Returns:
        Output tensor as torch tensor.
    """
    print(f"[DEBUG ttnn_runner] _run_op: Starting with {len(inputs)} inputs")
    num_inputs = len(inputs)
    shape = list(inputs[0].shape)
    print(f"[DEBUG ttnn_runner] Input shape: {shape}")

    # Calculate number of tiles.
    # Assume shape is [N, H, W] where H and W are tile multiples.
    num_tiles = 1
    for dim in shape[:-2]:
        num_tiles *= dim
    num_tiles *= (shape[-2] // TILE_HEIGHT) * (shape[-1] // TILE_WIDTH)

    # Create device tensors using to_dram (like test_axby.py).
    print(f"[DEBUG ttnn_runner] Creating device tensors")
    device_inputs = []

    for i, inp in enumerate(inputs):
        print(f"[DEBUG ttnn_runner] Converting input {i} to device tensor")
        device_inp = to_dram(inp.to(torch.bfloat16), device)
        device_inputs.append(device_inp)
    print(f"[DEBUG ttnn_runner] All inputs converted")

    # Create output tensor in DRAM using to_dram (like test_axby.py).
    print(f"[DEBUG ttnn_runner] Creating output tensor in DRAM")
    output_torch = torch.zeros(shape, dtype=torch.bfloat16)
    output_tensor = to_dram(output_torch, device)

    io_tensors = device_inputs + [output_tensor]
    print(f"[DEBUG ttnn_runner] Output tensor allocated")

    # Configure core grid (single core for simplicity).
    print(f"[DEBUG ttnn_runner] Configuring core grid")
    max_core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    # Get tile size.
    print(f"[DEBUG ttnn_runner] Getting tile size")
    tile_size_bytes = _get_tile_size_bytes(ttnn.bfloat16)
    print(f"[DEBUG ttnn_runner] Tile size: {tile_size_bytes} bytes")

    # Create CB descriptors.
    print(f"[DEBUG ttnn_runner] Creating CB descriptors")
    cb_descriptors = _create_cb_descriptors(
        core_grid=core_grid,
        num_inputs=num_inputs,
        tile_size_bytes=tile_size_bytes,
        num_tiles=1,  # Single buffering.
    )
    print(f"[DEBUG ttnn_runner] Created {len(cb_descriptors)} CB descriptors")

    # Build kernel descriptors.
    print(f"[DEBUG ttnn_runner] Building kernel descriptors")
    kernel_descriptors = []

    # Find reader and writer kernels.
    print(
        f"[DEBUG ttnn_runner] Finding reader and writer kernels from {len(noc_kernels)} NOC kernels"
    )
    reader_kernel = None
    writer_kernel = None
    for kernel in noc_kernels:
        print(f"[DEBUG ttnn_runner]   NOC kernel: {kernel.name}")
        if "reader" in kernel.name.lower():
            reader_kernel = kernel
        elif "writer" in kernel.name.lower():
            writer_kernel = kernel

    if reader_kernel is None or writer_kernel is None:
        raise ValueError("Could not find reader and writer kernels")
    print(
        f"[DEBUG ttnn_runner] Found reader: {reader_kernel.name}, writer: {writer_kernel.name}"
    )

    # Reader kernel descriptor.
    reader_rt_args = [[[num_tiles, 0]]]
    reader_common_args = [inp.buffer_address() for inp in device_inputs]

    # Compile-time args for reader: CB indices + TensorAccessorArgs for each input.
    # For binary: [CB_IN0, CB_IN1] + TensorAccessorArgs, for unary: [CB_IN0] + TensorAccessorArgs
    reader_ct_args = [CB_IN0] if num_inputs == 1 else [CB_IN0, CB_IN1]
    reader_ct_args.extend(_get_tensor_accessor_args(device_inputs))

    reader_descriptor = ttnn.KernelDescriptor(
        kernel_source=str(kernel_dir / f"{reader_kernel.name}.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=reader_common_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    kernel_descriptors.append(reader_descriptor)

    # Writer kernel descriptor.
    writer_rt_args = [[[num_tiles, 0]]]
    writer_common_args = [output_tensor.buffer_address()]

    # Compile-time args for writer: CB index + TensorAccessorArgs for output.
    writer_ct_args = [CB_OUT]
    writer_ct_args.extend(_get_tensor_accessor_args([output_tensor]))

    writer_descriptor = ttnn.KernelDescriptor(
        kernel_source=str(kernel_dir / f"{writer_kernel.name}.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=writer_common_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    kernel_descriptors.append(writer_descriptor)

    # Compute kernel descriptor.
    # Compile-time args are CB indices: [cb0, cb1, cb_out] for binary, [cb0, cb_out] for unary.
    if num_inputs == 2:
        compute_ct_args = [CB_IN0, CB_IN1, CB_OUT]
    else:
        compute_ct_args = [CB_IN0, CB_OUT]

    # Runtime args: [x][y][args]; currently unused so keep empty payload.
    compute_rt_args = [[[]]]

    compute_descriptor = ttnn.KernelDescriptor(
        kernel_source=str(kernel_dir / f"{compute_kernel.name}.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        common_runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )
    kernel_descriptors.append(compute_descriptor)

    # Create program descriptor.
    print(f"[DEBUG ttnn_runner] Creating program descriptor")
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        semaphores=[],
        cbs=cb_descriptors,
    )
    print(f"[DEBUG ttnn_runner] Program descriptor created")

    # Execute.
    print(f"[DEBUG ttnn_runner] Executing ttnn.generic_op")
    ttnn.generic_op(io_tensors, program_descriptor)
    print(f"[DEBUG ttnn_runner] Execution complete")

    # Use the original output_tensor (like test_axby.py uses out_t directly).
    # generic_op writes to the last tensor in io_tensors, which is output_tensor.
    print(f"[DEBUG ttnn_runner] Converting output to torch")
    result = ttnn.to_torch(output_tensor)
    print(f"[DEBUG ttnn_runner] Conversion complete")
    return result


def _get_tensor_accessor_args(tensors: List[Any]) -> List[int]:
    """
    Get compile-time args for tensor accessors.

    Args:
        tensors: List of device tensors.

    Returns:
        List of compile-time args.
    """
    args = []
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    return args
