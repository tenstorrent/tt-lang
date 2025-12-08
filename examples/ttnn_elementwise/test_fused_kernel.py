#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for fused operations: f(A + B) + C where f is exp or relu
All operations happen in DST registers without intermediate CB storage.
"""

import torch
import pytest
import ttnn
from loguru import logger

from utils import check_tensors_match, log_mismatch_diagnostics

# Configure logger to show INFO messages
logger.remove()  # Remove default handler
logger.add(lambda msg: print(msg, end=""), level="INFO")


@pytest.fixture(scope="session")
def device():
    """Create and yield a device, then close it after all tests."""
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


# Block configuration
BLOCK_H = 8
BLOCK_W = 8
BLOCK_SIZE = BLOCK_H * BLOCK_W  # 64 tiles per block

# Data format constants
BFLOAT16_SIZE = 2  # bytes per element


@pytest.mark.parametrize("num_tiles", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize(
    "double_buffer", [False, True], ids=["single_buf", "double_buf"]
)
@pytest.mark.parametrize(
    "single_core", [False, True], ids=["multi_core", "single_core"]
)
@pytest.mark.parametrize("unary_op", ["exp", "relu"], ids=["exp", "relu"])
def test_fused_kernel(device, num_tiles, double_buffer, single_core, unary_op):
    """Test fused f(A+B)+C operation with block processing, where f is exp or relu"""
    _run_fused_kernel_test(
        device, num_tiles, double_buffer, single_core, unary_op, BLOCK_H, BLOCK_W
    )


@pytest.mark.parametrize(
    "block_h,block_w", [(3, 3), (5, 5), (7, 3)], ids=["3x3", "5x5", "7x3"]
)
@pytest.mark.parametrize("unary_op", ["exp", "relu"], ids=["exp", "relu"])
def test_fused_kernel_remainder(device, block_h, block_w, unary_op):
    """Test fused kernel with block sizes not divisible by 4 (exercises remainder loop)"""
    block_size = block_h * block_w
    # Use num_tiles that gives us at least 2 full blocks
    num_tiles = block_size * 2
    _run_fused_kernel_test(
        device,
        num_tiles,
        double_buffer=False,
        single_core=True,
        unary_op=unary_op,
        block_h=block_h,
        block_w=block_w,
    )


def _run_fused_kernel_test(
    device, num_tiles, double_buffer, single_core, unary_op, block_h, block_w
):
    """Test fused f(A+B)+C operation with block processing, where f is exp or relu"""
    block_size = block_h * block_w

    shape = [1, num_tiles, 32, 32]

    # Create random input data in range [-1, 1]
    data_a = (torch.rand(shape) * 2 - 1).to(torch.bfloat16)
    data_b = (torch.rand(shape) * 2 - 1).to(torch.bfloat16)
    data_c = (torch.rand(shape) * 2 - 1).to(torch.bfloat16)

    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Create input tensors on device
    input_tensor_a = ttnn.from_torch(
        data_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        data_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )

    input_tensor_c = ttnn.from_torch(
        data_c,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )

    # Allocate output tensor
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        dram_memory_config,
    )

    io_tensors = [input_tensor_a, input_tensor_b, input_tensor_c, output_tensor]

    # Configure core grid
    if single_core:
        # Single core gets all tiles - useful for testing double-buffering with multiple blocks
        max_core = ttnn.CoreCoord(0, 0)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
        core_group_1 = core_grid
        core_group_2 = ttnn.CoreRangeSet([])
        work_per_core1 = num_tiles
    else:
        # Distribute work across multiple cores
        max_core = ttnn.CoreCoord(7, 7)
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
        (_, core_grid, core_group_1, core_group_2, work_per_core1, _) = (
            ttnn.split_work_to_cores(all_cores, num_tiles)
        )
        assert (
            len(core_group_2.ranges()) == 0
        ), "Fused kernel has number of tiles as compile time arg, does not support 2 core groups"

    input_cb_data_format = ttnn.bfloat16
    tile_height = 32
    tile_width = 32
    cb_page_size = tile_height * tile_width * BFLOAT16_SIZE  # 2KB per tile
    # CB must hold an entire block of tiles (double for double-buffering)
    num_buffers = 2 if double_buffer else 1
    cb_total_size = (
        block_size * cb_page_size * num_buffers
    )  # block_size tiles * 2KB * num_buffers

    # Define circular buffers
    in0_cb = 0
    in1_cb = 1
    in2_cb = 2
    out_cb = 16

    # CB format descriptors
    in0_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in0_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )
    in1_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in1_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )
    in2_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in2_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )

    # CB descriptors
    in0_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[in0_cb_format],
    )
    in1_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[in1_cb_format],
    )
    in2_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[in2_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[out_cb_format],
    )

    # Reader kernel compile-time and runtime args
    reader_compile_time_args = ttnn.TensorAccessorArgs(
        input_tensor_a
    ).get_compile_time_args()
    reader_compile_time_args.extend(
        ttnn.TensorAccessorArgs(input_tensor_b).get_compile_time_args()
    )
    reader_compile_time_args.extend(
        ttnn.TensorAccessorArgs(input_tensor_c).get_compile_time_args()
    )

    # Writer kernel compile-time args
    writer_compile_time_args = []
    writer_compile_time_args.extend(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )

    # Compute kernel compile-time args: none needed for this kernel
    compute_compile_time_args = []

    # Runtime args for all kernels
    num_x_cores = max_core.x + 1
    num_y_cores = max_core.y + 1
    reader_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    writer_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    compute_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]

    # Track current tile index for SPMD distribution
    current_tile = 0
    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = [
                    input_tensor_a.buffer_address(),
                    input_tensor_b.buffer_address(),
                    input_tensor_c.buffer_address(),
                    work_per_core1,
                    current_tile,  # start_id
                ]
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),
                    work_per_core1,
                    current_tile,
                ]
                compute_rt_args[x][y] = [work_per_core1]
                current_tile += work_per_core1

    # Pass block configuration via defines
    block_defines = [
        ("BLOCK_H", str(block_h)),
        ("BLOCK_W", str(block_w)),
    ]

    # Add unary operation define
    if unary_op == "relu":
        block_defines.append(("USE_RELU", "1"))

    # Select reader kernel based on buffering mode
    reader_kernel_path = (
        "kernels/dataflow/reader_ternary_db.cpp"
        if double_buffer
        else "kernels/dataflow/reader_ternary.cpp"
    )

    # Reader kernel descriptor
    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source=reader_kernel_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        defines=block_defines,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Writer kernel descriptor
    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="kernels/dataflow/writer_unary.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        defines=block_defines,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel descriptor
    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="kernels/compute/fused_elementwise.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        defines=block_defines,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(),
    )

    # Program descriptor
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[
            reader_kernel_descriptor,
            writer_kernel_descriptor,
            compute_kernel_descriptor,
        ],
        semaphores=[],
        cbs=[
            in0_cb_descriptor,
            in1_cb_descriptor,
            in2_cb_descriptor,
            out_cb_descriptor,
        ],
    )

    # Execute the fused operation
    output = ttnn.generic_op(io_tensors, program_descriptor)

    # Compute golden result using PyTorch: f(A + B) + C
    if unary_op == "relu":
        torch_result = torch.relu(data_a + data_b) + data_c
    else:  # exp
        torch_result = torch.exp(data_a + data_b) + data_c
    torch_output = ttnn.to_torch(output)

    logger.debug(f"Input A: {data_a.shape}, sample: {data_a[0, 0, 0, :5]}")
    logger.debug(f"Input B: {data_b.shape}, sample: {data_b[0, 0, 0, :5]}")
    logger.debug(f"Input C: {data_c.shape}, sample: {data_c[0, 0, 0, :5]}")
    logger.debug(f"Expected first tile output sample: {torch_result[0, 0, 0, :5]}")
    logger.debug(f"Actual first tile output sample: {torch_output[0, 0, 0, :5]}")
    if num_tiles > 1:
        logger.debug(f"Expected second tile output sample: {torch_result[0, 1, 0, :5]}")
        logger.debug(f"Actual second tile output sample: {torch_output[0, 1, 0, :5]}")

    # Check if results match
    matching = check_tensors_match(torch_result, torch_output)

    if not matching:
        log_mismatch_diagnostics(
            torch_result, torch_output, inputs={"A": data_a, "B": data_b, "C": data_c}
        )

    assert (
        matching
    ), f"Fused {unary_op} operation f(A+B)+C failed to match expected result"


if __name__ == "__main__":
    # For manual testing
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        test_fused_kernel(device, num_tiles=64)
        logger.info("Test passed!")
    finally:
        ttnn.close_device(device)
