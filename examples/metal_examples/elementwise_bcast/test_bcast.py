#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for elementwise add with broadcast on the second input.
"""

import sys
from pathlib import Path

import torch
import pytest
import ttnn
from loguru import logger

# Support both running as module and directly
try:
    from utils.correctness import assert_allclose
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from utils.correctness import assert_allclose

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
# Data format constants
BFLOAT16_SIZE = 2  # bytes per element


@pytest.mark.parametrize("bcast_dim", ["col", "row", "scalar"], ids=["col", "row", "scalar"])
@pytest.mark.parametrize("num_tiles", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize(
    "single_core", [False, True], ids=["multi_core", "single_core"]
)
def test_bcast_add(device, bcast_dim, num_tiles, single_core):
    """Test A + B where B is broadcast across columns, rows, or as a scalar."""
    _run_bcast_add_test(
        device, bcast_dim, num_tiles, single_core, BLOCK_H, BLOCK_W
    )


def _make_bcast_tile(bcast_dim):
    bcast_shape = [1, 1, 32, 32]
    data_b = torch.zeros(bcast_shape, dtype=torch.bfloat16)
    if bcast_dim == "col":
        data_b[..., :, 0] = (torch.rand([1, 1, 32]) * 2 - 1).to(torch.bfloat16)
    elif bcast_dim == "row":
        data_b[..., 0, :] = (torch.rand([1, 1, 32]) * 2 - 1).to(torch.bfloat16)
    elif bcast_dim == "scalar":
        data_b[..., 0, 0] = (torch.rand([1, 1]) * 2 - 1).to(torch.bfloat16)
    else:
        raise ValueError(f"Unsupported bcast_dim: {bcast_dim}")
    return data_b


def _bcast_golden(data_a, data_b, bcast_dim):
    if bcast_dim == "col":
        return data_a + data_b[..., :, :1]
    if bcast_dim == "row":
        return data_a + data_b[..., :1, :]
    if bcast_dim == "scalar":
        return data_a + data_b[..., :1, :1]
    raise ValueError(f"Unsupported bcast_dim: {bcast_dim}")


def _run_bcast_add_test(device, bcast_dim, num_tiles, single_core, block_h, block_w):
    """Test A + B with broadcast on B."""
    block_size = block_h * block_w

    shape = [1, num_tiles, 32, 32]
    data_a = (torch.rand(shape) * 2 - 1).to(torch.bfloat16)
    data_b = _make_bcast_tile(bcast_dim)

    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

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

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        dram_memory_config,
    )

    io_tensors = [input_tensor_a, input_tensor_b, output_tensor]

    if single_core:
        max_core = ttnn.CoreCoord(0, 0)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
        core_group_1 = core_grid
        core_group_2 = ttnn.CoreRangeSet([])
        work_per_core1 = num_tiles
    else:
        max_core = ttnn.CoreCoord(7, 7)
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
        (_, core_grid, core_group_1, core_group_2, work_per_core1, _) = (
            ttnn.split_work_to_cores(all_cores, num_tiles)
        )
        assert (
            len(core_group_2.ranges()) == 0
        ), "Bcast kernel does not support 2 core groups"

    input_cb_data_format = ttnn.bfloat16
    tile_height = 32
    tile_width = 32
    cb_page_size = tile_height * tile_width * BFLOAT16_SIZE
    cb_total_size = block_size * cb_page_size

    in0_cb = 0
    in1_cb = 1
    out_cb = 16

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
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )

    in0_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[in0_cb_format],
    )
    in1_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size,
        core_ranges=core_grid,
        format_descriptors=[in1_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[out_cb_format],
    )

    reader_compile_time_args = ttnn.TensorAccessorArgs(
        input_tensor_a
    ).get_compile_time_args()
    reader_compile_time_args.extend(
        ttnn.TensorAccessorArgs(input_tensor_b).get_compile_time_args()
    )
    writer_compile_time_args = []
    writer_compile_time_args.extend(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )

    compute_compile_time_args = []

    num_x_cores = max_core.x + 1
    num_y_cores = max_core.y + 1
    reader_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    writer_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    compute_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]

    current_tile = 0
    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = [
                    input_tensor_a.buffer_address(),
                    input_tensor_b.buffer_address(),
                    work_per_core1,
                    current_tile,
                ]
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),
                    work_per_core1,
                    current_tile,
                ]
                compute_rt_args[x][y] = [work_per_core1]
                current_tile += work_per_core1

    block_defines = [
        ("BLOCK_H", str(block_h)),
        ("BLOCK_W", str(block_w)),
    ]
    if bcast_dim == "row":
        block_defines.append(("BCAST_ROW", "1"))
    elif bcast_dim == "scalar":
        block_defines.append(("BCAST_SCALAR", "1"))

    reader_kernel_path = "kernels/dataflow/reader_binary_bcast_cols.cpp"

    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source=reader_kernel_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        defines=block_defines,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="kernels/dataflow/writer_unary.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        defines=block_defines,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="kernels/compute/bcast_add.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        defines=block_defines,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(),
    )

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
            out_cb_descriptor,
        ],
    )

    output = ttnn.generic_op(io_tensors, program_descriptor)

    torch_result = _bcast_golden(data_a, data_b, bcast_dim)
    torch_output = ttnn.to_torch(output)

    logger.debug(f"Broadcast tile sample: {data_b[0, 0, 0, :5]}")
    logger.debug(f"Output sample: {torch_output[0, 0, 0, :5]}")

    assert_allclose(torch_output, torch_result, rtol=5e-2, atol=1e-1)


if __name__ == "__main__":
    # For manual testing
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        test_bcast_add(device, bcast_dim="col", num_tiles=64, single_core=True)
        logger.info("Test passed!")
    finally:
        ttnn.close_device(device)
