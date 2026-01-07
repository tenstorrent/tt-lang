# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import pytest
import torch

from utils.correctness import assert_with_ulp


# (M * N) % (32 *32) == 0 for this implemention
@pytest.mark.parametrize(
    "M,K,N",
    [
        (640, 640, 640),
    ],
)
def test_multicore_matmul(M, K, N):
    # might be some l1 config stuff
    device = ttnn.open_device(device_id=0)
    assert (M * N) % (
        ttnn.TILE_SIZE * ttnn.TILE_SIZE
    ) == 0, "M*N must be multiple of TILE_SIZE*TILE_SIZE"
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    num_output_tiles_total = (M * N) // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)

    device_core_size = device.compute_with_storage_grid_size()
    upper_bound_core = ttnn.CoreCoord(device_core_size.x - 1, device_core_size.y - 1)
    device_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), upper_bound_core)]
    )
    print(
        f"core_grid: {device_core_grid}, num_output_tiles_total: {num_output_tiles_total}"
    )
    (_, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2) = (
        ttnn.split_work_to_cores(
            device_core_grid, num_output_tiles_total, row_wise=True
        )
    )
    print(
        f"all_cores: {all_cores}, core_group_1: {core_group_1}, core_group_2: {core_group_2}, work_per_core1: {work_per_core1}, work_per_core2: {work_per_core2}"
    )

    # allocate a, b and output tensors for matmul on device dram
    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG
    a_tensor = ttnn.rand(
        (M, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    b_tensor = ttnn.rand(
        (K, N),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    output_tensor = ttnn.empty(
        (M, N),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    dtype_size = 2  # bfloat16
    buffer_factor = 2
    cb_page_size = dtype_size * ttnn.TILE_SIZE * ttnn.TILE_SIZE
    cb_total_size = buffer_factor * cb_page_size

    a_cb = 0
    b_cb = 1
    out_cb = 16
    a_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=a_cb,
        data_format=ttnn.bfloat16,
        page_size=cb_page_size,
    )
    b_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=b_cb,
        data_format=ttnn.bfloat16,
        page_size=cb_page_size,
    )
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb,
        data_format=ttnn.bfloat16,
        page_size=cb_page_size,
    )

    a_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=all_cores,
        format_descriptors=[a_cb_format],
    )
    b_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=all_cores,
        format_descriptors=[b_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=all_cores,
        format_descriptors=[out_cb_format],
    )

    # TODO inconsistent metal access patterns for compile/runtime args
    reader_compile_time_args = ttnn.TensorAccessorArgs(a_tensor).get_compile_time_args()
    reader_compile_time_args.extend(
        ttnn.TensorAccessorArgs(b_tensor).get_compile_time_args()
    )
    writer_compile_time_args = ttnn.TensorAccessorArgs(
        output_tensor
    ).get_compile_time_args()

    # iterate over cores and assign work via runtime args
    # Both core groups should only be one core_range, but handling more just in case
    # will always be a smaller core grid than input grid, setting up runtime list
    # as the larger one to enable indexing in
    num_x_cores = upper_bound_core.x + 1
    num_y_cores = upper_bound_core.y + 1
    reader_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    writer_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    compute_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    current_tile = 0
    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                print(
                    f"Assigning core ({x},{y}) tile {current_tile} work_per_core1 {work_per_core1}"
                )
                reader_rt_args[x][y] = [
                    a_tensor.buffer_address(),
                    b_tensor.buffer_address(),
                    Mt,
                    Kt,
                    Nt,
                    current_tile,
                    work_per_core1,
                ]
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),
                    work_per_core1,
                    current_tile,
                ]
                compute_rt_args[x][y] = [work_per_core1, Kt]
                current_tile += work_per_core1

    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                print(
                    f"Assigning core ({x},{y}) tile {current_tile} work_per_core2 {work_per_core2}"
                )
                reader_rt_args[x][y] = [
                    a_tensor.buffer_address(),
                    b_tensor.buffer_address(),
                    Mt,
                    Kt,
                    Nt,
                    current_tile,
                    work_per_core2,
                ]
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),
                    work_per_core2,
                    current_tile,
                ]
                compute_rt_args[x][y] = [work_per_core2, Kt]
                current_tile += work_per_core2

    # Compute config init can't handle options, set here
    computeConfig = ttnn.ComputeConfigDescriptor()
    computeConfig.math_fidelity = ttnn.MathFidelity.HiFi4
    computeConfig.fp32_dest_acc_en = True
    computeConfig.math_approx_mode = False

    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/multicore_matmul/metal/kernels/mm_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/multicore_matmul/metal/kernels/mm_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/multicore_matmul/metal/kernels/mm_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[],
        runtime_args=compute_rt_args,
        config=computeConfig,
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[
            reader_kernel_descriptor,
            writer_kernel_descriptor,
            compute_kernel_descriptor,
        ],
        semaphores=[],
        cbs=[a_cb_descriptor, b_cb_descriptor, out_cb_descriptor],
    )

    print("Launching generic_op...")
    output = ttnn.generic_op([a_tensor, b_tensor, output_tensor], program_descriptor)
    print("Completed generic_op.")
    metal_output = ttnn.to_torch(output).to(torch.bfloat16)
    print(f"metal_output: {metal_output}")

    a_tensor_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_tensor_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    torch_output = torch.matmul(a_tensor_torch, b_tensor_torch)
    print(f"torch_output: {torch_output}")

    assert_with_ulp(torch_output, metal_output)

    ttnn.close_device(device)
