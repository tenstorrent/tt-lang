# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn
from ttl.utils.block_allocation import get_large_matmul_params
from ttl.utils.correctness import assert_with_ulp


def run_multicore_reuse_matmul(
    device,
    a_tensor,
    b_tensor,
    output_tensor,
    K_block_size,
    per_core_M,
    per_core_N,
    out_subblock_h,
    out_subblock_w,
):
    M = a_tensor.shape[0]
    K = a_tensor.shape[1]
    N = b_tensor.shape[1]
    assert a_tensor.shape[1] == b_tensor.shape[0], (
        "Incompatible matrix shapes for multiplication."
    )
    assert a_tensor.shape[0] == output_tensor.shape[0], (
        "Output matrix has incorrect number of rows."
    )
    assert b_tensor.shape[1] == output_tensor.shape[1], (
        "Output matrix has incorrect number of columns."
    )
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    assert per_core_M != 0, "get_large_matmul_params was not able to find a solution"
    assert Mt % per_core_M == 0, "per_core_M must divide Mt"
    assert Nt % per_core_N == 0, "per_core_N must divide Nt"
    assert Kt % K_block_size == 0, "K_block_size must divide Kt"
    num_blocks_y = Mt // per_core_M
    num_blocks_x = Nt // per_core_N

    device_core_size = device.compute_with_storage_grid_size()
    num_cores_x = device_core_size.x
    num_cores_y = device_core_size.y
    assert num_blocks_x <= num_cores_x and num_blocks_y <= num_cores_y, (
        "number of total blocks must be less than or equal to num cores in each dimension"
    )
    all_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_blocks_x - 1, num_blocks_y - 1),
            )
        ]
    )

    dtype_size = 2  # bfloat16
    cb_page_size = dtype_size * ttnn.TILE_SIZE * ttnn.TILE_SIZE

    a_cb = 0
    b_cb = 1
    out_cb = 16
    intermediate_cb = 24

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
    intermediate_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=intermediate_cb,
        data_format=ttnn.bfloat16,
        page_size=cb_page_size,
    )
    buffer_factor = 2
    a_cb_descriptor = ttnn.CBDescriptor(
        total_size=buffer_factor * cb_page_size * (per_core_M * K_block_size),
        core_ranges=all_cores,
        format_descriptors=[a_cb_format],
    )
    b_cb_descriptor = ttnn.CBDescriptor(
        total_size=buffer_factor * cb_page_size * (per_core_N * K_block_size),
        core_ranges=all_cores,
        format_descriptors=[b_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * (per_core_M * per_core_N),
        core_ranges=all_cores,
        format_descriptors=[out_cb_format],
    )
    intermediate_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * (per_core_M * per_core_N),
        core_ranges=all_cores,
        format_descriptors=[intermediate_cb_format],
    )

    num_blocks = Kt // K_block_size

    a_num_subblocks = per_core_M // out_subblock_h
    a_block_num_tiles = out_subblock_h * K_block_size * a_num_subblocks
    a_subblock_num_tiles = out_subblock_h * K_block_size

    b_num_subblocks = per_core_N // out_subblock_w
    b_block_num_tiles = out_subblock_w * K_block_size * b_num_subblocks
    b_per_core_w = out_subblock_w * b_num_subblocks

    out_subblock_num_tiles = out_subblock_h * out_subblock_w

    compute_compile_time_args = [
        K_block_size,
        a_num_subblocks,
        a_block_num_tiles,
        a_subblock_num_tiles,
        b_num_subblocks,
        b_block_num_tiles,
        b_per_core_w,
        num_blocks,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
    ]
    reader_compile_time_args = ttnn.TensorAccessorArgs(a_tensor).get_compile_time_args()
    reader_compile_time_args.extend(
        ttnn.TensorAccessorArgs(b_tensor).get_compile_time_args()
    )
    writer_compile_time_args = ttnn.TensorAccessorArgs(
        output_tensor
    ).get_compile_time_args()

    reader_rt_args = [[[] for _ in range(num_cores_y)] for _ in range(num_cores_x)]
    writer_rt_args = [[[] for _ in range(num_cores_y)] for _ in range(num_cores_x)]
    compute_rt_args = [[[] for _ in range(num_cores_y)] for _ in range(num_cores_x)]
    current_blk = 0
    for output_idx_y in range(num_blocks_y):
        for output_idx_x in range(num_blocks_x):
            core_x = current_blk % num_cores_x
            core_y = current_blk // num_cores_x
            reader_rt_args[core_x][core_y] = [
                a_tensor.buffer_address(),
                Kt * per_core_M * output_idx_y,
                1,
                Kt,
                K_block_size,
                K_block_size,
                per_core_M,
                K_block_size * per_core_M,
                b_tensor.buffer_address(),
                per_core_N * output_idx_x,
                1,
                Nt,
                K_block_size * Nt,
                per_core_N,
                K_block_size,
                per_core_N * K_block_size,
                Kt // K_block_size,
            ]
            writer_rt_args[core_x][core_y] = [
                output_tensor.buffer_address(),
                (output_idx_x * per_core_N) + (output_idx_y * per_core_M * Nt),
                1,
                Nt,
                out_subblock_w,
                out_subblock_h * Nt,
                out_subblock_w,
                out_subblock_h,
                out_subblock_w * out_subblock_h,
                per_core_N // out_subblock_w,
                per_core_M // out_subblock_h,
            ]
            current_blk += 1

    computeConfig = ttnn.ComputeConfigDescriptor()
    computeConfig.math_fidelity = ttnn.MathFidelity.HiFi4

    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/multicore_reuse_matmul/metal/kernels/reader_bmm_tile_layout.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/multicore_reuse_matmul/metal/kernels/writer_bmm_tile_layout.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/multicore_reuse_matmul/metal/kernels/bmm_large_block_zm.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=compute_compile_time_args,
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
        cbs=[
            a_cb_descriptor,
            b_cb_descriptor,
            out_cb_descriptor,
            intermediate_cb_descriptor,
        ],
    )

    return ttnn.generic_op([a_tensor, b_tensor, output_tensor], program_descriptor)


@pytest.mark.parametrize("M,K,N", [(640, 640, 640)])
def test_metal_matmul(M, K, N):
    device = ttnn.open_device(device_id=0)
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    K_block_size = 2

    device_core_size = device.compute_with_storage_grid_size()
    num_cores_x = device_core_size.x
    num_cores_y = device_core_size.y

    block_params = get_large_matmul_params(
        Mt, Nt, num_cores_y, num_cores_x, K_block_size
    )

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

    output = run_multicore_reuse_matmul(
        device,
        a_tensor,
        b_tensor,
        output_tensor,
        K_block_size,
        block_params.block_h,
        block_params.block_w,
        block_params.subblock_h,
        block_params.subblock_w,
    )
    metal_output = ttnn.to_torch(output).to(torch.bfloat16)

    a_tensor_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_tensor_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    torch_output = torch.matmul(a_tensor_torch, b_tensor_torch)

    assert_with_ulp(torch_output, metal_output)

    ttnn.close_device(device)
