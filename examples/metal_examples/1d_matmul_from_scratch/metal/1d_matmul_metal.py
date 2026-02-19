# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
1D Matmul Metal Example

This example demonstrates a 1D matmul where only input A (in0) is multicast across cores.
Unlike the 2D matmul where both inputs are multicast, here:
- in0 is broadcast from a single sender core (0,0) to all other cores
- in1 is read locally by each core (no multicast)
- Output is written locally by each core

The kernel configuration uses:
1. reader_bmm_tile_layout_in0_sender_padding.cpp - Single sender core (0,0) reads in0 and multicasts
2. reader_bmm_tile_layout_in0_receiver.cpp - All other cores receive in0 multicast
3. reader_bmm_tile_layout_in1_sender_writer_padding.cpp - All cores read in1 locally and write output
4. bmm_large_block_zm_fused_bias_activation.cpp - Compute kernel on all cores

Note: This implementation does not use bias, sharding, or activation fusion yet.
These features are supported by the kernels but not configured in this example.
"""

import pytest
import torch
import ttnn
from utils.block_allocation import get_large_matmul_params, num_cores_to_grid_ranges
from utils.correctness import assert_with_ulp

@pytest.mark.parametrize("M,N,K,n_blocks_per_core,block_m,block_n,block_k,subblock_h,subblock_w", [
    (32, 8 * 32, 32, 1, 1, 1, 1, 1, 1),
    (32, 8 * 32, 64, 1, 1, 1, 1, 1, 1),
    (64, 8 * 32, 32, 1, 1, 1, 1, 1, 1), # 2 broadcasts, failing
    (64, 8 * 32, 64, 1, 2, 1, 2, 2, 1),
    (8 * 32 * 2, 8 * 32, 8 * 32, 1, 16, 1, 8, 8, 1),
    (64, 8 * 32 * 2, 64, 2, 2, 1, 2, 2, 1),
    (64, 8 * 64, 64, 1, 2, 2, 2, 2, 2),
    (64, 8 * 64 * 2, 64, 2, 2, 2, 2, 2, 2),

    ])
def test_1d_matmul_metal(M, N, K, n_blocks_per_core, block_m, block_n, block_k, subblock_h, subblock_w):
    device = ttnn.open_device(device_id=0)
    # Allocate input and output tensors in DRAM
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

    Mt = M // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE

    device_core_size = device.compute_with_storage_grid_size()
    num_cores_x = device_core_size.x
    num_cores_y = device_core_size.y
    print(
        f"Device compute_with_storage_grid_size: ({num_cores_x}, {num_cores_y})"
    )
    num_worker_cores = Nt // (block_n * n_blocks_per_core)
    assert num_cores_x * num_cores_y >= num_worker_cores, "Not enough cores to run the test with the given number of blocks per core"
    assert Mt % block_m == 0, "block_m must divide Mt"
    assert Nt % block_n == 0, "block_n must divide Nt"
    assert Nt % (block_n * n_blocks_per_core) == 0, "number of n blocks split across cores must divide Nt"
    assert Kt % block_k == 0, "block_k must divide Kt"
    assert block_m * ttnn.TILE_SIZE % subblock_h == 0, "subblock_h must divide block_m"
    assert block_n * ttnn.TILE_SIZE % subblock_w == 0, "subblock_w must divide block_n"

    # For 1D matmul: Use single core (0,0) as sender for in0 multicast
    # All other cores are receivers and all cores do computation
    assert (
        num_worker_cores > 1
    ), "1D matmul requires multiple blocks to use all 4 kernels"

    # Single sender core at (0, 0) broadcasts to all other cores
    in0_sender_core = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
    )
    # All compute cores (entire grid used for computation)
    all_cores = ttnn.num_cores_to_corerangeset(num_worker_cores, ttnn.CoreCoord(num_cores_x, num_cores_y), row_wise=True)
    # Receiver cores are all cores except the single sender core (0,0)
    in0_receiver_cores = all_cores.subtract(in0_sender_core)

    # Circular buffer setup
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
        total_size=buffer_factor * cb_page_size * (block_m * block_k),
        core_ranges=all_cores,
        format_descriptors=[a_cb_format],
    )
    b_cb_descriptor = ttnn.CBDescriptor(
        total_size=buffer_factor * cb_page_size * (n_blocks_per_core * block_n * block_k),
        core_ranges=all_cores,
        format_descriptors=[b_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * (block_m * n_blocks_per_core * block_n),
        core_ranges=all_cores,
        format_descriptors=[out_cb_format],
    )
    intermediate_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * (block_m * n_blocks_per_core * block_n),
        core_ranges=all_cores,
        format_descriptors=[intermediate_cb_format],
    )
    in0_sender_semaphore_id = 0
    in0_receiver_semaphore_id = 1

    # KERNEL COMPILE TIME ARGS
    compute_compile_time_args = [
        block_k,  # in0_block_w
        block_m//subblock_h,  # in0_num_subblocks
        block_m * block_k,  # in0_block_num_tiles
        subblock_h * block_k,  # in0_subblock_num_tiles
        block_n//subblock_w,  # in1_num_subblocks
        block_n * block_k,  # in1_block_num_tiles
        block_n,  # in1_block_w
        Kt//block_k,  # num_blocks_inner_dim
        n_blocks_per_core,  # num_blocks_w_dim
        Mt//block_m,  # num_blocks_h_dim
        subblock_h,  # out_subblock_h
        subblock_w,  # out_subblock_w
        subblock_h * subblock_w,  # out_subblock_num_tiles
    ]
    # Compile time args for in0 sender
    in0_sender_compile_time_args = [
        1,  # in0_tensor_stride_w
        Kt,  # in0_tensor_stride_h
        block_k,  # in0_tensor_next_inner_dim_block_stride
        block_k * block_m,  # in0_tensor_next_h_dim_block_stride
        block_k,  # in0_block_w
        block_m,  # in0_block_h
        block_m * block_k,  # in0_block_num_tiles
        Kt//block_k,  # num_blocks_inner_dim
        n_blocks_per_core,  # num_blocks_w_dim
        Mt//block_m,  # num_blocks_h_dim
        in0_sender_semaphore_id,  # in0_mcast_sender_semaphore
        in0_receiver_semaphore_id,  # in0_mcast_receiver_semaphore
        num_worker_cores-1,  # in0_mcast_num_dests (per sender)
        num_worker_cores-1,  # in0_mcast_num_cores (per sender)
    ]
    # Add TensorAccessor compile time args for in0
    in0_sender_compile_time_args.extend(
        ttnn.TensorAccessorArgs(a_tensor).get_compile_time_args()
    )
    print(
        f"IN0_SENDER - COMPILE_TIME_ARGS ({len(in0_sender_compile_time_args)} args): {', '.join(map(str, in0_sender_compile_time_args))}"
    )

    # Compile time args for in0 receiver
    in0_receiver_compile_time_args = [
        block_m * block_k,  # in0_block_num_tiles
        Kt // block_k,  # num_blocks_inner_dim
        n_blocks_per_core,  # num_blocks_w_dim
        Mt//block_m,  # num_blocks_h_dim
        in0_sender_semaphore_id,  # in0_mcast_sender_semaphore
        in0_receiver_semaphore_id,  # in0_mcast_receiver_semaphore
    ]
    print(
        f"IN0_RECEIVER - COMPILE_TIME_ARGS ({len(in0_receiver_compile_time_args)} args): {', '.join(map(str, in0_receiver_compile_time_args))}"
    )

    # Compile time args for in1 reader + writer
    in1_writer_compile_time_args = [
        1,  # in1_tensor_stride_w
        Nt,  # in1_tensor_stride_h
        block_k * Nt,  # in1_tensor_next_block_stride
        block_n,  # in1_tensor_next_w_dim_block_stride
        block_n,  # in1_block_w
        block_k,  # in1_block_h
        block_n * block_k,  # in1_block_num_tiles
        Kt // block_k,  # num_blocks_inner_dim
        n_blocks_per_core,  # num_blocks_w_dim
        Mt // block_m,  # num_blocks_h_dim
        # Output tensor args
        1,  # out_tensor_stride_w
        Nt,  # out_tensor_stride_h  
        subblock_w,  # out_tensor_next_subblock_stride_w
        subblock_h * Nt,  # out_tensor_next_subblock_stride_h
        block_n,  # out_tensor_next_w_dim_block_stride
        block_m * Nt,  # out_tensor_next_h_dim_block_stride
        subblock_w,  # out_subblock_w
        subblock_h,  # out_subblock_h
        block_n // subblock_w,  # out_num_subblocks_w
        block_m // subblock_h,  # out_num_subblocks_h
        subblock_w * subblock_h,  # out_subblock_tile_count
    ]
    # Add TensorAccessor compile time args for in1
    in1_writer_compile_time_args.extend(
        ttnn.TensorAccessorArgs(b_tensor).get_compile_time_args()
    )
    # Add TensorAccessor compile time args for output
    in1_writer_compile_time_args.extend(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )

    print(
        f"IN1_SENDER_WRITER - COMPILE_TIME_ARGS ({len(in1_writer_compile_time_args)} args): {', '.join(map(str, in1_writer_compile_time_args))}"
    )


    # KERNEL RUNTIME ARGS
    # Setup runtime args for each core
    num_x_cores = num_cores_x if num_cores_x < num_worker_cores else num_worker_cores
    num_y_cores = num_worker_cores // num_cores_x if num_cores_x < num_worker_cores else 1

    in0_sender_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    in0_receiver_rt_args = [
        [[] for _ in range(num_y_cores)] for _ in range(num_x_cores)
    ]
    in1_writer_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    compute_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]

    total_receivers = num_worker_cores - 1
    print(
        f"1D matmul: Single sender at (0,0) multicasts to {total_receivers} receivers, across a grid of {num_x_cores} x {num_y_cores} cores"
    )

    noc_of_sender = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))

    # Assign work to cores
    worker_core_idx = 0
    for output_idx_y in range(num_y_cores):
        for output_idx_x in range(num_x_cores):
            # in0 sender args (only for core (0,0))
            # Single sender multicasts to all other cores in the grid
            if output_idx_x == 0 and output_idx_y == 0:
                # Multicast destinations: start from (1,0) if exists, or (0,1)
                mcast_start_core_noc = device.worker_core_from_logical_core(
                    ttnn.CoreCoord(1, 0)
                )
                mcast_end_core_noc = device.worker_core_from_logical_core(
                    ttnn.CoreCoord(num_x_cores - 1, num_y_cores - 1)
                )

                in0_sender_rt_args[output_idx_x][output_idx_y] = [
                    a_tensor.buffer_address(),  # in0_tensor_addr
                    0,  # in0_tensor_start_tile_id (start at tile 0)
                    mcast_start_core_noc.x,  # in0_mcast_dest_noc_start_x
                    mcast_start_core_noc.y,  # in0_mcast_dest_noc_start_y
                    mcast_end_core_noc.x,  # in0_mcast_dest_noc_end_x (all x)
                    mcast_end_core_noc.y,  # in0_mcast_dest_noc_end_y (all y)
                ]
                print(
                    f"IN0_SENDER - RUNTIME_ARGS for core ({output_idx_x}, {output_idx_y}):"
                )
                print(
                    f"IN0_SENDER_CORE - RUNTIME_ARGS ({len(in0_sender_rt_args[output_idx_x][output_idx_y])} args): {', '.join(map(str, in0_sender_rt_args[output_idx_x][output_idx_y]))}"
                )

            # in0 receiver args (for all cores except (0,0))
            if not (output_idx_x == 0 and output_idx_y == 0):
                in0_receiver_rt_args[output_idx_x][output_idx_y] = [
                    noc_of_sender.x,  # in0_mcast_sender_noc_x (sender is at x=0)
                    noc_of_sender.y,  # in0_mcast_sender_noc_y (sender is at y=0)
                ]
                print(
                    f"IN0_RECEIVER - RUNTIME_ARGS for core ({output_idx_x}, {output_idx_y}):"
                )
                print(
                    f"IN0_RECEIVER_CORE - RUNTIME_ARGS ({len(in0_receiver_rt_args[output_idx_x][output_idx_y])} args): {', '.join(map(str, in0_receiver_rt_args[output_idx_x][output_idx_y]))}"
                )

            # in1 reader + writer args (all cores)
            in1_writer_rt_args[output_idx_x][output_idx_y] = [
                b_tensor.buffer_address(),  # in1_tensor_addr
                worker_core_idx * n_blocks_per_core * block_n,  # in1_tensor_start_tile_id
                output_tensor.buffer_address(),  # out_tensor_addr
                worker_core_idx * n_blocks_per_core * block_n,  # out_tensor_start_tile_id
            ]

            print(
                f"IN1_SENDER_WRITER - RUNTIME_ARGS for core ({output_idx_x}, {output_idx_y}):"
            )
            print(
                f"IN1_SENDER_WRITER_CORE - RUNTIME_ARGS ({len(in1_writer_rt_args[output_idx_x][output_idx_y])} args): {', '.join(map(str, in1_writer_rt_args[output_idx_x][output_idx_y]))}"
            )
            worker_core_idx += 1

    # Compute config
    computeConfig = ttnn.ComputeConfigDescriptor()
    computeConfig.math_fidelity = ttnn.MathFidelity.HiFi4

    # Kernel descriptors
    in0_sender_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul_from_scratch/metal/kernels/sender_in0_interleaved.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_sender_core,
        compile_time_args=in0_sender_compile_time_args,
        runtime_args=in0_sender_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    in0_receiver_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul_from_scratch/metal/kernels/reciever_in0_interleaved.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_receiver_cores,
        compile_time_args=in0_receiver_compile_time_args,
        runtime_args=in0_receiver_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    in1_writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul_from_scratch/metal/kernels/reader_in1_writer_out_interleaved.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=in1_writer_compile_time_args,
        runtime_args=in1_writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul_from_scratch/metal/kernels/dummy_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=compute_compile_time_args,
        runtime_args=compute_rt_args,
        config=computeConfig,
    )

    # Semaphore descriptors for synchronization
    semaphore_descriptors = [
        ttnn.SemaphoreDescriptor(
            id=in0_sender_semaphore_id,
            initial_value=0,
            core_ranges=in0_sender_core,
        ),
        ttnn.SemaphoreDescriptor(
            id=in0_receiver_semaphore_id,
            initial_value=0,
            core_ranges=all_cores,
        ),
    ]

    kernels = [
        in0_sender_kernel_descriptor,
        in0_receiver_kernel_descriptor,
        in1_writer_kernel_descriptor,
        compute_kernel_descriptor,
    ]

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=semaphore_descriptors,
        cbs=[
            a_cb_descriptor,
            b_cb_descriptor,
            out_cb_descriptor,
            intermediate_cb_descriptor,
        ],
    )

    print("Launching 1D matmul generic_op...")
    output = ttnn.generic_op([a_tensor, b_tensor, output_tensor], program_descriptor)
    print("Completed generic_op.")

    # Verify correctness
    metal_output = ttnn.to_torch(output).to(torch.bfloat16)
    a_tensor_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_tensor_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    torch_output = torch.matmul(a_tensor_torch, b_tensor_torch)

    assert_with_ulp(torch_output, metal_output)
    print("test passed.")

    ttnn.close_device(device)

