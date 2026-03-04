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
"""

import pytest
import torch
import ttnn
from utils.correctness import assert_with_ulp

TS = ttnn.TILE_SIZE  # 32


@pytest.mark.parametrize(
    "M,N,K,n_blocks_per_core,block_m,block_n,block_k,subblock_h,subblock_w",
    [
        (TS, 2 * TS, TS, 1, 1, 1, 1, 1, 1),  # trivial base case
        (TS, 14 * TS, TS, 1, 1, 1, 1, 1, 1),  # just over 1 row for all arch
        (TS, 8 * TS, TS * 2, 1, 1, 1, 1, 1, 1),  # 2 blocks in k dim
        (TS * 2, 8 * TS, TS, 1, 1, 1, 1, 1, 1),  # 2 blocks in m dim
        (TS, 8 * TS * 2, TS, 2, 1, 1, 1, 1, 1),  # 2 blocks per core in n dim
        (TS * 6, 2 * TS, TS * 2, 1, 2, 1, 1, 2, 1),
        (
            TS,
            8 * TS * 2,
            TS * 2,
            2,
            1,
            1,
            1,
            1,
            1,
        ),  # 2 blocks per core in n dim, with 2 blocks in k dim
        (
            TS * 16,
            8 * TS,
            TS * 8,
            1,
            16,
            1,
            8,
            8,
            1,
        ),  # bigger blocks in m and k dims, with 2 subblocks per block in m/h dim
        (
            TS,
            8 * TS * 16,
            TS * 8,
            1,
            1,
            16,
            8,
            1,
            8,
        ),  # bigger blocks in n and k dims, with 2 subblocks per block in n/w dim
        (
            TS * 4,
            8 * TS * 4,
            TS * 4 * 2,
            1,
            4,
            4,
            2,
            2,
            2,
        ),  # 4 tile blocks, with 2 subblocks in each dim
        (
            TS * 4,
            8 * TS * 2 * 4,
            TS * 4 * 2,
            2,
            4,
            4,
            2,
            2,
            2,
        ),  # above but with 2 blocks per core in n dim
        (
            TS * 4,
            64 * TS * 2 * 4,
            TS * 4 * 2,
            2,
            4,
            4,
            2,
            2,
            2,
        ),  # above but all cores wh
        (
            TS * 8,
            120 * TS * 2 * 8,
            TS * 16,
            2,
            8,
            8,
            16,
            4,
            2,
        ),  # all cores small bh 640/768 L1 tile limit
        (
            TS * 8 * 2,
            120 * TS * 2 * 8,
            TS * 16,
            2,
            8,
            8,
            16,
            4,
            2,
        ),  # above, but with 2 blocks in m dim
    ],
)
def test_1d_matmul_metal(
    M, N, K, n_blocks_per_core, block_m, block_n, block_k, subblock_h, subblock_w
):
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
    print(f"Device compute_with_storage_grid_size: ({num_cores_x}, {num_cores_y})")
    num_worker_cores = Nt // (block_n * n_blocks_per_core)
    assert (
        num_cores_x * num_cores_y >= num_worker_cores
    ), "Not enough cores to run the test with the given number of blocks per core"
    assert Mt % block_m == 0, "block_m must divide Mt"
    assert Nt % block_n == 0, "block_n must divide Nt"
    assert (
        Nt % (block_n * n_blocks_per_core) == 0
    ), "number of n blocks split across cores must divide Nt"
    assert Kt % block_k == 0, "block_k must divide Kt"
    assert block_m % subblock_h == 0, "subblock_h must divide block_m"
    assert block_n % subblock_w == 0, "subblock_w must divide block_n"

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
    all_cores = ttnn.num_cores_to_corerangeset(
        num_worker_cores, ttnn.CoreCoord(num_cores_x, num_cores_y), row_wise=True
    )
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
        total_size=buffer_factor * cb_page_size * (block_n * block_k),
        core_ranges=all_cores,
        format_descriptors=[b_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * (block_m * block_n),
        core_ranges=all_cores,
        format_descriptors=[out_cb_format],
    )
    intermediate_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * (block_m * block_n),
        core_ranges=all_cores,
        format_descriptors=[intermediate_cb_format],
    )
    in0_sender_semaphore_id = 0
    in0_receiver_semaphore_id = 1

    # KERNEL COMPILE TIME ARGS
    in0_num_subblocks = block_m // subblock_h
    in0_block_num_tiles = block_m * block_k
    in0_subblock_num_tiles = subblock_h * block_k
    in1_num_subblocks = block_n // subblock_w
    in1_block_num_tiles = block_n * block_k
    num_blocks_inner_dim = Kt // block_k
    num_blocks_w_dim = n_blocks_per_core
    num_blocks_h_dim = Mt // block_m
    out_subblock_num_tiles = subblock_h * subblock_w
    compute_compile_time_args = [
        block_k,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        block_n,
        num_blocks_inner_dim,
        num_blocks_w_dim,
        num_blocks_h_dim,
        subblock_h,
        subblock_w,
        out_subblock_num_tiles,
    ]
    # Compile time args for in0 sender
    in0_sender_compile_time_args = [
        1,
        Kt,
        block_k,
        block_m * Kt,
        block_k,
        block_m,
        in0_block_num_tiles,
        num_blocks_inner_dim,
        num_blocks_w_dim,
        num_blocks_h_dim,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        num_worker_cores - 1,
        num_worker_cores - 1,
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
        in0_block_num_tiles,
        num_blocks_inner_dim,
        num_blocks_w_dim,
        num_blocks_h_dim,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
    ]
    print(
        f"IN0_RECEIVER - COMPILE_TIME_ARGS ({len(in0_receiver_compile_time_args)} args): {', '.join(map(str, in0_receiver_compile_time_args))}"
    )

    # Compile time args for in1 reader + writer
    in1_writer_compile_time_args = [
        1,
        Nt,
        block_k * Nt,
        block_n,
        block_n,
        block_k,
        in1_block_num_tiles,
        num_blocks_inner_dim,
        num_blocks_w_dim,
        num_blocks_h_dim,
        1,
        Nt,
        subblock_w,
        subblock_h * Nt,
        block_n,
        block_m * Nt,
        subblock_w,
        subblock_h,
        in1_num_subblocks,
        in0_num_subblocks,
        out_subblock_num_tiles,
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
    num_y_cores = (
        -(-num_worker_cores // num_cores_x) if num_cores_x < num_worker_cores else 1
    )

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
            if worker_core_idx >= num_worker_cores:
                break
            # in0 sender args (only for core (0,0))
            # Single sender multicasts to all other cores in the grid
            if output_idx_x == 0 and output_idx_y == 0:
                # NOTE: multicast nocs require perfect rectangular core regions
                # so when num_worker_cores % num_cores_x != 0, the last row of cores will be multicasted to, but not utilized
                mcast_end_core_noc = device.worker_core_from_logical_core(
                    ttnn.CoreCoord(num_x_cores - 1, num_y_cores - 1)
                )

                in0_sender_rt_args[output_idx_x][output_idx_y] = [
                    a_tensor.buffer_address(),
                    0,
                    noc_of_sender.x,
                    noc_of_sender.y,
                    mcast_end_core_noc.x,
                    mcast_end_core_noc.y,
                ]
                print(
                    f"IN0_SENDER - RUNTIME_ARGS for core ({output_idx_x}, {output_idx_y}), worker: {worker_core_idx}"
                )
                print(
                    f"IN0_SENDER_CORE - RUNTIME_ARGS ({len(in0_sender_rt_args[output_idx_x][output_idx_y])} args): {', '.join(map(str, in0_sender_rt_args[output_idx_x][output_idx_y]))}"
                )

            # in0 receiver args (for all cores except (0,0))
            if not (output_idx_x == 0 and output_idx_y == 0):
                in0_receiver_rt_args[output_idx_x][output_idx_y] = [
                    noc_of_sender.x,
                    noc_of_sender.y,
                ]
                print(
                    f"IN0_RECEIVER - RUNTIME_ARGS for core ({output_idx_x}, {output_idx_y}), worker: {worker_core_idx}"
                )
                print(
                    f"IN0_RECEIVER_CORE - RUNTIME_ARGS ({len(in0_receiver_rt_args[output_idx_x][output_idx_y])} args): {', '.join(map(str, in0_receiver_rt_args[output_idx_x][output_idx_y]))}"
                )

            # in1 reader + writer args (all cores)
            in1_writer_rt_args[output_idx_x][output_idx_y] = [
                b_tensor.buffer_address(),
                worker_core_idx * n_blocks_per_core * block_n,
                output_tensor.buffer_address(),
                worker_core_idx * n_blocks_per_core * block_n,
            ]

            print(
                f"IN1_SENDER_WRITER - RUNTIME_ARGS for core ({output_idx_x}, {output_idx_y}), worker: {worker_core_idx}"
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
        kernel_source="examples/metal_examples/1d_mcast_matmul/metal/kernels/sender_in0_interleaved.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_sender_core,
        compile_time_args=in0_sender_compile_time_args,
        runtime_args=in0_sender_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    in0_receiver_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_mcast_matmul/metal/kernels/reciever_in0_interleaved.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_receiver_cores,
        compile_time_args=in0_receiver_compile_time_args,
        runtime_args=in0_receiver_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    in1_writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_mcast_matmul/metal/kernels/reader_in1_writer_out_interleaved.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=in1_writer_compile_time_args,
        runtime_args=in1_writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_mcast_matmul/metal/kernels/reuse_compute.cpp",
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
