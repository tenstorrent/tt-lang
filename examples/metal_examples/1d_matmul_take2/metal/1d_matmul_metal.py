# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
1D Matmul Metal Example

This example demonstrates a 1D matmul where only input A (in0) is multicast across cores.
Unlike the 2D matmul where both inputs are multicast, here:
- in0 is broadcast from sender core(s) to receiver cores
- in1 is read locally by each core (no multicast)
- Output is written locally by each core

The kernel configuration uses:
1. reader_bmm_tile_layout_in0_sender_padding.cpp - Sender core reads in0 and multicasts
2. reader_bmm_tile_layout_in0_receiver.cpp - Receiver cores receive in0 multicast
3. reader_bmm_tile_layout_in1_sender_writer_padding.cpp - All cores read in1 locally and write output
4. bmm_large_block_zm_fused_bias_activation.cpp - Compute kernel on all cores

Note: This implementation does not use bias, sharding, or activation fusion yet.
These features are supported by the kernels but not configured in this example.
"""

import pytest
import torch
import ttnn
from ttlang.utils.block_allocation import get_large_matmul_params
from ttlang.utils.correctness import assert_with_ulp


@pytest.mark.parametrize("M,K,N", [(2560, 640, 640)])
def test_1d_matmul_metal(M, K, N):
    device = ttnn.open_device(device_id=0)
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    K_block_size = 2

    device_core_size = device.compute_with_storage_grid_size()
    print(
        f"Device compute_with_storage_grid_size: ({device_core_size.x}, {device_core_size.y})"
    )
    num_cores_x = device_core_size.x
    num_cores_y = device_core_size.y

    # Get block parameters for distributing work across cores
    block_params = get_large_matmul_params(
        Mt, Nt, num_cores_y, num_cores_x, K_block_size
    )
    per_core_M = block_params.block_h
    per_core_N = block_params.block_w
    out_subblock_h = block_params.subblock_h
    out_subblock_w = block_params.subblock_w
    assert per_core_M != 0, "get_large_matmul_params was not able to find a solution"
    print(
        f"per_core_M: {per_core_M}, per_core_N: {per_core_N}, out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}"
    )
    
    # Validate block parameters
    assert Mt % per_core_M == 0, "per_core_M must divide Mt"
    assert Nt % per_core_N == 0, "per_core_N must divide Nt"
    assert Kt % K_block_size == 0, "K_block_size must divide Kt"
    
    num_blocks_y = Mt // per_core_M
    num_blocks_x = Nt // per_core_N
    assert (
        num_blocks_x <= num_cores_x and num_blocks_y <= num_cores_y
    ), "number of total blocks must be less than or equal to num cores in each dimension"

    # For 1D matmul: Use first row (y=0) as sender cores for in0 multicast
    # All cores are receivers and do computation
    # We assume num_blocks_y > 1 (using all 4 kernels)
    assert num_blocks_y > 1, "1D matmul requires num_blocks_y > 1 to use all 4 kernels"
    
    in0_sender_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_blocks_x - 1, 0))]
    )
    
    # All compute cores (entire grid used for computation)
    all_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_blocks_x - 1, num_blocks_y - 1)
            )
        ]
    )
    
    # Receiver cores are all cores except the sender row
    # For 1D, in0 is sent along columns (y-direction)
    in0_receiver_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 1), ttnn.CoreCoord(num_blocks_x - 1, num_blocks_y - 1)
            )
        ]
    )

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

    # Compute kernel compile time args
    num_blocks = Kt // K_block_size

    a_num_subblocks = per_core_M // out_subblock_h
    a_block_num_tiles = out_subblock_h * K_block_size * a_num_subblocks
    a_subblock_num_tiles = out_subblock_h * K_block_size

    b_num_subblocks = per_core_N // out_subblock_w
    b_block_num_tiles = out_subblock_w * K_block_size * b_num_subblocks
    b_per_core_w = out_subblock_w * b_num_subblocks

    out_subblock_num_tiles = out_subblock_h * out_subblock_w
    out_block_num_tiles = per_core_M * per_core_N

    compute_compile_time_args = [
        K_block_size,                # in0_block_w
        a_num_subblocks,             # in0_num_subblocks
        a_block_num_tiles,           # in0_block_num_tiles
        a_subblock_num_tiles,        # in0_subblock_num_tiles
        b_num_subblocks,             # in1_num_subblocks
        b_block_num_tiles,           # in1_block_num_tiles
        b_per_core_w,                # in1_block_w
        num_blocks,                  # num_blocks_inner_dim
        1,                           # num_blocks_w_dim (always 1 per core in 1D)
        1,                           # num_blocks_h_dim (always 1 per core in 1D)
        out_subblock_h,              # out_subblock_h
        out_subblock_w,              # out_subblock_w
        out_subblock_num_tiles,      # out_subblock_num_tiles
        1,                           # batch
        out_block_num_tiles,         # out_block_num_tiles
        0,                           # untilize_out (false)
        0,                           # get_batch_from_reader (false, no sparsity)
        0,                           # in0_transpose_tile (false)
    ]

    # Semaphores for synchronization between sender and receiver
    # TODO: BIAS, SHARDING, ACTIVATION - These kernel features are not configured yet
    # The kernels support:
    # - FUSE_BIAS: Adding bias during computation
    # - IN0_SHARDED/IN1_SHARDED: Sharded memory layout
    # - SFPU_OP_INIT_ACTIVATION: Fused activation functions
    # These would require additional compile-time defines and runtime arguments

    in0_sender_semaphore_id = 0
    in0_receiver_semaphore_id = 1

    # Setup runtime args for each core
    num_x_cores = num_blocks_x
    num_y_cores = num_blocks_y
    
    in0_sender_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    in0_receiver_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    in1_writer_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]
    compute_rt_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]

    print(
        f"num_blocks_x: {num_blocks_x}, num_blocks_y: {num_blocks_y}, output tiles is {Mt}x{Nt}"
    )
    print(
        f"1D matmul: Each sender at y=0 multicasts to {num_blocks_y-1} receivers in its column"
    )

    # Assign work to cores
    for output_idx_y in range(num_blocks_y):
        for output_idx_x in range(num_blocks_x):
            # in0 sender args (only for y=0 cores)
            # Each sender multicasts to its column (same x, y from 1 to num_blocks_y-1)
            if output_idx_y == 0:
                in0_sender_rt_args[output_idx_x][output_idx_y] = [
                    a_tensor.buffer_address(),                    # in0_tensor_addr
                    Kt * per_core_M * output_idx_y,               # in0_tensor_start_tile_id
                    output_idx_x,                                 # in0_mcast_dest_noc_start_x (this column only)
                    1,                                            # in0_mcast_dest_noc_start_y (start at y=1)
                    output_idx_x,                                 # in0_mcast_dest_noc_end_x (same column)
                    num_blocks_y - 1,                             # in0_mcast_dest_noc_end_y (end at last y)
                    per_core_M,                                   # last_block_h (no padding in this example)
                    0,                                            # sparsity_addr (disabled)
                ]
                print(f"  Sender core [{output_idx_x},{output_idx_y}] multicasts to column x={output_idx_x}, y=[1,{num_blocks_y-1}]")

            # in0 receiver args (for y>0 cores)
            if output_idx_y > 0:
                in0_receiver_rt_args[output_idx_x][output_idx_y] = [
                    output_idx_x,  # in0_mcast_sender_noc_x (sender is at same x)
                    0,             # in0_mcast_sender_noc_y (always row 0)
                ]
                print(f"  Receiver core [{output_idx_x},{output_idx_y}] receives from sender at [{output_idx_x},0]")

            # in1 reader + writer args (all cores)
            in1_writer_rt_args[output_idx_x][output_idx_y] = [
                b_tensor.buffer_address(),                        # in1_tensor_addr
                per_core_N * output_idx_x,                        # in1_tensor_start_tile_id
                0,  # in1_mcast_dest_noc_start_x (not used for 1D, no in1 mcast)
                0,  # in1_mcast_dest_noc_start_y
                0,  # in1_mcast_dest_noc_end_x
                0,  # in1_mcast_dest_noc_end_y
                0,  # sparsity_addr (disabled)
                output_tensor.buffer_address(),                   # out_tensor_addr
                (output_idx_x * per_core_N) + (output_idx_y * per_core_M * Nt),  # out_tensor_start_tile_id
                per_core_N,                                       # last_block_w (no padding)
                a_num_subblocks,                                  # out_num_nonzero_subblocks_h
                out_subblock_h,                                   # out_last_subblock_h
                0,                                                # padded_block_tiles_h_skip
                b_num_subblocks,                                  # out_num_nonzero_subblocks_w
                b_num_subblocks,                                  # out_last_num_nonzero_subblocks_w
                out_subblock_w,                                   # out_last_subblock_w
                0,                                                # padded_subblock_tiles_addr_skip
                0,                                                # padded_block_tiles_w_skip
            ]

            print(
                f"core [{output_idx_x},{output_idx_y}] assigned output block slice [{output_idx_x * per_core_N}:{(output_idx_x + 1) * per_core_N}]x[{output_idx_y * per_core_M}:{(output_idx_y + 1) * per_core_M}]"
            )

    # Compile time args for in0 sender
    # Each sender multicasts to receivers in its column only
    in0_mcast_num_dests_per_sender = num_blocks_y - 1  # Number of receivers per sender
    in0_mcast_num_cores_per_sender = num_blocks_y - 1  # Total receivers in the column
    
    in0_sender_compile_time_args = [
        1,                                      # in0_tensor_stride_w
        Kt,                                     # in0_tensor_stride_h
        K_block_size,                           # in0_tensor_next_inner_dim_block_stride
        K_block_size * per_core_M,              # in0_tensor_next_h_dim_block_stride
        K_block_size,                           # in0_block_w
        per_core_M,                             # in0_block_h
        per_core_M * K_block_size,              # in0_block_num_tiles
        0,                                      # in0_last_ktile_w (no padding)
        0,                                      # extract_shard_sub_blocks (false)
        0,                                      # shard_width_in_tiles (not sharded)
        0,                                      # shard_height_in_tiles (not sharded)
        num_blocks,                             # num_blocks_inner_dim
        1,                                      # num_blocks_w_dim
        1,                                      # num_blocks_h_dim
        in0_sender_semaphore_id,                # in0_mcast_sender_semaphore
        in0_receiver_semaphore_id,              # in0_mcast_receiver_semaphore
        in0_mcast_num_dests_per_sender,         # in0_mcast_num_dests (per sender)
        in0_mcast_num_cores_per_sender,         # in0_mcast_num_cores (per sender)
        0,                                      # MtKt (batch, not used)
        1,                                      # batch
        0,                                      # batchB (sparsity, disabled)
        0,                                      # sparsity_pagesize
        0,                                      # bcast_A (false)
        0,                                      # get_batch_from_reader (false)
        0,                                      # fuse_op (false)
    ]
    # Add TensorAccessor compile time args for in0
    in0_sender_compile_time_args.extend(
        ttnn.TensorAccessorArgs(a_tensor).get_compile_time_args()
    )
    # Add TensorAccessor compile time args for sparsity (dummy, disabled)
    in0_sender_compile_time_args.extend([0] * 25)  # Placeholder for sparsity accessor

    # Compile time args for in0 receiver
    in0_receiver_compile_time_args = [
        per_core_M * K_block_size,              # in0_block_num_tiles
        num_blocks,                             # num_blocks_inner_dim
        1,                                      # num_blocks_w_dim
        1,                                      # num_blocks_h_dim
        in0_sender_semaphore_id,                # in0_mcast_sender_semaphore
        in0_receiver_semaphore_id,              # in0_mcast_receiver_semaphore
        1,                                      # batch
        0,                                      # get_batch_from_reader (false)
    ]

    # Compile time args for in1 reader + writer
    in1_writer_compile_time_args = [
        1,                                      # in1_tensor_stride_w
        Nt,                                     # in1_tensor_stride_h
        K_block_size * Nt,                      # in1_tensor_next_block_stride
        per_core_N,                             # in1_tensor_next_w_dim_block_stride
        per_core_N,                             # in1_block_w
        K_block_size,                           # in1_block_h
        per_core_N * K_block_size,              # in1_block_num_tiles
        num_blocks,                             # num_blocks_inner_dim
        1,                                      # num_blocks_w_dim
        1,                                      # num_blocks_h_dim
        0,                                      # in1_mcast_sender_semaphore (not used, no in1 mcast)
        0,                                      # in1_mcast_receiver_semaphore (not used)
        0,                                      # in1_mcast_num_dests (0 for 1D)
        0,                                      # in1_mcast_num_cores (0 for 1D)
        0,                                      # KtNt (batch, not used)
        1,                                      # batch
        0,                                      # bcast_B (false)
        0,                                      # batchB (sparsity, disabled)
        0,                                      # sparsity_pagesize
        # Output tensor args
        1,                                      # out_tensor_stride_w
        Nt,                                     # out_tensor_stride_h
        out_subblock_w,                         # out_tensor_next_subblock_stride_w
        out_subblock_h * Nt,                    # out_tensor_next_subblock_stride_h
        per_core_N,                             # out_tensor_next_w_dim_block_stride
        per_core_M * Nt,                        # out_tensor_next_h_dim_block_stride
        out_subblock_w,                         # out_subblock_w
        out_subblock_h,                         # out_subblock_h
        out_subblock_w * out_subblock_h,        # out_subblock_tile_count
        0,                                      # MtNt (batch, not used)
        0,                                      # fuse_op_all_gather (false)
        0,                                      # fuse_op_reduce_scatter (false)
    ]
    # Add TensorAccessor compile time args for in1
    in1_writer_compile_time_args.extend(
        ttnn.TensorAccessorArgs(b_tensor).get_compile_time_args()
    )
    # Add TensorAccessor compile time args for sparsity (dummy, disabled)
    in1_writer_compile_time_args.extend([0] * 25)  # Placeholder for sparsity accessor
    # Add TensorAccessor compile time args for output
    in1_writer_compile_time_args.extend(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )

    # Compute config
    computeConfig = ttnn.ComputeConfigDescriptor()
    computeConfig.math_fidelity = ttnn.MathFidelity.HiFi4

    # Define compile-time flags for kernels
    # TODO: Enable these when implementing bias/sharding/activation features
    # in0_sender_defines = ["SKIP_MCAST"]  # Would disable multicast for testing
    # in1_writer_defines = ["SKIP_MCAST"]  # Would disable in1 multicast (already disabled in 1D)
    # compute_defines = ["FUSE_BIAS", "SFPU_OP_INIT_ACTIVATION"]  # For bias and activation

    # Kernel descriptors
    in0_sender_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul_take2/metal/kernels/reader_bmm_tile_layout_in0_sender_padding.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_sender_cores,
        compile_time_args=in0_sender_compile_time_args,
        runtime_args=in0_sender_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    in0_receiver_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul_take2/metal/kernels/reader_bmm_tile_layout_in0_receiver.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_receiver_cores,
        compile_time_args=in0_receiver_compile_time_args,
        runtime_args=in0_receiver_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    in1_writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul_take2/metal/kernels/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=in1_writer_compile_time_args,
        runtime_args=in1_writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul_take2/metal/kernels/bmm_large_block_zm_fused_bias_activation.cpp",
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
            core_ranges=all_cores,
        ),
        ttnn.SemaphoreDescriptor(
            id=in0_receiver_semaphore_id,
            initial_value=0,
            core_ranges=all_cores,
        ),
    ]

    # Build kernel list - all 4 kernels
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
    print("✓ Test passed!")

    ttnn.close_device(device)
