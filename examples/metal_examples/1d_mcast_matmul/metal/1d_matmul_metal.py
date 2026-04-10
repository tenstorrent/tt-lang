# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
1D Matmul Metal Example

This example demonstrates a 1D matmul where only input A (in0) is multicast across nodes.
Unlike the 2D matmul where both inputs are multicast, here:
- in0 is broadcast from a single sender node (0,0) to all other nodes
- in1 is read locally by each node (no multicast)
- Output is written locally by each node
"""

import pytest
import torch
import ttnn
from utils.correctness import assert_with_ulp

TS = ttnn.TILE_SIZE  # 32


@pytest.mark.parametrize(
    "M,N,K,n_blocks_per_node,block_m,block_n,block_k,subblock_h,subblock_w",
    [
        (TS, 2 * TS, TS, 1, 1, 1, 1, 1, 1),  # trivial base case
        (TS, 14 * TS, TS, 1, 1, 1, 1, 1, 1),  # just over 1 row for all arch
        (TS, 8 * TS, TS * 2, 1, 1, 1, 1, 1, 1),  # 2 blocks in k dim
        (TS * 2, 8 * TS, TS, 1, 1, 1, 1, 1, 1),  # 2 blocks in m dim
        (TS, 8 * TS * 2, TS, 2, 1, 1, 1, 1, 1),  # 2 blocks per node in n dim
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
        ),  # 2 blocks per node in n dim, with 2 blocks in k dim
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
        ),  # above but with 2 blocks per node in n dim
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
        ),  # above but all nodes wh
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
        ),  # all nodes small bh 640/768 L1 tile limit
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
    M, N, K, n_blocks_per_node, block_m, block_n, block_k, subblock_h, subblock_w
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

    device_node_size = device.compute_with_storage_grid_size()
    num_nodes_x = device_node_size.x
    num_nodes_y = device_node_size.y
    print(f"Device compute_with_storage_grid_size: ({num_nodes_x}, {num_nodes_y})")
    num_worker_nodes = Nt // (block_n * n_blocks_per_node)
    assert (
        num_nodes_x * num_nodes_y >= num_worker_nodes
    ), "Not enough nodes to run the test with the given number of blocks per node"
    assert Mt % block_m == 0, "block_m must divide Mt"
    assert Nt % block_n == 0, "block_n must divide Nt"
    assert (
        Nt % (block_n * n_blocks_per_node) == 0
    ), "number of n blocks split across nodes must divide Nt"
    assert Kt % block_k == 0, "block_k must divide Kt"
    assert block_m % subblock_h == 0, "subblock_h must divide block_m"
    assert block_n % subblock_w == 0, "subblock_w must divide block_n"

    # For 1D matmul: Use single node (0,0) as sender for in0 multicast
    # All other nodes are receivers and all nodes do computation
    assert (
        num_worker_nodes > 1
    ), "1D matmul requires multiple blocks to use all 4 kernels"

    # Single sender node at (0, 0) broadcasts to all other nodes
    in0_sender_node = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
    )
    # All compute nodes (entire grid used for computation)
    all_nodes = ttnn.num_cores_to_corerangeset(
        num_worker_nodes, ttnn.CoreCoord(num_nodes_x, num_nodes_y), row_wise=True
    )
    # Receiver nodes are all nodes except the single sender node (0,0)
    in0_receiver_nodes = all_nodes.subtract(in0_sender_node)

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

    block_count = 2
    a_cb_descriptor = ttnn.CBDescriptor(
        total_size=block_count * cb_page_size * (block_m * block_k),
        core_ranges=all_nodes,
        format_descriptors=[a_cb_format],
    )
    b_cb_descriptor = ttnn.CBDescriptor(
        total_size=block_count * cb_page_size * (block_n * block_k),
        core_ranges=all_nodes,
        format_descriptors=[b_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * (block_m * block_n),
        core_ranges=all_nodes,
        format_descriptors=[out_cb_format],
    )
    intermediate_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * (block_m * block_n),
        core_ranges=all_nodes,
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
    num_blocks_w_dim = n_blocks_per_node
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
        num_worker_nodes - 1,
        num_worker_nodes - 1,
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
    # Setup runtime args for each node
    num_x_nodes = num_nodes_x if num_nodes_x < num_worker_nodes else num_worker_nodes
    num_y_nodes = (
        -(-num_worker_nodes // num_nodes_x) if num_nodes_x < num_worker_nodes else 1
    )

    in0_sender_rt_args = []
    in0_receiver_rt_args = []
    in1_writer_rt_args = []
    compute_rt_args = []

    total_receivers = num_worker_nodes - 1
    print(
        f"1D matmul: Single sender at (0,0) multicasts to {total_receivers} receivers, across a grid of {num_x_nodes} x {num_y_nodes} nodes"
    )

    noc_of_sender = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))

    # Assign work to nodes
    worker_node_idx = 0
    for output_idx_y in range(num_y_nodes):
        for output_idx_x in range(num_x_nodes):
            if worker_node_idx >= num_worker_nodes:
                break
            core = ttnn.CoreCoord(output_idx_x, output_idx_y)
            # in0 sender args (only for node (0,0))
            # Single sender multicasts to all other nodes in the grid
            if output_idx_x == 0 and output_idx_y == 0:
                # NOTE: multicast nocs require perfect rectangular node regions
                # so when num_worker_nodes % num_nodes_x != 0, the last row of nodes will be multicasted to, but not utilized
                mcast_end_node_noc = device.worker_core_from_logical_core(
                    ttnn.CoreCoord(num_x_nodes - 1, num_y_nodes - 1)
                )

                sender_args = [
                    a_tensor.buffer_address(),
                    0,
                    noc_of_sender.x,
                    noc_of_sender.y,
                    mcast_end_node_noc.x,
                    mcast_end_node_noc.y,
                ]
                in0_sender_rt_args.append((core, sender_args))
                print(
                    f"IN0_SENDER - RUNTIME_ARGS for node ({output_idx_x}, {output_idx_y}), worker: {worker_node_idx}"
                )
                print(
                    f"IN0_SENDER_CORE - RUNTIME_ARGS ({len(sender_args)} args): {', '.join(map(str, sender_args))}"
                )

            # in0 receiver args (for all nodes except (0,0))
            if not (output_idx_x == 0 and output_idx_y == 0):
                receiver_args = [
                    noc_of_sender.x,
                    noc_of_sender.y,
                ]
                in0_receiver_rt_args.append((core, receiver_args))
                print(
                    f"IN0_RECEIVER - RUNTIME_ARGS for node ({output_idx_x}, {output_idx_y}), worker: {worker_node_idx}"
                )
                print(
                    f"IN0_RECEIVER_CORE - RUNTIME_ARGS ({len(receiver_args)} args): {', '.join(map(str, receiver_args))}"
                )

            # in1 reader + writer args (all nodes)
            in1_writer_args = [
                b_tensor.buffer_address(),
                worker_node_idx * n_blocks_per_node * block_n,
                output_tensor.buffer_address(),
                worker_node_idx * n_blocks_per_node * block_n,
            ]
            in1_writer_rt_args.append((core, in1_writer_args))

            print(
                f"IN1_SENDER_WRITER - RUNTIME_ARGS for node ({output_idx_x}, {output_idx_y}), worker: {worker_node_idx}"
            )
            print(
                f"IN1_SENDER_WRITER_CORE - RUNTIME_ARGS ({len(in1_writer_args)} args): {', '.join(map(str, in1_writer_args))}"
            )
            worker_node_idx += 1

    # Compute config
    computeConfig = ttnn.ComputeConfigDescriptor()
    computeConfig.math_fidelity = ttnn.MathFidelity.HiFi4

    # Kernel descriptors
    in0_sender_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_mcast_matmul/metal/kernels/sender_in0_interleaved.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_sender_node,
        compile_time_args=in0_sender_compile_time_args,
        runtime_args=in0_sender_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    in0_receiver_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_mcast_matmul/metal/kernels/reciever_in0_interleaved.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_receiver_nodes,
        compile_time_args=in0_receiver_compile_time_args,
        runtime_args=in0_receiver_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    in1_writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_mcast_matmul/metal/kernels/reader_in1_writer_out_interleaved.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_nodes,
        compile_time_args=in1_writer_compile_time_args,
        runtime_args=in1_writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_mcast_matmul/metal/kernels/reuse_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_nodes,
        compile_time_args=compute_compile_time_args,
        runtime_args=compute_rt_args,
        config=computeConfig,
    )

    # Semaphore descriptors for synchronization
    semaphore_descriptors = [
        ttnn.SemaphoreDescriptor(
            id=in0_sender_semaphore_id,
            initial_value=0,
            core_ranges=in0_sender_node,
        ),
        ttnn.SemaphoreDescriptor(
            id=in0_receiver_semaphore_id,
            initial_value=0,
            core_ranges=all_nodes,
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
    print("Test passed!")

    ttnn.close_device(device)
