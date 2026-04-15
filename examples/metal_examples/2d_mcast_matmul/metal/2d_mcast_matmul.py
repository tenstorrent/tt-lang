# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn
from ttl.utils.block_allocation import get_large_matmul_params
from ttl.utils.correctness import assert_with_ulp


@pytest.mark.parametrize("M,K,N", [(3584, 768, 3072)])
def test_2d_mcast_matmul(M, K, N):
    device = ttnn.open_device(device_id=0)
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    in0_block_w = 2

    device_grid = device.compute_with_storage_grid_size()
    print(f"Device compute_with_storage_grid_size: ({device_grid.x}, {device_grid.y})")
    num_nodes_x = device_grid.x
    num_nodes_y = device_grid.y

    block_params = get_large_matmul_params(
        Mt, Nt, num_nodes_y, num_nodes_x, in0_block_w
    )
    per_node_M = block_params.block_h
    per_node_N = block_params.block_w
    out_subblock_h = block_params.subblock_h
    out_subblock_w = block_params.subblock_w
    assert per_node_M != 0, "get_large_matmul_params was not able to find a solution"
    print(
        f"per_node_M: {per_node_M}, per_node_N: {per_node_N}, out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}"
    )
    assert Mt % per_node_M == 0, "per_node_M must divide Mt"
    assert Nt % per_node_N == 0, "per_node_N must divide Nt"
    assert Kt % in0_block_w == 0, "in0_block_w must divide Kt"

    num_blocks_y = Mt // per_node_M
    num_blocks_x = Nt // per_node_N
    assert (
        num_blocks_x <= num_nodes_x and num_blocks_y <= num_nodes_y
    ), "number of total blocks must be less than or equal to num nodes in each dimension"
    assert (
        num_blocks_x >= 2 and num_blocks_y >= 2
    ), "2D mcast requires at least a 2x2 node grid"

    num_active_x = num_blocks_x
    num_active_y = num_blocks_y

    all_nodes = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_active_x - 1, num_active_y - 1),
            )
        ]
    )
    left_column = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_active_y - 1))]
    )
    all_except_left_column = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(1, 0),
                ttnn.CoreCoord(num_active_x - 1, num_active_y - 1),
            )
        ]
    )
    in0_sender_in1_sender = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
    )
    in0_sender_in1_receiver = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, num_active_y - 1))]
    )
    in0_receiver_in1_sender = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(num_active_x - 1, 0))]
    )
    in0_receiver_in1_receiver = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(1, 1),
                ttnn.CoreCoord(num_active_x - 1, num_active_y - 1),
            )
        ]
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

    in0_block_tiles = per_node_M * in0_block_w
    in1_block_tiles = per_node_N * in0_block_w
    out_block_tiles = per_node_M * per_node_N
    buffer_factor = 2
    a_cb_descriptor = ttnn.CBDescriptor(
        total_size=buffer_factor * cb_page_size * in0_block_tiles,
        core_ranges=all_nodes,
        format_descriptors=[a_cb_format],
    )
    b_cb_descriptor = ttnn.CBDescriptor(
        total_size=buffer_factor * cb_page_size * in1_block_tiles,
        core_ranges=all_nodes,
        format_descriptors=[b_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * out_block_tiles,
        core_ranges=all_nodes,
        format_descriptors=[out_cb_format],
    )
    intermediate_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * out_block_tiles,
        core_ranges=all_nodes,
        format_descriptors=[intermediate_cb_format],
    )

    # 4 semaphores for 2D mcast handshake
    in0_mcast_sender_semaphore_id = 0
    in0_mcast_receiver_semaphore_id = 1
    in1_mcast_sender_semaphore_id = 2
    in1_mcast_receiver_semaphore_id = 3

    semaphore_descriptors = [
        ttnn.SemaphoreDescriptor(
            id=in0_mcast_sender_semaphore_id,
            initial_value=0,
            core_ranges=all_nodes,
        ),
        ttnn.SemaphoreDescriptor(
            id=in0_mcast_receiver_semaphore_id,
            initial_value=0,
            core_ranges=all_nodes,
        ),
        ttnn.SemaphoreDescriptor(
            id=in1_mcast_sender_semaphore_id,
            initial_value=0,
            core_ranges=all_nodes,
        ),
        ttnn.SemaphoreDescriptor(
            id=in1_mcast_receiver_semaphore_id,
            initial_value=0,
            core_ranges=all_nodes,
        ),
    ]

    # Compute kernel compile time args
    in0_num_subblocks = per_node_M // out_subblock_h
    in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks
    in0_subblock_num_tiles = out_subblock_h * in0_block_w

    in1_num_subblocks = per_node_N // out_subblock_w
    in1_block_num_tiles_compute = out_subblock_w * in0_block_w * in1_num_subblocks
    in1_per_node_w = out_subblock_w * in1_num_subblocks

    num_blocks = Kt // in0_block_w
    out_subblock_num_tiles = out_subblock_h * out_subblock_w

    compute_compile_time_args = [
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles_compute,
        in1_per_node_w,
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

    reader_rt_args_corner = []
    reader_rt_args_left = []
    reader_rt_args_top = []
    reader_rt_args_interior = []
    writer_rt_args_left = []
    writer_rt_args_rest = []

    for node_idx_y in range(num_active_y):
        for node_idx_x in range(num_active_x):
            left_node = ttnn.CoreCoord(0, node_idx_y)
            left_node_plus_one = ttnn.CoreCoord(1, node_idx_y)
            right_node = ttnn.CoreCoord(num_active_x - 1, node_idx_y)
            top_node = ttnn.CoreCoord(node_idx_x, 0)
            top_node_plus_one = ttnn.CoreCoord(node_idx_x, 1)
            bottom_node = ttnn.CoreCoord(node_idx_x, num_active_y - 1)

            left_phys = device.worker_core_from_logical_core(left_node)
            left_plus_one_phys = device.worker_core_from_logical_core(
                left_node_plus_one
            )
            right_phys = device.worker_core_from_logical_core(right_node)
            top_phys = device.worker_core_from_logical_core(top_node)
            top_plus_one_phys = device.worker_core_from_logical_core(top_node_plus_one)
            bottom_phys = device.worker_core_from_logical_core(bottom_node)

            core = ttnn.CoreCoord(node_idx_x, node_idx_y)
            mm_reader_args = [
                a_tensor.buffer_address(),
                Kt * per_node_M * node_idx_y,  # in0 start tile
                1,  # in0 stride w
                Kt,  # in0 stride h
                in0_block_w,  # in0 next block stride
                in0_block_w,  # in0 block w
                per_node_M,  # in0 block h
                in0_block_w * per_node_M,  # in0 block num tiles
                b_tensor.buffer_address(),
                per_node_N * node_idx_x,  # in1 start tile
                1,  # in1 stride w
                Nt,  # in1 stride h
                in0_block_w * Nt,  # in1 next block stride
                per_node_N,  # in1 block w
                in0_block_w,  # in1 block h
                per_node_N * in0_block_w,  # in1 block num tiles
                Kt // in0_block_w,  # num blocks
                # in0 mcast args (rightward from left column)
                # NOTE: Physical NOC coords may be inverted from logical coords.
                # The kernel passes (end, start) to get_noc_multicast_addr to
                # produce the correct physical bounding box for the NOC in use.
                right_phys.x,  # in0_mcast_dest_noc_start
                right_phys.y,
                left_plus_one_phys.x,  # in0_mcast_dest_noc_end
                left_plus_one_phys.y,
                num_active_x - 1,  # in0 mcast num dests
                left_phys.x,
                left_phys.y,
                in0_mcast_sender_semaphore_id,
                in0_mcast_receiver_semaphore_id,
                # in1 mcast args (downward from top row)
                # NOTE: Same start/end convention as in0 above.
                bottom_phys.x,  # in1_mcast_dest_noc_start
                bottom_phys.y,
                top_plus_one_phys.x,  # in1_mcast_dest_noc_end
                top_plus_one_phys.y,
                num_active_y - 1,  # in1 mcast num dests
                top_phys.x,
                top_phys.y,
                in1_mcast_sender_semaphore_id,
                in1_mcast_receiver_semaphore_id,
            ]

            writer_args = [
                output_tensor.buffer_address(),
                node_idx_x * per_node_N + node_idx_y * per_node_M * Nt,
                1,  # stride w
                Nt,  # stride h
                out_subblock_w,  # next subblock stride w
                out_subblock_h * Nt,  # next subblock stride h
                out_subblock_w,
                out_subblock_h,
                out_subblock_w * out_subblock_h,
                per_node_N // out_subblock_w,  # num subblocks w
                per_node_M // out_subblock_h,  # num subblocks h
            ]

            if node_idx_x == 0 and node_idx_y == 0:
                reader_rt_args_corner.append((core, mm_reader_args))
                writer_rt_args_left.append((core, writer_args))
            elif node_idx_x == 0:
                reader_rt_args_left.append((core, mm_reader_args))
                writer_rt_args_left.append((core, writer_args))
            elif node_idx_y == 0:
                reader_rt_args_top.append((core, mm_reader_args))
                writer_rt_args_rest.append((core, writer_args))
            else:
                reader_rt_args_interior.append((core, mm_reader_args))
                writer_rt_args_rest.append((core, writer_args))

    # Left column (in0 senders): reader on RISCV_1/NOC0, writer on RISCV_0/NOC1
    # Non-left column (in0 receivers): reader on RISCV_1/NOC1, writer on RISCV_0/NOC0
    reader_config_noc0 = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default,
    )
    reader_config_noc1 = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_1_default,
    )
    writer_config_noc0 = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.RISCV_0_default,
    )
    writer_config_noc1 = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.RISCV_1_default,
    )

    computeConfig = ttnn.ComputeConfigDescriptor()
    computeConfig.math_fidelity = ttnn.MathFidelity.HiFi4

    reader_corner = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/2d_mcast_matmul/metal/kernels/reader_bmm_tile_layout_in0_sender_in1_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_sender_in1_sender,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args_corner,
        config=reader_config_noc0,
    )
    reader_left = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/2d_mcast_matmul/metal/kernels/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_sender_in1_receiver,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args_left,
        config=reader_config_noc0,
    )
    reader_top = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/2d_mcast_matmul/metal/kernels/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_receiver_in1_sender,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args_top,
        config=reader_config_noc1,
    )
    reader_interior = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/2d_mcast_matmul/metal/kernels/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=in0_receiver_in1_receiver,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args_interior,
        config=reader_config_noc1,
    )
    writer_left_col = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/2d_mcast_matmul/metal/kernels/writer_bmm_tile_layout.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=left_column,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args_left,
        config=writer_config_noc1,
    )
    writer_rest = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/2d_mcast_matmul/metal/kernels/writer_bmm_tile_layout.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_except_left_column,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args_rest,
        config=writer_config_noc0,
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/2d_mcast_matmul/metal/kernels/bmm_large_block_zm.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_nodes,
        compile_time_args=compute_compile_time_args,
        runtime_args=[],
        config=computeConfig,
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[
            reader_corner,
            reader_left,
            reader_top,
            reader_interior,
            writer_left_col,
            writer_rest,
            compute_kernel,
        ],
        semaphores=semaphore_descriptors,
        cbs=[
            a_cb_descriptor,
            b_cb_descriptor,
            out_cb_descriptor,
            intermediate_cb_descriptor,
        ],
    )

    print("Launching generic_op...")
    output = ttnn.generic_op([a_tensor, b_tensor, output_tensor], program_descriptor)
    print("Completed generic_op.")
    metal_output = ttnn.to_torch(output).to(torch.bfloat16)

    a_tensor_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_tensor_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    torch_output = torch.matmul(a_tensor_torch, b_tensor_torch)

    assert_with_ulp(torch_output, metal_output)

    ttnn.close_device(device)
