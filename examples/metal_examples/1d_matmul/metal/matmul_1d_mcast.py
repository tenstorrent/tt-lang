# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import pytest
import torch

from ttlang.utils.correctness import assert_with_ulp


@pytest.mark.parametrize("M,K,N", [(1280, 1024, 2080)])
def test_1d_mcast_matmul(M, K, N):
    device = ttnn.open_device(device_id=0)
    
    # Tile dimensions
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    
    # Grid configuration - use full Blackhole compute grid
    device_core_size = device.compute_with_storage_grid_size()
    num_cores_x = device_core_size.x  # 13 for Blackhole
    num_cores_y = device_core_size.y  # 10 for Blackhole
    print(f"Device grid: {num_cores_x}x{num_cores_y} = {num_cores_x * num_cores_y} cores")
    
    # Per-core work distribution
    per_core_M = Mt // num_cores_y  # 80 / 10 = 8 tiles per core
    per_core_N = Nt // num_cores_x  # 130 / 13 = 10 tiles per core
    
    assert Mt % num_cores_y == 0, f"Mt ({Mt}) must be divisible by num_cores_y ({num_cores_y})"
    assert Nt % num_cores_x == 0, f"Nt ({Nt}) must be divisible by num_cores_x ({num_cores_x})"
    
    # K-dimension blocking
    in0_block_w = 2  # K-tiles per block (32 / 16 = 2, for 16 K-blocks)
    assert Kt % in0_block_w == 0, f"Kt ({Kt}) must be divisible by in0_block_w ({in0_block_w})"
    num_blocks = Kt // in0_block_w  # Number of K-dimension blocks
    
    # Output subblocking (must satisfy out_subblock_h * out_subblock_w <= 8)
    out_subblock_h = 2  # Must divide per_core_M (4)
    out_subblock_w = 1  # Must divide per_core_N (5)
    assert per_core_M % out_subblock_h == 0
    assert per_core_N % out_subblock_w == 0
    assert out_subblock_h * out_subblock_w <= 8
    
    # Sharding configuration for input 0 (activations)
    # Each core holds per_core_M rows and full K columns
    shard_width_in_tiles = Kt  # Full K dimension (64 tiles)
    shard_height_in_tiles = per_core_M  # 8 tiles = 256 rows per core
    shard_shape = [shard_height_in_tiles * ttnn.TILE_SIZE, K]  # [256, 2048] per core
    
    print(f"per_core_M: {per_core_M}, per_core_N: {per_core_N}")
    print(f"in0_block_w: {in0_block_w}, num_blocks: {num_blocks}")
    print(f"out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}")
    print(f"shard_shape: {shard_shape}")
    
    # Core grid for computation
    all_cores = ttnn.CoreRangeSet([
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1)
        )
    ])
    
    # Allocate tensors
    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG
    
    # Input 0: Block-sharded activations in L1
    shard_spec = ttnn.ShardSpec(
        all_cores,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        shard_spec
    )
    
    # Create input tensors - first in DRAM, then convert to sharded
    a_tensor_dram = ttnn.rand(
        (M, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    
    # Convert to block-sharded in L1
    a_tensor = ttnn.to_memory_config(a_tensor_dram, sharded_memory_config)
    
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
    
    # Circular buffer configuration
    dtype_size = 2  # bfloat16
    cb_page_size = dtype_size * ttnn.TILE_SIZE * ttnn.TILE_SIZE
    
    # CB indices
    a_mcast_cb = 0  # in0 multicast buffer
    a_shard_cb = 2  # in0 shard buffer (globally allocated)
    b_cb = 1        # in1 weights
    out_cb = 16     # output
    intermediate_cb = 24  # L1 accumulation
    semaphore_cb = 6      # semaphore for mcast sync
    
    # CB format descriptors
    a_mcast_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=a_mcast_cb,
        data_format=ttnn.bfloat16,
        page_size=cb_page_size,
    )
    a_shard_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=a_shard_cb,
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
    
    # Buffer sizing
    mcast_input_buffering_depth = 2
    in0_block_tiles = per_core_M * in0_block_w
    in1_block_tiles = per_core_N * in0_block_w
    out_block_tiles = per_core_M * per_core_N
    
    # CB descriptors
    a_mcast_cb_descriptor = ttnn.CBDescriptor(
        total_size=mcast_input_buffering_depth * cb_page_size * in0_block_tiles,
        core_ranges=all_cores,
        format_descriptors=[a_mcast_cb_format],
    )
    
    a_shard_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * (per_core_M * shard_width_in_tiles),
        core_ranges=all_cores,
        format_descriptors=[a_shard_cb_format],
    )
    # Mark as globally allocated (points to sharded buffer)
    a_shard_cb_descriptor.set_globally_allocated_address(a_tensor.buffer_address())
    
    b_cb_descriptor = ttnn.CBDescriptor(
        total_size=mcast_input_buffering_depth * cb_page_size * in1_block_tiles,
        core_ranges=all_cores,
        format_descriptors=[b_cb_format],
    )
    
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * out_block_tiles,
        core_ranges=all_cores,
        format_descriptors=[out_cb_format],
    )
    
    intermediate_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_page_size * out_block_tiles,
        core_ranges=all_cores,
        format_descriptors=[intermediate_cb_format],
    )
    
    # Semaphores for multicast synchronization
    in0_mcast_sender_semaphore_id = 0
    in0_mcast_receiver_semaphore_id = 1
    
    in0_mcast_sender_semaphore = ttnn.SemaphoreDescriptor(
        address=in0_mcast_sender_semaphore_id,
        initial_value=0,
        core_ranges=all_cores,
    )
    
    in0_mcast_receiver_semaphore = ttnn.SemaphoreDescriptor(
        address=in0_mcast_receiver_semaphore_id,
        initial_value=0,
        core_ranges=all_cores,
    )
    
    # Compile-time args for reader kernel (in0 sender/receiver)
    in0_block_num_tiles = in0_block_tiles
    in0_block_size_bytes = in0_block_num_tiles * cb_page_size
    in0_last_ktile_w = in0_block_w  # No partial K blocks in this example
    num_blocks_w_dim = num_cores_x
    num_blocks_h_dim = num_cores_y
    in0_mcast_num_dests = num_cores_x * num_cores_y
    in0_mcast_num_cores = num_cores_x * num_cores_y
    transpose_mcast = 0  # Not transposing
    in0_block_h = per_core_M
    batch = 1
    fuse_op = 0  # No fused ops
    
    reader_in0_compile_time_args = [
        1,  # core_has_output_block_work
        1,  # core_in_in0_receiver_mcast_grid
        in0_block_num_tiles,
        in0_block_size_bytes,
        in0_last_ktile_w,
        num_blocks,
        num_blocks_w_dim,
        num_blocks_h_dim,
        in0_mcast_sender_semaphore_id,
        in0_mcast_receiver_semaphore_id,
        in0_mcast_num_dests,
        in0_mcast_num_cores,
        num_cores_x,
        num_cores_y,
        transpose_mcast,
        shard_width_in_tiles,
        shard_height_in_tiles,
        in0_block_w,
        in0_block_h,
        batch,
        fuse_op,
    ]
    
    # Compile-time args for writer kernel (in1 reader + output writer)
    in1_tensor_stride_w = 1
    in1_tensor_stride_h = Nt
    in1_tensor_next_block_stride = in0_block_w * Nt
    in1_tensor_next_w_dim_block_stride = per_core_N
    in1_block_w = per_core_N
    in1_block_h = in0_block_w
    in1_block_num_tiles = in1_block_tiles
    in1_mcast_sender_semaphore_id = 2  # Not used (SKIP_MCAST)
    in1_mcast_receiver_semaphore_id = 3  # Not used
    in1_mcast_num_dests = 0  # SKIP_MCAST
    in1_mcast_num_cores = 1
    KtNt = Kt * Nt
    bcast_B = 0
    batchB = 1
    sparsity_pagesize = 0
    out_tensor_stride_w = 1
    out_tensor_stride_h = Nt
    out_tensor_next_subblock_stride_w = out_subblock_w
    out_tensor_next_subblock_stride_h = out_subblock_h * Nt
    out_tensor_next_w_dim_block_stride = per_core_N
    out_tensor_next_h_dim_block_stride = per_core_M * Nt
    out_subblock_tile_count = out_subblock_h * out_subblock_w
    MtNt = Mt * Nt
    in3_tensor_stride_w = 0  # No bias
    fuse_op_all_gather = 0
    fuse_op_reduce_scatter = 0
    
    writer_compile_time_args = [
        in1_tensor_stride_w,
        in1_tensor_stride_h,
        in1_tensor_next_block_stride,
        in1_tensor_next_w_dim_block_stride,
        in1_block_w,
        in1_block_h,
        in1_block_num_tiles,
        num_blocks,
        num_blocks_w_dim,
        num_blocks_h_dim,
        in1_mcast_sender_semaphore_id,
        in1_mcast_receiver_semaphore_id,
        in1_mcast_num_dests,
        in1_mcast_num_cores,
        KtNt,
        batch,
        bcast_B,
        batchB,
        sparsity_pagesize,
        out_tensor_stride_w,
        out_tensor_stride_h,
        out_tensor_next_subblock_stride_w,
        out_tensor_next_subblock_stride_h,
        out_tensor_next_w_dim_block_stride,
        out_tensor_next_h_dim_block_stride,
        out_subblock_w,
        out_subblock_h,
        out_subblock_tile_count,
        MtNt,
        in3_tensor_stride_w,
        fuse_op_all_gather,
        fuse_op_reduce_scatter,
    ]
    writer_compile_time_args.extend(ttnn.TensorAccessorArgs(b_tensor).get_compile_time_args())
    writer_compile_time_args.extend([0] * 4)  # sparsity accessor (zeros)
    writer_compile_time_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_compile_time_args.extend([0] * 4)  # bias accessor (zeros)
    
    # Compute kernel compile-time args
    in0_num_subblocks = per_core_M // out_subblock_h
    in0_subblock_num_tiles = out_subblock_h * in0_block_w
    in1_num_subblocks = per_core_N // out_subblock_w
    in1_per_core_w = per_core_N
    out_num_blocks_x = 1  # Each core handles one output block
    out_num_blocks_y = 1
    out_subblock_num_tiles = out_subblock_h * out_subblock_w
    untilize_out = 0
    in0_transpose_tile = 0
    
    compute_compile_time_args = [
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        out_num_blocks_x,
        out_num_blocks_y,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        batch,
        out_block_tiles,
        untilize_out,
        0,  # get_batch_from_reader
        in0_transpose_tile,
    ]
    
    # Get NOC coordinates for all cores
    noc_coords_x = []
    noc_coords_y = []
    for x in range(num_cores_x):
        for y in range(num_cores_y):
            coord = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            if x == 0:  # Only add y coords once
                noc_coords_y.append(coord.y)
        noc_coords_x.append(coord.x)
    
    # Runtime args per core
    reader_rt_args = [[[] for _ in range(num_cores_y)] for _ in range(num_cores_x)]
    writer_rt_args = [[[] for _ in range(num_cores_y)] for _ in range(num_cores_x)]
    compute_rt_args = [[[] for _ in range(num_cores_y)] for _ in range(num_cores_x)]
    
    for core_x in range(num_cores_x):
        for core_y in range(num_cores_y):
            # Reader runtime args (in0 sender/receiver)
            sender_id = core_y * num_cores_x + core_x
            in0_mcast_dest_noc_start_x = noc_coords_x[0]
            in0_mcast_dest_noc_start_y = noc_coords_y[0]
            in0_mcast_dest_noc_end_x = noc_coords_x[-1]
            in0_mcast_dest_noc_end_y = noc_coords_y[-1]
            
            reader_rt_args[core_x][core_y] = [
                sender_id,
                in0_mcast_dest_noc_start_x,
                in0_mcast_dest_noc_start_y,
                in0_mcast_dest_noc_end_x,
                in0_mcast_dest_noc_end_y,
            ]
            reader_rt_args[core_x][core_y].extend(noc_coords_x)
            reader_rt_args[core_x][core_y].extend(noc_coords_y)
            
            # Writer runtime args (in1 reader + output writer)
            in1_tensor_start_tile_id = core_x * per_core_N
            out_tensor_start_tile_id = (core_x * per_core_N) + (core_y * per_core_M * Nt)
            
            # Padding args (no padding in this example)
            last_block_w = in0_block_w
            out_num_nonzero_subblocks_h = in0_num_subblocks
            out_last_subblock_h = out_subblock_h
            padded_block_tiles_h_skip = 0
            out_num_nonzero_subblocks_w = in1_num_subblocks
            out_last_num_nonzero_subblocks_w = out_num_nonzero_subblocks_w
            out_last_subblock_w = out_subblock_w
            padded_subblock_tiles_addr_skip = 0
            padded_block_tiles_w_skip = 0
            
            writer_rt_args[core_x][core_y] = [
                b_tensor.buffer_address(),
                in1_tensor_start_tile_id,
                0,  # in1_mcast_dest_noc_start_x (unused with SKIP_MCAST)
                0,  # in1_mcast_dest_noc_start_y
                0,  # in1_mcast_dest_noc_end_x
                0,  # in1_mcast_dest_noc_end_y
                0,  # sparsity_addr
                output_tensor.buffer_address(),
                out_tensor_start_tile_id,
                last_block_w,
                out_num_nonzero_subblocks_h,
                out_last_subblock_h,
                padded_block_tiles_h_skip,
                out_num_nonzero_subblocks_w,
                out_last_num_nonzero_subblocks_w,
                out_last_subblock_w,
                padded_subblock_tiles_addr_skip,
                padded_block_tiles_w_skip,
            ]
            
            # Compute has no runtime args
            compute_rt_args[core_x][core_y] = []
    
    # Compute config
    computeConfig = ttnn.ComputeConfigDescriptor()
    computeConfig.math_fidelity = ttnn.MathFidelity.HiFi4
    computeConfig.fp32_dest_acc_en = True
    computeConfig.math_approx_mode = False
    
    # Reader config (RISCV_1)
    readerConfig = ttnn.ReaderConfigDescriptor()
    readerConfig.processor = ttnn.DataMovementProcessor.RISCV_1
    readerConfig.noc = ttnn.NOC.NOC_1
    
    # Writer config (RISCV_0)
    writerConfig = ttnn.WriterConfigDescriptor()
    writerConfig.processor = ttnn.DataMovementProcessor.RISCV_0
    writerConfig.noc = ttnn.NOC.NOC_0
    writerConfig.defines = {"SKIP_MCAST": "1"}  # We skip mcast for in1
    
    # Kernel descriptors
    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul/metal/kernels/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=reader_in0_compile_time_args,
        runtime_args=reader_rt_args,
        config=readerConfig,
    )
    
    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul/metal/kernels/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args,
        config=writerConfig,
    )
    
    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/1d_matmul/metal/kernels/bmm_large_block_zm_fused_bias_activation.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=compute_compile_time_args,
        runtime_args=compute_rt_args,
        config=computeConfig,
    )
    
    # Program descriptor
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[
            reader_kernel_descriptor,
            writer_kernel_descriptor,
            compute_kernel_descriptor,
        ],
        semaphores=[
            in0_mcast_sender_semaphore,
            in0_mcast_receiver_semaphore,
        ],
        cbs=[
            a_mcast_cb_descriptor,
            b_cb_descriptor,
            a_shard_cb_descriptor,
            out_cb_descriptor,
            intermediate_cb_descriptor,
        ],
    )
    
    print("Launching generic_op...")
    output = ttnn.generic_op([a_tensor, b_tensor, output_tensor], program_descriptor)
    print("Completed generic_op.")
    
    # Verify correctness
    metal_output = ttnn.to_torch(output).to(torch.bfloat16)
    a_tensor_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_tensor_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    torch_output = torch.matmul(a_tensor_torch, b_tensor_torch)
    
    assert_with_ulp(torch_output, metal_output)
    
    ttnn.close_device(device)
