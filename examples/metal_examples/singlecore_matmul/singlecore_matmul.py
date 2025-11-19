import ttnn
import pytest
import torch

# works for single tile, not for multiple
@pytest.mark.parametrize("M,K,N", [(640, 640, 640)])
def test_singlecore_matmul(M, K, N): 
    # might be some l1 config stuff
    device = ttnn.open_device(device_id=0)

    # single core setup
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # find py hw constants in ttnn
    Mt = M // 32
    Kt = K // 32
    Nt = N // 32

    #   shape_a = [Mt, Kt, 32, 32]
    #   shape_b = [Kt, Nt, 32, 32]
    #   shape_out = [Mt, Nt, 32, 32]

    # don't get to specify the page size, but shoudl get correct size of 1 tile
    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG
    # allocate a, b and output tensors for matmul on device dram
    # these shapes are wrong, need to reflect tiled layout
    a_tensor = ttnn.rand((M, K), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=dram_memory_config)
    b_tensor = ttnn.rand((K, N), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=dram_memory_config)
    output_tensor = ttnn.allocate_tensor_on_device(
        (M, N),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        dram_memory_config,
    )

    cb_total_size = 2 * 2 * 1024  # tt::DataFormat::Float16_b hard coded to have size 2 * 1024
    cb_page_size = 2 * 1024

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
        core_ranges=core_grid,
        format_descriptors=[a_cb_format],
    )
    b_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[b_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[out_cb_format],
    )








    reader_compile_time_args = ttnn.TensorAccessorArgs(a_tensor).get_compile_time_args()
    reader_compile_time_args.extend(ttnn.TensorAccessorArgs(b_tensor).get_compile_time_args())
    writer_compile_time_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    compute_compile_time_args = [Mt, Kt, Nt]
    reader_rt_args = [a_tensor.buffer_address(), b_tensor.buffer_address(), Mt, Kt, Nt]
    writer_rt_args = [output_tensor.buffer_address(), Mt, Nt]


    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/singlecore_matmul/mm_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        runtime_args=[[reader_rt_args]],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/singlecore_matmul/mm_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        runtime_args=[[writer_rt_args]],
        config=ttnn.WriterConfigDescriptor(),
    )
    # Compute config init can't handle options, set here
    computeConfig = ttnn.ComputeConfigDescriptor()
    computeConfig.math_fidelity = ttnn.MathFidelity.HiFi4
    computeConfig.fp32_dest_acc_en = True
    computeConfig.math_approx_mode = False
    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/singlecore_matmul/mm_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        runtime_args=[[[]]],
        config=computeConfig,
    )





    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
        semaphores=[],
        cbs=[a_cb_descriptor, b_cb_descriptor, out_cb_descriptor],
    )

    output = ttnn.generic_op([a_tensor, b_tensor, output_tensor], program_descriptor)

    ttnnComputeConfig = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
    )
    g_output = ttnn.matmul(a_tensor, b_tensor, compute_kernel_config=ttnnComputeConfig, core_grid=ttnn.CoreGrid(y=1, x=1))

    torch_output = ttnn.to_torch(output)
    torch_g_output = ttnn.to_torch(g_output)

    #torch.set_printoptions(threshold=32*32) 

    print(f"torch_golden: {torch_g_output}")
    print(f"torch_output: {torch_output}")

    matching = torch.allclose(torch_g_output, torch_output)
    if not matching:
        diff = torch.abs(torch_g_output - torch_output)
        print(f"Max difference: {torch.max(diff)}")
    print(f"Tensors are matching: {matching}")


    ttnn.close_device(device)
    assert matching