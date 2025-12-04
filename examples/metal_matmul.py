# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import pytest
import torch

from utils import assert_with_ulp


# works for single tile, not for multiple
# @pytest.mark.parametrize("M,K,N", [(640, 640, 640)])
@pytest.mark.parametrize("M,K,N", [(128, 128, 128), (256, 256, 256), (512, 512, 512)])
def test_singlecore_matmul(M, K, N):
    device = ttnn.open_device(device_id=0)
    # single core grid
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ttnn py hw constants for tile size?
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

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

    a_cb = 0
    b_cb = 1
    out_cb = 16
    cb_page_size = 2 * ttnn.TILE_SIZE * ttnn.TILE_SIZE
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
    cb_total_size = 2 * cb_page_size
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
    reader_compile_time_args.extend(
        ttnn.TensorAccessorArgs(b_tensor).get_compile_time_args()
    )
    writer_compile_time_args = ttnn.TensorAccessorArgs(
        output_tensor
    ).get_compile_time_args()
    compute_compile_time_args = [Mt, Kt, Nt]
    reader_rt_args = [a_tensor.buffer_address(), b_tensor.buffer_address(), Mt, Kt, Nt]
    writer_rt_args = [output_tensor.buffer_address(), Mt, Nt]

    # Compute config init can't handle options, set here
    computeConfig = ttnn.ComputeConfigDescriptor()
    computeConfig.math_fidelity = ttnn.MathFidelity.HiFi4
    computeConfig.fp32_dest_acc_en = True
    computeConfig.math_approx_mode = False

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
    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="examples/metal_examples/singlecore_matmul/mm_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        runtime_args=[[[]]],
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

    output = ttnn.generic_op([a_tensor, b_tensor, output_tensor], program_descriptor)
    metal_output = ttnn.to_torch(output).to(torch.bfloat16)

    a_tensor_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_tensor_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    torch_output = torch.matmul(a_tensor_torch, b_tensor_torch)

    assert_with_ulp(torch_output, metal_output)

    ttnn.close_device(device)


from ttl import Program, make_circular_buffer_like, copy


@ttl.kernel()
def singlecore_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    blk_size = ttnn.TILE_SIZE
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    a_cb = make_circular_buffer_like(a, shape=(blk_size, blk_size), buffer_factor=2)
    b_cb = make_circular_buffer_like(b, shape=(blk_size, blk_size), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(blk_size, blk_size), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        for _ in range(a.shape[0] // blk_size):  # m
            for _ in range(b.shape[1] // blk_size):  # n
                with out_cb.reserve() as out_blk:
                    for _ in range(a.shape[1] // blk_size):  # k
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        out_blk += a_blk @ b_blk
                        a_cb.pop()
                        b_cb.pop()

    @ttl.datamovement()
    def mm_reader():
        for m in range(0, a.shape[0], blk_size):
            for n in range(0, b.shape[1], blk_size):
                for k in range(0, a.shape[1], blk_size):
                    with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                        copy(a[m : (m + blk_size), k : (k + blk_size)], a_blk)
                        copy(b[k : (k + blk_size), n : (n + blk_size)], b_blk)

    @ttl.datamovement()
    def mm_writer():
        for m in range(0, a.shape[0], blk_size):
            for n in range(0, b.shape[1], blk_size):
                out_blk = out_cb.wait()
                copy(out_blk, out[m : (m + blk_size), n : (n + blk_size)])
                out_cb.pop()

    return Program(mm_compute, mm_reader, mm_writer)(a, b, out)


def test_singlecore_matmul():
    """Test singlecore matmul kernel."""
    device = ttnn.open_device(device_id=0)
    M, K, N = 256, 256, 256
    a = ttnn.rand((M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    c = ttnn.empty((M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    singlecore_matmul(a, b, c)

    golden = torch.matmul(
        ttnn.to_torch(a).to(torch.bfloat16), ttnn.to_torch(b).to(torch.bfloat16)
    )
    result = ttnn.to_torch(c).to(torch.bfloat16)
    assert_with_ulp(golden, result)

    ttnn.close_device(device)
