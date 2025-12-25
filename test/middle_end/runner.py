# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test runner for middle-end tests.

Executes compiled TTL kernels via ttnn.generic_op and validates results.
"""

from typing import List, Tuple, Optional
import os

import torch

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None

from .op_specs import AnyOpSpec
from .config_specs import TestConfig, BufferType
from .ttl_builder import build_ttl_module
from .compile_utils import compile_and_translate, write_kernels, CompiledKernels
from .test_utils import (
    TTLCompileException,
    TTLRuntimeException,
    TTLGoldenException,
    compare_tensors,
)


# Constants for CB configuration.
BFLOAT16_TILE_SIZE = 32 * 32 * 2  # 2KB per tile.
FLOAT32_TILE_SIZE = 32 * 32 * 4  # 4KB per tile.


def get_tile_size_bytes(dtype: torch.dtype) -> int:
    """Get the tile size in bytes for the given dtype."""
    if dtype == torch.bfloat16:
        return BFLOAT16_TILE_SIZE
    elif dtype == torch.float32:
        return FLOAT32_TILE_SIZE
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def torch_dtype_to_ttnn(dtype: torch.dtype):
    """Convert torch dtype to ttnn dtype."""
    if dtype == torch.bfloat16:
        return ttnn.bfloat16
    elif dtype == torch.float32:
        return ttnn.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def create_input_tensors(
    op: AnyOpSpec,
    config: TestConfig,
    device,
) -> Tuple[List[torch.Tensor], List]:
    """
    Create input tensors on device.

    Args:
        op: Operation spec (determines number of inputs and input_range).
        config: Test configuration.
        device: TTNN device.

    Returns:
        Tuple of (torch tensors, ttnn tensors on device).
    """
    shape = list(config.tensor_shape)
    dtype = config.dtype
    ttnn_dtype = torch_dtype_to_ttnn(dtype)

    torch_tensors = []
    device_tensors = []

    # Determine input range from op spec or use default.
    if op.input_range is not None:
        min_val, max_val = op.input_range
    else:
        # Default range [-1, 1].
        min_val, max_val = -1.0, 1.0

    for i in range(op.arity):
        # Create random input data in the specified range.
        t = torch.rand(shape, dtype=dtype) * (max_val - min_val) + min_val

        torch_tensors.append(t)

        # Convert to TTNN tensor on device.
        device_tensor = ttnn.from_torch(
            t,
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        device_tensors.append(device_tensor)

    return torch_tensors, device_tensors


def create_output_tensor(config: TestConfig, device):
    """Create an output tensor on device."""
    shape = list(config.tensor_shape)
    ttnn_dtype = torch_dtype_to_ttnn(config.dtype)

    # Allocate output tensor.
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn_dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    return output


def build_cb_descriptors(
    op: AnyOpSpec,
    config: TestConfig,
    core_ranges,
) -> List:
    """
    Build circular buffer descriptors.

    Args:
        op: Operation spec.
        config: Test configuration.
        core_ranges: TTNN CoreRangeSet.

    Returns:
        List of ttnn.CBDescriptor objects.
    """
    tile_size = get_tile_size_bytes(config.dtype)
    num_tiles = config.num_tiles
    ttnn_dtype = torch_dtype_to_ttnn(config.dtype)

    # Total CB size depends on buffer factor.
    cb_total_size = num_tiles * tile_size * config.buffer_factor

    descriptors = []

    # Input CBs (one per input).
    for i in range(op.arity):
        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=i,
            data_format=ttnn_dtype,
            page_size=tile_size,
        )
        cb_desc = ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_ranges,
            format_descriptors=[cb_format],
        )
        descriptors.append(cb_desc)

    # Output CB.
    output_cb_index = op.arity
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=output_cb_index,
        data_format=ttnn_dtype,
        page_size=tile_size,
    )
    out_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_ranges,
        format_descriptors=[out_cb_format],
    )
    descriptors.append(out_cb_desc)

    return descriptors


def build_kernel_descriptors(
    kernels: CompiledKernels,
    op: AnyOpSpec,
    config: TestConfig,
    core_ranges,
    input_tensors: List,
    output_tensor,
) -> List:
    """
    Build kernel descriptors for ttnn.generic_op.

    Args:
        kernels: Compiled kernel C++ sources.
        op: Operation specification.
        config: Test configuration.
        core_ranges: TTNN CoreRangeSet.
        input_tensors: List of input device tensors.
        output_tensor: Output device tensor.

    Returns:
        List of ttnn.KernelDescriptor objects.
    """
    # Write kernels to build directory.
    kernel_paths = write_kernels(kernels, op.name, str(config))

    # For single-core execution.
    num_tiles = config.num_tiles
    start_id = 0

    # Build reader runtime args.
    reader_rt_args = [t.buffer_address() for t in input_tensors]
    reader_rt_args.extend([num_tiles, start_id])

    # Build writer runtime args.
    writer_rt_args = [output_tensor.buffer_address(), num_tiles, start_id]

    # Build compute runtime args.
    compute_rt_args = [num_tiles]

    # Reader compile-time args (TensorAccessorArgs for each input).
    reader_ct_args = []
    for t in input_tensors:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    # Writer compile-time args.
    writer_ct_args = list(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )

    # Compute compile-time args (CB indices).
    compute_ct_args = list(range(len(input_tensors) + 1))  # CB indices 0, 1, ..., N

    descriptors = []

    # Reader kernel.
    reader_desc = ttnn.KernelDescriptor(
        kernel_source=kernel_paths[kernels.reader_name],
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=[[reader_rt_args]],  # [cores][core_ranges][args]
        config=ttnn.ReaderConfigDescriptor(),
    )
    descriptors.append(reader_desc)

    # Compute kernel.
    compute_desc = ttnn.KernelDescriptor(
        kernel_source=kernel_paths[kernels.compute_name],
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=compute_ct_args,
        runtime_args=[[compute_rt_args]],
        config=ttnn.ComputeConfigDescriptor(),
    )
    descriptors.append(compute_desc)

    # Writer kernel.
    writer_desc = ttnn.KernelDescriptor(
        kernel_source=kernel_paths[kernels.writer_name],
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=[[writer_rt_args]],
        config=ttnn.WriterConfigDescriptor(),
    )
    descriptors.append(writer_desc)

    return descriptors


def run_compute_test(
    op: AnyOpSpec,
    config: TestConfig,
    device,
    system_desc_path: Optional[str] = None,
    verbose: bool = False,
    dump_mlir: bool = False,
    pcc_threshold: float = 0.99,
    check_pcc: bool = False,
):
    """
    Execute a compute operation test.

    This is the main entry point for running a single test case.

    Args:
        op: Operation specification.
        config: Test configuration.
        device: TTNN device.
        system_desc_path: Path to system descriptor (auto-detected if None).
        verbose: Enable verbose output.
        dump_mlir: Save generated MLIR to build directory.
        pcc_threshold: Minimum PCC for pass (if check_pcc=True).
        check_pcc: Use PCC-based comparison instead of allclose.

    Raises:
        TTLCompileException: If compilation fails.
        TTLRuntimeException: If runtime execution fails.
        TTLGoldenException: If golden comparison fails.
    """
    if not TTNN_AVAILABLE:
        raise TTLRuntimeException("ttnn not available")

    if verbose:
        print(f"Running test: {op.name} with {config}")

    # 1. Create input tensors on device.
    torch_inputs, device_inputs = create_input_tensors(op, config, device)
    output_tensor = create_output_tensor(config, device)

    # 2. Compute golden result using torch.
    golden = op.golden(*torch_inputs)

    # 3. Build TTL module.
    try:
        module = build_ttl_module(op, config)
    except Exception as e:
        raise TTLCompileException(f"Failed to build TTL module: {e}") from e

    if verbose or dump_mlir:
        mlir_str = str(module)
        if verbose:
            print("Generated TTL module:")
            print(mlir_str)
        if dump_mlir:
            _dump_mlir(op.name, str(config), mlir_str)

    # 4. Compile and translate to C++.
    try:
        cache_key = f"{op.name}_{config}"
        kernels = compile_and_translate(module, system_desc_path, cache_key)
    except Exception as e:
        raise TTLCompileException(f"Failed to compile TTL module: {e}") from e

    if verbose:
        print(
            f"Compiled kernels: {kernels.reader_name}, "
            f"{kernels.compute_name}, {kernels.writer_name}"
        )

    # 5. Build program descriptor.
    # Single core for now.
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_ranges = ttnn.CoreRangeSet([core_range])

    cb_descriptors = build_cb_descriptors(op, config, core_ranges)
    kernel_descriptors = build_kernel_descriptors(
        kernels, op, config, core_ranges, device_inputs, output_tensor
    )

    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=[],
    )

    # 6. Execute.
    try:
        io_tensors = device_inputs + [output_tensor]
        result = ttnn.generic_op(io_tensors, program)
    except Exception as e:
        raise TTLRuntimeException(f"Runtime execution failed: {e}") from e

    # 7. Read back result.
    result_torch = ttnn.to_torch(result)

    # 8. Validate.
    comparison = compare_tensors(
        golden,
        result_torch,
        pcc_threshold=pcc_threshold,
        atol=1e-1,
        rtol=5e-2,
        check_pcc=check_pcc,
    )

    if not comparison.passed:
        raise TTLGoldenException(
            f"Test failed for {op.name} with {config}:\n"
            f"  {comparison.message}\n"
            f"  PCC: {comparison.pcc:.4f}\n"
            f"  Max abs diff: {comparison.max_abs_diff:.6f}\n"
            f"  Mean abs error: {comparison.mean_abs_error:.6f}\n"
            f"  Cosine sim: {comparison.cosine_sim:.4f}\n"
            f"  Result sample: {result_torch.flatten()[:5]}\n"
            f"  Golden sample: {golden.flatten()[:5]}"
        )

    if verbose:
        print(
            f"PASS: {op.name} with {config} "
            f"(pcc={comparison.pcc:.4f}, max_diff={comparison.max_abs_diff:.4f})"
        )


def _dump_mlir(op_name: str, config_name: str, mlir_str: str) -> str:
    """Save generated MLIR to build directory for debugging."""
    build_dir = os.environ.get("BUILD_DIR", "build")
    output_dir = os.path.join(build_dir, "test", "middle_end", op_name, config_name)
    os.makedirs(output_dir, exist_ok=True)

    mlir_path = os.path.join(output_dir, "module.mlir")
    with open(mlir_path, "w") as f:
        f.write(mlir_str)
    return mlir_path


def run_binary_test(op: AnyOpSpec, config: TestConfig, device):
    """Convenience wrapper for binary operation tests."""
    assert op.arity == 2, f"Expected binary op, got arity {op.arity}"
    run_compute_test(op, config, device)


def run_unary_test(op: AnyOpSpec, config: TestConfig, device):
    """Convenience wrapper for unary operation tests."""
    assert op.arity == 1, f"Expected unary op, got arity {op.arity}"
    run_compute_test(op, config, device)
