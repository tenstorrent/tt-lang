# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared kernel execution logic for tt-lang.

Provides functions for building kernel descriptors, CB descriptors, and
executing kernels on device via ttnn.generic_op. Used by both the Python
DSL (CompiledTTNNKernel) and ME2E tests.

This module provides a single reusable implementation of kernel argument
building and execution.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

ttnn = None  # Lazy-loaded via _ensure_ttnn()


def _ensure_ttnn():
    """Lazy import of ttnn."""
    global ttnn
    if ttnn is not None:
        return ttnn
    try:
        import ttnn as _ttnn

        ttnn = _ttnn
    except (ModuleNotFoundError, ImportError):
        pass
    return ttnn


from .dtype_utils import tile_bytes_from_dtype, torch_dtype_to_ttnn_datatype


@dataclass
class KernelSpec:
    """Specification for a single kernel to execute.

    Attributes:
        path: Path to the kernel C++ source file.
        thread_type: Type of kernel ("compute", "noc", or "ethernet").
        tensor_indices: List of global tensor indices this kernel accesses.
            For DM kernels, these determine which buffer addresses go in
            common_runtime_args, in order.
        config: Kernel config descriptor (ComputeConfigDescriptor,
            ReaderConfigDescriptor, WriterConfigDescriptor, or EthernetConfigDescriptor).
    """

    path: str
    thread_type: str
    tensor_indices: List[int]
    config: Any


def build_tensor_accessor_args(tensors: List[Any]) -> List[int]:
    """
    Build compile-time args for tensor accessors.

    Args:
        tensors: List of ttnn.Tensor objects on device.

    Returns:
        List of compile-time args (flattened TensorAccessorArgs for all tensors).
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    args = []
    for tensor in tensors:
        tensor_args = ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
        args.extend(tensor_args)
    return args


def build_kernel_descriptors(
    kernel_specs: List[KernelSpec],
    tensors: List[Any],
    tensor_accessor_args: List[int],
    core_ranges: Any,
    grid_cols: int,
    grid_rows: int,
    num_cbs: int,
) -> List[Any]:
    """
    Build kernel descriptors for ttnn.generic_op.

    Args:
        kernel_specs: List of kernel specifications.
        tensors: List of ttnn.Tensor objects. Position in this list determines
            the global tensor index. Individual kernels access subsets via
            tensor_indices in each KernelSpec.
        tensor_accessor_args: Flattened compile-time args from all tensors.
        core_ranges: ttnn.CoreRangeSet for kernel execution.
        grid_cols: Number of grid columns (x dimension).
        grid_rows: Number of grid rows (y dimension).
        num_cbs: Total number of circular buffers (including intermediate CBs).

    Returns:
        List of ttnn.KernelDescriptor objects.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    kernel_descriptors = []

    # CB indices are 0, 1, 2, ... for each CB (including intermediate CBs).
    cb_indices = list(range(num_cbs))

    for spec in kernel_specs:
        # Build common_runtime_args using tensor_indices.
        # C++ indexes by function-local position, we provide addresses in that order.
        common_runtime_args = [
            tensors[idx].buffer_address() for idx in spec.tensor_indices
        ]

        # Compute kernels only need CB indices.
        # DM kernels need CB indices + TensorAccessorArgs config.
        if spec.thread_type == "compute":
            kernel_compile_time_args = cb_indices
        else:
            kernel_compile_time_args = cb_indices + list(tensor_accessor_args)

        kernel_desc = ttnn.KernelDescriptor(
            kernel_source=spec.path,
            core_ranges=core_ranges,
            compile_time_args=kernel_compile_time_args,
            common_runtime_args=common_runtime_args,
            config=spec.config,
        )
        kernel_descriptors.append(kernel_desc)

    return kernel_descriptors


def build_cb_descriptors(
    tensors: List[Any],
    cb_configs: List[Any],
    core_ranges: Any,
) -> List[Any]:
    """
    Build circular buffer descriptors for ttnn.generic_op.

    Args:
        tensors: List of ttnn.Tensor objects. Each tensor's position (0, 1, 2, ...)
            corresponds to its CB index. For intermediate CBs (not backed by
            input/output tensors), pass None in the corresponding position.
        cb_configs: List of CircularBuffer objects for each CB, indexed by CB index.
            Each CB has shape, block_count, tensor (for dtype), and _cb_index attributes.
        core_ranges: ttnn.CoreRangeSet for CB allocation.

    Returns:
        List of ttnn.CBDescriptor objects.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    cb_descriptors = []
    for i, cb in enumerate(cb_configs):
        if cb is None:
            raise ValueError(
                f"Missing CB config for index {i}. "
                f"All CB indices must have associated CircularBuffer configurations."
            )

        # Get dtype from CB's reference tensor.
        ref_tensor = cb.tensor
        if hasattr(ref_tensor, "dtype") and hasattr(ref_tensor.dtype, "name"):
            data_format = ref_tensor.dtype
        else:
            data_format = torch_dtype_to_ttnn_datatype(ref_tensor.dtype)

        page_size = tile_bytes_from_dtype(data_format)
        num_tiles = cb.shape[0] * cb.shape[1] * cb.block_count
        total_size = num_tiles * page_size

        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=i,
            data_format=data_format,
            page_size=page_size,
        )
        cb_desc = ttnn.CBDescriptor(
            total_size=total_size,
            core_ranges=core_ranges,
            format_descriptors=[cb_format],
        )
        cb_descriptors.append(cb_desc)

    return cb_descriptors


def run_kernel_on_device(
    kernel_specs: List[KernelSpec],
    tensors: List[Any],
    cb_configs: List[Any],
    core_ranges: Any,
    program_hash: int = None,
) -> Any:
    """
    Execute kernels on device using ttnn.generic_op.

    This is the main entry point for kernel execution. It builds all
    descriptors and runs the program.

    Args:
        kernel_specs: List of kernel specifications (path, thread_type, tensor_indices, config).
        tensors: List of ttnn.Tensor objects. Position in this list determines the
            global tensor index. Individual kernels access subsets via tensor_indices
            in each KernelSpec.
        cb_configs: List of CircularBuffer objects for each CB, indexed by CB index.
            Includes both tensor-backed CBs and intermediate CBs. Each CB has shape,
            block_count, tensor (for dtype), and _cb_index attributes.
        core_ranges: ttnn.CoreRangeSet for kernel execution.
        program_hash: Hash for tt-metal program cache (not yet used).

    Returns:
        Result from ttnn.generic_op (typically None or output tensor).
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    # Build tensor accessor args.
    tensor_accessor_args = build_tensor_accessor_args(tensors)

    # Get grid dimensions from core_ranges.
    grid_size = core_ranges.bounding_box().grid_size()
    grid_cols = grid_size.x
    grid_rows = grid_size.y

    # Build kernel descriptors.
    kernel_descriptors = build_kernel_descriptors(
        kernel_specs=kernel_specs,
        tensors=tensors,
        tensor_accessor_args=tensor_accessor_args,
        core_ranges=core_ranges,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        num_cbs=len(cb_configs),
    )

    # Build CB descriptors.
    cb_descriptors = build_cb_descriptors(
        tensors=tensors,
        cb_configs=cb_configs,
        core_ranges=core_ranges,
    )

    # Build and execute program.
    # TODO: Enable custom_program_hash once tt-metal exposes it in Python bindings.
    # See tt-metal/ttnn/cpp/ttnn-nanobind/program_descriptors.cpp - needs to add
    # custom_program_hash parameter to ProgramDescriptor binding.
    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=[],
        # custom_program_hash=program_hash,
    )

    # ttnn.generic_op requires io_tensors to contain at least one input
    # and one output (size >= 2).  Output-only kernels (e.g. fill with no
    # input tensor) have only the output tensor; duplicate it so the runtime
    # sees [out, out].  The first copy acts as a dummy input that no kernel
    # thread actually reads.
    # TODO: Remove this workaround if ttnn.generic_op relaxes the >= 2
    # tensor requirement
    io_tensors = list(tensors)
    if not io_tensors:
        raise ValueError("kernel must have at least one output tensor")
    if len(io_tensors) < 2:
        io_tensors = [io_tensors[-1]] + io_tensors  # Duplicate output tensor as input

    return ttnn.generic_op(io_tensors, program)


def _dtype_to_ttnn_str(data_format) -> str:
    """Convert a data format to ttnn.dtype string for code emission."""
    dtype_str = str(data_format)
    if "bfloat16" in dtype_str.lower():
        return "ttnn.bfloat16"
    elif "float32" in dtype_str.lower():
        return "ttnn.float32"
    elif "float16" in dtype_str.lower():
        return "ttnn.float16"
    elif "uint32" in dtype_str.lower():
        return "ttnn.uint32"
    elif "uint16" in dtype_str.lower():
        return "ttnn.uint16"
    elif "int32" in dtype_str.lower():
        return "ttnn.int32"
    return "ttnn.bfloat16"


def emit_runner_source(
    kernel_specs: List[KernelSpec],
    cb_configs: List[Any],
    grid_cols: int,
    grid_rows: int,
    num_tensors: int,
    kernel_name: str = "kernel",
) -> str:
    """
    Emit Python source code for a standalone runner that invokes ttnn.generic_op.

    Generates a ready-to-use Python file with all the CB and kernel
    descriptor setup. Tensor-specific values (buffer addresses, accessor args)
    are marked with TODO comments for the user to fill in.
    """
    lines = []

    lines.append("# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC")
    lines.append("# SPDX-License-Identifier: Apache-2.0")
    lines.append("")
    lines.append(f'"""Auto-generated runner for {kernel_name}."""')
    lines.append("")
    lines.append("import ttnn")
    lines.append("")

    lines.append(f"GRID_COLS = {grid_cols}")
    lines.append(f"GRID_ROWS = {grid_rows}")
    lines.append(f"NUM_TENSORS = {num_tensors}")
    lines.append("")

    lines.append("KERNEL_PATHS = [")
    for spec in kernel_specs:
        lines.append(f'    ("{spec.path}", "{spec.thread_type}"),')
    lines.append("]")
    lines.append("")

    lines.append("KERNEL_TENSOR_INDICES = [")
    for spec in kernel_specs:
        lines.append(f"    {spec.tensor_indices!r},  # {spec.thread_type}")
    lines.append("]")
    lines.append("")

    lines.append("CB_CONFIGS = [")
    for i, cb in enumerate(cb_configs):
        if cb is None:
            lines.append(f"    None,  # CB {i}")
            continue
        ref_tensor = cb.tensor
        if hasattr(ref_tensor, "dtype") and hasattr(ref_tensor.dtype, "name"):
            data_format = ref_tensor.dtype
        else:
            data_format = torch_dtype_to_ttnn_datatype(ref_tensor.dtype)
        page_size = tile_bytes_from_dtype(data_format)
        dtype_str = _dtype_to_ttnn_str(data_format)
        num_tiles = cb.shape[0] * cb.shape[1] * cb.block_count
        total_size = num_tiles * page_size
        lines.append(
            f"    ({cb.shape!r}, {cb.block_count}, {dtype_str}, {page_size}, {total_size}),  # CB {i}"
        )
    lines.append("]")
    lines.append("")

    lines.append("")
    lines.append("def run(tensors, device=None):")
    lines.append(f'    """Run the {kernel_name} on device."""')
    lines.append(
        f"    assert len(tensors) == {num_tensors}, f'Expected {num_tensors} tensors, got {{len(tensors)}}'"
    )
    lines.append("")
    lines.append("    if device is None:")
    lines.append("        device = tensors[0].device()")
    lines.append("")

    lines.append("    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(")
    lines.append("        ttnn.CoreCoord(0, 0),")
    lines.append("        ttnn.CoreCoord(GRID_COLS - 1, GRID_ROWS - 1)")
    lines.append("    )])")
    lines.append("")

    lines.append("    tensor_accessor_args = []")
    lines.append("    for tensor in tensors:")
    lines.append(
        "        tensor_accessor_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())"
    )
    lines.append("")

    lines.append("    cb_descriptors = []")
    lines.append(
        "    for i, (shape, block_count, dtype, page_size, total_size) in enumerate(CB_CONFIGS):"
    )
    lines.append("        cb_format = ttnn.CBFormatDescriptor(")
    lines.append("            buffer_index=i,")
    lines.append("            data_format=dtype,")
    lines.append("            page_size=page_size,")
    lines.append("        )")
    lines.append("        cb_desc = ttnn.CBDescriptor(")
    lines.append("            total_size=total_size,")
    lines.append("            core_ranges=core_ranges,")
    lines.append("            format_descriptors=[cb_format],")
    lines.append("        )")
    lines.append("        cb_descriptors.append(cb_desc)")
    lines.append("")

    lines.append(f"    cb_indices = list(range({len(cb_configs)}))")
    lines.append("    kernel_descriptors = []")
    lines.append("    noc_idx = 0")
    lines.append("")
    lines.append(
        "    for kernel_idx, (kernel_path, thread_type) in enumerate(KERNEL_PATHS):"
    )
    lines.append("        tensor_indices = KERNEL_TENSOR_INDICES[kernel_idx]")
    lines.append(
        "        common_runtime_args = [tensors[idx].buffer_address() for idx in tensor_indices]"
    )
    lines.append("")
    lines.append("        if thread_type == 'compute':")
    lines.append("            compile_time_args = cb_indices")
    lines.append("            config = ttnn.ComputeConfigDescriptor()")
    lines.append("        else:")
    lines.append("            compile_time_args = cb_indices + tensor_accessor_args")
    lines.append("            if noc_idx == 0:")
    lines.append("                config = ttnn.ReaderConfigDescriptor()")
    lines.append("            else:")
    lines.append("                config = ttnn.WriterConfigDescriptor()")
    lines.append("            noc_idx += 1")
    lines.append("")
    lines.append("        kernel_desc = ttnn.KernelDescriptor(")
    lines.append("            kernel_source=kernel_path,")
    lines.append("            core_ranges=core_ranges,")
    lines.append("            compile_time_args=compile_time_args,")
    lines.append("            common_runtime_args=common_runtime_args,")
    lines.append("            config=config,")
    lines.append("        )")
    lines.append("        kernel_descriptors.append(kernel_desc)")
    lines.append("")

    lines.append("    program = ttnn.ProgramDescriptor(")
    lines.append("        kernels=kernel_descriptors,")
    lines.append("        cbs=cb_descriptors,")
    lines.append("        semaphores=[],")
    lines.append("    )")
    lines.append("")
    lines.append("    return ttnn.generic_op(list(tensors), program)")
    lines.append("")

    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append('    print("Runner generated. See run() function for usage.")')
    lines.append("")

    return "\n".join(lines)


def emit_runner_file(
    kernel_specs: List[KernelSpec],
    cb_configs: List[Any],
    grid_cols: int,
    grid_rows: int,
    num_tensors: int,
    output_path: str,
    kernel_name: str = "kernel",
) -> str:
    """
    Emit a Python runner file for the compiled kernel.

    Returns the output path.
    """
    import os

    source = emit_runner_source(
        kernel_specs=kernel_specs,
        cb_configs=cb_configs,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        num_tensors=num_tensors,
        kernel_name=kernel_name,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(source)

    print(f"Runner written to {output_path}")
    return output_path


__all__ = [
    "KernelSpec",
    "build_tensor_accessor_args",
    "build_kernel_descriptors",
    "build_cb_descriptors",
    "run_kernel_on_device",
    "emit_runner_source",
    "emit_runner_file",
]
