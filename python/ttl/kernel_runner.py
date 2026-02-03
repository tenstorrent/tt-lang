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
import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import ttnn
except (ModuleNotFoundError, ImportError):
    ttnn = None

from .dtype_utils import tile_bytes_from_dtype, torch_dtype_to_ttnn_datatype


@dataclass
class PipeConnection:
    """Information about a pipe connection from the pipe graph.

    Attributes:
        srcX, srcY: Sender core coordinates.
        dstStartX, dstStartY: Receiver start coordinates.
        dstEndX, dstEndY: Receiver end coordinates.
        receiverCBIndex: CB index used by the receiver.
        runtimeArgSlot: Slot in runtime args for this pipe's receiver CB address.
    """

    srcX: int
    srcY: int
    dstStartX: int
    dstStartY: int
    dstEndX: int
    dstEndY: int
    receiverCBIndex: int
    runtimeArgSlot: int


def load_pipe_graph(json_path: Optional[str] = None) -> Optional[List[PipeConnection]]:
    """Load pipe graph from JSON file.

    Args:
        json_path: Path to the JSON file. If None, checks TTLANG_PIPE_GRAPH_JSON
            environment variable.

    Returns:
        List of PipeConnection objects, or None if no pipe graph available.
    """
    if json_path is None:
        json_path = os.environ.get("TTLANG_PIPE_GRAPH_JSON")

    if not json_path or not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        pipes = []
        for p in data.get("pipes", []):
            pipes.append(
                PipeConnection(
                    srcX=p["srcX"],
                    srcY=p["srcY"],
                    dstStartX=p["dstStartX"],
                    dstStartY=p["dstStartY"],
                    dstEndX=p["dstEndX"],
                    dstEndY=p["dstEndY"],
                    receiverCBIndex=p["receiverCBIndex"],
                    runtimeArgSlot=p["runtimeArgSlot"],
                )
            )
        return pipes
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Failed to load pipe graph from {json_path}: {e}")
        return None


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
    pipe_graph: Optional[List[PipeConnection]] = None,
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
        pipe_graph: Optional pipe graph for populating sender runtime args.

    Returns:
        List of ttnn.KernelDescriptor objects.
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    kernel_descriptors = []

    # CB indices are 0, 1, 2, ... for each CB (including intermediate CBs).
    cb_indices = list(range(num_cbs))

    # Debug: print pipe graph if available
    if pipe_graph and os.environ.get("TTLANG_DEBUG_PIPE_GRAPH"):
        print(f"=== Pipe Graph ({len(pipe_graph)} connections) ===")
        for p in pipe_graph:
            print(
                f"  Pipe: src=({p.srcX},{p.srcY}) -> dst=({p.dstStartX},{p.dstStartY})-({p.dstEndX},{p.dstEndY})"
            )
            print(
                f"    receiverCBIndex={p.receiverCBIndex}, runtimeArgSlot={p.runtimeArgSlot}"
            )

    for spec in kernel_specs:
        # runtime_args structure: [x][y][args_per_core].
        # Each core gets an empty arg list (we use my_x/my_y for indexing).
        runtime_args = [[[] for _ in range(grid_rows)] for _ in range(grid_cols)]

        # Populate runtime args for sender cores based on pipe graph.
        # For gather patterns, senders need the receiver's CB info.
        if pipe_graph and spec.thread_type != "compute":
            for p in pipe_graph:
                if 0 <= p.srcX < grid_cols and 0 <= p.srcY < grid_rows:
                    # Add receiver CB index to sender's runtime args.
                    # The kernel will use this to look up the CB address.
                    # Ensure runtime_args[x][y] has enough slots
                    while len(runtime_args[p.srcX][p.srcY]) <= p.runtimeArgSlot:
                        runtime_args[p.srcX][p.srcY].append(0)
                    runtime_args[p.srcX][p.srcY][p.runtimeArgSlot] = p.receiverCBIndex

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
            runtime_args=runtime_args,
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
            Each CB has shape, buffer_factor, tensor (for dtype), and _cb_index attributes.
        core_ranges: ttnn.CoreRangeSet for CB allocation.

    Returns:
        List of ttnn.CBDescriptor objects.
    """
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
        num_tiles = cb.shape[0] * cb.shape[1] * cb.buffer_factor
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
            buffer_factor, tensor (for dtype), and _cb_index attributes.
        core_ranges: ttnn.CoreRangeSet for kernel execution.
        program_hash: Hash for tt-metal program cache (not yet used).

    Returns:
        Result from ttnn.generic_op (typically None or output tensor).
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    # Build tensor accessor args.
    tensor_accessor_args = build_tensor_accessor_args(tensors)

    # Get grid dimensions from core_ranges.
    grid_size = core_ranges.bounding_box().grid_size()
    grid_cols = grid_size.x
    grid_rows = grid_size.y

    # Load pipe graph for sender runtime args (gather patterns).
    pipe_graph = load_pipe_graph()

    # Build kernel descriptors.
    kernel_descriptors = build_kernel_descriptors(
        kernel_specs=kernel_specs,
        tensors=tensors,
        tensor_accessor_args=tensor_accessor_args,
        core_ranges=core_ranges,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        num_cbs=len(cb_configs),
        pipe_graph=pipe_graph,
    )

    # Build CB descriptors.
    cb_descriptors = build_cb_descriptors(
        tensors=tensors,
        cb_configs=cb_configs,
        core_ranges=core_ranges,
    )

    # Build semaphore descriptors for pipe synchronization.
    # Pipe semaphores are indexed by srcX coordinate (see getPipeSemaphoreIndex).
    # We need to allocate enough semaphores for all pipe sources.
    semaphore_descriptors = []
    if pipe_graph:
        max_sem_idx = max(p.srcX for p in pipe_graph)
        # Allocate semaphores 0 through max_sem_idx, initialized to 0
        for sem_id in range(max_sem_idx + 1):
            semaphore_descriptors.append(
                ttnn.SemaphoreDescriptor(
                    sem_id, core_ranges=core_ranges, initial_value=0
                )
            )

    # Build and execute program.
    # TODO: Enable custom_program_hash once tt-metal exposes it in Python bindings.
    # See tt-metal/ttnn/cpp/ttnn-nanobind/program_descriptors.cpp - needs to add
    # custom_program_hash parameter to ProgramDescriptor binding.
    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=semaphore_descriptors,
        # custom_program_hash=program_hash,
    )

    return ttnn.generic_op(list(tensors), program)


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
    return "ttnn.bfloat16"  # default


def _config_type_to_str(thread_type: str, kernel_idx: int, noc_idx: int) -> str:
    """Convert thread type to config descriptor string."""
    if thread_type == "compute":
        return "ttnn.ComputeConfigDescriptor()"
    elif thread_type == "noc":
        if noc_idx == 0:
            return "ttnn.ReaderConfigDescriptor()"
        else:
            return "ttnn.WriterConfigDescriptor()"
    return "ttnn.ReaderConfigDescriptor()"


def emit_runner_source(
    kernel_specs: List[KernelSpec],
    cb_configs: List[Any],
    grid_cols: int,
    grid_rows: int,
    num_tensors: int,
    pipe_graph: Optional[List[PipeConnection]] = None,
    kernel_name: str = "kernel",
) -> str:
    """
    Emit Python source code for a standalone runner that invokes ttnn.generic_op.

    This generates a ready-to-use Python file with all the CB and kernel
    descriptor setup. Tensor-specific values (buffer addresses, accessor args)
    are marked with TODO comments for the user to fill in.

    Args:
        kernel_specs: List of kernel specifications.
        cb_configs: List of CircularBuffer objects for each CB.
        grid_cols: Number of grid columns (x dimension).
        grid_rows: Number of grid rows (y dimension).
        num_tensors: Number of tensors the kernel expects.
        pipe_graph: Optional pipe graph for semaphore setup.
        kernel_name: Name for the generated runner.

    Returns:
        Python source code as a string.
    """
    lines = []

    # Header
    lines.append("# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC")
    lines.append("# SPDX-License-Identifier: Apache-2.0")
    lines.append("")
    lines.append(f'"""Auto-generated runner for {kernel_name}."""')
    lines.append("")
    lines.append("import ttnn")
    lines.append("")

    # Grid dimensions
    lines.append(f"GRID_COLS = {grid_cols}")
    lines.append(f"GRID_ROWS = {grid_rows}")
    lines.append(f"NUM_TENSORS = {num_tensors}")
    lines.append("")

    # Kernel paths
    lines.append("KERNEL_PATHS = [")
    for spec in kernel_specs:
        lines.append(f'    ("{spec.path}", "{spec.thread_type}"),')
    lines.append("]")
    lines.append("")

    # Tensor indices for each kernel
    lines.append("# Tensor indices: which global tensor indices each kernel accesses")
    lines.append("# Used to build common_runtime_args = [tensors[i].buffer_address() for i in indices]")
    lines.append("KERNEL_TENSOR_INDICES = [")
    for i, spec in enumerate(kernel_specs):
        lines.append(f"    {spec.tensor_indices!r},  # {spec.thread_type}")
    lines.append("]")
    lines.append("")

    # CB configurations
    lines.append("# CB configurations: (shape, buffer_factor, dtype_str, page_size)")
    lines.append("CB_CONFIGS = [")
    for i, cb in enumerate(cb_configs):
        if cb is None:
            lines.append(f"    None,  # CB {i} - missing config")
            continue
        ref_tensor = cb.tensor
        if hasattr(ref_tensor, "dtype") and hasattr(ref_tensor.dtype, "name"):
            data_format = ref_tensor.dtype
        else:
            data_format = torch_dtype_to_ttnn_datatype(ref_tensor.dtype)
        page_size = tile_bytes_from_dtype(data_format)
        dtype_str = _dtype_to_ttnn_str(data_format)
        num_tiles = cb.shape[0] * cb.shape[1] * cb.buffer_factor
        total_size = num_tiles * page_size
        lines.append(
            f"    ({cb.shape!r}, {cb.buffer_factor}, {dtype_str}, {page_size}, {total_size}),  # CB {i}"
        )
    lines.append("]")
    lines.append("")

    # Semaphore info if pipes exist
    if pipe_graph:
        max_sem_idx = max(p.srcX for p in pipe_graph)
        lines.append(f"NUM_SEMAPHORES = {max_sem_idx + 1}  # For pipe synchronization")
    else:
        lines.append("NUM_SEMAPHORES = 0")
    lines.append("")

    # The run function
    lines.append("")
    lines.append("def run(tensors, device=None):")
    lines.append('    """')
    lines.append(f"    Run the {kernel_name} on device.")
    lines.append("")
    lines.append("    Args:")
    lines.append(f"        tensors: List of {num_tensors} ttnn.Tensor objects on device.")
    lines.append("        device: Optional device (inferred from tensors if not provided).")
    lines.append('    """')
    lines.append(f"    assert len(tensors) == {num_tensors}, f'Expected {num_tensors} tensors, got {{len(tensors)}}'")
    lines.append("")
    lines.append("    if device is None:")
    lines.append("        device = tensors[0].device()")
    lines.append("")

    # Core ranges
    lines.append("    # Build core ranges")
    lines.append("    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(")
    lines.append("        ttnn.CoreCoord(0, 0),")
    lines.append("        ttnn.CoreCoord(GRID_COLS - 1, GRID_ROWS - 1)")
    lines.append("    )])")
    lines.append("")

    # Build tensor accessor args
    lines.append("    # Build tensor accessor args (compile-time args for DM kernels)")
    lines.append("    tensor_accessor_args = []")
    lines.append("    for tensor in tensors:")
    lines.append("        tensor_accessor_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())")
    lines.append("")

    # CB descriptors
    lines.append("    # Build CB descriptors")
    lines.append("    cb_descriptors = []")
    lines.append("    for i, (shape, buffer_factor, dtype, page_size, total_size) in enumerate(CB_CONFIGS):")
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

    # Kernel descriptors
    lines.append("    # Build kernel descriptors")
    lines.append(f"    cb_indices = list(range({len(cb_configs)}))")
    lines.append("    kernel_descriptors = []")
    lines.append("    noc_idx = 0")
    lines.append("")
    lines.append("    for kernel_idx, (kernel_path, thread_type) in enumerate(KERNEL_PATHS):")
    lines.append("        tensor_indices = KERNEL_TENSOR_INDICES[kernel_idx]")
    lines.append("")
    lines.append("        # runtime_args: [x][y][args] - per-core args")
    lines.append("        runtime_args = [[[] for _ in range(GRID_ROWS)] for _ in range(GRID_COLS)]")
    lines.append("")
    lines.append("        # common_runtime_args: buffer addresses in tensor_indices order")
    lines.append("        common_runtime_args = [tensors[idx].buffer_address() for idx in tensor_indices]")
    lines.append("")
    lines.append("        # compile_time_args: CB indices (+ tensor accessor args for DM kernels)")
    lines.append("        if thread_type == 'compute':")
    lines.append("            compile_time_args = cb_indices")
    lines.append("            config = ttnn.ComputeConfigDescriptor()")
    lines.append("            # TODO: Configure compute options if needed:")
    lines.append("            # config.fp32_dest_acc_en = True")
    lines.append("            # config.math_fidelity = ttnn.MathFidelity.HiFi4")
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
    lines.append("            runtime_args=runtime_args,")
    lines.append("            common_runtime_args=common_runtime_args,")
    lines.append("            config=config,")
    lines.append("        )")
    lines.append("        kernel_descriptors.append(kernel_desc)")
    lines.append("")

    # Semaphore descriptors
    lines.append("    # Build semaphore descriptors")
    lines.append("    semaphore_descriptors = []")
    lines.append("    for sem_id in range(NUM_SEMAPHORES):")
    lines.append("        semaphore_descriptors.append(")
    lines.append("            ttnn.SemaphoreDescriptor(sem_id, core_ranges=core_ranges, initial_value=0)")
    lines.append("        )")
    lines.append("")

    # Program and generic_op
    lines.append("    # Build program and execute")
    lines.append("    program = ttnn.ProgramDescriptor(")
    lines.append("        kernels=kernel_descriptors,")
    lines.append("        cbs=cb_descriptors,")
    lines.append("        semaphores=semaphore_descriptors,")
    lines.append("    )")
    lines.append("")
    lines.append("    return ttnn.generic_op(list(tensors), program)")
    lines.append("")

    # Main block for standalone usage
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    # Example usage - fill in your tensor setup")
    lines.append("    #")
    lines.append("    # device = ttnn.open_device(device_id=0)")
    lines.append("    #")
    lines.append(f"    # # Create {num_tensors} tensors")
    lines.append("    # tensors = [")
    lines.append("    #     ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16,")
    lines.append("    #                     layout=ttnn.TILE_LAYOUT, device=device,")
    lines.append("    #                     memory_config=ttnn.DRAM_MEMORY_CONFIG),")
    lines.append("    #     # ... more tensors ...")
    lines.append("    # ]")
    lines.append("    #")
    lines.append("    # result = run(tensors)")
    lines.append("    #")
    lines.append("    # ttnn.close_device(device)")
    lines.append("    #")
    lines.append('    print("Runner generated. See comments above for usage.")')
    lines.append("")

    return "\n".join(lines)


def emit_runner_file(
    kernel_specs: List[KernelSpec],
    cb_configs: List[Any],
    grid_cols: int,
    grid_rows: int,
    num_tensors: int,
    output_path: str,
    pipe_graph: Optional[List[PipeConnection]] = None,
    kernel_name: str = "kernel",
) -> str:
    """
    Emit a Python runner file for the compiled kernel.

    Args:
        kernel_specs: List of kernel specifications.
        cb_configs: List of CircularBuffer objects for each CB.
        grid_cols: Number of grid columns.
        grid_rows: Number of grid rows.
        num_tensors: Number of tensors the kernel expects.
        output_path: Path to write the runner file.
        pipe_graph: Optional pipe graph for semaphore setup.
        kernel_name: Name for the generated runner.

    Returns:
        The output path.
    """
    source = emit_runner_source(
        kernel_specs=kernel_specs,
        cb_configs=cb_configs,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        num_tensors=num_tensors,
        pipe_graph=pipe_graph,
        kernel_name=kernel_name,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(source)

    print(f"Runner written to {output_path}")
    return output_path


__all__ = [
    "KernelSpec",
    "PipeConnection",
    "load_pipe_graph",
    "build_tensor_accessor_args",
    "build_kernel_descriptors",
    "build_cb_descriptors",
    "run_kernel_on_device",
    "emit_runner_source",
    "emit_runner_file",
]
