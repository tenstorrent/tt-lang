# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Kernel specification and execution for E2E tests.

Supports N data movement (NOC) threads and single compute thread.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple

from ttmlir.ir import Module
from ttmlir.passes import ttkernel_to_cpp_by_name, get_ttkernel_names


class ThreadType(Enum):
    """Kernel thread type."""

    NOC = "noc"  # Data movement thread
    COMPUTE = "compute"  # Compute thread


@dataclass
class KernelSpec:
    """Specification for a single kernel."""

    name: str
    thread_type: ThreadType
    source: str  # C++ source code
    compile_args: List[int] = field(default_factory=list)
    runtime_args: List[int] = field(default_factory=list)


def translate_module_to_kernels(
    module: Module,
) -> Tuple[List[KernelSpec], KernelSpec]:
    """
    Translate compiled TTKernel module to C++ kernel specs.

    Args:
        module: Compiled module (after pass pipeline).

    Returns:
        Tuple of (noc_kernels, compute_kernel).
        noc_kernels is a list of data movement kernels.
        compute_kernel is the single compute kernel.
    """
    # Get kernel names and types.
    kernel_info = get_ttkernel_names(module)

    noc_kernels = []
    compute_kernel = None

    for name, thread_type_str in kernel_info:
        cpp = ttkernel_to_cpp_by_name(module, name)

        if thread_type_str == "compute":
            thread_type = ThreadType.COMPUTE
        elif thread_type_str == "noc":
            thread_type = ThreadType.NOC
        else:
            raise ValueError(f"Unknown thread type: {thread_type_str}")

        spec = KernelSpec(name=name, thread_type=thread_type, source=cpp)

        if thread_type == ThreadType.COMPUTE:
            if compute_kernel is not None:
                raise ValueError("Multiple compute kernels found")
            compute_kernel = spec
        else:
            noc_kernels.append(spec)

    if compute_kernel is None:
        raise ValueError("No compute kernel found")

    return noc_kernels, compute_kernel


def _shim_tensor_accessor_args(source: str, kernel_name: str) -> str:
    """
    Shim TensorAccessorArgs and CB indices to use correct pattern.
    
    Temporary workaround until compiler emits correct CTA offsets and CB indices.
    
    1. Replaces TensorAccessorArgs<42, 0>() and TensorAccessorArgs<43, 0>()
       with TensorAccessorArgs<0>() and proper chaining for interleaved tensors.
    2. Rewrites get_compile_time_arg_val indices to be per-kernel (not global).
    
    Args:
        source: C++ kernel source.
        kernel_name: Kernel function name.
        
    Returns:
        Shimmed source with correct TensorAccessorArgs and CB index pattern.
    """
    import re
    
    # For interleaved tensors, TensorAccessorArgs should use single template param
    # (the CTA offset), not <row_stride, col_stride>.
    # Pattern: TensorAccessorArgs<42, 0>() or TensorAccessorArgs<43, 0>()
    
    # Binary reader: two tensors
    if "reader_binary" in kernel_name.lower():
        # First tensor: TensorAccessorArgs<42, 0>() -> TensorAccessorArgs<0>()
        source = re.sub(
            r'TensorAccessorArgs\s+(\w+)\s*=\s*TensorAccessorArgs<42,\s*0>\(\);',
            r'TensorAccessorArgs \1 = TensorAccessorArgs<0>();',
            source
        )
        # Second tensor: TensorAccessorArgs<43, 0>() -> TensorAccessorArgs<1>()
        # Note: For interleaved, offset is just tensor index since each uses 1 CTA slot
        source = re.sub(
            r'TensorAccessorArgs\s+(\w+)\s*=\s*TensorAccessorArgs<43,\s*0>\(\);',
            r'TensorAccessorArgs \1 = TensorAccessorArgs<1>();',
            source
        )
        # CB indices: no change needed (uses 0, 1)
    
    # Unary reader: single tensor
    elif "reader_unary" in kernel_name.lower():
        # Single tensor: TensorAccessorArgs<42, 0>() -> TensorAccessorArgs<0>()
        source = re.sub(
            r'TensorAccessorArgs\s+(\w+)\s*=\s*TensorAccessorArgs<42,\s*0>\(\);',
            r'TensorAccessorArgs \1 = TensorAccessorArgs<0>();',
            source
        )
        # CB index: no change needed (uses 0)
    
    # Writer: single tensor, but CB index needs adjustment
    elif "writer" in kernel_name.lower():
        # Single tensor: TensorAccessorArgs<42, 0>() -> TensorAccessorArgs<0>()
        source = re.sub(
            r'TensorAccessorArgs\s+(\w+)\s*=\s*TensorAccessorArgs<42,\s*0>\(\);',
            r'TensorAccessorArgs \1 = TensorAccessorArgs<0>();',
            source
        )
        # CB index: rewrite get_compile_time_arg_val(1) -> get_compile_time_arg_val(0) for unary
        # or get_compile_time_arg_val(2) -> get_compile_time_arg_val(0) for binary
        source = re.sub(
            r'get_compile_time_arg_val\(([12])\)',
            r'get_compile_time_arg_val(0)',
            source
        )
    
    return source


def write_kernels(
    noc_kernels: List[KernelSpec],
    compute_kernel: KernelSpec,
    output_dir: Path,
) -> dict:
    """
    Write kernel C++ sources to output directory.

    Args:
        noc_kernels: List of NOC kernel specs.
        compute_kernel: Compute kernel spec.
        output_dir: Directory to write kernels.

    Returns:
        Dict mapping kernel name to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for kernel in noc_kernels + [compute_kernel]:
        # Apply TensorAccessorArgs shim for NOC kernels (reader/writer).
        # TODO: Remove this shim once compiler emits correct CTA offsets.
        source = kernel.source
        if kernel.thread_type == ThreadType.NOC:
            source = _shim_tensor_accessor_args(source, kernel.name)
        
        path = output_dir / f"{kernel.name}.cpp"
        with open(path, "w") as f:
            f.write(source)
        paths[kernel.name] = str(path)

    return paths
