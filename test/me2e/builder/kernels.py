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
    Shim TensorAccessorArgs to use InterleavedAddrGenFast pattern.

    The compiler generates kernels using TensorAccessorArgs which require specific
    compile-time arg formats. This shim replaces TensorAccessorArgs with the simpler
    InterleavedAddrGenFast pattern that works with basic CB index + base address args.

    This is a temporary workaround until proper kernel generation is implemented.

    Args:
        source: C++ kernel source.
        kernel_name: Kernel function name.

    Returns:
        Shimmed source with InterleavedAddrGenFast pattern.
    """
    import re

    # Track which common_arg indices map to which tensors.
    # The generated code uses get_common_arg_val to get base addresses.
    # We need to replace TensorAccessorArgs with InterleavedAddrGenFast.

    # Replace TensorAccessorArgs instantiation and TensorAccessor construction
    # with InterleavedAddrGenFast pattern.
    #
    # Original pattern:
    #   auto tensor_accessor_args_N = TensorAccessorArgs<X, Y>();
    #   TensorAccessor vM = TensorAccessor(tensor_accessor_args_N, base_addr, tile_size);
    #
    # New pattern:
    #   // (remove TensorAccessorArgs line)
    #   InterleavedAddrGenFast<true> vM = {.bank_base_address = base_addr, .page_size = tile_size};

    # Step 1: Remove TensorAccessorArgs variable declarations.
    source = re.sub(
        r"auto\s+tensor_accessor_args_\w+\s*=\s*TensorAccessorArgs<[^>]+>\(\);\s*\n",
        "",
        source,
    )

    # Step 2: Replace TensorAccessor construction with InterleavedAddrGenFast.
    # Pattern: TensorAccessor vN = TensorAccessor(tensor_accessor_args_M, base_addr_var, tile_size_var);
    # Replace with: InterleavedAddrGenFast<true> vN = {.bank_base_address = (uint32_t)base_addr_var, .page_size = (uint32_t)tile_size_var};
    # The casts are needed because the generated code uses int32_t but InterleavedAddrGenFast expects uint32_t.
    source = re.sub(
        r"TensorAccessor\s+(\w+)\s*=\s*TensorAccessor\(tensor_accessor_args_\w+,\s*(\w+),\s*(\w+)\);",
        r"InterleavedAddrGenFast<true> \1 = {.bank_base_address = (uint32_t)\2, .page_size = (uint32_t)\3};",
        source,
    )

    # Fix CB index for writer: rewrite get_compile_time_arg_val(N) to (0) for single-output writer.
    if "writer" in kernel_name.lower():
        # The compiler may emit various CB indices (2, 16, etc.) for the output CB.
        # For a single-output writer with shimmed code, we want index 0.
        source = re.sub(
            r"get_compile_time_arg_val\(\d+\)",
            r"get_compile_time_arg_val(0)",
            source,
        )

        # CRITICAL FIX: The compiler doesn't emit cb_wait_front before get_read_ptr!
        # This is a bug in the ttl.cb_wait -> emitc conversion.
        # We need to add cb_wait_front(cb, 1) before get_read_ptr(cb).
        # Pattern: get_read_ptr(get_compile_time_arg_val(0))
        # Insert: cb_wait_front(get_compile_time_arg_val(0), 1); before it.
        source = re.sub(
            r"(int32_t \w+ = get_read_ptr\(get_compile_time_arg_val\(0\)\);)",
            r"cb_wait_front(get_compile_time_arg_val(0), 1);\n  \1",
            source,
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
    import os

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    debug_kernels = os.environ.get("TTLANG_DEBUG_KERNELS", "0") == "1"

    for kernel in noc_kernels + [compute_kernel]:
        # Apply TensorAccessorArgs shim for NOC kernels (reader/writer).
        # TODO: Remove this shim once compiler emits correct CTA offsets.
        source = kernel.source
        if kernel.thread_type == ThreadType.NOC:
            if debug_kernels:
                print(f"\n[DEBUG kernels] Original {kernel.name}.cpp:")
                print(source[:2000])
            source = _shim_tensor_accessor_args(source, kernel.name)
            if debug_kernels:
                print(f"\n[DEBUG kernels] After shim {kernel.name}.cpp:")
                print(source[:2000])

        path = output_dir / f"{kernel.name}.cpp"
        with open(path, "w") as f:
            f.write(source)
        paths[kernel.name] = str(path)

    return paths
