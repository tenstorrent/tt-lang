# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compilation utilities for middle-end tests.

Provides pass pipeline execution and C++ translation from TTL MLIR.
"""

import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ttmlir.ir import Module
from ttmlir.passmanager import PassManager
from ttmlir.passes import ttkernel_to_cpp_by_name, get_ttkernel_names


# Cache for compiled kernels to avoid redundant compilation.
_kernel_cache: Dict[str, str] = {}


@dataclass
class CompiledKernels:
    """Container for compiled kernel C++ sources."""

    reader_cpp: str
    reader_name: str
    compute_cpp: str
    compute_name: str
    writer_cpp: str
    writer_name: str

    def get_kernel_info(self) -> List[Tuple[str, str, str]]:
        """
        Return list of (name, thread_type, cpp_source) tuples.

        Thread types: "noc" for reader/writer, "compute" for compute.
        """
        return [
            (self.reader_name, "noc", self.reader_cpp),
            (self.compute_name, "compute", self.compute_cpp),
            (self.writer_name, "noc", self.writer_cpp),
        ]


def get_system_desc_path() -> str:
    """Get the system descriptor path, generating if needed."""
    path = os.environ.get("SYSTEM_DESC_PATH")
    if path and os.path.exists(path):
        return path

    try:
        from _ttmlir_runtime import runtime

        system_desc = runtime.get_current_system_desc()
        generated_path = "/tmp/ttlang_me2e_system.ttsys"
        system_desc.store(generated_path)
        return generated_path
    except (ImportError, Exception) as e:
        raise RuntimeError(f"Cannot get system descriptor: {e}")


def compile_ttl_module(
    module: Module, system_desc_path: Optional[str] = None
) -> Module:
    """
    Run the TTL-to-TTKernel pass pipeline on the module.

    Args:
        module: TTL MLIR module to compile.
        system_desc_path: Path to system descriptor (auto-detected if not provided).

    Returns:
        Compiled module with TTKernel/EmitC ops.
    """
    if system_desc_path is None:
        system_desc_path = get_system_desc_path()

    # Build the pass pipeline.
    # This follows the pipeline from compute_with_data_movement.mlir and d2m_api.py.
    pipeline_passes = [
        # Register device.
        f"ttcore-register-device{{system-desc-path={system_desc_path}}}",
        # TTL to compute conversion.
        "func.func(convert-ttl-to-compute)",
        "func.func(ttl-tile-and-assign-dst)",
        "func.func(ttl-insert-tile-regs-sync)",
        "func.func(ttl-lower-to-loops)",
        "func.func(ttl-annotate-cb-associations)",
        # TTL to TTKernel conversion.
        "convert-ttl-to-ttkernel",
        # Cleanup.
        "canonicalize",
        "cse",
        # Lower affine and convert to EmitC.
        "lower-affine",
        "convert-ttkernel-to-emitc",
        "canonicalize",
    ]

    pipeline_str = f"builtin.module({','.join(pipeline_passes)})"

    pm = PassManager.parse(pipeline_str)
    pm.enable_verifier(True)

    # Enable verbose output if requested.
    if os.environ.get("TTLANG_VERBOSE_PASSES"):
        module.context.enable_multithreading(False)
        pm.enable_ir_printing(
            print_after_all=True,
            print_before_all=True,
            print_after_failure=True,
        )

    pm.run(module.operation)

    return module


def translate_to_cpp(module: Module) -> CompiledKernels:
    """
    Translate compiled TTKernel module to C++ source strings.

    Args:
        module: Compiled module (after pass pipeline).

    Returns:
        CompiledKernels containing C++ source for each kernel.
    """
    # Get kernel names and types.
    kernel_info = get_ttkernel_names(module)

    if len(kernel_info) != 3:
        raise ValueError(
            f"Expected 3 kernels (reader, compute, writer), got {len(kernel_info)}: "
            f"{[name for name, _ in kernel_info]}"
        )

    # Sort kernels by type: reader (noc), compute, writer (noc).
    # The order in kernel_info should match definition order in module.
    reader_name = None
    compute_name = None
    writer_name = None

    for name, thread_type in kernel_info:
        if thread_type == "compute":
            compute_name = name
        elif thread_type == "noc":
            if reader_name is None:
                reader_name = name
            else:
                writer_name = name

    if not all([reader_name, compute_name, writer_name]):
        raise ValueError(f"Could not identify all kernel types from: {kernel_info}")

    # Translate each kernel to C++.
    reader_cpp = ttkernel_to_cpp_by_name(module, reader_name)
    compute_cpp = ttkernel_to_cpp_by_name(module, compute_name)
    writer_cpp = ttkernel_to_cpp_by_name(module, writer_name)

    return CompiledKernels(
        reader_cpp=reader_cpp,
        reader_name=reader_name,
        compute_cpp=compute_cpp,
        compute_name=compute_name,
        writer_cpp=writer_cpp,
        writer_name=writer_name,
    )


def compile_and_translate(
    module: Module,
    system_desc_path: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> CompiledKernels:
    """
    Compile TTL module and translate to C++.

    Combines compile_ttl_module and translate_to_cpp with optional caching.

    Args:
        module: TTL MLIR module.
        system_desc_path: Path to system descriptor.
        cache_key: Optional cache key for kernel reuse.

    Returns:
        CompiledKernels with C++ sources.
    """
    # Check cache.
    if cache_key and cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    # Compile.
    compiled_module = compile_ttl_module(module, system_desc_path)

    # Translate.
    kernels = translate_to_cpp(compiled_module)

    # Cache if key provided.
    if cache_key:
        _kernel_cache[cache_key] = kernels

    return kernels


def get_kernel_output_dir(
    op_name: str, config_name: str, build_dir: Optional[str] = None
) -> str:
    """
    Get the output directory for generated kernel sources.

    Directory structure: <build_dir>/test/middle_end/<op_name>/<config_name>/

    Args:
        op_name: Name of the operation (e.g., "add", "exp").
        config_name: Configuration string (e.g., "4x4_bfloat16_sb").
        build_dir: Base build directory. Defaults to ./build or BUILD_DIR env var.

    Returns:
        Path to the kernel output directory.
    """
    if build_dir is None:
        build_dir = os.environ.get("BUILD_DIR", "build")

    # Construct path: build/test/middle_end/<op>/<config>/
    output_dir = os.path.join(build_dir, "test", "middle_end", op_name, config_name)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def write_kernels(
    kernels: CompiledKernels,
    op_name: str,
    config_name: str,
    build_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Write kernel C++ sources to build directory.

    Directory structure: <build_dir>/test/middle_end/<op_name>/<config_name>/

    Args:
        kernels: CompiledKernels to write.
        op_name: Name of the operation (e.g., "add", "exp").
        config_name: Configuration string (e.g., "4x4_bfloat16_sb").
        build_dir: Base build directory. Defaults to ./build or BUILD_DIR env var.

    Returns:
        Dict mapping kernel name to file path.
    """
    output_dir = get_kernel_output_dir(op_name, config_name, build_dir)
    paths = {}

    for name, thread_type, cpp_source in kernels.get_kernel_info():
        path = os.path.join(output_dir, f"{name}.cpp")
        with open(path, "w") as f:
            f.write(cpp_source)
        paths[name] = path

    return paths


def clear_cache():
    """Clear the kernel compilation cache."""
    global _kernel_cache
    _kernel_cache = {}
