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
        path = output_dir / f"{kernel.name}.cpp"
        with open(path, "w") as f:
            f.write(kernel.source)
        paths[kernel.name] = str(path)

    return paths
