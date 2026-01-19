# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Kernel specification and execution for E2E tests.

Supports N data movement (NOC) threads and single compute thread.
Extracts tensor indices from compiled MLIR for proper argument building.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple

from ttmlir.ir import Module, ArrayAttr, IntegerAttr
from ttmlir.passes import ttkernel_to_cpp_by_name, get_ttkernel_names
from ttmlir.dialects import func


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
    tensor_indices: List[int] = field(default_factory=list)  # Global tensor indices
    compile_args: List[int] = field(default_factory=list)
    runtime_args: List[int] = field(default_factory=list)


def _get_kernel_tensor_indices(module: Module, kernel_name: str) -> List[int]:
    """
    Extract tensor indices from a kernel function's ttl.crta_indices attribute.

    The ttl.crta_indices attribute on each function specifies which global
    tensor indices that kernel accesses for building common_runtime_args.

    Args:
        module: Compiled module containing the kernel function.
        kernel_name: Name of the kernel function.

    Returns:
        List of global tensor indices accessed by the kernel.
    """
    # Find the function in the module.
    for op in module.body.operations:
        if isinstance(op, func.FuncOp) and op.name.value == kernel_name:
            # Check for ttl.crta_indices attribute.
            if "ttl.crta_indices" in op.attributes:
                crta_attr = op.attributes["ttl.crta_indices"]
                if isinstance(crta_attr, ArrayAttr):
                    return [int(IntegerAttr(idx).value) for idx in crta_attr]
            return []
    return []


def translate_module_to_kernels(
    module: Module,
) -> Tuple[List[KernelSpec], KernelSpec]:
    """
    Translate compiled TTKernel module to C++ kernel specs.

    Extracts tensor indices from ttl.crta_indices attributes for proper
    argument building.
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

        # Extract tensor indices from MLIR attributes.
        tensor_indices = _get_kernel_tensor_indices(module, name)

        spec = KernelSpec(
            name=name,
            thread_type=thread_type,
            source=cpp,
            tensor_indices=tensor_indices,
        )

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
    Write kernel C++ sources and metadata to output directory.

    Clears the output directory first to avoid stale files, then writes
    kernel_metadata.json with tensor_indices for each kernel.

    Args:
        noc_kernels: List of NOC kernel specs.
        compute_kernel: Compute kernel spec.
        output_dir: Directory to write kernels.

    Returns:
        Dict mapping kernel name to file path.
    """
    import json
    import shutil

    # Remove existing directory to avoid stale metadata/kernels.
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    metadata = {}

    for kernel in noc_kernels + [compute_kernel]:
        path = output_dir / f"{kernel.name}.cpp"
        with open(path, "w") as f:
            f.write(kernel.source)
        paths[kernel.name] = str(path)

        # Save metadata for each kernel.
        metadata[kernel.name] = {
            "thread_type": kernel.thread_type.value,
            "tensor_indices": kernel.tensor_indices,
        }

    # Write metadata file.
    metadata_path = output_dir / "kernel_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return paths


def load_kernel_metadata(kernel_dir: Path) -> dict:
    """
    Load kernel metadata from kernel_metadata.json.

    Args:
        kernel_dir: Directory containing kernel files and metadata.

    Returns:
        Dict mapping kernel name to metadata (thread_type, tensor_indices).
    """
    import json

    metadata_path = kernel_dir / "kernel_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {}
