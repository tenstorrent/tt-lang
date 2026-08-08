# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Kernel specification and execution for ME2E tests.

Supports N data movement (NOC) threads and single compute thread.
Extracts tensor indices from compiled MLIR for proper argument building.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple

from ttl.ir import Module, ArrayAttr, DenseI32ArrayAttr, IntegerAttr
from ttl.passes import ttkernel_to_cpp_by_name, get_ttkernel_names
from ttl.dialects import func


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
    fp32_dest_acc_en: bool = False
    dst_full_sync_en: bool = False
    unpack_to_dest_fp32: List[int] = field(default_factory=list)


def _get_kernel_func(module: Module, kernel_name: str) -> func.FuncOp:
    """Return the compiled function for a kernel symbol."""
    for operation in module.body.operations:
        if isinstance(operation, func.FuncOp) and operation.name.value == kernel_name:
            return operation
    raise ValueError(f"Kernel function '{kernel_name}' not found")


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
    operation = _get_kernel_func(module, kernel_name)
    if "ttl.crta_indices" in operation.attributes:
        crta_attr = operation.attributes["ttl.crta_indices"]
        if isinstance(crta_attr, ArrayAttr):
            return [int(IntegerAttr(index).value) for index in crta_attr]
    return []


def _get_kernel_bool_attr(module: Module, kernel_name: str, attr_name: str) -> bool:
    """Return a required compiler-generated boolean kernel attribute."""
    operation = _get_kernel_func(module, kernel_name)
    attribute = operation.attributes.get(attr_name)
    if attribute is None:
        raise ValueError(
            f"Required compiler-generated attribute '{attr_name}' is missing "
            f"from compute kernel '{kernel_name}'"
        )
    attribute_text = str(attribute).strip()
    if attribute_text == "true":
        return True
    if attribute_text == "false":
        return False
    raise ValueError(
        f"Expected boolean attribute '{attr_name}' on kernel '{kernel_name}', "
        f"got {attribute_text!r}"
    )


def _get_kernel_i32_array_attr(
    module: Module, kernel_name: str, attr_name: str
) -> List[int]:
    """Return a compiler-generated dense i32 array kernel attribute."""
    operation = _get_kernel_func(module, kernel_name)
    attribute = operation.attributes.get(attr_name)
    if attribute is None:
        raise ValueError(
            f"Required compiler-generated attribute '{attr_name}' is missing "
            f"from compute kernel '{kernel_name}'"
        )
    if not isinstance(attribute, DenseI32ArrayAttr):
        raise ValueError(
            f"Expected dense i32 array attribute '{attr_name}' on kernel "
            f"'{kernel_name}', got {attribute}"
        )
    return list(attribute)


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
            spec.fp32_dest_acc_en = _get_kernel_bool_attr(
                module, name, "fp32_dest_acc_en"
            )
            spec.dst_full_sync_en = _get_kernel_bool_attr(
                module, name, "dst_full_sync_en"
            )
            spec.unpack_to_dest_fp32 = _get_kernel_i32_array_attr(
                module, name, "ttl.unpack_to_dest_fp32"
            )
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
            "fp32_dest_acc_en": kernel.fp32_dest_acc_en,
            "dst_full_sync_en": kernel.dst_full_sync_en,
            "unpack_to_dest_fp32": kernel.unpack_to_dest_fp32,
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
