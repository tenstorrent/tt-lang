# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Runner implementation for declarative ME2E tests.

Provides run_compute_test function that executes compute operations using ttnn.generic_op.
Follows the elementwise example pattern.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .builder.kernels import (
    KernelSpec,
    ThreadType,
    translate_module_to_kernels,
    write_kernels,
)
from .builder.pipeline import compile_ttl_to_ttkernel
from .builder.ttl_builder import build_e2e_module
from .builder.ttnn_runner import run_binary_op, run_unary_op
from .config_specs import TestConfig
from .op_specs import ComputeOpSpec
from ttlang_test_utils import assert_with_ulp

# Kernel cache to avoid redundant compilation.
_kernel_cache: Dict[str, str] = {}


def get_compute_kernel(
    op: ComputeOpSpec, config: TestConfig, device: Optional[Any] = None
) -> str:
    """
    Generate or retrieve cached compute kernel C++ source.

    Args:
        op: Compute operation specification.
        config: Test configuration.
        device: Optional TTNN device for architecture detection.

    Returns:
        C++ source code for the compute kernel.
    """
    cache_key = (
        f"{op.name}_{op.ttl_op}_{config.block_h}x{config.block_w}_{config.dtype}"
    )
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    # Build ME2E TTL module.
    e2e_config = config.to_e2e_config()
    module = build_e2e_module(op.name, op.arity, e2e_config)

    # Run TTL pass pipeline to get EmitC.
    compiled_module = compile_ttl_to_ttkernel(module, device)

    # Translate to C++ kernels.
    noc_kernels, compute_kernel = translate_module_to_kernels(compiled_module)

    # Cache the compute kernel source.
    _kernel_cache[cache_key] = compute_kernel.source

    return compute_kernel.source


def run_compute_test(
    op: ComputeOpSpec,
    config: TestConfig,
    device: Any,
) -> None:
    """
    Execute a compute test using ttnn.generic_op pattern.

    Args:
        op: Compute operation specification.
        config: Test configuration.
        device: TTNN device.

    Raises:
        AssertionError: If the computed result doesn't match the golden reference.
    """
    # Set seed for reproducible test inputs.
    seed = int(os.environ.get("TTLANG_TEST_SEED", "42"))
    torch.manual_seed(seed)

    e2e_config = config.to_e2e_config()

    # 1. Create tensors on device (following elementwise pattern).
    # Use input range if specified (for ops with domain constraints).
    input_range = op.input_range or (-1.0, 1.0)
    lo, hi = input_range

    torch_inputs = []
    for _ in range(op.arity):
        t = torch.rand(e2e_config.tensor_shape, dtype=e2e_config.dtype) * (hi - lo) + lo
        torch_inputs.append(t)

    # Compute golden using torch reference.
    if op.arity == 1:
        golden = op.golden(torch_inputs[0])
    else:
        golden = op.golden(torch_inputs[0], torch_inputs[1])

    # 2. Get or generate compute kernel.
    compute_cpp = get_compute_kernel(op, config, device)

    # 3. Build full ME2E module to get reader/writer kernels.
    # We need the full module to extract all kernels (reader, compute, writer).
    module = build_e2e_module(op.name, op.arity, e2e_config)
    compiled_module = compile_ttl_to_ttkernel(module, device)
    noc_kernels, compute_kernel_spec = translate_module_to_kernels(compiled_module)

    # Replace compute kernel source with cached/generated one.
    compute_kernel_spec = KernelSpec(
        name=compute_kernel_spec.name,
        thread_type=ThreadType.COMPUTE,
        source=compute_cpp,
    )

    # 4. Write kernels to temporary directory.
    user = os.environ.get("USER", "default")
    kernel_dir = Path(tempfile.mkdtemp(prefix=f"ttlang_kernels_{user}_{op.name}_"))
    write_kernels(noc_kernels, compute_kernel_spec, kernel_dir)

    # 5. Execute on device.
    try:
        if op.arity == 2:
            result = run_binary_op(
                device=device,
                noc_kernels=noc_kernels,
                compute_kernel=compute_kernel_spec,
                input_a=torch_inputs[0],
                input_b=torch_inputs[1],
                kernel_dir=kernel_dir,
            )
        else:
            result = run_unary_op(
                device=device,
                noc_kernels=noc_kernels,
                compute_kernel=compute_kernel_spec,
                input_a=torch_inputs[0],
                kernel_dir=kernel_dir,
            )

        # 6. Validate against golden.
        assert_with_ulp(result, golden)

    finally:
        # Cleanup temporary kernel directory.
        import shutil

        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
