# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MLIR pass pipeline execution for ME2E tests.

Provides compilation from TTL dialect to TTKernel dialect.
"""

import os
from typing import Any, Optional

from ttmlir.ir import Module
from ttmlir.passmanager import PassManager

from .device_arch import get_mock_arch_from_device


def compile_ttl_to_ttkernel(module: Module, device: Optional[Any] = None) -> Module:
    """
    Run the TTL-to-TTKernel pass pipeline on the module.

    Mirrors the pipeline from TTLPipelines.cpp but with proper nesting.

    Args:
        module: TTL MLIR module to compile.
        device: Optional TTNN device for architecture detection.

    Returns:
        Compiled module with TTKernel/EmitC ops.
    """
    # Always use mock architecture detected from device.
    mock_arch = get_mock_arch_from_device(device)
    device_pass = f"ttcore-register-device{{mock-system-desc-arch={mock_arch}}}"

    pipeline_str = (
        f"builtin.module("
        f"{device_pass},"
        # TTL to compute conversion (runs on each function).
        f"func.func(convert-ttl-to-compute,"
        f"ttl-assign-dst,"
        f"ttl-insert-tile-regs-sync,"
        f"ttl-lower-to-loops,"
        f"ttl-annotate-cb-associations),"
        # TTL to TTKernel conversion (module-level pass).
        f"convert-ttl-to-ttkernel,"
        f"canonicalize,"
        f"cse,"
        # Lower to EmitC.
        f"lower-affine,"
        f"convert-ttkernel-to-emitc,"
        f"canonicalize"
        f")"
    )

    pm = PassManager.parse(pipeline_str, context=module.context)
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
