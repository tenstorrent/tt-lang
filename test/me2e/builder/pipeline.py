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

from ttl.ttl_api import CompilerOptions

from .device_arch import get_mock_arch_from_device


def compile_ttl_to_ttkernel(
    module: Module,
    device: Optional[Any] = None,
    compiler_options: Optional[CompilerOptions] = None,
) -> Module:
    """
    Run the TTL-to-TTKernel pass pipeline on the module.

    Mirrors the pipeline from TTLPipelines.cpp but with proper nesting.

    Args:
        module: TTL MLIR module to compile.
        device: Optional TTNN device for architecture detection.
        compiler_options: Compiler pipeline options.

    Returns:
        Compiled module with TTKernel/EmitC ops.
    """
    opts = compiler_options or CompilerOptions()

    # Always use mock architecture detected from device.
    mock_arch = get_mock_arch_from_device(device)
    device_pass = f"ttcore-register-device{{mock-system-desc-arch={mock_arch}}}"

    # Build function-level passes.
    assign_dst_pass = "ttl-assign-dst"
    if not opts.enable_fpu_binary_ops:
        assign_dst_pass = "ttl-assign-dst{enable-fpu-binary-ops=0}"

    func_passes = [
        "convert-ttl-to-compute",
        assign_dst_pass,
    ]
    if opts.maximize_dst:
        func_passes.append("ttl-subblock-compute-for-dst")
    func_passes += [
        "ttl-insert-tile-regs-sync",
        "ttl-lower-to-loops",
    ]
    if opts.maximize_dst:
        func_passes.append("ttl-schedule-operations")
    func_passes.append("ttl-annotate-cb-associations")
    func_passes_str = ",".join(func_passes)

    pipeline_str = (
        f"builtin.module("
        f"{device_pass},"
        f"func.func({func_passes_str}),"
        # TTL to TTKernel conversion (module-level pass).
        f"convert-ttl-to-ttkernel,"
        # Insert minimal init ops before compute ops.
        f"ttkernel-insert-inits,"
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
