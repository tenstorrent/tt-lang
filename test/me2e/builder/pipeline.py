# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MLIR pass pipeline execution for ME2E tests.

Provides compilation from TTL dialect to TTKernel dialect.
"""

import os
from typing import Any, Optional

from ttl.ir import Module
from ttl.passmanager import PassManager


def compile_ttl_to_ttkernel(
    module: Module,
    device: Optional[Any] = None,
    maximize_dst: bool = True,
    enable_fpu_binary_ops: bool = True,
) -> Module:
    """
    Run the TTL-to-TTKernel pass pipeline on the module.

    Mirrors the pipeline from TTLPipelines.cpp.

    Args:
        module: TTL MLIR module to compile.
        device: Optional TTNN device (unused, kept for API compat).
        maximize_dst: Enable DST maximization (subblocking + scheduling).
        enable_fpu_binary_ops: Enable FPU binary op detection (add_tiles, etc).

    Returns:
        Compiled module with TTKernel/EmitC ops.
    """
    fpu_flag = int(enable_fpu_binary_ops)
    assign_dst_pass = f"ttl-assign-dst{{enable-fpu-binary-ops={fpu_flag}}}"

    # Build per-function passes.
    func_passes = [
        "convert-ttl-to-compute",
        assign_dst_pass,
    ]
    if maximize_dst:
        func_passes.append("ttl-subblock-compute-for-dst")
    func_passes.append("ttl-insert-tile-regs-sync")
    func_passes.append("ttl-lower-to-loops")
    if maximize_dst:
        func_passes.append("ttl-schedule-operations")
    func_passes.append("ttl-annotate-cb-associations")

    func_pipeline = ",".join(func_passes)

    pipeline_str = (
        f"builtin.module("
        f"func.func({func_pipeline}),"
        f"convert-ttl-to-ttkernel,"
        f"ttkernel-insert-inits,"
        f"canonicalize,"
        f"cse,"
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
