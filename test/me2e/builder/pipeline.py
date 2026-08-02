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
    specialize_cores: bool = False,
) -> Module:
    """
    Run the TTL-to-TTKernel pass pipeline on the module.

    Mirrors the pipeline from TTLPipelines.cpp.

    Args:
        module: TTL MLIR module to compile.
        device: Optional TTNN device (unused, kept for API compat).
        maximize_dst: Enable DST maximization (subblocking + scheduling).
        enable_fpu_binary_ops: Enable FPU binary op detection (add_tiles, etc).
        specialize_cores: Clone kernels that branch on a core coordinate
            once per launch coordinate (ttkernel-specialize-cores).

    Returns:
        Compiled module with TTKernel/EmitC ops.
    """
    pipeline_options = " ".join(
        [
            f"maximize-dst={str(maximize_dst).lower()}",
            f"enable-fpu-binary-ops={str(enable_fpu_binary_ops).lower()}",
            f"specialize-cores={str(specialize_cores).lower()}",
            "lower-to-emitc=true",
        ]
    )
    pipeline_str = f"builtin.module(ttl-to-ttkernel-pipeline{{{pipeline_options}}})"

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
