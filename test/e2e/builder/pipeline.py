# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MLIR pass pipeline execution for E2E tests.

Provides compilation from TTL dialect to TTKernel dialect.
"""

import os
from typing import Optional

from ttmlir.ir import Module
from ttmlir.passmanager import PassManager

from .system_desc import get_system_desc_path


def compile_ttl_to_ttkernel(
    module: Module, system_desc_path: Optional[str] = None
) -> Module:
    """
    Run the TTL-to-TTKernel pass pipeline on the module.

    Mirrors the pipeline from TTLPipelines.cpp but with proper nesting.

    Args:
        module: TTL MLIR module to compile.
        system_desc_path: Path to system descriptor (auto-detected if not provided).

    Returns:
        Compiled module with TTKernel/EmitC ops.
    """
    if system_desc_path is None:
        system_desc_path = get_system_desc_path()

    # Build the pass pipeline matching TTLPipelines.cpp.
    # func.func passes run on each function, module passes run on module.
    pipeline_str = (
        f"builtin.module("
        f"ttcore-register-device{{system-desc-path={system_desc_path}}},"
        # TTL to compute conversion (runs on each function).
        f"func.func(convert-ttl-to-compute,"
        f"ttl-tile-and-assign-dst,"
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
