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


def get_system_desc_path() -> str:
    """Get the system descriptor path, generating if needed."""
    path = os.environ.get("SYSTEM_DESC_PATH")
    if path and os.path.exists(path):
        return path

    try:
        from _ttmlir_runtime import runtime

        system_desc = runtime.get_current_system_desc()
        generated_path = "/tmp/ttlang_e2e_system.ttsys"
        system_desc.store(generated_path)
        return generated_path
    except (ImportError, Exception) as e:
        raise RuntimeError(f"Cannot get system descriptor: {e}")


def compile_ttl_to_ttkernel(
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
