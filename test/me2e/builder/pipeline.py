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
    fpu_flag = int(enable_fpu_binary_ops)
    set_compute_config_pass = (
        f"ttl-set-compute-kernel-config{{enable-fpu-binary-ops={fpu_flag}}}"
    )

    # Verification observes the complete high-level schedule after automatic
    # synchronization. Transport formation then precedes DFB acquire coalescing
    # so grouped acquisitions use the materialized transfer span.
    pre_transport_func_passes = [
        "ttl-materialize-loop-state",
        "ttl-insert-intermediate-dfbs",
        "ttl-insert-copy-wait",
        "ttl-insert-cb-sync",
    ]
    post_transport_func_passes = [
        "ttl-coalesce-dfb-acquires",
        "ttl-annotate-l1-acc-loops",
        "convert-ttl-to-compute",
    ]
    lowering_func_passes = [
        set_compute_config_pass,
        "ttl-assign-dst",
    ]
    if maximize_dst:
        lowering_func_passes.append("ttl-subblock-compute-for-dst")
    dst_acc_str = "true" if maximize_dst else "false"
    lowering_func_passes.append(
        f"ttl-lower-to-loops{{dst-accumulation={dst_acc_str}}}"
    )
    if maximize_dst:
        lowering_func_passes.append("ttl-schedule-operations")
    pre_transport_pipeline = ",".join(pre_transport_func_passes)
    post_transport_pipeline = ",".join(post_transport_func_passes)
    lowering_func_pipeline = ",".join(lowering_func_passes)

    specialize_passes = ""
    if specialize_cores:
        specialize_passes = "ttkernel-specialize-cores,canonicalize,cse,"

    pipeline_str = (
        f"builtin.module("
        f"func.func({pre_transport_pipeline}),"
        f"ttl-verify-pipenet,"
        f"ttl-form-pipe-transports,"
        f"func.func({post_transport_pipeline}),"
        f"ttl-finalize-dfb-indices,"
        f"func.func({lowering_func_pipeline}),"
        f"func.func(ttl-annotate-cb-associations),"
        f"ttl-verify-dfb-spsc,"
        f"ttl-erase-pipenet-scopes,"
        f"ttl-validate-cb-budget,"
        f"convert-ttl-to-ttkernel,"
        f"ttkernel-insert-inits,"
        f"canonicalize,"
        f"cse,"
        f"{specialize_passes}"
        f"lower-affine,"
        f"func.func(convert-ttkernel-to-emitc),"
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
