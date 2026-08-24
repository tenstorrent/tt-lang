# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MLIR pass pipeline execution for ME2E tests.

Provides compilation from TTL dialect to TTKernel dialect.
"""

import os
from typing import Any, Optional

from ttl.dialects import ttcore
from ttl.ir import Module
from ttl.passmanager import PassManager

from .device_arch import get_mock_arch_from_device


_TTCORE_ARCH_BY_DEVICE_NAME = {
    "blackhole": ttcore.Arch.Blackhole,
    "wormhole_b0": ttcore.Arch.WormholeB0,
}


def compile_ttl_to_ttkernel(
    module: Module,
    device: Optional[Any] = None,
    maximize_dst: bool = True,
    accumulation_strategy: str = "auto",
    enable_fpu_binary_ops: bool = True,
    specialize_cores: bool = False,
) -> Module:
    """
    Run the TTL-to-TTKernel pass pipeline on the module.

    Mirrors the pipeline from TTLPipelines.cpp.

    Args:
        module: TTL MLIR module to compile.
        device: Optional TTNN device used to select target capabilities. A
            compiler-only invocation without a device uses a Wormhole mock.
        maximize_dst: Enable DST maximization (subblocking + scheduling).
        accumulation_strategy: Accumulation storage strategy.
        enable_fpu_binary_ops: Allow FPU strategy selection for add/sub/mul.
        specialize_cores: Run the ttkernel-specialize-and-annotate-dfb-use
            sub-pipeline. Clones kernels that branch on a core coordinate
            once per launch coordinate, then records surviving DFB uses.

    Returns:
        Compiled module with TTKernel/EmitC ops.
    """
    target_arch = get_mock_arch_from_device(device)
    module.operation.attributes["ttl.target_arch"] = ttcore.ir.ArchAttr.get(
        module.context, int(_TTCORE_ARCH_BY_DEVICE_NAME[target_arch])
    )

    pipeline_options = " ".join(
        [
            f"maximize-dst={str(maximize_dst).lower()}",
            f"accumulation-strategy={accumulation_strategy}",
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
