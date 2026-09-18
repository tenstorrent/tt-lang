# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Verify diagnostics for invalid external DFB effect integer expressions.
# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s boolean 2>&1 | FileCheck %s --check-prefix=BOOLEAN
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s zero-divisor 2>&1 | FileCheck %s --check-prefix=ZERO-DIVISOR
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s zero-modulo 2>&1 | FileCheck %s --check-prefix=ZERO-MODULO
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s unsupported 2>&1 | FileCheck %s --check-prefix=OPERATOR

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/static_effect.hpp"
MODE = sys.argv[1]


def make_invalid_static_effect(mode):
    if mode == "boolean":

        @ttl.operation(grid=(1, 1))
        def invalid_static_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "static_effect",
                dfb_dependencies=[source],
                dfb_effects=[ttl.DFBEffect.wait(source, tiles=True)],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "zero-divisor":

        @ttl.operation(grid=(1, 1))
        def invalid_static_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "static_effect",
                dfb_dependencies=[source],
                dfb_effects=[ttl.DFBEffect.wait(source, tiles=1 // 0)],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "zero-modulo":

        @ttl.operation(grid=(1, 1))
        def invalid_static_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "static_effect",
                dfb_dependencies=[source],
                dfb_effects=[ttl.DFBEffect.wait(source, tiles=1 % 0)],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "unsupported":

        @ttl.operation(grid=(1, 1))
        def invalid_static_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "static_effect",
                dfb_dependencies=[source],
                dfb_effects=[ttl.DFBEffect.wait(source, tiles=2**0)],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    else:
        raise ValueError(f"unsupported mode: {mode}")

    return invalid_static_effect


invalid_static_effect = make_invalid_static_effect(MODE)


# BOOLEAN: TTLangCompileError: error: ttl.call_extern_func() effect tiles must be a statically resolvable integer
# ZERO-DIVISOR: TTLangCompileError: error: ttl.call_extern_func() effect tiles divisor must be nonzero
# ZERO-MODULO: TTLangCompileError: error: ttl.call_extern_func() effect tiles divisor must be nonzero
# OPERATOR: TTLangCompileError: error: ttl.call_extern_func() effect tiles must be a statically resolvable integer


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_static_effect(input_tensor)
