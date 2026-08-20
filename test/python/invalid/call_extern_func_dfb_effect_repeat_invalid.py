# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s negative 2>&1 | FileCheck %s --check-prefix=NEGATIVE
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s dynamic 2>&1 | FileCheck %s --check-prefix=DYNAMIC
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s empty 2>&1 | FileCheck %s --check-prefix=EMPTY
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s expression 2>&1 | FileCheck %s --check-prefix=EXPRESSION
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s limit 2>&1 | FileCheck %s --check-prefix=LIMIT
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s nested_limit 2>&1 | FileCheck %s --check-prefix=NESTED-LIMIT
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s cumulative_limit 2>&1 | FileCheck %s --check-prefix=CUMULATIVE-LIMIT
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s huge_count 2>&1 | FileCheck %s --check-prefix=HUGE-COUNT
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s keyword_arguments 2>&1 | FileCheck %s --check-prefix=KEYWORD-ARGUMENTS

"""Verify repeated external DFB effect syntax and expansion limits."""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/repeated_effect.hpp"
HUGE_REPEAT_COUNT = 10**1000
MODE = sys.argv[1]


def make_invalid_repeated_effect(mode):
    if mode == "negative":

        @ttl.operation(grid=(1, 1))
        def invalid_repeated_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "repeated_effect",
                dfb_dependencies=[source],
                dfb_effects=[
                    ttl.DFBEffect.repeat(
                        -1,
                        [ttl.DFBEffect.wait(source, tiles=1)],
                    )
                ],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "dynamic":

        @ttl.operation(grid=(1, 1))
        def invalid_repeated_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            node_x, _ = ttl.node(dims=2)
            ttl.call_extern_func(
                FAKE_HEADER,
                "repeated_effect",
                dfb_dependencies=[source],
                dfb_effects=[
                    ttl.DFBEffect.repeat(
                        node_x + 1,
                        [ttl.DFBEffect.wait(source, tiles=1)],
                    )
                ],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "empty":

        @ttl.operation(grid=(1, 1))
        def invalid_repeated_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "repeated_effect",
                dfb_dependencies=[source],
                dfb_effects=[ttl.DFBEffect.repeat(2, [])],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "expression":

        @ttl.operation(grid=(1, 1))
        def invalid_repeated_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "repeated_effect",
                dfb_dependencies=[source],
                dfb_effects=[ttl.DFBEffect.wait(source, tiles=1)]
                + [ttl.DFBEffect.pop(source, tiles=1)],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "limit":

        @ttl.operation(grid=(1, 1))
        def invalid_repeated_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "repeated_effect",
                dfb_dependencies=[source],
                dfb_effects=[
                    ttl.DFBEffect.repeat(
                        4097,
                        [ttl.DFBEffect.wait(source, tiles=1)],
                    )
                ],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "nested_limit":

        @ttl.operation(grid=(1, 1))
        def invalid_repeated_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "repeated_effect",
                dfb_dependencies=[source],
                dfb_effects=[
                    ttl.DFBEffect.repeat(
                        65,
                        [
                            ttl.DFBEffect.repeat(
                                64,
                                [ttl.DFBEffect.wait(source, tiles=1)],
                            )
                        ],
                    )
                ],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "cumulative_limit":

        @ttl.operation(grid=(1, 1))
        def invalid_repeated_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "repeated_effect",
                dfb_dependencies=[source],
                dfb_effects=[
                    ttl.DFBEffect.repeat(
                        2048,
                        [
                            ttl.DFBEffect.wait(source, tiles=1),
                            ttl.DFBEffect.pop(source, tiles=1),
                        ],
                    ),
                    ttl.DFBEffect.wait(source, tiles=1),
                ],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "huge_count":

        @ttl.operation(grid=(1, 1))
        def invalid_repeated_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "repeated_effect",
                dfb_dependencies=[source],
                dfb_effects=[
                    ttl.DFBEffect.repeat(
                        HUGE_REPEAT_COUNT,
                        [ttl.DFBEffect.wait(source, tiles=1)],
                    )
                ],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    elif mode == "keyword_arguments":

        @ttl.operation(grid=(1, 1))
        def invalid_repeated_effect(input_tensor):
            source = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=2
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "repeated_effect",
                dfb_dependencies=[source],
                dfb_effects=[
                    ttl.DFBEffect.repeat(
                        count=2,
                        effects=[ttl.DFBEffect.wait(source, tiles=1)],
                    )
                ],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    else:
        raise ValueError(f"unsupported mode: {mode}")

    return invalid_repeated_effect


invalid_repeated_effect = make_invalid_repeated_effect(MODE)


# NEGATIVE: TTLangCompileError: error: ttl.DFBEffect.repeat() count must be nonnegative
# DYNAMIC: TTLangCompileError: error: ttl.call_extern_func() effect repeat count must be a statically resolvable integer
# EMPTY: TTLangCompileError: error: ttl.DFBEffect.repeat() effects must not be empty
# EXPRESSION: TTLangCompileError: error: ttl.call_extern_func() dfb_effects and ttl.DFBEffect.repeat() effects must be lists
# LIMIT: TTLangCompileError: error: ttl.call_extern_func() dfb_effects may contain at most 4096 expanded effects
# NESTED-LIMIT: TTLangCompileError: error: ttl.call_extern_func() dfb_effects may contain at most 4096 expanded effects
# CUMULATIVE-LIMIT: TTLangCompileError: error: ttl.call_extern_func() dfb_effects may contain at most 4096 expanded effects
# HUGE-COUNT: TTLangCompileError: error: ttl.call_extern_func() dfb_effects may contain at most 4096 expanded effects
# KEYWORD-ARGUMENTS: TTLangCompileError: error: ttl.DFBEffect.repeat() requires count and effects arguments


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_repeated_effect(input_tensor)
