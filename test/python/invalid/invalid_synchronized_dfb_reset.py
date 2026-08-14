# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s declaration 2>&1 | FileCheck %s --check-prefix=DECLARATION
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s element 2>&1 | FileCheck %s --check-prefix=ELEMENT
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s duplicate 2>&1 | FileCheck %s --check-prefix=DUPLICATE
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s raw 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s paired 2>&1 | FileCheck %s --check-prefix=PAIRED
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s effects 2>&1 | FileCheck %s --check-prefix=EFFECTS
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s accesses 2>&1 | FileCheck %s --check-prefix=ACCESSES
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s selection 2>&1 | FileCheck %s --check-prefix=SELECTION
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s global 2>&1 | FileCheck %s --check-prefix=GLOBAL
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s scope 2>&1 | FileCheck %s --check-prefix=SCOPE
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s all-targets 2>&1 | FileCheck %s --check-prefix=ALL-TARGETS

"""Verify invalid synchronized DFB reset declarations and uses."""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/synchronized_dfb_reset.hpp"
MODE = sys.argv[1]
GLOBAL_RESET = ttl.DFBReset(participants=(ttl.KernelKind.COMPUTE,))

if MODE == "declaration":
    ttl.DFBReset(participants=[ttl.KernelKind.COMPUTE])
elif MODE == "element":
    ttl.DFBReset(participants=(17,))
elif MODE == "duplicate":
    ttl.DFBReset(participants=(ttl.KernelKind.COMPUTE, ttl.KernelKind.COMPUTE))
elif MODE == "scope":
    ttl.DFBReset(participants=(ttl.KernelKind.COMPUTE,), scope="all-local")


def make_invalid_operation(mode):
    participants = (
        ttl.KernelKind.COMPUTE,
        ttl.KernelKind.DATA_MOVEMENT,
    )
    reset = ttl.DFBReset(participants=participants)

    if mode == "raw":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            ttl.call_extern_func(
                FAKE_HEADER,
                "reset",
                func_args=[target],
                dfb_reset=17,
                dfb_reset_targets=[target],
                kernel=ttl.KernelKind.COMPUTE,
            )

    elif mode == "paired":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            ttl.call_extern_func(
                FAKE_HEADER,
                "reset",
                func_args=[target],
                dfb_reset=reset,
                kernel=(
                    ttl.KernelKind.COMPUTE,
                    ttl.KernelKind.DATA_MOVEMENT,
                ),
            )

    elif mode == "all-targets":
        all_local_reset = ttl.DFBReset(
            participants=participants,
            scope=ttl.DFBResetScope.ALL_LOCAL,
        )

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            ttl.call_extern_func(
                FAKE_HEADER,
                "reset",
                func_args=[target],
                dfb_reset=all_local_reset,
                dfb_reset_targets=[target],
                kernel=(
                    ttl.KernelKind.COMPUTE,
                    ttl.KernelKind.DATA_MOVEMENT,
                ),
            )

    elif mode == "effects":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            ttl.call_extern_func(
                FAKE_HEADER,
                "reset",
                func_args=[target],
                dfb_effects=[ttl.DFBEffect.reserve(target, tiles=1)],
                dfb_reset=reset,
                dfb_reset_targets=[target],
                kernel=(
                    ttl.KernelKind.COMPUTE,
                    ttl.KernelKind.DATA_MOVEMENT,
                ),
            )

    elif mode == "accesses":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            ttl.call_extern_func(
                FAKE_HEADER,
                "reset",
                func_args=[target],
                dfb_accesses=[ttl.DFBAccess.interface_preserved(target)],
                dfb_reset=reset,
                dfb_reset_targets=[target],
                kernel=(
                    ttl.KernelKind.COMPUTE,
                    ttl.KernelKind.DATA_MOVEMENT,
                ),
            )

    elif mode == "selection":
        selection_reset = ttl.DFBReset(participants=(ttl.KernelKind.COMPUTE,))

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            ttl.call_extern_func(
                FAKE_HEADER,
                "reset",
                func_args=[target],
                dfb_reset=selection_reset,
                dfb_reset_targets=[target],
                kernel=ttl.KernelKind.DATA_MOVEMENT,
            )

    else:
        raise ValueError(f"unsupported mode: {mode}")

    return invalid_operation


if MODE == "global":

    @ttl.operation(grid=(1, 1))
    def invalid_operation(input_tensor):
        target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        ttl.call_extern_func(
            FAKE_HEADER,
            "reset",
            func_args=[target],
            dfb_reset=GLOBAL_RESET,
            dfb_reset_targets=[target],
            kernel=ttl.KernelKind.COMPUTE,
        )

elif MODE not in {"declaration", "element", "duplicate", "scope"}:
    invalid_operation = make_invalid_operation(MODE)


# DECLARATION: TypeError: DFBReset participants must be a nonempty tuple
# ELEMENT: TypeError: DFBReset participants must contain only KernelKind or Kernel values, got int
# DUPLICATE: ValueError: DFBReset participants must be distinct
# SCOPE: TypeError: DFBReset scope must be a ttl.DFBResetScope, got str
# RAW: ValueError: @ttl.operation split: call_extern_func dfb_reset must be a DFBReset captured by the enclosing operation
# PAIRED: TTLangCompileError: error: ttl.call_extern_func() targeted DFB reset requires dfb_reset_targets
# ALL-TARGETS: TTLangCompileError: error: ttl.call_extern_func() all-local DFB reset cannot declare dfb_reset_targets
# EFFECTS: TTLangCompileError: error: ttl.call_extern_func() synchronized DFB reset cannot return a value or declare protocol effects or unknown DFB access
# ACCESSES: TTLangCompileError: error: ttl.call_extern_func() synchronized DFB reset cannot declare non-transactional accesses
# SELECTION: ValueError: @ttl.operation split: external-call kernel selection contains a logical kernel outside the DFBReset participant set
# GLOBAL: ValueError: @ttl.operation 'invalid_operation': DFBReset 'GLOBAL_RESET' must be created by an enclosing factory


if __name__ == "__main__" and MODE not in {
    "declaration",
    "element",
    "duplicate",
    "scope",
}:
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_operation(input_tensor)
