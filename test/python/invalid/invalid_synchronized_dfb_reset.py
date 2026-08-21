# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s declaration 2>&1 | FileCheck %s --check-prefix=DECLARATION
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s element 2>&1 | FileCheck %s --check-prefix=ELEMENT
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s duplicate 2>&1 | FileCheck %s --check-prefix=DUPLICATE
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s participants 2>&1 | FileCheck %s --check-prefix=PARTICIPANTS
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s raw 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s missing-dfbs 2>&1 | FileCheck %s --check-prefix=MISSING-DFBS
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s all-dfbs 2>&1 | FileCheck %s --check-prefix=ALL-DFBS
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s empty 2>&1 | FileCheck %s --check-prefix=EMPTY
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s duplicate-dfbs 2>&1 | FileCheck %s --check-prefix=DUPLICATE-DFBS
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s non-dfb 2>&1 | FileCheck %s --check-prefix=NON-DFB
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s global 2>&1 | FileCheck %s --check-prefix=GLOBAL

"""Verify invalid synchronized DFB reset declarations and uses."""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn

MODE = sys.argv[1]

if MODE == "declaration":
    ttl.DFBReset(participants=[ttl.KernelKind.COMPUTE])
elif MODE == "element":
    ttl.DFBReset(participants=(ttl.KernelKind.COMPUTE,))
elif MODE == "duplicate":
    duplicate_compute = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    ttl.DFBReset(participants=(duplicate_compute, duplicate_compute, reader, writer))
elif MODE == "participants":
    ttl.DFBReset(participants=(ttl.Kernel(ttl.KernelKind.COMPUTE),))


def make_invalid_operation(mode):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))

    if mode == "raw":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            ttl.reset_dfbs(17, dfbs=[target])

    elif mode == "missing-dfbs":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            ttl.reset_dfbs(reset)

    elif mode == "all-dfbs":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            ttl.reset_all_dfbs(reset, dfbs=[target])

    elif mode == "empty":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            ttl.reset_dfbs(reset, dfbs=[])

    elif mode == "duplicate-dfbs":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            ttl.reset_dfbs(reset, dfbs=[target, target])

    elif mode == "non-dfb":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            ttl.reset_dfbs(reset, dfbs=[1])

    else:
        raise ValueError(f"unsupported mode: {mode}")

    return invalid_operation


if MODE == "global":
    global_compute = ttl.Kernel(ttl.KernelKind.COMPUTE)
    global_reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    global_writer = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    global_reset = ttl.DFBReset(
        participants=(global_compute, global_reader, global_writer)
    )

    @ttl.operation(grid=(1, 1))
    def invalid_operation(input_tensor):
        target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        ttl.reset_dfbs(global_reset, dfbs=[target])

elif MODE not in {"declaration", "element", "duplicate", "participants"}:
    invalid_operation = make_invalid_operation(MODE)


# DECLARATION: TypeError: DFBReset participants must be a nonempty tuple
# ELEMENT: TypeError: DFBReset participants must contain only Kernel values, got KernelKind
# DUPLICATE: ValueError: DFBReset participants must be distinct
# PARTICIPANTS: ValueError: DFBReset participants must contain one compute kernel and two data movement kernels
# RAW: ValueError: @ttl.operation split: ttl.reset_dfbs reset must be a DFBReset captured by the enclosing operation
# MISSING-DFBS: TTLangCompileError: error: ttl.reset_dfbs() requires the dfbs keyword argument
# ALL-DFBS: TTLangCompileError: error: ttl.reset_all_dfbs() does not accept keyword arguments
# EMPTY: TTLangCompileError: error: ttl.reset_dfbs() dfbs must be a nonempty list
# DUPLICATE-DFBS: TTLangCompileError: error: ttl.reset_dfbs() dfbs must be distinct
# NON-DFB: TTLangCompileError: error: ttl.reset_dfbs() dfbs element must be a DFB
# GLOBAL: ValueError: @ttl.operation 'invalid_operation': DFBReset 'global_reset' must be created by an enclosing factory


if __name__ == "__main__" and MODE not in {
    "declaration",
    "element",
    "duplicate",
    "participants",
}:
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_operation(input_tensor)
