# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CPP < %t.output

"""Verify synchronized DFB reset identity through composition and splitting."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/synchronized_dfb_reset.hpp"


def make_reset_operation():
    second_data_movement = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(
            second_data_movement,
            ttl.KernelKind.DATA_MOVEMENT,
            ttl.KernelKind.COMPUTE,
        )
    )
    all_local_reset = ttl.DFBReset(
        participants=(
            second_data_movement,
            ttl.KernelKind.DATA_MOVEMENT,
            ttl.KernelKind.COMPUTE,
        ),
        scope=ttl.DFBResetScope.ALL_LOCAL,
    )

    @ttl.operation()
    def reset_helper(target: ttl.DFB):
        ttl.call_extern_func(
            FAKE_HEADER,
            "reset",
            func_args=[target],
            dfb_reset=reset,
            dfb_reset_targets=[target],
        )
        ttl.call_extern_func(
            FAKE_HEADER,
            "reset_all",
            dfb_reset=all_local_reset,
        )

    @ttl.operation(grid=(1, 1))
    def synchronized_reset_operation(input_tensor):
        target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        reset_helper(target)

    return synchronized_reset_operation


synchronized_reset_operation = make_reset_operation()


# The captured reset is replicated to exactly its three logical-kernel
# participants. Participant representation is canonical despite source order.
# INITIAL-LABEL: func.func @synchronized_reset_operation__trisc
# INITIAL-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = compute>
# INITIAL: ttl.opaque_call "reset" dfb_reset <{{[0-9]+}}, all_local = false, participants[<kind = compute>, <kind = data_movement>, <kind = data_movement, identity = "[[$SECOND:[^"]+]]"
# INITIAL-NEXT: ttl.opaque_call "reset_all" dfb_reset <{{[0-9]+}}, all_local = true, participants[<kind = compute>, <kind = data_movement>, <kind = data_movement, identity = "[[$SECOND]]"
# INITIAL-LABEL: func.func @synchronized_reset_operation__ncrisc
# INITIAL-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>
# INITIAL: ttl.opaque_call "reset" dfb_reset <{{[0-9]+}}, all_local = false, participants[<kind = compute>, <kind = data_movement>, <kind = data_movement, identity = "[[$SECOND]]"
# INITIAL-NEXT: ttl.opaque_call "reset_all" dfb_reset <{{[0-9]+}}, all_local = true, participants[<kind = compute>, <kind = data_movement>, <kind = data_movement, identity = "[[$SECOND]]"
# INITIAL-LABEL: func.func @synchronized_reset_operation__brisc
# INITIAL-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "[[$SECOND]]"
# INITIAL: ttl.opaque_call "reset" dfb_reset <{{[0-9]+}}, all_local = false, participants[<kind = compute>, <kind = data_movement>, <kind = data_movement, identity = "[[$SECOND]]"
# INITIAL-NEXT: ttl.opaque_call "reset_all" dfb_reset <{{[0-9]+}}, all_local = true, participants[<kind = compute>, <kind = data_movement>, <kind = data_movement, identity = "[[$SECOND]]"

# Reset metadata does not add generated C++ arguments beyond the explicit DFB
# index argument.
# CPP-DAG: reset(get_compile_time_arg_val(0));

# An all-local reset has the same identity in every participant and has no DFB
# operand solely for compiler metadata.
# CPP-DAG: reset_all();


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    synchronized_reset_operation(input_tensor)
