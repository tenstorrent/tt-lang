# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CPP < %t.output

"""Verify built-in synchronized DFB reset composition and lowering."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import ttl_api


def _blackhole_compile_target(_runtime_args):
    return "blackhole"


ttl_api._device_target_arch = _blackhole_compile_target


def make_reset_operation():
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    selected_reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )
    all_reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation()
    def reset_helper(target: ttl.DFB):
        ttl.reset_dfbs(selected_reset, dfbs=[target])
        ttl.reset_all_dfbs(all_reset)

    @ttl.operation(grid=(1, 1))
    def synchronized_reset_operation(input_tensor):
        target = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        reset_helper(target)

    return synchronized_reset_operation


synchronized_reset_operation = make_reset_operation()


# One source call is replicated to exactly its three logical-kernel
# participants. Participant representation is canonical despite source order.
# INITIAL-LABEL: func.func @synchronized_reset_operation__trisc
# INITIAL-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "dfb_reset_participant_compute_0_1_0"
# INITIAL: ttl.reset_dfbs <{{[0-9]+}}, participants[<kind = compute, identity = "dfb_reset_participant_compute_0_1_0"
# INITIAL-NEXT: ttl.reset_all_dfbs <{{[0-9]+}}, participants[<kind = compute, identity = "dfb_reset_participant_compute_0_1_0"
# INITIAL-LABEL: func.func @synchronized_reset_operation__ncrisc
# INITIAL-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "dfb_reset_participant_data_movement_0_1_0"
# INITIAL: ttl.reset_dfbs <{{[0-9]+}}, participants[<kind = compute, identity = "dfb_reset_participant_compute_0_1_0"
# INITIAL-NEXT: ttl.reset_all_dfbs <{{[0-9]+}}, participants[<kind = compute, identity = "dfb_reset_participant_compute_0_1_0"
# INITIAL-LABEL: func.func @synchronized_reset_operation__brisc
# INITIAL-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "dfb_reset_participant_data_movement_0_1_1"
# INITIAL: ttl.reset_dfbs <{{[0-9]+}}, participants[<kind = compute, identity = "dfb_reset_participant_compute_0_1_0"
# INITIAL-NEXT: ttl.reset_all_dfbs <{{[0-9]+}}, participants[<kind = compute, identity = "dfb_reset_participant_compute_0_1_0"

# The built-in lowering supplies the shared state address and physical-index
# masks; no user reset helper is required.
# CPP-COUNT-6: experimental::reset_dfb_interfaces(


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    synchronized_reset_operation(input_tensor)
