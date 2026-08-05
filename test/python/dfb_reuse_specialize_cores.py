# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s --no-ttl-specialize-cores > %t.generic 2>&1
# RUN: FileCheck %s --check-prefix=GENERIC-CPP < %t.generic
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s --ttl-specialize-cores > %t.specialized 2>&1
# RUN: FileCheck %s --check-prefix=SPECIALIZED-CPP < %t.specialized

"""Verify that per-core specialization preserves reused DFB indices in C++."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn

import ttl
from ttlang_test_utils import to_dram


@ttl.operation(grid=(2, 1))
def dfb_reuse_specialize_cores(input_tensor, output_tensor):
    first_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    acknowledgment_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=2
    )
    second_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=2
    )
    output_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def compute():
        with first_dfb.wait():
            pass

        with acknowledgment_dfb.reserve() as acknowledgment:
            acknowledgment.store(
                ttl.block.fill(
                    0,
                    shape=acknowledgment.shape,
                    dtype=acknowledgment.dtype,
                )
            )

        with second_dfb.wait() as second_block, output_dfb.reserve() as output_block:
            output_block.store(second_block)

    @ttl.datamovement()
    def dm_read():
        node_x, _ = ttl.node(dims=2)
        with first_dfb.reserve() as first_block:
            transaction = ttl.copy(input_tensor[0, 0], first_block)
            if node_x == 0:
                transaction = ttl.copy(input_tensor[0, 1], first_block)
            transaction.wait()

        with acknowledgment_dfb.wait():
            pass

        with second_dfb.reserve() as second_block:
            ttl.copy(input_tensor[0, node_x], second_block).wait()

    @ttl.datamovement()
    def dm_write():
        node_x, _ = ttl.node(dims=2)
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, output_tensor[0, node_x]).wait()


# The unspecialized reader retains one coordinate branch. The two reserve calls
# around the acknowledgment wait use the same physical DFB index because the
# acknowledgment proves that the first lifetime ends before the second begins.
# GENERIC-CPP-LABEL: === dm_read kernel written to {{.*}} ===
# GENERIC-CPP: CircularBuffer [[ACK:cb_ctarg_[0-9]+]](get_compile_time_arg_val([[ACK_INDEX:[0-9]+]]))
# GENERIC-CPP-NEXT: CircularBuffer [[REUSED:cb_ctarg_[0-9]+]](get_compile_time_arg_val([[REUSED_INDEX:[0-9]+]]))
# GENERIC-CPP: [[REUSED]].reserve_back
# GENERIC-CPP: [[ACK]].wait_front
# GENERIC-CPP: [[REUSED]].reserve_back

# Specialization emits one reader clone per launch coordinate. Both clones must
# preserve the same physical DFB assignment after their branches are folded.
# SPECIALIZED-CPP-LABEL: === dm_read_c0_0 kernel written to {{.*}} ===
# SPECIALIZED-CPP: CircularBuffer [[C0_ACK:cb_ctarg_[0-9]+]](get_compile_time_arg_val([[C0_ACK_INDEX:[0-9]+]]))
# SPECIALIZED-CPP-NEXT: CircularBuffer [[C0_REUSED:cb_ctarg_[0-9]+]](get_compile_time_arg_val([[C0_REUSED_INDEX:[0-9]+]]))
# SPECIALIZED-CPP: [[C0_REUSED]].reserve_back
# SPECIALIZED-CPP: [[C0_ACK]].wait_front
# SPECIALIZED-CPP: [[C0_REUSED]].reserve_back
# SPECIALIZED-CPP-LABEL: === dm_read_c1_0 kernel written to {{.*}} ===
# SPECIALIZED-CPP: CircularBuffer [[C1_ACK:cb_ctarg_[0-9]+]](get_compile_time_arg_val([[C1_ACK_INDEX:[0-9]+]]))
# SPECIALIZED-CPP-NEXT: CircularBuffer [[C1_REUSED:cb_ctarg_[0-9]+]](get_compile_time_arg_val([[C1_REUSED_INDEX:[0-9]+]]))
# SPECIALIZED-CPP: [[C1_REUSED]].reserve_back
# SPECIALIZED-CPP: [[C1_ACK]].wait_front
# SPECIALIZED-CPP: [[C1_REUSED]].reserve_back


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        tensor_shape = (32, 64)
        input_tensor = to_dram(torch.zeros(tensor_shape, dtype=torch.bfloat16), device)
        output_tensor = to_dram(torch.zeros(tensor_shape, dtype=torch.bfloat16), device)
        dfb_reuse_specialize_cores(input_tensor, output_tensor)
    finally:
        ttnn.close_device(device)
