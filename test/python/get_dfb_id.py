# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Verify that ttl.get_dfb_id() in template_args emits ttl.get_dfb_id in the
initial MLIR and lowers to a raw integer literal in the generated C++.

The compute thread calls ttl.call_extern_func with DFB IDs and a literal
constant as template args. The same DFBs are direct function arguments so the
allocator can retain their storage dependencies. Stub data-movement threads
satisfy the TTNN interop 3-kernel requirement.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


FAKE_HEADER = "/dev/null/fake_shim.hpp"


@ttl.operation(grid=(1, 1))
def get_dfb_id_kernel(inp):
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    scratch = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        ttl.call_extern_func(
            FAKE_HEADER,
            "my_shim",
            template_args=[ttl.get_dfb_id(scratch), ttl.get_dfb_id(in_dfb), 1],
            func_args=[scratch, in_dfb],
        )

    @ttl.datamovement()
    def dm_read():
        blk = in_dfb.reserve()
        tx = ttl.copy(inp[0, 0], blk)
        tx.wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        blk = in_dfb.wait()
        blk.pop()


# =============================================================================
# Initial IR checks -- verify ttl.get_dfb_id SSA values feed opaque_call
# =============================================================================

# CHECK-LABEL: func.func @compute
# CHECK-DAG: %[[SCRATCH:.*]] = ttl.bind_cb{cb_index = 1
# CHECK-DAG: %[[IN_CB:.*]] = ttl.bind_cb{cb_index = 0

# get_dfb_id results flow as SSA values into opaque_call template_args
# CHECK: %[[ID_SCRATCH:.*]] = ttl.get_dfb_id %[[SCRATCH]] : <
# CHECK: %[[ID_IN:.*]] = ttl.get_dfb_id %[[IN_CB]] : <
# CHECK: %[[C1:.*]] = arith.constant 1 : i32

# The opaque_call receives template arg values as SSA operands
# CHECK: ttl.opaque_call "my_shim" template_args(%[[ID_SCRATCH]], %[[ID_IN]], %[[C1]]) (%[[SCRATCH]], %[[IN_CB]])

# =============================================================================
# C++ output checks -- verify raw integers in template args
# =============================================================================

# CHECK-CPP: === compute kernel written to {{.*}} ===
# CHECK-CPP: my_shim<[[SCRATCH_INDEX:[0-9]+]], [[IN_INDEX:[0-9]+]], 1>
# CHECK-CPP-SAME: (get_compile_time_arg_val([[SCRATCH_INDEX]]), get_compile_time_arg_val([[IN_INDEX]]))


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)

        get_dfb_id_kernel(inp)

    finally:
        ttnn.close_device(device)
