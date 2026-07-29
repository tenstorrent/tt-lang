# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Verify direct DFB and ttnn-created semaphore template args in call_extern_func.

The frontend accepts DFBs directly in template_args, materializes them through
the opaque-call pipeline, and lowers DFB/semaphore values to integer template
literals.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


FAKE_HEADER = "/dev/null/fake_shim.hpp"


@ttl.operation(grid=(1, 1))
def extern_dfb_template_kernel(inp, global_sem):
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    scratch_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        ttl.call_extern_func(
            FAKE_HEADER,
            "my_shim",
            template_args=[scratch_dfb, in_dfb, 1, global_sem],
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


# CHECK-LABEL: func.func @compute
# CHECK-DAG: %[[SCRATCH:.*]] = ttl.bind_cb{cb_index = 1
# CHECK-DAG: %[[IN_CB:.*]] = ttl.bind_cb{cb_index = 0

# Frontend auto-detects DFB template args and materializes get_dfb_id.
# CHECK: %[[ID_SCRATCH:.*]] = ttl.get_dfb_id %[[SCRATCH]] : <
# CHECK: %[[ID_IN:.*]] = ttl.get_dfb_id %[[IN_CB]] : <
# CHECK: %[[C1:.*]] = arith.constant 1 : i32
# CHECK: %[[SEMAPHORE_ADDR:.*]] = arith.constant {{[0-9]+}} : i32
# CHECK: ttl.opaque_call "my_shim" template_args(%[[ID_SCRATCH]], %[[ID_IN]], %[[C1]], %[[SEMAPHORE_ADDR]])

# CHECK-CPP: === compute kernel written to {{.*}} ===
# CHECK-CPP: my_shim<1, 0, 1, {{[0-9]+}}>()


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
        core_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
        )
        global_sem = ttnn.create_global_semaphore(device, core_ranges, 0)
        extern_dfb_template_kernel(inp, global_sem)
    finally:
        ttnn.close_device(device)
