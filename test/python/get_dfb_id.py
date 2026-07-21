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
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


FAKE_HEADER = "/dev/null/fake_shim.hpp"


@ttl.operation(grid=(1, 1))
def get_dfb_id_kernel(inp, out):
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    scratch = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.datamovement()
    def dm():
        blk = in_dfb.reserve()
        tx = ttl.copy(inp[0, 0], blk)
        tx.wait()
        blk.push()

        ttl.call_extern_func(
            FAKE_HEADER,
            "my_shim",
            template_args=[ttl.get_dfb_id(scratch), ttl.get_dfb_id(out_dfb), 1],
            include_paths=[],
        )

        out_blk = out_dfb.wait()
        tx2 = ttl.copy(out_blk, out[0, 0])
        tx2.wait()
        out_blk.pop()


# =============================================================================
# Initial IR checks -- verify ttl.get_dfb_id SSA values feed opaque_call
# =============================================================================

# CHECK-LABEL: func.func @dm
# CHECK-DAG: %[[SCRATCH:.*]] = ttl.bind_cb{cb_index = 2
# CHECK-DAG: %[[OUT_CB:.*]] = ttl.bind_cb{cb_index = 1

# get_dfb_id results flow as SSA values into opaque_call template_args
# CHECK: %[[ID_SCRATCH:.*]] = ttl.get_dfb_id %[[SCRATCH]] : !ttl.cb<
# CHECK: %[[ID_OUT:.*]] = ttl.get_dfb_id %[[OUT_CB]] : !ttl.cb<
# CHECK: %[[C1:.*]] = arith.constant 1 : i32

# The opaque_call receives template arg values as SSA operands
# CHECK: ttl.opaque_call "my_shim" template_args(%[[ID_SCRATCH]], %[[ID_OUT]], %[[C1]])

# =============================================================================
# C++ output checks -- verify raw integers in template args
# =============================================================================

# CHECK-CPP: === dm kernel written to {{.*}} ===
# CHECK-CPP: my_shim<2, 1, 1>()


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        get_dfb_id_kernel(inp, out)

    finally:
        ttnn.close_device(device)
