# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""Compile a rank-4 reduction over a leading block dimension."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn

import ttl
from ttlang_test_utils import to_l1


@ttl.operation(grid=(1, 1))
def reduce_rank4_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 2, 1, 2), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1, 1, 2), block_count=2)

    @ttl.compute()
    def reduce_rank4_compute():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            result = ttl.math.reduce_sum(inp_block, dims=[1], shape=(1, 1, 1, 2))
            out_block.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as block:
            ttl.copy(inp[0:1, 0:2, 0:1, 0:2], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as block:
            ttl.copy(block, out[0:1, 0:1, 0:1, 0:2]).wait()


# CHECK-LABEL: func.func @reduce_rank4_compute
# CHECK: %[[INPUT:.*]] = ttl.cb_wait {{.*}} -> tensor<1x2x1x2x!ttcore.tile<32x32, bf16>>
# CHECK: %[[RESULT:.*]] = ttl.reduce {{.*}} [1] : {{.*}} -> tensor<1x1x1x2x!ttcore.tile<32x32, bf16>>
# CHECK: ttl.store %[[RESULT]]

# CHECK-CPP-LABEL: === reduce_rank4_compute kernel written to {{.*}} ===
# CHECK-CPP: void kernel_main()
# CHECK-CPP: fill_tile(
# CHECK-CPP: for (size_t
# CHECK-CPP: add_binary_tile(


device = ttnn.open_device(device_id=0)
try:
    inp = to_l1(torch.ones((1, 2, 32, 64), dtype=torch.bfloat16), device)
    out = to_l1(torch.zeros((1, 1, 32, 64), dtype=torch.bfloat16), device)
    reduce_rank4_kernel(inp, out)
finally:
    ttnn.close_device(device)
