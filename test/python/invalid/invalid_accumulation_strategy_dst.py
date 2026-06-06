# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: forced DST lowering rejects a tensor recurrence when the loop
needs three contribution tiles but the contribution DFB can hold only two.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl
from ttlang_test_utils import to_l1

TILE = 32
N_ITERS = 3


# CHECK: cannot lower tensor accumulation scope to DST
# CHECK: expected a DST-compatible same-type additive recurrence
# CHECK: select the automatic accumulation strategy or l1-pack
@ttl.operation(grid=(1, 1))
def invalid_dst_strategy_kernel(initial, delta, out):
    initial_dfb = ttl.make_dataflow_buffer_like(initial, shape=(1, 1), block_count=2)
    delta_dfb = ttl.make_dataflow_buffer_like(delta, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with initial_dfb.wait() as acc:
            for _ in range(N_ITERS):
                with delta_dfb.wait() as delta_blk:
                    acc = acc + delta_blk

            with out_dfb.reserve() as out_blk:
                out_blk.store(acc)

    @ttl.datamovement()
    def reader():
        with initial_dfb.reserve() as initial_blk:
            ttl.copy(initial[0:1, 0:1], initial_blk).wait()
        for _ in range(N_ITERS):
            with delta_dfb.reserve() as delta_blk:
                ttl.copy(delta[0:1, 0:1], delta_blk).wait()

    @ttl.datamovement()
    def writer():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0:1, 0:1]).wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        initial = to_l1(torch.full((TILE, TILE), -4.0, dtype=torch.bfloat16), device)
        delta = to_l1(torch.full((TILE, TILE), 1.0, dtype=torch.bfloat16), device)
        out = to_l1(torch.zeros((TILE, TILE), dtype=torch.bfloat16), device)
        invalid_dst_strategy_kernel(
            initial, delta, out, options="--ttl-accumulation-strategy=dst"
        )
    finally:
        ttnn.close_device(device)
