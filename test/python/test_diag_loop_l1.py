# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin

# =============================================================================
# DIAGNOSTIC: Simple loop with L1 TensorAccessor
# =============================================================================
# Tests if loops work with proper CB synchronization.
# Uses L1 (not DRAM) to isolate loop behavior from DRAM issues.
# =============================================================================

import torch
from ttlang.d2m_api import *

NUM_ITERS = 2  # Small number of iterations


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_diag_loop_l1(lhs, rhs, out):
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)
    num_iters = NUM_ITERS

    @compute()
    def add_compute(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        for i in range(num_iters):
            l = lhs_cb.wait()
            r = rhs_cb.wait()
            o = out_cb.reserve()
            result = l + r
            o.store(result)
            lhs_cb.pop()
            rhs_cb.pop()
            out_cb.push()

    @datamovement()
    def dm_lhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        for i in range(num_iters):
            lhs_shard = lhs_cb.reserve()
            tx = dma(lhs_accessor[0, 0], lhs_shard)
            tx.wait()
            lhs_cb.push()

    @datamovement()
    def dm_rhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        for i in range(num_iters):
            rhs_shard = rhs_cb.reserve()
            tx = dma(rhs_accessor[0, 0], rhs_shard)
            tx.wait()
            rhs_cb.push()

    return Program(add_compute, dm_lhs, dm_rhs)(lhs, rhs, out)


lhs = torch.full((32, 32), 2.0)
rhs = torch.full((32, 32), 3.0)
out = torch.full((32, 32), -999.0)

print(f"=== DIAG: Loop L1 ({NUM_ITERS} iters) ===")
test_diag_loop_l1(lhs, rhs, out)
print(f"Result: {out[0,0].item()}")
print("DONE - did not hang")
