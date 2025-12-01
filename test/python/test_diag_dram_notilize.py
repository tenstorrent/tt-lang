# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin

# =============================================================================
# DIAGNOSTIC: DRAM TensorAccessor without tilize
# =============================================================================
# Tests if DRAM -> L1 DMA works. Data may be garbage (not tilized) but
# if it doesn't hang, the DRAM read path is functional.
# =============================================================================

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_diag_dram_notilize(lhs, rhs, out):
    # DRAM-backed accessors
    lhs_accessor = TensorAccessor(lhs, memory_space="DRAM")
    rhs_accessor = TensorAccessor(rhs, memory_space="DRAM")

    @compute()
    def add_compute(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r  # Data is scalar format, result will be garbage but shouldn't hang
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @datamovement()
    def dm_lhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()
        lhs_cb.push()

    @datamovement()
    def dm_rhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_accessor[0, 0], rhs_shard)
        tx.wait()
        rhs_cb.push()

    return Program(add_compute, dm_lhs, dm_rhs)(lhs, rhs, out)


lhs = torch.full((32, 32), 2.0)
rhs = torch.full((32, 32), 3.0)
out = torch.full((32, 32), -999.0)

print("=== DIAG: DRAM no tilize ===")
test_diag_dram_notilize(lhs, rhs, out)
print(f"Result: {out[0,0].item()} (data may be garbage, but didn't hang)")
print("DONE")
