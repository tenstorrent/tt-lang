# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin

# =============================================================================
# DIAGNOSTIC: DRAM with just DM (no compute add)
# =============================================================================
# Tests DRAM -> L1 data movement only. Compute just passes data through.
# No arithmetic - just store what we get.
# =============================================================================

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def test_diag_dram_dmonly(lhs, out):
    lhs_accessor = TensorAccessor(lhs, memory_space="DRAM")

    @compute()
    def passthrough(lhs_cb: CircularBuffer, out_cb: CircularBuffer):
        # Just consume from CB - don't try to store (types won't match)
        l = lhs_cb.wait()
        lhs_cb.pop()
        # Output not used - just testing if DM works

    @datamovement()
    def dm_read(lhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()
        # NO push - matching working L1 pattern

    return Program(passthrough, dm_read)(lhs, out)


lhs = torch.full((32, 32), 7.0)
out = torch.full((32, 32), -999.0)

print("=== DIAG: DRAM DM only (no add) ===")
test_diag_dram_dmonly(lhs, out)
print(f"Result: {out[0,0].item()} (may be garbage)")
print("DONE")
