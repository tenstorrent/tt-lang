# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin

# =============================================================================
# DIAGNOSTIC: DRAM round-trip with explicit L1 staging buffer
# =============================================================================
# Tests DRAM -> L1 (staging) -> tilize -> L1 (output) data flow.
# Uses explicit L1 staging buffer for the CB backing.
# =============================================================================

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def test_diag_dram_dmonly(lhs, staging, out):
    # lhs is DRAM source, staging is L1 CB buffer, out is L1 output
    lhs_accessor = TensorAccessor(lhs, memory_space="DRAM")

    @datamovement()
    def dm_read(staging_cb: CircularBuffer, out_cb: CircularBuffer):
        # DMA from DRAM into L1 staging CB
        shard = staging_cb.reserve()
        tx = dma(lhs_accessor[0, 0], shard)
        tx.wait()
        staging_cb.push()

    @compute()
    def compute_kern(staging_cb: CircularBuffer, out_cb: CircularBuffer):
        # Wait for data in staging CB, tilize to output CB
        data = staging_cb.wait()
        out_tile = out_cb.reserve()
        tilize(data, out_tile)
        staging_cb.pop()
        out_cb.push()

    @datamovement()
    def dm_out(staging_cb: CircularBuffer, out_cb: CircularBuffer):
        # Consume output CB (CB state persists between programs)
        out_shard = out_cb.wait()
        out_cb.pop()

    return Program(compute_kern, dm_read, dm_out)(staging, out)


lhs = torch.full((32, 32), 7.0)
staging = torch.empty((32, 32))  # L1 staging buffer
out = torch.full((32, 32), -999.0)

print("=== DIAG: DRAM round-trip with explicit L1 staging ===")
test_diag_dram_dmonly(lhs, staging, out)
print(f"Input: 7.0, Output: {out[0,0].item()}")
if torch.allclose(out, lhs, rtol=1e-2, atol=1e-2):
    print("PASS: Data round-tripped correctly")
else:
    print(f"FAIL: Expected 7.0, got {out.min().item()} to {out.max().item()}")
print("DONE")
