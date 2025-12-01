# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin

# =============================================================================
# DIAGNOSTIC: DRAM round-trip with tilize/untilize (no compute, no loop)
# =============================================================================
# Tests DRAM -> L1 -> DRAM data round-trip using tilize/untilize.
# No arithmetic, no loops - just verify data moves correctly.
# =============================================================================

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_diag_dram_dmonly(lhs, lhs_tiled, out):
    lhs_accessor = TensorAccessor(lhs, memory_space="DRAM")

    @compute()
    def passthrough(lhs_scalar_cb: CircularBuffer, lhs_tiled_cb: CircularBuffer, out_cb: CircularBuffer):
        # Wait for scalar data from DM
        l_scalar = lhs_scalar_cb.wait()

        # Tilize: scalar -> tiled
        l_tiled = lhs_tiled_cb.reserve()
        tilize(l_scalar, l_tiled)
        lhs_tiled_cb.push()
        lhs_scalar_cb.pop()

        # Pass tiled data to output (framework will untilize)
        l_tile = lhs_tiled_cb.wait()
        o = out_cb.reserve()
        o.store(l_tile)
        out_cb.push()
        lhs_tiled_cb.pop()

    @datamovement()
    def dm_read(lhs_scalar_cb: CircularBuffer, lhs_tiled_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_shard = lhs_scalar_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()
        lhs_scalar_cb.push()

    return Program(passthrough, dm_read)(lhs, lhs_tiled, out)


lhs = torch.full((32, 32), 7.0)
lhs_tiled = torch.zeros((32, 32))
out = torch.full((32, 32), -999.0)

print("=== DIAG: DRAM round-trip (tilize, no loop) ===")
test_diag_dram_dmonly(lhs, lhs_tiled, out)
print(f"Input: 7.0, Output: {out[0,0].item()}")
if torch.allclose(out, lhs, rtol=1e-2, atol=1e-2):
    print("PASS: Data round-tripped correctly")
else:
    print(f"FAIL: Expected 7.0, got {out.min().item()} to {out.max().item()}")
print("DONE")
