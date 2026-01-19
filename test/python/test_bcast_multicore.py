# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Broadcast multicore test: scoping-based tile reuse pattern.

Key feature: Different tensor sizes with scoping-based broadcast. The L1 tensor
(256x256) has 1 tile per core, while DRAM tensors (1024x1024) have 16 tiles per
core. The single L1 tile is loaded once and held in scope while iterating over
all 16 DRAM tiles, effectively "broadcasting" the L1 value.

Features:
- 2MB DRAM tensors (1024x1024): a, b, out1, out2 - 4x4 tiles per core
- 128KB L1 tensors (256x256): c, out3 - 1x1 tile per core
- 8x8 multicore grid with dynamic indexing via core(dims=2)
- 1x1 CB shapes for all CBs (shapes match in binary ops)
- L1 tile 'c' held in outer scope, reused across 16 DRAM iterations
- 20 fused ops across 3 outputs
"""

import pytest
import torch
import ttnn
from ttlang_test_utils import to_dram, to_l1

import ttl

pytestmark = pytest.mark.requires_ttnn

TILE_SIZE = 32
GRID_ROWS = 8
GRID_COLS = 8

# DRAM tensors: 4x4 tiles per core = 1024x1024 total
DRAM_CB_ROWS = 4
DRAM_CB_COLS = 4
DRAM_SHAPE = (
    GRID_ROWS * DRAM_CB_ROWS * TILE_SIZE,
    GRID_COLS * DRAM_CB_COLS * TILE_SIZE,
)

# L1 tensors: 1x1 tile per core = 256x256 total
L1_CB_ROWS = 1
L1_CB_COLS = 1
L1_SHAPE = (
    GRID_ROWS * L1_CB_ROWS * TILE_SIZE,
    GRID_COLS * L1_CB_COLS * TILE_SIZE,
)


@ttl.kernel(grid=(8, 8))
def bcast_kernel(a, b, c, out1, out2, out3):
    """
    Multicore kernel with 20 fused ops across 3 outputs.
    DRAM tensors (a, b, out1, out2): 4x4 tiles per core, processed 1x1 at a time
    L1 tensors (c, out3): 1x1 tile per core

    Key feature: L1 tile 'c' is held in scope and broadcast across all 16 DRAM
    iterations. This tests scoping-based broadcast with different tensor sizes.
    """
    # All CBs use 1x1 shape - broadcast achieved via scoping
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out1_cb = ttl.make_circular_buffer_like(out1, shape=(1, 1), buffer_factor=2)
    out2_cb = ttl.make_circular_buffer_like(out2, shape=(1, 1), buffer_factor=2)

    # L1 CBs: 1x1 tile
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out3_cb = ttl.make_circular_buffer_like(out3, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def fused_compute():
        # Wait for L1 tile once - it will be broadcast across all DRAM iterations
        with c_cb.wait() as cv:
            # Process DRAM tiles (4x4 = 16 iterations), broadcasting cv
            for _ in range(4):
                for _ in range(4):
                    with (
                        a_cb.wait() as av,
                        b_cb.wait() as bv,
                        out1_cb.reserve() as o1,
                        out2_cb.reserve() as o2,
                    ):
                        # out1 = f(a, b): 7 ops (pure DRAM)
                        v1 = ttl.math.sigmoid(av)  # 1
                        v1 = ttl.math.tanh(v1)  # 2
                        v1 = v1 + bv  # 3
                        v1 = ttl.math.sigmoid(v1)  # 4
                        v1 = ttl.math.tanh(v1)  # 5
                        v1 = ttl.math.abs(v1)  # 6
                        v1 = ttl.math.relu(v1)  # 7
                        o1.store(v1)

                        # out2 = g(b, c): 6 ops (DRAM + broadcast L1)
                        v2 = ttl.math.tanh(bv)  # 8
                        v2 = v2 + cv  # 9 - broadcast L1 tile to each DRAM tile
                        v2 = ttl.math.sigmoid(v2)  # 10
                        v2 = ttl.math.tanh(v2)  # 11
                        v2 = ttl.math.abs(v2)  # 12
                        v2 = ttl.math.sigmoid(v2)  # 13
                        o2.store(v2)

            # Process L1 output (cv still in scope from outer with)
            with out3_cb.reserve() as o3:
                # out3 = h(c): 7 ops (pure L1)
                v3 = ttl.math.relu(cv)  # 14
                v3 = ttl.math.sigmoid(v3)  # 15
                v3 = ttl.math.tanh(v3)  # 16
                v3 = ttl.math.abs(v3)  # 17
                v3 = ttl.math.sigmoid(v3)  # 18
                v3 = ttl.math.tanh(v3)  # 19
                v3 = ttl.math.relu(v3)  # 20
                o3.store(v3)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)

        # Read L1 tile FIRST (so it's available for broadcast)
        with c_cb.reserve() as c_blk:
            tx_c = ttl.copy(c[y, x], c_blk)
            tx_c.wait()

        # Read DRAM tiles (4x4 per core)
        for local_r in range(4):
            for local_c in range(4):
                with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                    dram_row = y * 4 + local_r
                    dram_col = x * 4 + local_c
                    tx_a = ttl.copy(a[dram_row, dram_col], a_blk)
                    tx_b = ttl.copy(b[dram_row, dram_col], b_blk)
                    tx_a.wait()
                    tx_b.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)

        # Write DRAM tiles (4x4 per core)
        for local_r in range(4):
            for local_c in range(4):
                with out1_cb.wait() as o1_blk, out2_cb.wait() as o2_blk:
                    dram_row = y * 4 + local_r
                    dram_col = x * 4 + local_c
                    tx1 = ttl.copy(o1_blk, out1[dram_row, dram_col])
                    tx2 = ttl.copy(o2_blk, out2[dram_row, dram_col])
                    tx1.wait()
                    tx2.wait()

        # Write L1 tile (1 per core)
        with out3_cb.wait() as o3_blk:
            tx3 = ttl.copy(o3_blk, out3[y, x])
            tx3.wait()

    return ttl.Program(fused_compute, dm_read, dm_write)(a, b, c, out1, out2, out3)


def compute_expected_dram(a, b, c):
    """Compute expected DRAM outputs using torch."""
    a_f, b_f, c_f = a.float(), b.float(), c.float()

    # out1 = f(a, b): 7 ops
    v1 = torch.sigmoid(a_f)
    v1 = torch.tanh(v1)
    v1 = v1 + b_f
    v1 = torch.sigmoid(v1)
    v1 = torch.tanh(v1)
    v1 = torch.abs(v1)
    exp1 = torch.relu(v1)

    # out2 = g(b, c): 6 ops with broadcast
    # c is 256x256, b is 1024x1024
    # Each c tile is broadcast to 4x4 region of b tiles
    # c[i,j] maps to b[i*4:(i+1)*4, j*4:(j+1)*4] in tile space
    # In element space: c element at [i,j] maps to b elements at [i*4:i*4+128, j*4:j*4+128]
    # Simpler: repeat c by 4x4 to match b's shape
    c_broadcast = c_f.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)
    v2 = torch.tanh(b_f)
    v2 = v2 + c_broadcast  # broadcast L1 tile to each DRAM tile
    v2 = torch.sigmoid(v2)
    v2 = torch.tanh(v2)
    v2 = torch.abs(v2)
    exp2 = torch.sigmoid(v2)

    return exp1, exp2


def compute_expected_l1(c):
    """Compute expected L1 output using torch."""
    c_f = c.float()

    # out3 = h(c): 7 ops
    v3 = torch.relu(c_f)
    v3 = torch.sigmoid(v3)
    v3 = torch.tanh(v3)
    v3 = torch.abs(v3)
    v3 = torch.sigmoid(v3)
    v3 = torch.tanh(v3)
    exp3 = torch.relu(v3)

    return exp3


def test_bcast_multicore(device):
    """Test scoping-based broadcast: L1 tile reused across DRAM iterations."""
    # Random DRAM inputs
    a_torch = torch.rand(DRAM_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    b_torch = torch.rand(DRAM_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    out1_torch = torch.zeros(DRAM_SHAPE, dtype=torch.bfloat16)
    out2_torch = torch.zeros(DRAM_SHAPE, dtype=torch.bfloat16)

    # Random L1 inputs
    c_torch = torch.rand(L1_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    out3_torch = torch.zeros(L1_SHAPE, dtype=torch.bfloat16)

    exp1, exp2 = compute_expected_dram(a_torch, b_torch, c_torch)
    exp3 = compute_expected_l1(c_torch)

    # DRAM tensors
    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out1 = to_dram(out1_torch, device)
    out2 = to_dram(out2_torch, device)

    # L1 tensors
    c = to_l1(c_torch, device)
    out3 = to_l1(out3_torch, device)

    bcast_kernel(a, b, c, out1, out2, out3)

    # Verify grid_size
    x_size, y_size = ttl.grid_size(dims=2)
    assert (x_size, y_size) == (GRID_COLS, GRID_ROWS)

    # Verify results
    result1 = ttnn.to_torch(out1)
    result2 = ttnn.to_torch(out2)
    result3 = ttnn.to_torch(out3)

    assert torch.allclose(result1.float(), exp1, rtol=0.05, atol=0.1)
    assert torch.allclose(result2.float(), exp2, rtol=0.05, atol=0.1)
    assert torch.allclose(result3.float(), exp3, rtol=0.05, atol=0.1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
