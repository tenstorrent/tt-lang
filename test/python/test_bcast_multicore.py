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

Also tests varying CB granularity (1x1, 2x2) with ttl.math.broadcast op.
"""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram, to_l1

import ttl

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


# =============================================================================
# Granularity-parameterized broadcast tests
# =============================================================================


def make_bcast_granularity_kernel(granularity: int):
    """Factory to create broadcast kernels with different CB granularities.

    Tests ttl.math.broadcast with varying CB block sizes on multicore grid.
    Uses row broadcast (dims=[0]) pattern.
    """

    @ttl.kernel(grid=(8, 8))
    def bcast_granularity_kernel(inp, out):
        """Multicore broadcast kernel with parameterized granularity."""
        block_rows = granularity
        block_cols = granularity

        grid_x, grid_y = ttl.grid_size(dims=2)
        rows_per_core = inp.shape[0] // TILE_SIZE // grid_x // block_rows
        cols_per_core = inp.shape[1] // TILE_SIZE // grid_y // block_cols

        inp_cb = ttl.make_circular_buffer_like(
            inp, shape=(block_rows, block_cols), buffer_factor=2
        )
        out_cb = ttl.make_circular_buffer_like(
            out, shape=(block_rows, block_cols), buffer_factor=2
        )

        @ttl.compute()
        def compute_fn():
            for _ in range(rows_per_core):
                for _ in range(cols_per_core):
                    with inp_cb.wait() as i, out_cb.reserve() as o:
                        result = ttl.math.broadcast(i, o, dims=[0])
                        o.store(result)

        @ttl.datamovement()
        def dm_read():
            core_x, core_y = ttl.core(dims=2)
            for core_row in range(rows_per_core):
                row = core_x * rows_per_core + core_row
                start_row = row * block_rows
                end_row = (row + 1) * block_rows
                for core_col in range(cols_per_core):
                    col = core_y * cols_per_core + core_col
                    start_col = col * block_cols
                    end_col = (col + 1) * block_cols
                    with inp_cb.reserve() as blk:
                        tx = ttl.copy(inp[start_row:end_row, start_col:end_col], blk)
                        tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, core_y = ttl.core(dims=2)
            for core_row in range(rows_per_core):
                row = core_x * rows_per_core + core_row
                start_row = row * block_rows
                end_row = (row + 1) * block_rows
                for core_col in range(cols_per_core):
                    col = core_y * cols_per_core + core_col
                    start_col = col * block_cols
                    end_col = (col + 1) * block_cols
                    with out_cb.wait() as blk:
                        tx = ttl.copy(blk, out[start_row:end_row, start_col:end_col])
                        tx.wait()

    return bcast_granularity_kernel


def create_row_bcast_input_multitile(shape, value, dtype=torch.bfloat16):
    """Create tensor with first row of each tile filled (for row broadcast)."""
    t = torch.zeros(shape, dtype=dtype)
    rows, cols = shape
    for tile_row in range(0, rows, TILE_SIZE):
        t[tile_row, :] = value
    return t


def expected_row_bcast_result(shape, value, dtype=torch.bfloat16):
    """Expected result: all elements filled with broadcast value."""
    return torch.full(shape, value, dtype=dtype)


# Pre-create kernels for each granularity (avoids recompilation in parametrize)
_bcast_kernel_g1 = make_bcast_granularity_kernel(1)
_bcast_kernel_g2 = make_bcast_granularity_kernel(2)


@pytest.mark.parametrize(
    "granularity,kernel",
    [
        (1, _bcast_kernel_g1),
        (2, _bcast_kernel_g2),
    ],
    ids=["granularity_1x1", "granularity_2x2"],
)
def test_bcast_multicore_granularity(device, granularity, kernel):
    """Test ttl.math.broadcast with different CB granularities on 8x8 grid.

    Validates that broadcast works correctly with varying CB block sizes:
    - granularity=1: 1x1 tile blocks (baseline)
    - granularity=2: 2x2 tile blocks (4 tiles per block)

    Each test uses shape divisible by grid and granularity:
    - 8x8 grid * granularity * 32 tile_size
    """
    # Shape must be divisible by grid_size * granularity * tile_size
    shape_dim = GRID_ROWS * granularity * TILE_SIZE  # 256 for g=1, 512 for g=2
    shape = (shape_dim, shape_dim)

    value = 3.5
    inp_torch = create_row_bcast_input_multitile(shape, value)
    out_torch = torch.zeros(shape, dtype=torch.bfloat16)
    expected = expected_row_bcast_result(shape, value)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
