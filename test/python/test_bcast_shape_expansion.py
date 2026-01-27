# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test bcast shape expansion: input CB smaller than output CB.

This tests the feature where bcast can accept mismatched input/output shapes
and automatically generates loops to broadcast tiles from the smaller input
to fill the larger output.

Examples:
  - col bcast: (2, 1) -> (2, 2) - broadcast column to fill 2 columns
  - row bcast: (1, 2) -> (2, 2) - broadcast row to fill 2 rows
  - scalar bcast: (1, 1) -> (2, 2) - broadcast scalar to fill all tiles
"""
import sys

import pytest
import torch

try:
    import ttnn
    import ttl

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")

TILE_SIZE = 32


def to_l1(t, device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def assert_allclose(actual, expected, rtol=1e-2, atol=1e-2):
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        mismatch = ~torch.isclose(actual, expected, rtol=rtol, atol=atol)
        mismatch_indices = torch.nonzero(mismatch)
        msg = f"Tensors not close. Mismatches: {mismatch_indices.shape[0]}\n"
        for idx in mismatch_indices[:5]:
            i, j = idx[0].item(), idx[1].item()
            msg += f"  [{i},{j}]: got {actual[i,j]:.4f}, expected {expected[i,j]:.4f}\n"
        raise AssertionError(msg)


# =============================================================================
# Kernels with shape expansion
# =============================================================================


@ttl.kernel(grid=(1, 1))
def bcast_col_expand_kernel(inp, out):
    """Col bcast with shape expansion: (2, 1) -> (2, 2)."""
    # Input CB is (2, 1) - one column of tiles
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 1), buffer_factor=2)
    # Output CB is (2, 2) - full tile grid
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, out_cb.reserve() as o:
            # bcast handles the (2,1) -> (2,2) expansion internally
            result = ttl.math.broadcast(i, o, dims=[1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            # Read only column 0 (2 tiles)
            tx = ttl.copy(inp[0:2, 0:1], inp_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            # Write full 2x2 grid
            tx = ttl.copy(out_blk, out[0:2, 0:2])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def bcast_row_expand_kernel(inp, out):
    """Row bcast with shape expansion: (1, 2) -> (2, 2)."""
    # Input CB is (1, 2) - one row of tiles
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 2), buffer_factor=2)
    # Output CB is (2, 2) - full tile grid
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, out_cb.reserve() as o:
            result = ttl.math.broadcast(i, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            # Read only row 0 (2 tiles)
            tx = ttl.copy(inp[0:1, 0:2], inp_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0:2, 0:2])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def bcast_scalar_expand_kernel(inp, out):
    """Scalar bcast with shape expansion: (1, 1) -> (2, 2)."""
    # Input CB is (1, 1) - single tile
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    # Output CB is (2, 2) - full tile grid
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, out_cb.reserve() as o:
            result = ttl.math.broadcast(i, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            # Read only tile (0,0)
            tx = ttl.copy(inp[0:1, 0:1], inp_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0:2, 0:2])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def mul_add_bcast_expand_kernel(a, b, c, out):
    """Compute (a * b) + bcast(c) with c having shape expansion.

    a, b: full (2, 2) tile grid
    c: column vector (2, 1) that gets col-broadcast to (2, 2)
    """
    a_cb = ttl.make_circular_buffer_like(a, shape=(2, 2), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(2, 2), buffer_factor=2)
    # c input is (2, 1) - will be expanded by bcast
    c_cb = ttl.make_circular_buffer_like(c, shape=(2, 1), buffer_factor=2)
    # c_bcast output is (2, 2)
    c_bcast_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        # Stage 1: Bcast c from (2,1) to (2,2)
        with c_cb.wait() as c_tile, c_bcast_cb.reserve() as c_out:
            c_bcast = ttl.math.broadcast(c_tile, c_out, dims=[1])
            c_out.store(c_bcast)

        # Stage 2: Compute (a * b) + c_bcast
        with (
            a_cb.wait() as a_tile,
            b_cb.wait() as b_tile,
            c_bcast_cb.wait() as c_bcast_tile,
            out_cb.reserve() as o,
        ):
            ab = a_tile * b_tile
            result = ab + c_bcast_tile
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as a_blk:
            tx_a = ttl.copy(a[0:2, 0:2], a_blk)
            tx_a.wait()

        with b_cb.reserve() as b_blk:
            tx_b = ttl.copy(b[0:2, 0:2], b_blk)
            tx_b.wait()

        # Only read column 0 for c
        with c_cb.reserve() as c_blk:
            tx_c = ttl.copy(c[0:2, 0:1], c_blk)
            tx_c.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0:2, 0:2])
            tx.wait()


# =============================================================================
# Test Cases
# =============================================================================


class TestBcastShapeExpansion:
    """Test bcast with shape expansion (smaller input -> larger output)."""

    def test_col_bcast_expand(self, device):
        """Test col bcast (2,1) -> (2,2): column broadcast with expansion."""
        # Input: 64x32 tensor (2 tiles in col, 1 tile in row)
        # Only column 0 of each tile has values
        inp_torch = torch.zeros((64, 32), dtype=torch.bfloat16)
        inp_torch[:, 0] = 5.0  # All rows, column 0 = 5

        # Output: 64x64 tensor (2x2 tiles)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        # Expected: all values should be 5.0 (col broadcast fills all columns)
        expected = torch.full((64, 64), 5.0, dtype=torch.bfloat16)

        print("\n=== Col Bcast Shape Expansion Test: (2,1) -> (2,2) ===")
        print("Input (64x32, col 0 = 5.0):")
        print(inp_torch)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_col_expand_kernel(inp, out)
        result = ttnn.to_torch(out)

        print("Output (64x64, should be all 5.0):")
        print(result)
        print(f"Unique values: {torch.unique(result).tolist()}")

        assert_allclose(result.float(), expected.float())

    def test_row_bcast_expand(self, device):
        """Test row bcast (1,2) -> (2,2): row broadcast with expansion."""
        # Input: 32x64 tensor (1 tile in row, 2 tiles in col)
        # Only row 0 has values
        inp_torch = torch.zeros((32, 64), dtype=torch.bfloat16)
        inp_torch[0, :] = 3.0  # Row 0 = 3

        # Output: 64x64 tensor (2x2 tiles)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        # Expected: all values should be 3.0 (row broadcast fills all rows)
        expected = torch.full((64, 64), 3.0, dtype=torch.bfloat16)

        print("\n=== Row Bcast Shape Expansion Test: (1,2) -> (2,2) ===")
        print("Input (32x64, row 0 = 3.0):")
        print(inp_torch)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_row_expand_kernel(inp, out)
        result = ttnn.to_torch(out)

        print("Output (64x64, should be all 3.0):")
        print(result)
        print(f"Unique values: {torch.unique(result).tolist()}")

        assert_allclose(result.float(), expected.float())

    def test_scalar_bcast_expand(self, device):
        """Test scalar bcast (1,1) -> (2,2): scalar broadcast with expansion."""
        # Input: 32x32 tensor (1 tile)
        # Only [0,0] has value
        inp_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        inp_torch[0, 0] = 7.0

        # Output: 64x64 tensor (2x2 tiles)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        # Expected: all values should be 7.0
        expected = torch.full((64, 64), 7.0, dtype=torch.bfloat16)

        print("\n=== Scalar Bcast Shape Expansion Test: (1,1) -> (2,2) ===")
        print("Input (32x32, [0,0] = 7.0):")
        print(inp_torch)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_scalar_expand_kernel(inp, out)
        result = ttnn.to_torch(out)

        print("Output (64x64, should be all 7.0):")
        print(result)
        print(f"Unique values: {torch.unique(result).tolist()}")

        assert_allclose(result.float(), expected.float())

    def test_mul_add_bcast_expand(self, device):
        """Test (a * b) + bcast(c) where c has shape expansion.

        a = 2.0 everywhere (2x2 tiles)
        b = 3.0 everywhere (2x2 tiles)
        c = 1.0 in column 0 (2x1 tiles, expanded to 2x2 via col bcast)

        Expected: (2 * 3) + 1 = 7.0 for all elements
        """
        a_torch = torch.full((64, 64), 2.0, dtype=torch.bfloat16)
        b_torch = torch.full((64, 64), 3.0, dtype=torch.bfloat16)

        # c: only column 0 has values (for col broadcast)
        c_torch = torch.zeros((64, 32), dtype=torch.bfloat16)
        c_torch[:, 0] = 1.0

        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
        expected = torch.full((64, 64), 7.0, dtype=torch.bfloat16)

        print("\n=== Mul Add Bcast Expand Test: (a * b) + bcast(c) ===")
        print("Input a (all 2.0):")
        print(a_torch)
        print("Input b (all 3.0):")
        print(b_torch)
        print("Input c (col 0 = 1.0, shape 64x32):")
        print(c_torch)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)
        out = to_l1(out_torch, device)

        mul_add_bcast_expand_kernel(a, b, c, out)
        result = ttnn.to_torch(out)

        print("Output (should be all 7.0):")
        print(result)
        print(
            f"Expected: all 7.0, Actual unique values: {torch.unique(result).tolist()}"
        )

        assert_allclose(result.float(), expected.float())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
