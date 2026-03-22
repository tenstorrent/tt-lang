# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-tile tests for reduce_sum and reduce_max.

Tests various tensor shapes (2x2, 1x4, 4x1) to verify reduce operations
work correctly across tile boundaries.

Hardware tile format note:
  Each 32x32 tile is internally 4 faces of 16x16. For REDUCE_ROW (dims=[1]),
  each face reduces independently and results land in column 0 of each face.
  After untilize, face 0 col 0 → row 0 elements 0-15 (valid), face 2 col 0 →
  rows 16-31 col 0 (NOT row 0 elements 16-31). So REDUCE_ROW only yields 16
  valid values per tile in row 0. REDUCE_COL (dims=[0]) yields all 32 values.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl


# =============================================================================
# Multi-tile reduce kernels: 2x2 tile grid (64x64 tensor)
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_dim0_2x2(inp, scaler, out):
    """Reduce-sum dims=[0] on 2x2 tile grid."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0:2, 0:2], inp_blk)
        tx.wait()
        inp_blk.push()
        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0:2, 0:2], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_blk.pop()


@ttl.kernel(grid=(1, 1))
def reduce_sum_dim1_2x2(inp, scaler, out):
    """Reduce-sum dims=[1] on 2x2 tile grid."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0:2, 0:2], inp_blk)
        tx.wait()
        inp_blk.push()
        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0:2, 0:2], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_blk.pop()


@ttl.kernel(grid=(1, 1))
def reduce_max_dim0_2x2(inp, scaler, out):
    """Reduce-max dims=[0] on 2x2 tile grid."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_max(i, s, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0:2, 0:2], inp_blk)
        tx.wait()
        inp_blk.push()
        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0:2, 0:2], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_blk.pop()


@ttl.kernel(grid=(1, 1))
def reduce_max_dim1_2x2(inp, scaler, out):
    """Reduce-max dims=[1] on 2x2 tile grid."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_max(i, s, o, dims=[1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0:2, 0:2], inp_blk)
        tx.wait()
        inp_blk.push()
        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0:2, 0:2], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_blk.pop()


# =============================================================================
# Wide tensor kernels: 1x4 tile grid (32x128 tensor)
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_dim0_1x4(inp, scaler, out):
    """Reduce-sum dims=[0] on 1x4 tile grid (32x128)."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 4), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 4), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 4), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0:1, 0:4], inp_blk)
        tx.wait()
        inp_blk.push()
        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0:1, 0:4], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:1, 0:4])
        tx.wait()
        out_blk.pop()


@ttl.kernel(grid=(1, 1))
def reduce_sum_dim1_1x4(inp, scaler, out):
    """Reduce-sum dims=[1] on 1x4 tile grid (32x128)."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 4), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 4), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 4), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0:1, 0:4], inp_blk)
        tx.wait()
        inp_blk.push()
        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0:1, 0:4], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:1, 0:4])
        tx.wait()
        out_blk.pop()


# =============================================================================
# Tall tensor kernels: 4x1 tile grid (128x32 tensor)
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_dim0_4x1(inp, scaler, out):
    """Reduce-sum dims=[0] on 4x1 tile grid (128x32)."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(4, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(4, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(4, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0:4, 0:1], inp_blk)
        tx.wait()
        inp_blk.push()
        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0:4, 0:1], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:4, 0:1])
        tx.wait()
        out_blk.pop()


@ttl.kernel(grid=(1, 1))
def reduce_sum_dim1_4x1(inp, scaler, out):
    """Reduce-sum dims=[1] on 4x1 tile grid (128x32)."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(4, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(4, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(4, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0:4, 0:1], inp_blk)
        tx.wait()
        inp_blk.push()
        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0:4, 0:1], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:4, 0:1])
        tx.wait()
        out_blk.pop()


# =============================================================================
# Tests
# =============================================================================


@pytest.fixture
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


class TestReduceSum2x2:
    """Reduce on 2x2 tile grid (64x64 tensor).

    This tests that reduce works across tile boundaries. Each tile is reduced
    independently - the result is a per-tile reduction, NOT a global reduction
    across the full 64x64 tensor.
    """

    def test_reduce_sum_dim0_2x2(self, device):
        """dims=[0]: column-wise sum within each tile."""
        torch.manual_seed(42)
        inp_torch = torch.randn((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_dim0_2x2(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Each tile is reduced independently. Check tile (0,0): rows 0-31, cols 0-31
        tile00_inp = inp_torch[0:32, 0:32]
        expected_tile00_row0 = tile00_inp.float().sum(dim=0).bfloat16()

        print("\n=== Reduce Sum dims=[0] 2x2 ===")
        print(f"Tile(0,0) expected row 0: {expected_tile00_row0[0:8].tolist()}")
        print(f"Tile(0,0) got row 0:      {result[0, 0:8].tolist()}")

        assert_allclose(
            result[0, 0:32].float(), expected_tile00_row0.float(),
            rtol=0.05, atol=0.5
        )

        # Check tile (0,1): rows 0-31, cols 32-63
        tile01_inp = inp_torch[0:32, 32:64]
        expected_tile01_row0 = tile01_inp.float().sum(dim=0).bfloat16()
        assert_allclose(
            result[0, 32:64].float(), expected_tile01_row0.float(),
            rtol=0.05, atol=0.5
        )

        # Check tile (1,0): rows 32-63, cols 0-31
        tile10_inp = inp_torch[32:64, 0:32]
        expected_tile10_row0 = tile10_inp.float().sum(dim=0).bfloat16()
        assert_allclose(
            result[32, 0:32].float(), expected_tile10_row0.float(),
            rtol=0.05, atol=0.5
        )

    def test_reduce_sum_dim1_2x2(self, device):
        """dims=[1]: row-wise sum within each tile.

        Due to tile face layout, only first 16 of 32 row reductions per tile
        are valid in row 0. Validate those.
        """
        torch.manual_seed(42)
        inp_torch = torch.randn((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_dim1_2x2(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Check tile (0,0): row-wise sums, first 16 valid
        tile00_inp = inp_torch[0:32, 0:32]
        expected_tile00 = tile00_inp.float().sum(dim=1).bfloat16()

        print("\n=== Reduce Sum dims=[1] 2x2 ===")
        print(f"Tile(0,0) expected first 16: {expected_tile00[0:8].tolist()}")
        print(f"Tile(0,0) got row 0[:16]:    {result[0, 0:8].tolist()}")

        assert_allclose(
            result[0, 0:16].float(), expected_tile00[:16].float(),
            rtol=0.1, atol=1.0
        )

        # Check tile (1,0): rows 32-63, row-wise sums
        tile10_inp = inp_torch[32:64, 0:32]
        expected_tile10 = tile10_inp.float().sum(dim=1).bfloat16()
        assert_allclose(
            result[32, 0:16].float(), expected_tile10[:16].float(),
            rtol=0.1, atol=1.0
        )


class TestReduceMax2x2:
    """Reduce max on 2x2 tile grid (64x64 tensor)."""

    def test_reduce_max_dim0_2x2(self, device):
        """dims=[0]: column-wise max within each tile."""
        torch.manual_seed(42)
        inp_torch = torch.randn((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_max_dim0_2x2(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Check all 4 tiles
        for tr in range(2):
            for tc in range(2):
                tile_inp = inp_torch[tr*32:(tr+1)*32, tc*32:(tc+1)*32]
                expected = tile_inp.float().max(dim=0).values.bfloat16()
                actual = result[tr*32, tc*32:(tc+1)*32]
                assert_allclose(
                    actual.float(), expected.float(),
                    rtol=0.05, atol=0.1
                )

    def test_reduce_max_dim1_2x2(self, device):
        """dims=[1]: row-wise max within each tile (first 16 valid)."""
        torch.manual_seed(42)
        inp_torch = torch.randn((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_max_dim1_2x2(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Check tile (0,0) and (1,0) - first 16 values valid per tile
        for tr in range(2):
            tile_inp = inp_torch[tr*32:(tr+1)*32, 0:32]
            expected = tile_inp.float().max(dim=1).values.bfloat16()
            actual = result[tr*32, 0:16]
            assert_allclose(
                actual.float(), expected[:16].float(),
                rtol=0.05, atol=0.1
            )


class TestReduceWide:
    """Reduce on 1x4 tile grid (32x128 tensor).

    Tests that wide tensors (many column tiles) reduce correctly.
    This is the shape pattern used in RMSNorm/softmax where we reduce
    across the hidden dimension.
    """

    def test_reduce_sum_dim0_1x4(self, device):
        """dims=[0]: column-wise sum, 4 column tiles."""
        torch.manual_seed(42)
        inp_torch = torch.randn((32, 128), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 128), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 128), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_dim0_1x4(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Check each of the 4 tiles
        for tc in range(4):
            tile_inp = inp_torch[0:32, tc*32:(tc+1)*32]
            expected = tile_inp.float().sum(dim=0).bfloat16()
            actual = result[0, tc*32:(tc+1)*32]
            assert_allclose(
                actual.float(), expected.float(),
                rtol=0.05, atol=0.5
            )

    def test_reduce_sum_dim1_1x4(self, device):
        """dims=[1]: row-wise sum, 4 column tiles.

        Each tile independently reduces its 32 columns. The result for each
        tile is the per-row sum of that tile's 32 columns, placed in row 0
        (first 16 valid per tile due to face layout).
        """
        torch.manual_seed(42)
        inp_torch = torch.randn((32, 128), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 128), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 128), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_dim1_1x4(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Check tile 0 (cols 0-31): row-wise sums, first 16 valid
        tile0_inp = inp_torch[0:32, 0:32]
        expected_tile0 = tile0_inp.float().sum(dim=1).bfloat16()

        print("\n=== Reduce Sum dims=[1] 1x4 ===")
        print(f"Tile 0 expected[:8]: {expected_tile0[:8].tolist()}")
        print(f"Tile 0 got row0[:8]: {result[0, 0:8].tolist()}")

        assert_allclose(
            result[0, 0:16].float(), expected_tile0[:16].float(),
            rtol=0.1, atol=1.0
        )


class TestReduceTall:
    """Reduce on 4x1 tile grid (128x32 tensor).

    Tests that tall tensors (many row tiles) reduce correctly.
    """

    def test_reduce_sum_dim0_4x1(self, device):
        """dims=[0]: column-wise sum, 4 row tiles.

        Each tile independently reduces its 32 rows. Result in row 0 of
        each tile (all 32 columns valid for REDUCE_COL).
        """
        torch.manual_seed(42)
        inp_torch = torch.randn((128, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((128, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((128, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_dim0_4x1(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Check each of the 4 tiles
        for tr in range(4):
            tile_inp = inp_torch[tr*32:(tr+1)*32, 0:32]
            expected = tile_inp.float().sum(dim=0).bfloat16()
            actual = result[tr*32, 0:32]
            assert_allclose(
                actual.float(), expected.float(),
                rtol=0.05, atol=0.5
            )

    def test_reduce_sum_dim1_4x1(self, device):
        """dims=[1]: row-wise sum, 4 row tiles (first 16 valid per tile)."""
        torch.manual_seed(42)
        inp_torch = torch.randn((128, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((128, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((128, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_dim1_4x1(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Check each of the 4 tiles - first 16 row reductions valid
        for tr in range(4):
            tile_inp = inp_torch[tr*32:(tr+1)*32, 0:32]
            expected = tile_inp.float().sum(dim=1).bfloat16()
            actual = result[tr*32, 0:16]
            assert_allclose(
                actual.float(), expected[:16].float(),
                rtol=0.1, atol=1.0
            )


class TestReduceWithScaler:
    """Test multi-tile reduce with non-unit scaler (e.g., for mean computation)."""

    def test_reduce_sum_mean_2x2(self, device):
        """Compute mean via reduce_sum with 1/32 scaler on 2x2 tiles."""
        torch.manual_seed(42)
        inp_torch = torch.randn((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.full((64, 64), 1.0 / 32.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_dim0_2x2(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Check tile (0,0): should be column-wise mean
        tile00_inp = inp_torch[0:32, 0:32]
        expected = tile00_inp.float().mean(dim=0).bfloat16()

        print("\n=== Reduce Sum Mean 2x2 ===")
        print(f"Expected: {expected[0:8].tolist()}")
        print(f"Got:      {result[0, 0:8].tolist()}")

        assert_allclose(
            result[0, 0:32].float(), expected.float(),
            rtol=0.05, atol=0.1
        )
