# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TTL matmul operations.

Tests the tile_matmul op which performs C += A * B at the tile level.
Specifically tests multi-tile accumulation in the K dimension which is
critical for correctness (DST must persist across K iterations).

Test cases:
1. Single-tile: [32x32] @ [32x32] = [32x32] (baseline)
2. K-accumulation: [32x64] @ [64x32] = [32x32] (2 tiles in K, must accumulate)
3. M-tiling: [64x32] @ [32x32] = [64x32] (2 output tiles in M)
4. N-tiling: [32x32] @ [32x64] = [32x64] (2 output tiles in N)
5. Full multi-tile: [64x64] @ [64x64] = [64x64] (all dimensions multi-tile)
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
# Single-tile matmul kernel: [1x1] @ [1x1] = [1x1] tiles
# =============================================================================


@ttl.kernel(grid=(1, 1))
def matmul_single_tile_kernel(a, b, c):
    """Single tile matmul: C += A * B where all are 1x1 tiles."""
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as a_tile, b_cb.wait() as b_tile, c_cb.reserve() as c_out:
            result = ttl.math.matmul(a_tile, b_tile, c_out)
            c_out.store(result)

    @ttl.datamovement()
    def dm_read():
        a_blk = a_cb.reserve()
        tx_a = ttl.copy(a[0, 0], a_blk)
        tx_a.wait()
        a_cb.push()

        b_blk = b_cb.reserve()
        tx_b = ttl.copy(b[0, 0], b_blk)
        tx_b.wait()
        b_cb.push()

    @ttl.datamovement()
    def dm_write():
        c_blk = c_cb.wait()
        tx = ttl.copy(c_blk, c[0, 0])
        tx.wait()
        c_cb.pop()


# =============================================================================
# K-dimension accumulation kernel: [1x2] @ [2x1] = [1x1] tiles
# This is the CRITICAL test for multi-tile correctness.
# The K loop iterates twice and DST must accumulate across iterations.
# =============================================================================


@ttl.kernel(grid=(1, 1))
def matmul_k_accum_kernel(a, b, c):
    """K-dimension accumulation: C[1,1] = A[1,2] @ B[2,1].

    A has 2 tiles in the K dimension (cols), B has 2 tiles in K dimension (rows).
    Output is a single tile that accumulates both A[:, k] * B[k, :] products.
    """
    # A: [M=1, K=2] tiles, B: [K=2, N=1] tiles, C: [M=1, N=1] tiles
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 2), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(2, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as a_tile, b_cb.wait() as b_tile, c_cb.reserve() as c_out:
            result = ttl.math.matmul(a_tile, b_tile, c_out)
            c_out.store(result)

    @ttl.datamovement()
    def dm_read():
        # Load all A tiles: [0,0] and [0,1]
        a_blk = a_cb.reserve()
        tx_a = ttl.copy(a[0:1, 0:2], a_blk)
        tx_a.wait()
        a_cb.push()

        # Load all B tiles: [0,0] and [1,0]
        b_blk = b_cb.reserve()
        tx_b = ttl.copy(b[0:2, 0:1], b_blk)
        tx_b.wait()
        b_cb.push()

    @ttl.datamovement()
    def dm_write():
        c_blk = c_cb.wait()
        tx = ttl.copy(c_blk, c[0, 0])
        tx.wait()
        c_cb.pop()


# =============================================================================
# M-dimension tiling kernel: [2x1] @ [1x1] = [2x1] tiles
# =============================================================================


@ttl.kernel(grid=(1, 1))
def matmul_m_tiling_kernel(a, b, c):
    """M-dimension tiling: C[2,1] = A[2,1] @ B[1,1].

    A has 2 tiles in M dimension (rows), output has 2 tiles in M.
    """
    a_cb = ttl.make_circular_buffer_like(a, shape=(2, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(2, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as a_tile, b_cb.wait() as b_tile, c_cb.reserve() as c_out:
            result = ttl.math.matmul(a_tile, b_tile, c_out)
            c_out.store(result)

    @ttl.datamovement()
    def dm_read():
        a_blk = a_cb.reserve()
        tx_a = ttl.copy(a[0:2, 0:1], a_blk)
        tx_a.wait()
        a_cb.push()

        b_blk = b_cb.reserve()
        tx_b = ttl.copy(b[0, 0], b_blk)
        tx_b.wait()
        b_cb.push()

    @ttl.datamovement()
    def dm_write():
        c_blk = c_cb.wait()
        tx = ttl.copy(c_blk, c[0:2, 0:1])
        tx.wait()
        c_cb.pop()


# =============================================================================
# N-dimension tiling kernel: [1x1] @ [1x2] = [1x2] tiles
# =============================================================================


@ttl.kernel(grid=(1, 1))
def matmul_n_tiling_kernel(a, b, c):
    """N-dimension tiling: C[1,2] = A[1,1] @ B[1,2].

    B has 2 tiles in N dimension (cols), output has 2 tiles in N.
    """
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 2), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as a_tile, b_cb.wait() as b_tile, c_cb.reserve() as c_out:
            result = ttl.math.matmul(a_tile, b_tile, c_out)
            c_out.store(result)

    @ttl.datamovement()
    def dm_read():
        a_blk = a_cb.reserve()
        tx_a = ttl.copy(a[0, 0], a_blk)
        tx_a.wait()
        a_cb.push()

        b_blk = b_cb.reserve()
        tx_b = ttl.copy(b[0:1, 0:2], b_blk)
        tx_b.wait()
        b_cb.push()

    @ttl.datamovement()
    def dm_write():
        c_blk = c_cb.wait()
        tx = ttl.copy(c_blk, c[0:1, 0:2])
        tx.wait()
        c_cb.pop()


# =============================================================================
# Full multi-tile kernel: [2x2] @ [2x2] = [2x2] tiles
# Tests all dimensions: M, N, K all have multiple tiles
# =============================================================================


@ttl.kernel(grid=(1, 1))
def matmul_full_multitile_kernel(a, b, c):
    """Full multi-tile: C[2,2] = A[2,2] @ B[2,2].

    All dimensions have 2 tiles. K-dimension requires accumulation.
    """
    a_cb = ttl.make_circular_buffer_like(a, shape=(2, 2), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(2, 2), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as a_tile, b_cb.wait() as b_tile, c_cb.reserve() as c_out:
            result = ttl.math.matmul(a_tile, b_tile, c_out)
            c_out.store(result)

    @ttl.datamovement()
    def dm_read():
        a_blk = a_cb.reserve()
        tx_a = ttl.copy(a[0:2, 0:2], a_blk)
        tx_a.wait()
        a_cb.push()

        b_blk = b_cb.reserve()
        tx_b = ttl.copy(b[0:2, 0:2], b_blk)
        tx_b.wait()
        b_cb.push()

    @ttl.datamovement()
    def dm_write():
        c_blk = c_cb.wait()
        tx = ttl.copy(c_blk, c[0:2, 0:2])
        tx.wait()
        c_cb.pop()


# =============================================================================
# Test Cases
# =============================================================================


class TestMatmulSingleTile:
    """Test single-tile matmul (baseline)."""

    def test_matmul_identity(self, device):
        """Test matmul with identity-like values: A=1, B=1 -> C=32."""
        # Each tile is 32x32. With A=all ones and B=all ones,
        # C[i,j] = sum_k(A[i,k] * B[k,j]) = 32 * 1 * 1 = 32
        a_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        b_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        c_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        expected = torch.full((32, 32), 32.0, dtype=torch.bfloat16)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)

        matmul_single_tile_kernel(a, b, c)
        result = ttnn.to_torch(c)

        print("\n=== Single Tile Matmul Test ===")
        print(f"Expected C[0,0] = 32.0, got {result[0, 0].item()}")

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-1)

    def test_matmul_scaled(self, device):
        """Test matmul with scaled values: A=2, B=0.5 -> C=32."""
        a_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        b_torch = torch.full((32, 32), 0.5, dtype=torch.bfloat16)
        c_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # C[i,j] = sum_k(2 * 0.5) = 32 * 1 = 32
        expected = torch.full((32, 32), 32.0, dtype=torch.bfloat16)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)

        matmul_single_tile_kernel(a, b, c)
        result = ttnn.to_torch(c)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-1)


class TestMatmulKAccumulation:
    """Test K-dimension accumulation - CRITICAL for multi-tile correctness.

    This tests the case where DST must persist across iterations of the
    K loop. If DST is cleared between iterations, the result will be wrong.
    """

    def test_k_accum_ones(self, device):
        """Test K-accumulation with all ones.

        A: [32, 64] (1x2 tiles), B: [64, 32] (2x1 tiles), C: [32, 32] (1x1 tile)
        With all ones: C[i,j] = sum over k of A[i,k] * B[k,j] = 64 * 1 = 64

        This is 2x the single-tile result because we have 2 tiles in K.
        """
        a_torch = torch.ones((32, 64), dtype=torch.bfloat16)
        b_torch = torch.ones((64, 32), dtype=torch.bfloat16)
        c_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: 64 (not 32!) because K has 2 tiles that must accumulate
        expected = torch.full((32, 32), 64.0, dtype=torch.bfloat16)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)

        matmul_k_accum_kernel(a, b, c)
        result = ttnn.to_torch(c)

        print("\n=== K-Accumulation Test (CRITICAL) ===")
        print(f"Expected C[0,0] = 64.0 (from 2 K-tiles), got {result[0, 0].item()}")
        print("If this shows 32.0, K-accumulation is broken!")

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-1)

    def test_k_accum_distinct_tiles(self, device):
        """Test K-accumulation with distinct values per K-tile.

        A[:, 0:32] = 1.0, A[:, 32:64] = 2.0  (K-tile 0 has 1s, K-tile 1 has 2s)
        B[0:32, :] = 1.0, B[32:64, :] = 1.0  (both K-tiles in B have 1s)

        C[i,j] = A[i, k0]*B[k0, j] + A[i, k1]*B[k1, j]
               = 32*(1*1) + 32*(2*1) = 32 + 64 = 96
        """
        a_torch = torch.zeros((32, 64), dtype=torch.bfloat16)
        a_torch[:, 0:32] = 1.0  # K-tile 0
        a_torch[:, 32:64] = 2.0  # K-tile 1

        b_torch = torch.ones((64, 32), dtype=torch.bfloat16)
        c_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: 32*1 + 32*2 = 96
        expected = torch.full((32, 32), 96.0, dtype=torch.bfloat16)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)

        matmul_k_accum_kernel(a, b, c)
        result = ttnn.to_torch(c)

        print("\n=== K-Accumulation Distinct Tiles Test ===")
        print(f"A K-tile 0 = 1.0, A K-tile 1 = 2.0, B = 1.0")
        print(f"Expected: 32*1 + 32*2 = 96, got {result[0, 0].item()}")

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-1)


class TestMatmulMTiling:
    """Test M-dimension tiling (multiple output rows)."""

    def test_m_tiling_ones(self, device):
        """Test M-tiling with all ones.

        A: [64, 32] (2x1 tiles), B: [32, 32] (1x1 tile), C: [64, 32] (2x1 tiles)
        Each output tile computes independently.
        """
        a_torch = torch.ones((64, 32), dtype=torch.bfloat16)
        b_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        c_torch = torch.zeros((64, 32), dtype=torch.bfloat16)

        # Each output tile: C[i,j] = 32 * 1 = 32
        expected = torch.full((64, 32), 32.0, dtype=torch.bfloat16)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)

        matmul_m_tiling_kernel(a, b, c)
        result = ttnn.to_torch(c)

        print("\n=== M-Tiling Test ===")
        print(f"Expected all 32.0, got C[0,0]={result[0, 0].item()}, C[32,0]={result[32, 0].item()}")

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-1)

    def test_m_tiling_distinct(self, device):
        """Test M-tiling with distinct values per M-tile.

        A[0:32, :] = 1.0, A[32:64, :] = 2.0
        B = 1.0

        C[0:32, :] = 32*1 = 32
        C[32:64, :] = 32*2 = 64
        """
        a_torch = torch.zeros((64, 32), dtype=torch.bfloat16)
        a_torch[0:32, :] = 1.0  # M-tile 0
        a_torch[32:64, :] = 2.0  # M-tile 1

        b_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        c_torch = torch.zeros((64, 32), dtype=torch.bfloat16)

        expected = torch.zeros((64, 32), dtype=torch.bfloat16)
        expected[0:32, :] = 32.0
        expected[32:64, :] = 64.0

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)

        matmul_m_tiling_kernel(a, b, c)
        result = ttnn.to_torch(c)

        print("\n=== M-Tiling Distinct Test ===")
        print(f"Expected C[0,0]=32, C[32,0]=64")
        print(f"Got C[0,0]={result[0, 0].item()}, C[32,0]={result[32, 0].item()}")

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-1)


class TestMatmulNTiling:
    """Test N-dimension tiling (multiple output columns)."""

    def test_n_tiling_ones(self, device):
        """Test N-tiling with all ones.

        A: [32, 32] (1x1 tile), B: [32, 64] (1x2 tiles), C: [32, 64] (1x2 tiles)
        """
        a_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        b_torch = torch.ones((32, 64), dtype=torch.bfloat16)
        c_torch = torch.zeros((32, 64), dtype=torch.bfloat16)

        expected = torch.full((32, 64), 32.0, dtype=torch.bfloat16)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)

        matmul_n_tiling_kernel(a, b, c)
        result = ttnn.to_torch(c)

        print("\n=== N-Tiling Test ===")
        print(f"Expected all 32.0, got C[0,0]={result[0, 0].item()}, C[0,32]={result[0, 32].item()}")

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-1)

    def test_n_tiling_distinct(self, device):
        """Test N-tiling with distinct values per N-tile.

        A = 1.0
        B[:, 0:32] = 1.0, B[:, 32:64] = 2.0

        C[:, 0:32] = 32*1 = 32
        C[:, 32:64] = 32*2 = 64
        """
        a_torch = torch.ones((32, 32), dtype=torch.bfloat16)

        b_torch = torch.zeros((32, 64), dtype=torch.bfloat16)
        b_torch[:, 0:32] = 1.0  # N-tile 0
        b_torch[:, 32:64] = 2.0  # N-tile 1

        c_torch = torch.zeros((32, 64), dtype=torch.bfloat16)

        expected = torch.zeros((32, 64), dtype=torch.bfloat16)
        expected[:, 0:32] = 32.0
        expected[:, 32:64] = 64.0

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)

        matmul_n_tiling_kernel(a, b, c)
        result = ttnn.to_torch(c)

        print("\n=== N-Tiling Distinct Test ===")
        print(f"Expected C[0,0]=32, C[0,32]=64")
        print(f"Got C[0,0]={result[0, 0].item()}, C[0,32]={result[0, 32].item()}")

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-1)


class TestMatmulFullMultitile:
    """Test full multi-tile matmul (M, N, K all have 2 tiles)."""

    def test_full_multitile_ones(self, device):
        """Test full multi-tile with all ones.

        A: [64, 64], B: [64, 64], C: [64, 64]
        All 2x2 tiles. K has 2 tiles so each output tile accumulates 2 products.

        C[i,j] = sum_k(A[i,k] * B[k,j]) = 64 * 1 = 64
        """
        a_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        b_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        c_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        # With K=2 tiles, each output = 64 (2 * 32)
        expected = torch.full((64, 64), 64.0, dtype=torch.bfloat16)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)

        matmul_full_multitile_kernel(a, b, c)
        result = ttnn.to_torch(c)

        print("\n=== Full Multi-tile Test ===")
        print(f"Expected all 64.0 (2 K-tiles accumulated)")
        print(f"Got C[0,0]={result[0, 0].item()}, C[32,32]={result[32, 32].item()}")

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-1)

    def test_full_multitile_pytorch_reference(self, device):
        """Test full multi-tile against PyTorch matmul reference.

        Uses random-ish values to catch any indexing bugs.
        """
        # Create tensors with distinguishable values per tile
        a_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
        b_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        # Fill A tiles with distinct values
        a_torch[0:32, 0:32] = 0.1   # A[0,0]
        a_torch[0:32, 32:64] = 0.2  # A[0,1]
        a_torch[32:64, 0:32] = 0.3  # A[1,0]
        a_torch[32:64, 32:64] = 0.4  # A[1,1]

        # Fill B tiles with distinct values
        b_torch[0:32, 0:32] = 1.0   # B[0,0]
        b_torch[0:32, 32:64] = 2.0  # B[0,1]
        b_torch[32:64, 0:32] = 3.0  # B[1,0]
        b_torch[32:64, 32:64] = 4.0  # B[1,1]

        c_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        # Compute expected with PyTorch
        expected = torch.matmul(a_torch.float(), b_torch.float()).to(torch.bfloat16)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)

        matmul_full_multitile_kernel(a, b, c)
        result = ttnn.to_torch(c)

        print("\n=== Full Multi-tile PyTorch Reference Test ===")
        print(f"A tiles: [0.1, 0.2; 0.3, 0.4], B tiles: [1, 2; 3, 4]")
        print(f"Expected C[0,0]={expected[0, 0].item():.2f}, got {result[0, 0].item():.2f}")
        print(f"Expected C[32,32]={expected[32, 32].item():.2f}, got {result[32, 32].item():.2f}")

        assert_allclose(result.float(), expected.float(), rtol=1e-1, atol=1.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
