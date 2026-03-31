# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul subblocking tests: verify correct results when matmul output exceeds
DST register capacity and is auto-subblocked.

Covers bf16 and f32 data types, various block shapes (square, non-square,
prime dimensions), fused patterns (matmul + bias, matmul + relu), and
boundary cases (exact DST fit).
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v --tb=short

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

import ttl

TILE = 32


# =============================================================================
# Kernel factories
# =============================================================================


def make_matmul_kernel(m_tiles, n_tiles, k_tiles=1, dtype=torch.bfloat16):
    """Create a matmul kernel with the given tile block dimensions."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out):
        a_dfb = ttl.make_dataflow_buffer_like(
            a, shape=(m_tiles, k_tiles), buffer_factor=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b, shape=(k_tiles, n_tiles), buffer_factor=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(m_tiles, n_tiles), buffer_factor=2
        )

        @ttl.compute()
        def compute_fn():
            with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                with out_dfb.reserve() as o:
                    o.store(a_blk @ b_blk)

        @ttl.datamovement()
        def dm_read():
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:m_tiles, 0:k_tiles], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[0:k_tiles, 0:n_tiles], blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:m_tiles, 0:n_tiles])
                tx.wait()

    return kernel


def make_matmul_bias_kernel(m_tiles, n_tiles):
    """Create a matmul + bias kernel: out = (a @ b) + c."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, c, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(m_tiles, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, n_tiles), buffer_factor=2)
        c_dfb = ttl.make_dataflow_buffer_like(
            c, shape=(m_tiles, n_tiles), buffer_factor=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(m_tiles, n_tiles), buffer_factor=2
        )

        @ttl.compute()
        def compute_fn():
            with (
                a_dfb.wait() as a_blk,
                b_dfb.wait() as b_blk,
                c_dfb.wait() as c_blk,
            ):
                with out_dfb.reserve() as o:
                    o.store((a_blk @ b_blk) + c_blk)

        @ttl.datamovement()
        def dm_read():
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:m_tiles, 0], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[0, 0:n_tiles], blk)
                tx.wait()
            with c_dfb.reserve() as blk:
                tx = ttl.copy(c[0:m_tiles, 0:n_tiles], blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:m_tiles, 0:n_tiles])
                tx.wait()

    return kernel


def make_matmul_relu_kernel(m_tiles, n_tiles):
    """Create a matmul + relu kernel: out = relu(a @ b)."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(m_tiles, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, n_tiles), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(m_tiles, n_tiles), buffer_factor=2
        )

        @ttl.compute()
        def compute_fn():
            with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                with out_dfb.reserve() as o:
                    o.store(ttl.math.relu(a_blk @ b_blk))

        @ttl.datamovement()
        def dm_read():
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:m_tiles, 0], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[0, 0:n_tiles], blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:m_tiles, 0:n_tiles])
                tx.wait()

    return kernel


# =============================================================================
# Test cases
# =============================================================================


class TestMatmulSubblock:
    """Matmul with output exceeding DST capacity (auto-subblocked)."""

    @pytest.mark.parametrize(
        "m,n,desc",
        [
            (3, 3, "square_prime"),
            (4, 4, "square_power2"),
            (3, 4, "non_square_MltN"),
            (4, 3, "non_square_MgtN"),
            (8, 8, "large_8x8"),
            (6, 12, "large_6x12"),
            (16, 4, "tall_16x4"),
            (4, 16, "wide_4x16"),
        ],
    )
    def test_bf16(self, device, m, n, desc):
        """bf16 matmul exceeding DST capacity of 8 tiles."""

        M, K, N = m * TILE, TILE, n * TILE
        a_t = torch.randn(M, K, dtype=torch.bfloat16)
        b_t = torch.randn(K, N, dtype=torch.bfloat16)
        expected = a_t @ b_t

        a = to_dram(a_t, device)
        b = to_dram(b_t, device)
        out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

        kernel = make_matmul_kernel(m, n)
        kernel(a, b, out)
        result = ttnn.to_torch(out)

        assert_pcc(expected, result)

    @pytest.mark.parametrize(
        "m,n,desc",
        [
            (2, 3, "exceeds_f32_cap4"),
            (3, 3, "square_prime_f32"),
            (3, 2, "non_square_f32"),
            (4, 4, "large_4x4_f32"),
            (8, 2, "tall_8x2_f32"),
            (2, 8, "wide_2x8_f32"),
        ],
    )
    def test_f32(self, device, m, n, desc):
        """f32 matmul exceeding DST capacity of 4 tiles."""

        M, K, N = m * TILE, TILE, n * TILE
        a_t = torch.randn(M, K, dtype=torch.float32)
        b_t = torch.randn(K, N, dtype=torch.float32)
        expected = a_t @ b_t

        a = to_dram(a_t, device)
        b = to_dram(b_t, device)
        out = to_dram(torch.zeros(M, N, dtype=torch.float32), device)

        kernel = make_matmul_kernel(m, n, dtype=torch.float32)
        kernel(a, b, out)
        result = ttnn.to_torch(out)

        assert_pcc(expected, result, threshold=0.999)


class TestMatmulSubblockExactFit:
    """Matmul that exactly fits DST (no subblocking needed)."""

    def test_2x4_bf16(self, device):
        """2x4=8 tiles = bf16 DST capacity. Must NOT subblock."""

        M, K, N = 2 * TILE, TILE, 4 * TILE
        a_t = torch.randn(M, K, dtype=torch.bfloat16)
        b_t = torch.randn(K, N, dtype=torch.bfloat16)
        expected = a_t @ b_t

        a = to_dram(a_t, device)
        b = to_dram(b_t, device)
        out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

        kernel = make_matmul_kernel(2, 4)
        kernel(a, b, out)
        result = ttnn.to_torch(out)

        assert_pcc(expected, result)

    def test_2x2_f32(self, device):
        """2x2=4 tiles = f32 DST capacity. Must NOT subblock."""

        M, K, N = 2 * TILE, TILE, 2 * TILE
        a_t = torch.randn(M, K, dtype=torch.float32)
        b_t = torch.randn(K, N, dtype=torch.float32)
        expected = a_t @ b_t

        a = to_dram(a_t, device)
        b = to_dram(b_t, device)
        out = to_dram(torch.zeros(M, N, dtype=torch.float32), device)

        kernel = make_matmul_kernel(2, 2, dtype=torch.float32)
        kernel(a, b, out)
        result = ttnn.to_torch(out)

        assert_pcc(expected, result, threshold=0.999)


class TestMatmulFusedSubblock:
    """Fused matmul patterns with output exceeding DST capacity."""

    def test_matmul_bias_3x3(self, device):
        """(a @ b) + c with 3x3=9 > 8 DST tiles."""

        M, K, N = 3 * TILE, TILE, 3 * TILE
        a_t = torch.randn(M, K, dtype=torch.bfloat16)
        b_t = torch.randn(K, N, dtype=torch.bfloat16)
        c_t = torch.randn(M, N, dtype=torch.bfloat16)
        expected = (a_t @ b_t) + c_t

        a = to_dram(a_t, device)
        b = to_dram(b_t, device)
        c = to_dram(c_t, device)
        out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

        kernel = make_matmul_bias_kernel(3, 3)
        kernel(a, b, c, out)
        result = ttnn.to_torch(out)

        assert_pcc(expected, result)

    def test_matmul_relu_3x3(self, device):
        """relu(a @ b) with 3x3=9 > 8 DST tiles."""

        M, K, N = 3 * TILE, TILE, 3 * TILE
        a_t = torch.randn(M, K, dtype=torch.bfloat16)
        b_t = torch.randn(K, N, dtype=torch.bfloat16)
        expected = torch.relu(a_t @ b_t)

        a = to_dram(a_t, device)
        b = to_dram(b_t, device)
        out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

        kernel = make_matmul_relu_kernel(3, 3)
        kernel(a, b, out)
        result = ttnn.to_torch(out)

        assert_pcc(expected, result)

    def test_matmul_bias_4x3_f32(self, device):
        """(a @ b) + c with 4x3=12 > 4 DST tiles (f32)."""

        M, K, N = 4 * TILE, TILE, 3 * TILE
        a_t = torch.randn(M, K, dtype=torch.float32)
        b_t = torch.randn(K, N, dtype=torch.float32)
        c_t = torch.randn(M, N, dtype=torch.float32)
        expected = (a_t @ b_t) + c_t

        a = to_dram(a_t, device)
        b = to_dram(b_t, device)
        c = to_dram(c_t, device)
        out = to_dram(torch.zeros(M, N, dtype=torch.float32), device)

        kernel = make_matmul_bias_kernel(4, 3)
        kernel(a, b, c, out)
        result = ttnn.to_torch(out)

        assert_pcc(expected, result, threshold=0.999)
