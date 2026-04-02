# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for broadcast + add pattern (FPU binary add with broadcast operand).

Tests tile_bcast col followed by tile_add in a single fused compute block.
The broadcast reads from CB and writes to DST, then tile_add operates on the
broadcast result (DST) and another CB input (via copy_tile).

Exercises DST subblocking at larger tile domains (4x4 exceeds DST capacity of
8 tiles for bf16).
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v --tb=short

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl


# =============================================================================
# Kernel definitions
# =============================================================================


@ttl.operation(grid=(1, 1))
def bcast_col_add_1x1_kernel(a, b, out):
    """Compute bcast_col(b) + a on a 1x1 tile."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with (
            b_dfb.wait() as b_tile,
            a_dfb.wait() as a_tile,
            out_dfb.reserve() as o,
        ):
            b_bcast = ttl.math.broadcast(b_tile, o, dims=[1])
            result = b_bcast + a_tile
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        a_blk = a_dfb.reserve()
        tx_a = ttl.copy(a[0, 0], a_blk)
        tx_a.wait()
        a_blk.push()

        b_blk = b_dfb.reserve()
        tx_b = ttl.copy(b[0, 0], b_blk)
        tx_b.wait()
        b_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_blk.pop()


@ttl.operation(grid=(1, 1))
def bcast_col_add_2x2_kernel(a, b, out):
    """Compute bcast_col(b) + a on a 2x2 tile grid."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(2, 2), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with (
            b_dfb.wait() as b_tile,
            a_dfb.wait() as a_tile,
            out_dfb.reserve() as o,
        ):
            b_bcast = ttl.math.broadcast(b_tile, o, dims=[1])
            result = b_bcast + a_tile
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        a_blk = a_dfb.reserve()
        tx_a = ttl.copy(a[0:2, 0:2], a_blk)
        tx_a.wait()
        a_blk.push()

        b_blk = b_dfb.reserve()
        tx_b = ttl.copy(b[0:2, 0:2], b_blk)
        tx_b.wait()
        b_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_blk.pop()


@ttl.operation(grid=(1, 1))
def bcast_col_add_4x4_kernel(a, b, out):
    """Compute bcast_col(b) + a on a 4x4 tile grid (requires DST subblocking)."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(4, 4), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(4, 4), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(4, 4), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with (
            b_dfb.wait() as b_tile,
            a_dfb.wait() as a_tile,
            out_dfb.reserve() as o,
        ):
            b_bcast = ttl.math.broadcast(b_tile, o, dims=[1])
            result = b_bcast + a_tile
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        a_blk = a_dfb.reserve()
        tx_a = ttl.copy(a[0:4, 0:4], a_blk)
        tx_a.wait()
        a_blk.push()

        b_blk = b_dfb.reserve()
        tx_b = ttl.copy(b[0:4, 0:4], b_blk)
        tx_b.wait()
        b_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:4, 0:4])
        tx.wait()
        out_blk.pop()


# =============================================================================
# Golden computation
# =============================================================================


def col_broadcast_add_golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute golden for tile_bcast col + tile_add.

    tile_bcast col broadcasts column 0 of each 32x32 tile across all 32
    columns of that tile. The result is then added element-wise with A.
    """
    result = a.clone()
    tile_h, tile_w = 32, 32
    rows, cols = a.shape
    for tr in range(0, rows, tile_h):
        for tc in range(0, cols, tile_w):
            b_col = b[tr : tr + tile_h, tc : tc + 1]  # (32, 1)
            b_bcast = b_col.expand(tile_h, tile_w)  # (32, 32)
            result[tr : tr + tile_h, tc : tc + tile_w] = (
                a[tr : tr + tile_h, tc : tc + tile_w] + b_bcast
            )
    return result


# =============================================================================
# Test cases
# =============================================================================


class TestBcastColAdd:
    """Test bcast_col(B) + A at various tile grid sizes."""

    def test_1x1(self, device):
        """Single tile: bcast_col(B) + A."""
        torch.manual_seed(42)
        a_torch = torch.rand((32, 32), dtype=torch.bfloat16) * 2 - 1
        b_torch = torch.rand((32, 32), dtype=torch.bfloat16) * 2 - 1
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        expected = col_broadcast_add_golden(a_torch, b_torch)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        out = to_l1(out_torch, device)

        bcast_col_add_1x1_kernel(a, b, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_2x2(self, device):
        """2x2 tile grid (4 tiles): bcast_col(B) + A."""
        torch.manual_seed(42)
        shape = (64, 64)
        a_torch = torch.rand(shape, dtype=torch.bfloat16) * 2 - 1
        b_torch = torch.rand(shape, dtype=torch.bfloat16) * 2 - 1
        out_torch = torch.zeros(shape, dtype=torch.bfloat16)
        expected = col_broadcast_add_golden(a_torch, b_torch)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        out = to_l1(out_torch, device)

        bcast_col_add_2x2_kernel(a, b, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_4x4(self, device):
        """4x4 tile grid (16 tiles): bcast_col(B) + A, requires DST subblocking."""
        torch.manual_seed(42)
        shape = (128, 128)
        a_torch = torch.rand(shape, dtype=torch.bfloat16) * 2 - 1
        b_torch = torch.rand(shape, dtype=torch.bfloat16) * 2 - 1
        out_torch = torch.zeros(shape, dtype=torch.bfloat16)
        expected = col_broadcast_add_golden(a_torch, b_torch)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        out = to_l1(out_torch, device)

        bcast_col_add_4x4_kernel(a, b, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)
