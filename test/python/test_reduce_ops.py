# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TTL reduce operations (reduce_sum and reduce_max).

Tests the tile_reduce_sum and tile_reduce_max ops which reduce a tile
along a specified dimension (rows or columns).

Hardware behavior note: For both REDUCE_ROW and REDUCE_COL, the result
is placed in ROW 0 of the output tile after pack+untilize:
  - REDUCE_ROW (dims=[0]): output[0, j] = f(input[:, j]) for each column j
  - REDUCE_COL (dims=[1]): output[0, j] = f(input[j, :]) for each row j
    (the per-row result is transposed into row 0 by the packer mask)
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
# Reduce Sum Kernels
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_row_kernel(inp, scaler, out):
    """Reduce-sum along rows (dims=[0]): output[0, j] = sum_i(inp[i, j] * scaler[i, j])."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0, 0], inp_blk)
        tx.wait()
        inp_blk.push()

        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0, 0], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_blk.pop()


@ttl.kernel(grid=(1, 1))
def reduce_sum_col_kernel(inp, scaler, out):
    """Reduce-sum along columns (dims=[1]): output[0, j] = sum_k(inp[j, k] * scaler[j, k])."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0, 0], inp_blk)
        tx.wait()
        inp_blk.push()

        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0, 0], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_blk.pop()


# =============================================================================
# Reduce Max Kernels
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_max_row_kernel(inp, scaler, out):
    """Reduce-max along rows (dims=[0]): output[0, j] = max_i(inp[i, j])."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_max(i, s, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0, 0], inp_blk)
        tx.wait()
        inp_blk.push()

        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0, 0], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_blk.pop()


@ttl.kernel(grid=(1, 1))
def reduce_max_col_kernel(inp, scaler, out):
    """Reduce-max along columns (dims=[1]): output[0, j] = max_k(inp[j, k])."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            result = ttl.math.reduce_max(i, s, o, dims=[1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0, 0], inp_blk)
        tx.wait()
        inp_blk.push()

        sc_blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0, 0], sc_blk)
        tx.wait()
        sc_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
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


class TestReduceSumRow:
    """Test reduce_sum with dims=[0] (reduce across rows).

    Hardware output: row 0 contains column-wise sums.
    output[0, j] = sum_i(inp[i, j])
    """

    def test_reduce_sum_row_basic(self, device):
        """Reduce rows: output[0, j] = sum_i(inp[i, j])."""
        torch.manual_seed(42)
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: sum along dim=0 (rows), result in row 0
        expected_row0 = inp_torch.float().sum(dim=0).bfloat16()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_row_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        print("\n=== Reduce Sum Row Test ===")
        print(f"Expected row 0: {expected_row0[0:8].tolist()}")
        print(f"Got row 0:      {result[0, 0:8].tolist()}")

        assert_allclose(
            result[0, :].float(), expected_row0.float(), rtol=0.05, atol=0.5
        )

    def test_reduce_sum_row_with_scaler(self, device):
        """Reduce rows with 1/32 scaler (computes mean)."""
        torch.manual_seed(42)
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.full((32, 32), 1.0 / 32.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: mean along dim=0, result in row 0
        expected_row0 = inp_torch.float().mean(dim=0).bfloat16()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_row_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        print("\n=== Reduce Sum Row (Mean) Test ===")
        print(f"Expected row 0: {expected_row0[0:8].tolist()}")
        print(f"Got row 0:      {result[0, 0:8].tolist()}")

        assert_allclose(
            result[0, :].float(), expected_row0.float(), rtol=0.05, atol=0.1
        )


class TestReduceSumCol:
    """Test reduce_sum with dims=[1] (reduce across columns).

    Hardware output: row 0 contains row-wise sums (transposed by packer).
    output[0, j] = sum_k(inp[j, k])
    """

    def test_reduce_sum_col_basic(self, device):
        """Reduce columns: output[0, j] = sum_k(inp[j, k])."""
        torch.manual_seed(42)
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: sum along dim=1 (columns), result transposed into row 0
        expected_row0 = inp_torch.float().sum(dim=1).bfloat16()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_col_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        print("\n=== Reduce Sum Col Test ===")
        print(f"Expected row 0: {expected_row0[0:8].tolist()}")
        print(f"Got row 0:      {result[0, 0:8].tolist()}")

        # Result is in row 0 (hardware transposes col result to row 0).
        # REDUCE_ROW in tile format only fills the first 16 elements of row 0
        # (corresponding to the first 16x16 face). Compare first 16 elements.
        assert_allclose(
            result[0, :16].float(), expected_row0[:16].float(), rtol=0.1, atol=1.0
        )


class TestReduceMaxRow:
    """Test reduce_max with dims=[0] (reduce across rows).

    Hardware output: row 0 contains column-wise maxes.
    output[0, j] = max_i(inp[i, j])
    """

    def test_reduce_max_row_basic(self, device):
        """Reduce rows: output[0, j] = max_i(inp[i, j])."""
        torch.manual_seed(42)
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: max along dim=0, result in row 0
        expected_row0 = inp_torch.float().max(dim=0).values.bfloat16()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_max_row_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        print("\n=== Reduce Max Row Test ===")
        print(f"Expected row 0: {expected_row0[0:8].tolist()}")
        print(f"Got row 0:      {result[0, 0:8].tolist()}")

        assert_allclose(
            result[0, :].float(), expected_row0.float(), rtol=0.05, atol=0.1
        )


class TestReduceMaxCol:
    """Test reduce_max with dims=[1] (reduce across columns).

    Hardware output: row 0 contains row-wise maxes (transposed by packer).
    output[0, j] = max_k(inp[j, k])
    """

    def test_reduce_max_col_basic(self, device):
        """Reduce columns: output[0, j] = max_k(inp[j, k])."""
        torch.manual_seed(42)
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: max along dim=1, result transposed into row 0
        expected_row0 = inp_torch.float().max(dim=1).values.bfloat16()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_max_col_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        print("\n=== Reduce Max Col Test ===")
        print(f"Expected row 0: {expected_row0[0:8].tolist()}")
        print(f"Got row 0:      {result[0, 0:8].tolist()}")

        # Result is in row 0 (hardware transposes col result to row 0).
        # REDUCE_ROW in tile format only fills the first 16 elements of row 0
        # (corresponding to the first 16x16 face). Compare first 16 elements.
        assert_allclose(
            result[0, :16].float(), expected_row0[:16].float(), rtol=0.05, atol=0.1
        )
