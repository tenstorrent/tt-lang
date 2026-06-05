# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for raw_element_read/write on f32 and bf16 tensors.

Covers four access patterns at both precisions:

  1. Element copy  -- read one position, write to another.
  2. Constant write -- write a literal float to an element position.
     For bf16 blocks the f32 literal is implicitly truncated.
  3. Pairwise sort  -- compare two elements via float32_greater /
     bfloat16_greater and conditionally swap them.
  4. Conditional equality write -- copy a row element-by-element and
     overwrite positions that match a reference value (KV-cache
     update pattern, exercises arith.cmpi eq lowering).

Each pattern has separate kernel definitions for f32 and bf16 because
the L1 pointer width (32-bit vs 16-bit) and comparison helpers differ.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_l1


# =============================================================================
# Pattern 1: Element copy  (raw_element_simple pattern)
# =============================================================================


@ttl.operation(grid=(1, 1))
def f32_element_copy_kernel(inp, out):
    """Copy f32 element [0,5] from input to output [0,0]."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            val = ttl.raw_element_read(rblk, 0, 5)
            with out_dfb.reserve() as wblk:
                ttl.raw_element_write(wblk, 0, 0, val)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


@ttl.operation(grid=(1, 1))
def bf16_element_copy_kernel(inp, out):
    """Copy bf16 element [0,5] from input to output [0,0]."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            val = ttl.raw_element_read(rblk, 0, 5)
            with out_dfb.reserve() as wblk:
                ttl.raw_element_write(wblk, 0, 0, val)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def test_f32_element_copy(device):
    """f32 raw_element_read/write round-trips a single element."""
    inp_torch = torch.randn(32, 32, dtype=torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_element_copy_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(inp_torch[0, 5].item(), abs=1e-5)


def test_bf16_element_copy(device):
    """bf16 raw_element_read/write round-trips a single element."""
    inp_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_element_copy_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(inp_torch[0, 5].item(), abs=1e-2)


# =============================================================================
# Pattern 2: Constant write  (raw_element_constants pattern)
# =============================================================================


@ttl.operation(grid=(1, 1))
def f32_constant_write_kernel(out):
    """Write a constant (3.14) to f32 output element [0,0]."""
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        val = 3.14
        with out_dfb.reserve() as wblk:
            ttl.raw_element_write(wblk, 0, 0, val)
            tx = ttl.copy(wblk, out[0, 0])
            tx.wait()


@ttl.operation(grid=(1, 1))
def bf16_constant_write_kernel(out):
    """Write a constant (3.14) to bf16 output element [0,0].

    The f32 literal is implicitly truncated to bf16 by the DSL.
    """
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        val = 3.14
        with out_dfb.reserve() as wblk:
            ttl.raw_element_write(wblk, 0, 0, val)
            tx = ttl.copy(wblk, out[0, 0])
            tx.wait()


def test_f32_constant_write(device):
    """f32 raw_element_write places a float constant in the output tile."""
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_constant_write_kernel(out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(3.14, abs=1e-5)


def test_bf16_constant_write(device):
    """bf16 raw_element_write truncates an f32 literal and writes it."""
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_constant_write_kernel(out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(3.14, abs=5e-2)


# =============================================================================
# Pattern 3: Pairwise sort  (raw_element_topk compare-and-swap pattern)
# =============================================================================


@ttl.operation(grid=(1, 1))
def f32_sort_pair_kernel(inp, out):
    """Sort f32 elements [0,0] and [0,1] via float32_greater."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            with out_dfb.reserve() as wblk:
                a = ttl.raw_element_read(rblk, 0, 0)
                b = ttl.raw_element_read(rblk, 0, 1)
                ttl.raw_element_write(wblk, 0, 0, a)
                ttl.raw_element_write(wblk, 0, 1, b)
                if a > b:
                    ttl.raw_element_write(wblk, 0, 0, b)
                    ttl.raw_element_write(wblk, 0, 1, a)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


@ttl.operation(grid=(1, 1))
def bf16_sort_pair_kernel(inp, out):
    """Sort bf16 elements [0,0] and [0,1] via bfloat16_greater."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            with out_dfb.reserve() as wblk:
                a = ttl.raw_element_read(rblk, 0, 0)
                b = ttl.raw_element_read(rblk, 0, 1)
                ttl.raw_element_write(wblk, 0, 0, a)
                ttl.raw_element_write(wblk, 0, 1, b)
                if a > b:
                    ttl.raw_element_write(wblk, 0, 0, b)
                    ttl.raw_element_write(wblk, 0, 1, a)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def _make_sort_pair_input(a_val, b_val, dtype):
    """Build a 32x32 tile with a_val at [0,0] and b_val at [0,1]."""
    t = torch.zeros(32, 32, dtype=dtype)
    t[0, 0] = a_val
    t[0, 1] = b_val
    return t


def test_f32_sort_pair_swap(device):
    """float32_greater correctly swaps when a > b."""
    inp_torch = _make_sort_pair_input(5.0, 2.0, torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_sort_pair_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(2.0, abs=1e-5)
    assert result[0, 1].item() == pytest.approx(5.0, abs=1e-5)


def test_f32_sort_pair_no_swap(device):
    """float32_greater does not swap when a <= b."""
    inp_torch = _make_sort_pair_input(2.0, 5.0, torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_sort_pair_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(2.0, abs=1e-5)
    assert result[0, 1].item() == pytest.approx(5.0, abs=1e-5)


def test_bf16_sort_pair_swap(device):
    """bfloat16_greater correctly swaps when a > b."""
    inp_torch = _make_sort_pair_input(5.0, 2.0, torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_sort_pair_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(2.0, abs=1e-2)
    assert result[0, 1].item() == pytest.approx(5.0, abs=1e-2)


def test_bf16_sort_pair_no_swap(device):
    """bfloat16_greater does not swap when a <= b."""
    inp_torch = _make_sort_pair_input(2.0, 5.0, torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_sort_pair_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(2.0, abs=1e-2)
    assert result[0, 1].item() == pytest.approx(5.0, abs=1e-2)


# =============================================================================
# Pattern 4: Conditional equality write  (raw_element_kv_cache pattern)
# =============================================================================


@ttl.operation(grid=(1, 1))
def f32_kv_cache_kernel(inp, out):
    """Copy row 0 from input to output, overwriting positions that match row 1 col 0.

    Row 0 is the cache row, row 1 col 0 is the reference value. When a
    cache element equals the reference, overwrite it with the reference
    (identity, but exercises the arith.cmpi eq comparison path).
    """
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            with out_dfb.reserve() as wblk:
                new_val = ttl.raw_element_read(rblk, 1, 0)

                for c in range(32):
                    cache_val = ttl.raw_element_read(rblk, 0, c)
                    ttl.raw_element_write(wblk, 0, c, cache_val)
                    if cache_val == new_val:
                        ttl.raw_element_write(wblk, 0, c, new_val)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


@ttl.operation(grid=(1, 1))
def bf16_kv_cache_kernel(inp, out):
    """Copy row 0 from input to output, overwriting positions that match row 1 col 0.

    Same as the f32 variant but for bf16 tensors.
    """
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            with out_dfb.reserve() as wblk:
                new_val = ttl.raw_element_read(rblk, 1, 0)

                for c in range(32):
                    cache_val = ttl.raw_element_read(rblk, 0, c)
                    ttl.raw_element_write(wblk, 0, c, cache_val)
                    if cache_val == new_val:
                        ttl.raw_element_write(wblk, 0, c, new_val)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def _make_kv_cache_input(dtype):
    """Build input tile: row 0 has [7, 3, 7, 5, ...0], row 1 col 0 has 7.

    The kernel should copy row 0 to output and overwrite positions
    where the value equals 7.0 (the reference from row 1 col 0). Since
    the overwrite is with the same value, output row 0 equals input row 0.
    """
    t = torch.zeros(32, 32, dtype=dtype)
    t[0, 0] = 7.0
    t[0, 1] = 3.0
    t[0, 2] = 7.0
    t[0, 3] = 5.0
    t[1, 0] = 7.0
    return t


def test_f32_kv_cache(device):
    """f32 conditional equality write copies row 0 faithfully."""
    inp_torch = _make_kv_cache_input(torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_kv_cache_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(7.0, abs=1e-5)
    assert result[0, 1].item() == pytest.approx(3.0, abs=1e-5)
    assert result[0, 2].item() == pytest.approx(7.0, abs=1e-5)
    assert result[0, 3].item() == pytest.approx(5.0, abs=1e-5)


def test_bf16_kv_cache(device):
    """bf16 conditional equality write copies row 0 faithfully."""
    inp_torch = _make_kv_cache_input(torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_kv_cache_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(7.0, abs=1e-2)
    assert result[0, 1].item() == pytest.approx(3.0, abs=1e-2)
    assert result[0, 2].item() == pytest.approx(7.0, abs=1e-2)
    assert result[0, 3].item() == pytest.approx(5.0, abs=1e-2)
