# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for raw_element_read/write on f32 and bf16 tensors.

Covers six access patterns at both precisions:

  1. Element copy  -- read one position, write to another.
  2. Constant write -- write a literal float to an element position.
     For bf16 blocks the f32 literal is implicitly truncated.
  3. Pairwise sort (ogt)  -- compare two elements via float32_greater /
     bfloat16_greater and conditionally swap them. Extended with
     negative/mixed-sign test vectors.
  4. Conditional equality write (oeq) -- copy a row element-by-element
     and overwrite positions that match a reference value (KV-cache
     update pattern).
  5. Min-pair (olt) -- exercises the operand-swap path in
     LowerScalarCmpF via less-than comparison.
  6. Filter not-equal (one) -- replace zero-valued elements with a
     sentinel, exercising the arith.cmpf one predicate.

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
    """bf16 raw_element_write truncates an f32 literal and writes it.

    The DSL inserts arith.truncf (not round-to-nearest), so the result
    may differ from torch's bf16 cast by up to one ULP.
    """
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_constant_write_kernel(out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(3.14, abs=0.02)


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


# =============================================================================
# Pattern 5: Min-pair via olt  (exercises operand-swap path in LowerScalarCmpF)
# =============================================================================


@ttl.operation(grid=(1, 1))
def f32_min_pair_kernel(inp, out):
    """Find the minimum of elements [0,0] and [0,1] via less-than comparison."""
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
                if a < b:
                    ttl.raw_element_write(wblk, 0, 0, a)
                    ttl.raw_element_write(wblk, 0, 1, b)
                else:
                    ttl.raw_element_write(wblk, 0, 0, b)
                    ttl.raw_element_write(wblk, 0, 1, a)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


@ttl.operation(grid=(1, 1))
def bf16_min_pair_kernel(inp, out):
    """Find the minimum of bf16 elements [0,0] and [0,1] via less-than."""
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
                if a < b:
                    ttl.raw_element_write(wblk, 0, 0, a)
                    ttl.raw_element_write(wblk, 0, 1, b)
                else:
                    ttl.raw_element_write(wblk, 0, 0, b)
                    ttl.raw_element_write(wblk, 0, 1, a)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def test_f32_min_pair(device):
    """f32 olt correctly places the minimum at [0,0]."""
    inp_torch = _make_sort_pair_input(5.0, 2.0, torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_min_pair_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(2.0, abs=1e-5)
    assert result[0, 1].item() == pytest.approx(5.0, abs=1e-5)


def test_bf16_min_pair(device):
    """bf16 olt correctly places the minimum at [0,0]."""
    inp_torch = _make_sort_pair_input(5.0, 2.0, torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_min_pair_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(2.0, abs=1e-2)
    assert result[0, 1].item() == pytest.approx(5.0, abs=1e-2)


# =============================================================================
# Pattern 6: Filter not-equal  (exercises arith.cmpi ne / one predicate)
# =============================================================================


@ttl.operation(grid=(1, 1))
def f32_filter_ne_kernel(inp, out):
    """Replace zero-valued elements in row 0 with a sentinel (-1.0).

    Copies row 0 from input to output. Positions equal to zero are
    overwritten with -1.0. Exercises the arith.cmpf one (not-equal)
    comparison path.
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
                zero = 0.0
                sentinel = -1.0
                for c in range(8):
                    val = ttl.raw_element_read(rblk, 0, c)
                    if val != zero:
                        ttl.raw_element_write(wblk, 0, c, val)
                    else:
                        ttl.raw_element_write(wblk, 0, c, sentinel)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


@ttl.operation(grid=(1, 1))
def bf16_filter_ne_kernel(inp, out):
    """Replace zero-valued bf16 elements in row 0 with a sentinel.

    Reads the reference zero value from row 1 col 0 and the sentinel
    from row 1 col 1 to avoid f32/bf16 type mismatch with constants.
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
                zero_ref = ttl.raw_element_read(rblk, 1, 0)
                sentinel_ref = ttl.raw_element_read(rblk, 1, 1)
                for c in range(8):
                    val = ttl.raw_element_read(rblk, 0, c)
                    if val != zero_ref:
                        ttl.raw_element_write(wblk, 0, c, val)
                    else:
                        ttl.raw_element_write(wblk, 0, c, sentinel_ref)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def _make_filter_ne_input(dtype):
    """Build input: row 0 = [3, 0, 7, 0, 1, 0, 5, 2] + zeros.

    For bf16 variant: row 1 col 0 = 0.0 (reference zero),
    row 1 col 1 = -1.0 (sentinel value).
    """
    t = torch.zeros(32, 32, dtype=dtype)
    t[0, 0] = 3.0
    t[0, 2] = 7.0
    t[0, 4] = 1.0
    t[0, 6] = 5.0
    t[0, 7] = 2.0
    t[1, 1] = -1.0
    return t


def test_f32_filter_ne(device):
    """f32 not-equal replaces zero positions with sentinel."""
    inp_torch = _make_filter_ne_input(torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_filter_ne_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(3.0, abs=1e-5)
    assert result[0, 1].item() == pytest.approx(-1.0, abs=1e-5)
    assert result[0, 2].item() == pytest.approx(7.0, abs=1e-5)
    assert result[0, 3].item() == pytest.approx(-1.0, abs=1e-5)
    assert result[0, 4].item() == pytest.approx(1.0, abs=1e-5)
    assert result[0, 5].item() == pytest.approx(-1.0, abs=1e-5)
    assert result[0, 6].item() == pytest.approx(5.0, abs=1e-5)
    assert result[0, 7].item() == pytest.approx(2.0, abs=1e-5)


def test_bf16_filter_ne(device):
    """bf16 not-equal replaces zero positions with sentinel."""
    inp_torch = _make_filter_ne_input(torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_filter_ne_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(3.0, abs=1e-2)
    assert result[0, 1].item() == pytest.approx(-1.0, abs=1e-2)
    assert result[0, 2].item() == pytest.approx(7.0, abs=1e-2)
    assert result[0, 3].item() == pytest.approx(-1.0, abs=1e-2)
    assert result[0, 4].item() == pytest.approx(1.0, abs=1e-2)
    assert result[0, 5].item() == pytest.approx(-1.0, abs=1e-2)
    assert result[0, 6].item() == pytest.approx(5.0, abs=1e-2)
    assert result[0, 7].item() == pytest.approx(2.0, abs=1e-2)


# =============================================================================
# Pattern 3 extended: Negative/mixed-sign/zero test vectors for sort-pair (4c)
# =============================================================================


@pytest.mark.parametrize(
    "a_val,b_val,expect_first,expect_second",
    [
        (-3.0, -1.0, -3.0, -1.0),
        (-2.0, 4.0, -2.0, 4.0),
        (4.0, -2.0, -2.0, 4.0),
    ],
    ids=["both-negative", "mixed-neg-pos", "mixed-pos-neg"],
)
def test_f32_sort_pair_signed(device, a_val, b_val, expect_first, expect_second):
    """f32 sort-pair with negative and mixed-sign inputs."""
    inp_torch = _make_sort_pair_input(a_val, b_val, torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_sort_pair_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(expect_first, abs=1e-5)
    assert result[0, 1].item() == pytest.approx(expect_second, abs=1e-5)


@pytest.mark.parametrize(
    "a_val,b_val,expect_first,expect_second",
    [
        (-3.0, -1.0, -3.0, -1.0),
        (-2.0, 4.0, -2.0, 4.0),
        (4.0, -2.0, -2.0, 4.0),
    ],
    ids=["both-negative", "mixed-neg-pos", "mixed-pos-neg"],
)
def test_bf16_sort_pair_signed(device, a_val, b_val, expect_first, expect_second):
    """bf16 sort-pair with negative and mixed-sign inputs (sign-magnitude guard)."""
    inp_torch = _make_sort_pair_input(a_val, b_val, torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_sort_pair_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(expect_first, abs=1e-1)
    assert result[0, 1].item() == pytest.approx(expect_second, abs=1e-1)


# =============================================================================
# Pattern 7: Row-scan argmax
# =============================================================================


@ttl.operation(grid=(1, 1))
def f32_argmax_row_kernel(inp, out):
    """Scan 32 elements in row 0, write the maximum to output [0,0]."""
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
                max_val = ttl.raw_element_read(rblk, 0, 0)
                for c in range(1, 32):
                    val = ttl.raw_element_read(rblk, 0, c)
                    if val > max_val:
                        max_val = val
                ttl.raw_element_write(wblk, 0, 0, max_val)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


@ttl.operation(grid=(1, 1))
def bf16_argmax_row_kernel(inp, out):
    """Scan 32 bf16 elements in row 0, write the maximum to output [0,0]."""
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
                max_val = ttl.raw_element_read(rblk, 0, 0)
                for c in range(1, 32):
                    val = ttl.raw_element_read(rblk, 0, c)
                    if val > max_val:
                        max_val = val
                ttl.raw_element_write(wblk, 0, 0, max_val)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def _make_argmax_row_input(dtype):
    """Build input with mixed positive/negative values in row 0.

    Row 0: [-5, 3, -1, 8, -2, 0, 7, -4, 1, 6, ...zeros].
    Expected max = 8.0 at index 3.
    """
    t = torch.zeros(32, 32, dtype=dtype)
    row_vals = [-5.0, 3.0, -1.0, 8.0, -2.0, 0.0, 7.0, -4.0, 1.0, 6.0]
    for i, v in enumerate(row_vals):
        t[0, i] = v
    return t


@pytest.mark.xfail(reason="loop carry error")
def test_f32_argmax_row(device):
    """f32 row-scan argmax finds the maximum across mixed-sign values."""
    inp_torch = _make_argmax_row_input(torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_argmax_row_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(8.0, abs=1e-5)


@pytest.mark.xfail(reason="loop carry error")
def test_bf16_argmax_row(device):
    """bf16 row-scan argmax finds the maximum across mixed-sign values."""
    inp_torch = _make_argmax_row_input(torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_argmax_row_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(8.0, abs=1e-1)
