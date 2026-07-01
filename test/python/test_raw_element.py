# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for raw_element_read/write on f32 and bf16 tensors.

Covers seven access patterns:

  1. Element copy -- read one position, write to another
     (datamovement-only, compute is a no-op).
  2. Constant write -- write a literal float to an element position.
     For bf16 the f32 literal is implicitly truncated.
  3. Pairwise sort (ogt) -- compare two elements via greater-than and
     conditionally swap. Extended with negative/mixed-sign vectors (3b).
  4. Min-pair (olt) -- exercises the operand-swap path in
     LowerScalarCmpF via less-than comparison.
  5. Compute-then-read -- compute negates a tile and stores to a CB;
     the writer thread element_reads from the computed result.
  6. Write-then-compute -- the reader copies a tile and element_writes
     a modified value; compute negates the modified tile.
  7. Row-scan argmax -- scan 32 elements and write the maximum.
     Currently xfail (ISSUE #380). Includes a ttnn.max comparison.
"""

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
# Pattern 4: Min-pair via olt  (exercises operand-swap path in LowerScalarCmpF)
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
# Pattern 3b: Negative/mixed-sign/zero test vectors for sort-pair
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
# Pattern 5: Compute-then-read (compute does math, writer element_reads result)
# =============================================================================


@ttl.operation(grid=(1, 1))
def f32_compute_then_read_kernel(inp, out):
    """Negate a tile in compute, then element_read from the computed result."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    computed_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x, computed_dfb.reserve() as o:
            o.store(ttl.math.neg(x))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with computed_dfb.wait() as cblk:
            val = ttl.raw_element_read(cblk, 0, 5)
            with out_dfb.reserve() as wblk:
                ttl.raw_element_write(wblk, 0, 0, val)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


@ttl.operation(grid=(1, 1))
def bf16_compute_then_read_kernel(inp, out):
    """Negate a bf16 tile in compute, then element_read the result."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    computed_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x, computed_dfb.reserve() as o:
            o.store(ttl.math.neg(x))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with computed_dfb.wait() as cblk:
            val = ttl.raw_element_read(cblk, 0, 5)
            with out_dfb.reserve() as wblk:
                ttl.raw_element_write(wblk, 0, 0, val)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def test_f32_compute_then_read(device):
    """f32 element_read from a CB written by compute (neg)."""
    inp_torch = torch.randn(32, 32, dtype=torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_compute_then_read_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    expected = -inp_torch[0, 5].item()
    assert result[0, 0].item() == pytest.approx(expected, abs=1e-5)


def test_bf16_compute_then_read(device):
    """bf16 element_read from a CB written by compute (neg)."""
    inp_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_compute_then_read_kernel(inp, out)
    result = ttnn.to_torch(out)

    expected = -inp_torch[0, 5].item()
    assert result[0, 0].item() == pytest.approx(expected, abs=1e-2)


# =============================================================================
# Pattern 6: Write-then-compute (reader element_writes, compute does math)
# =============================================================================


@ttl.operation(grid=(1, 1))
def f32_write_then_compute_kernel(inp, out):
    """Reader copies a tile and overwrites [0,0] with 42.0; compute negates."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(ttl.math.neg(x))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()
            ttl.raw_element_write(blk, 0, 0, 42.0)

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


@ttl.operation(grid=(1, 1))
def bf16_write_then_compute_kernel(inp, out):
    """Reader copies a bf16 tile and overwrites [0,0] with 42.0; compute negates."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(ttl.math.neg(x))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()
            ttl.raw_element_write(blk, 0, 0, 42.0)

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_f32_write_then_compute(device):
    """f32 element_write in reader followed by compute neg."""
    inp_torch = torch.randn(32, 32, dtype=torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_write_then_compute_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    modified = inp_torch.clone()
    modified[0, 0] = 42.0
    expected = -modified

    assert result[0, 0].item() == pytest.approx(-42.0, abs=1e-5)
    assert result[0, 5].item() == pytest.approx(expected[0, 5].item(), abs=1e-5)


def test_bf16_write_then_compute(device):
    """bf16 element_write in reader followed by compute neg."""
    inp_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_write_then_compute_kernel(inp, out)
    result = ttnn.to_torch(out)

    modified = inp_torch.clone()
    modified[0, 0] = 42.0
    expected = -modified

    assert result[0, 0].item() == pytest.approx(-42.0, abs=1e-1)
    assert result[0, 5].item() == pytest.approx(expected[0, 5].item(), abs=1e-1)


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


@pytest.mark.xfail(reason="conditional assignment not exiting scope ISSUE #380")
def test_f32_argmax_row(device):
    """f32 row-scan argmax finds the maximum across mixed-sign values."""
    inp_torch = _make_argmax_row_input(torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_argmax_row_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(8.0, abs=1e-5)


@pytest.mark.xfail(reason="conditional assignment not exiting scope ISSUE #380")
def test_bf16_argmax_row(device):
    """bf16 row-scan argmax finds the maximum across mixed-sign values."""
    inp_torch = _make_argmax_row_input(torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_argmax_row_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert result[0, 0].item() == pytest.approx(8.0, abs=1e-1)


# =============================================================================
# Pattern 7b: ttnn.max comparison for row-scan argmax
# =============================================================================


@pytest.mark.xfail(reason="conditional assignment not exiting scope ISSUE #380")
def test_f32_argmax_vs_ttnn_max(device):
    """f32 row-scan max matches ttnn.max on the same input.

    The kernel scans row 0 for the maximum value.  Since all non-row-0
    elements are zero and the row-0 maximum (8.0) is positive, the
    global ttnn.max must agree.
    """
    inp_torch = _make_argmax_row_input(torch.float32)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    f32_argmax_row_kernel(inp, out)
    kernel_result = ttnn.to_torch(out).float()

    ttnn_max = ttnn.to_torch(ttnn.max(inp)).float()
    torch_max = inp_torch.max().item()

    assert kernel_result[0, 0].item() == pytest.approx(torch_max, abs=1e-5)
    assert ttnn_max.item() == pytest.approx(torch_max, abs=1e-5)


@pytest.mark.xfail(reason="conditional assignment not exiting scope ISSUE #380")
def test_bf16_argmax_vs_ttnn_max(device):
    """bf16 row-scan max matches ttnn.max on the same input."""
    inp_torch = _make_argmax_row_input(torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    bf16_argmax_row_kernel(inp, out)
    kernel_result = ttnn.to_torch(out)

    ttnn_max = ttnn.to_torch(ttnn.max(inp))
    torch_max = inp_torch.float().max().item()

    assert kernel_result[0, 0].item() == pytest.approx(torch_max, abs=1e-1)
    assert ttnn_max.float().item() == pytest.approx(torch_max, abs=1e-1)
