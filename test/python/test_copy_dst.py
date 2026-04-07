# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for copy_dst legalization when a DST intermediate is reused.

When a computed intermediate (DST register value) is referenced multiple times
in a single store expression, the compiler emits ttl.copy_dst to preserve
the value before a destructive unary. These tests validate that copy_dst is
correctly lowered through the pipeline and produces numerically correct results.

Reproduces: https://github.com/tenstorrent/tt-lang/issues/384
Regression: https://github.com/tenstorrent/tt-lang/issues/443
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram


@ttl.operation(grid=(1, 1))
def dst_intermediate_reuse_kernel(a, b, out):
    """x = a*b (DST intermediate), then x * rsqrt(abs(x)).

    x is referenced twice: once for abs (in-place unary), once for the outer
    multiply. This forces copy_dst insertion for the intermediate.
    """
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
            x = av * bv  # x is a DST intermediate
            # x referenced twice: once for abs, once for outer multiply
            o.store(x * ttl.math.rsqrt(ttl.math.abs(x)))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_dst_intermediate_reuse(device):
    """Validate x * rsqrt(abs(x)) where x is a DST intermediate (a*b)."""
    # Use uniform [0.5, 2.0] to avoid zeros (rsqrt(0) = inf -> NaN).
    torch.manual_seed(42)
    a_t = torch.empty(32, 32, dtype=torch.bfloat16).uniform_(0.5, 2.0)
    b_t = torch.empty(32, 32, dtype=torch.bfloat16).uniform_(0.5, 2.0)
    out_t = torch.zeros(32, 32, dtype=torch.bfloat16)

    out_dev = to_dram(out_t, device)
    dst_intermediate_reuse_kernel(to_dram(a_t, device), to_dram(b_t, device), out_dev)

    x = a_t.float() * b_t.float()
    expected = x * torch.rsqrt(torch.abs(x))
    result = ttnn.to_torch(out_dev)
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@ttl.operation(grid=(1, 1))
def silu_on_intermediate_kernel(a, b, out):
    """SiLU on computed intermediate: y = a+b, then y * sigmoid(y).

    Regression test for https://github.com/tenstorrent/tt-lang/issues/443.
    The intermediate y lives in DST. sigmoid(y) is a destructive unary that
    clobbers its DST input. The compiler must emit copy_dst to preserve y
    for the outer multiply. With copy_dest_values args swapped, the copy
    goes the wrong direction and the multiply reads stale data.
    """
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
            y = av + bv
            o.store(y * ttl.math.sigmoid(y))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_silu_on_intermediate(device):
    """SiLU on computed intermediate must match torch reference (issue #443)."""
    a_t = torch.randn(32, 32, dtype=torch.bfloat16)
    b_t = torch.randn(32, 32, dtype=torch.bfloat16)
    out_t = torch.zeros(32, 32, dtype=torch.bfloat16)

    out_dev = to_dram(out_t, device)
    silu_on_intermediate_kernel(to_dram(a_t, device), to_dram(b_t, device), out_dev)

    y = a_t.float() + b_t.float()
    expected = y * torch.sigmoid(y)
    result = ttnn.to_torch(out_dev)
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)
