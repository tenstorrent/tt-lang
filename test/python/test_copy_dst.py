# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for copy_dst legalization when a DST intermediate is reused.

When a computed intermediate (DST register value) is referenced multiple times
in a single store expression, the compiler emits ttl.copy_dst to preserve
the value before a destructive unary. This test validates that copy_dst is
correctly lowered through the pipeline.

Reproduces: https://github.com/tenstorrent/tt-lang/issues/384
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram


@ttl.kernel(grid=(1, 1))
def dst_intermediate_reuse_kernel(a, b, out):
    """x = a*b (DST intermediate), then x * rsqrt(abs(x)).

    x is referenced twice: once for abs (in-place unary), once for the outer
    multiply. This forces copy_dst insertion for the intermediate.
    """
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

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


@pytest.mark.skip(
    reason="copy_dest_values.h not included in tt-mlir TTKernelToCpp.cpp (#384)"
)
def test_dst_intermediate_reuse(device):
    """Compile kernel where DST intermediate is reused (triggers copy_dst)."""
    a_t = torch.randn(32, 32, dtype=torch.bfloat16)
    b_t = torch.randn(32, 32, dtype=torch.bfloat16)
    out_t = torch.zeros(32, 32, dtype=torch.bfloat16)

    dst_intermediate_reuse_kernel(
        to_dram(a_t, device), to_dram(b_t, device), to_dram(out_t, device)
    )
