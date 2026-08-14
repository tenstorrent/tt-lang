# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression: an inline ``ttl.block.fill(k)`` operand under DST pressure.

The kernel computes ``out = tanh(a) * sigmoid(a) * ttl.block.fill(k)`` (k a
compile-time constant) and must match the golden. ``a`` has two consumers so
AssignDST inserts a copy_tile for the sigmoid subtree; the expression has to be
this deep for the inline fill to share that copy_tile's DST slot (shallower forms
like ``(a + b) * fill(k)`` don't exercise the hazard).

Parameterized over bf16 and f32: f32 halves the usable DST slots (8 -> 4), so the
allocation pressure that forces the fill to share a slot differs between them.
Both consumers of ``a`` are SFPU (tanh/sigmoid) so a single f32 input CB is legal;
an FPU+SFPU mix on one CB is rejected in f32.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram

K = 3.0

_DTYPE_PARAMS = [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float32, id="f32"),
]
_DTYPE_TOL = {
    torch.bfloat16: dict(rtol=5e-2, atol=1.0),
    torch.float32: dict(rtol=1e-3, atol=1e-3),
}


@ttl.operation(grid=(1, 1))
def fill_inline_kernel(a, out):
    """out = tanh(a) * sigmoid(a) * fill(K)  -- the regression."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av:
            with out_dfb.reserve() as o:
                core = ttl.math.tanh(av) * ttl.math.sigmoid(av)
                # Inline fill as an operand to the outer multiply -- clobbered.
                # Typing the fill from o covers both dtypes with one kernel: a
                # thread closure may only capture ints, floats, tensors and
                # buffers, so the dtype cannot be passed in as a parameter.
                o.store(core * ttl.block.fill(K, shape=o.shape, dtype=o.dtype))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as a_blk:
            ttl.copy(a[0, 0], a_blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def _golden(a, k):
    return torch.tanh(a.float()) * torch.sigmoid(a.float()) * float(k)


def _inputs(dtype):
    torch.manual_seed(0)
    return torch.randn(32, 32, dtype=dtype)


@pytest.mark.parametrize("dtype", _DTYPE_PARAMS)
def test_fill_inline_matches_golden(device, dtype):
    """The inline fill constant survives DST scheduling and reaches the mul."""
    a = _inputs(dtype)
    expected = _golden(a, K).to(dtype)

    a_t = to_dram(a, device)
    out_t = to_dram(torch.zeros(32, 32, dtype=dtype), device)

    fill_inline_kernel(a_t, out_t)

    result = ttnn.to_torch(out_t)
    assert_allclose(result.float(), expected.float(), **_DTYPE_TOL[dtype])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-rxX", "--tb=short"]))
