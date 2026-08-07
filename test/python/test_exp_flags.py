# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""End-to-end coverage for the ttl.math.exp hardware flags.

Flags must be passed as literal keyword arguments in the ``ttl.math.exp`` call
(the TTL AST compiler cannot resolve ``**kwargs`` splats), so each variant has
its own operation.
"""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

EXP_SCALE = 2.0


@ttl.operation(grid=(1, 1))
def exp_default_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


@ttl.operation(grid=(1, 1))
def exp_approx_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x, approx=True))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


@ttl.operation(grid=(1, 1))
def exp_approx_skip_clamp_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x, approx=True, skip_clamp_check=True))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


@ttl.operation(grid=(1, 1))
def exp_scale_literal_mul_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x * 2.0))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


@ttl.operation(grid=(1, 1))
def exp_scale_variable_mul_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x * EXP_SCALE))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


@ttl.operation(grid=(1, 1))
def exp_scale_keyword_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x, scale=EXP_SCALE))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


def _run(kernel, inp_t, device):
    tile = ttnn.TILE_SIZE
    in_t = to_l1(inp_t, device)
    out_t = to_l1(torch.zeros(tile, tile, dtype=torch.bfloat16), device)
    kernel(in_t, out_t)
    return ttnn.to_torch(out_t).reshape(tile, tile).to(torch.bfloat16)


def _rand_tile():
    tile = ttnn.TILE_SIZE
    return (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)


def test_exp_default(device):
    inp_t = _rand_tile()
    expected = torch.exp(inp_t.float()).to(torch.bfloat16)
    got = _run(exp_default_kernel, inp_t, device)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_exp_approx(device):
    inp_t = _rand_tile()
    expected = torch.exp(inp_t.float()).to(torch.bfloat16)
    got = _run(exp_approx_kernel, inp_t, device)
    # Approximate mode: relax tolerance.
    assert_allclose(got, expected, rtol=5e-2, atol=5e-2)


def test_exp_approx_skip_clamp(device):
    # approx + skip_clamp_check (InputClamping::None) is a speed-over-accuracy
    # mode: it skips the clamp the approximate SFPU exp normally relies on, so
    # the result is not close to the exact exp (it saturates). We only verify
    # the flags plumb through, compile, and run, producing finite positive
    # outputs (exp is always > 0) rather than asserting close numerics.
    inp_t = _rand_tile()
    got = _run(exp_approx_skip_clamp_kernel, inp_t, device)
    assert torch.isfinite(got).all()
    assert (got > 0).all()


def test_exp_scaled_by_literal_mul(device):
    inp_t = _rand_tile()
    expected = torch.exp(2.0 * inp_t.float()).to(torch.bfloat16)
    got = _run(exp_scale_literal_mul_kernel, inp_t, device)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_exp_scaled_by_variable_mul(device):
    inp_t = _rand_tile()
    expected = torch.exp(EXP_SCALE * inp_t.float()).to(torch.bfloat16)
    got = _run(exp_scale_variable_mul_kernel, inp_t, device)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_exp_scaled_by_keyword(device):
    inp_t = _rand_tile()
    expected = torch.exp(EXP_SCALE * inp_t.float()).to(torch.bfloat16)
    got = _run(exp_scale_keyword_kernel, inp_t, device)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)
