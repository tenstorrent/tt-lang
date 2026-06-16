# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""End-to-end coverage for the ttl.exp hardware flags (approx / scale /
skip_clamp_check). Each flag combination compiles and runs a single-tile exp
atom on device and is compared against a torch reference. The approximation
flags only loosen the tolerance; ``scale`` changes the result to exp(scale*x).
"""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1


def _make_exp_atom(**exp_kwargs):
    @ttl.atom(grid=(1, 1))
    def _atom(in_t, out_t):
        a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

        a_blk = a_cb.reserve()
        ttl.copy(in_t[0:1, 0:1], a_blk)

        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.exp(x, **exp_kwargs))

        out_done = out_cb.wait()
        ttl.copy(out_done, out_t[0:1, 0:1])

    return _atom


def _run(atom, inp_t, device):
    tile = ttnn.TILE_SIZE
    in_t = to_l1(inp_t, device)
    out_t = to_l1(torch.zeros(tile, tile, dtype=torch.bfloat16), device)
    atom(in_t, out_t)
    return ttnn.to_torch(out_t).reshape(tile, tile).to(torch.bfloat16)


def test_exp_default(device):
    tile = ttnn.TILE_SIZE
    inp_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)
    expected = torch.exp(inp_t.float()).to(torch.bfloat16)
    got = _run(_make_exp_atom(), inp_t, device)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_exp_approx(device):
    tile = ttnn.TILE_SIZE
    inp_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)
    expected = torch.exp(inp_t.float()).to(torch.bfloat16)
    got = _run(_make_exp_atom(approx=True), inp_t, device)
    # Approximate mode: relax tolerance.
    assert_allclose(got, expected, rtol=5e-2, atol=5e-2)


def test_exp_approx_skip_clamp(device):
    tile = ttnn.TILE_SIZE
    inp_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)
    expected = torch.exp(inp_t.float()).to(torch.bfloat16)
    got = _run(_make_exp_atom(approx=True, skip_clamp_check=True), inp_t, device)
    assert_allclose(got, expected, rtol=5e-2, atol=5e-2)


def test_exp_scale(device):
    tile = ttnn.TILE_SIZE
    inp_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)
    scale = 2.0
    expected = torch.exp(scale * inp_t.float()).to(torch.bfloat16)
    got = _run(_make_exp_atom(scale=scale), inp_t, device)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)
