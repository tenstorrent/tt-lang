# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""@ttl.atom-in-@ttl.atom inlining. A small DFB-parameter helper atom is
inlined into a larger atom; the helper takes its DFBs as ttl.DFB
parameters (it has no way to be supplied buffers except by inlining).
Also checks that a callee declaring its own DFB is rejected at the outer
atom's decoration time."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1


@ttl.atom()
def _exp_block(inp: ttl.DFB, out: ttl.DFB):
    """Per-tile exp; declares no DFBs of its own (takes them as params)."""
    x = inp.wait()
    r = out.reserve()
    r.store(ttl.exp(x))


@ttl.atom(grid=(1, 1))
def atom_outer_exp(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    a_blk = a_cb.reserve()
    ttl.copy(in_t[0:1, 0:1], a_blk)

    _exp_block(a_cb, out_cb)  # inlined at decoration time

    out_done = out_cb.wait()
    ttl.copy(out_done, out_t[0:1, 0:1])


def test_callee_with_dfb_decl_rejected():
    """A callee that declares its own DFB cannot be inlined."""

    @ttl.atom()
    def _declares_dfb(inp: ttl.DFB, out: ttl.DFB):
        scratch = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        x = inp.wait()
        r = out.reserve()
        r.store(x)

    with pytest.raises(ValueError, match="make_dataflow_buffer_like"):

        @ttl.atom(grid=(1, 1))
        def _outer_bad(in_t, out_t):
            a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
            out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)
            _declares_dfb(a_cb, out_cb)


def test_callee_with_make_dfb_decl_rejected():
    """A callee that declares its own DFB via make_dfb cannot be inlined."""

    @ttl.atom()
    def _declares_make_dfb(inp: ttl.DFB, out: ttl.DFB):
        scratch = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        x = inp.wait()
        r = out.reserve()
        r.store(x)

    with pytest.raises(ValueError, match="make_dfb"):

        @ttl.atom(grid=(1, 1))
        def _outer_bad(in_t, out_t):
            a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
            out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)
            _declares_make_dfb(a_cb, out_cb)


def test_atom_outer_exp(device):
    tile = ttnn.TILE_SIZE
    inp_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)
    expected = torch.exp(inp_t.float()).to(torch.bfloat16)

    in_t = to_l1(inp_t, device)
    out_t = to_l1(torch.zeros(tile, tile, dtype=torch.bfloat16), device)

    atom_outer_exp(in_t, out_t)

    got = ttnn.to_torch(out_t).reshape(tile, tile).to(torch.bfloat16)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)
