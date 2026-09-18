# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Unified @ttl.operation with a tensor-less DFB built via ttl.make_dfb. The output
buffer is declared from a dtype name string rather than a borrowed
tensor; compute writes exp(x) into it and data movement drains it."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1


@ttl.operation(grid=(1, 1))
def atom_make_dfb_exp(inp, out):
    in_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)

    a_blk = in_cb.reserve()
    ttl.copy(inp[0:1, 0:1], a_blk)

    o = out_cb.reserve()
    x = in_cb.wait()
    o.store(ttl.exp(x))

    out_done = out_cb.wait()
    ttl.copy(out_done, out[0:1, 0:1])


def test_atom_make_dfb_exp(device):
    tile = ttnn.TILE_SIZE
    inp_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)
    expected = torch.exp(inp_t.float()).to(torch.bfloat16)

    inp = to_l1(inp_t, device)
    out = to_l1(torch.zeros(tile, tile, dtype=torch.bfloat16), device)

    atom_make_dfb_exp(inp, out)

    got = ttnn.to_torch(out).reshape(tile, tile).to(torch.bfloat16)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)
