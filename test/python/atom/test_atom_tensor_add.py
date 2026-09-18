# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Unified @ttl.operation tensor add: a single body (no explicit thread
functions) that reads two ttnn tensors through DFBs, adds them, and
writes the result. Exercises the thread splitter end to end."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1


@ttl.operation(grid=(1, 1), math_fidelity="LoFi")
def atom_tensor_add(a, b, out):
    a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_cb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    a_blk = a_cb.reserve()
    ttl.copy(a[0:1, 0:1], a_blk)
    b_blk = b_cb.reserve()
    ttl.copy(b[0:1, 0:1], b_blk)

    s = out_cb.reserve()
    a_in = a_cb.wait()
    b_in = b_cb.wait()
    s.store(a_in + b_in)

    out_done = out_cb.wait()
    ttl.copy(out_done, out[0:1, 0:1])


def test_atom_tensor_add(device):
    tile = ttnn.TILE_SIZE
    a_t = torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5
    b_t = torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5
    expected = (a_t.float() + b_t.float()).to(torch.bfloat16)

    a = to_l1(a_t, device)
    b = to_l1(b_t, device)
    out = to_l1(torch.zeros(tile, tile, dtype=torch.bfloat16), device)

    atom_tensor_add(a, b, out)

    got = ttnn.to_torch(out).reshape(tile, tile).to(torch.bfloat16)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)
