# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Focused test for ttl.ops.mcast: broadcast one tile down a column of cores.

The source core (0, 0) stages an input tile and multicasts it to every row;
each row drains its received block out. All output row-blocks must equal the
input tile. Exercises the mcast op (if_src send + if_dst receive) and the
INCLUDE_SRC loopback in isolation.
"""

import pytest
import torch

import ttl
from ttl.ops.mcast import mcast, mcast_cols

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram

TILE = ttnn.TILE_SIZE
NROWS = 4
BM, BK = 1, 1


@ttl.atom(grid=(1, NROWS))
def atom_bcast(inp, out):
    stage = ttl.make_dataflow_buffer_like(inp, shape=(BM, BK), block_count=2)
    recv = ttl.make_dataflow_buffer_like(inp, shape=(BM, BK), block_count=2)
    net = ttl.PipeNet(mcast_cols(NROWS, 1))

    _col, row = ttl.node(dims=2)

    mcast(net, inp[0:BM, 0:BK], stage, recv)

    blk = recv.wait()
    ttl.copy(blk, out[row * BM : row * BM + BM, 0:BK])


def test_mcast_broadcast(device):
    inp_t = torch.randn(BM * TILE, BK * TILE, dtype=torch.bfloat16)
    out_t = torch.zeros(NROWS * BM * TILE, BK * TILE, dtype=torch.bfloat16)

    inp = to_dram(inp_t, device)
    out = to_dram(out_t, device)

    atom_bcast(inp, out)

    got = ttnn.to_torch(out).reshape(NROWS * BM * TILE, BK * TILE).to(torch.bfloat16)
    for r in range(NROWS):
        assert_allclose(got[r * TILE : (r + 1) * TILE], inp_t, rtol=1e-2, atol=1e-2)
