# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""SUMMA matmul as a single @ttl.atom, composed from the ttl.ops.mcast op.

Each core (col_c, row_c) owns one BM x BN output block. A is multicast across
the NP cores of each row; B is multicast down the MP rows of each column.
Every core accumulates the full K range locally and writes its block out. The
broadcasts are the mcast op, inlined twice (A row-broadcast, B column-
broadcast), instead of hand-written if_src/if_dst callbacks.
"""

import pytest
import torch

import ttl
from ttl.ops.mcast import mcast, mcast_cols, mcast_rows

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = ttnn.TILE_SIZE
BM, BN, BK = 2, 2, 2

NP = 2  # N blocks / grid cols
MP = 2  # M blocks / grid rows

M = MP * BM * TILE  # 128
N = NP * BN * TILE  # 128
K_BLOCKS = 2
K = K_BLOCKS * BK * TILE  # 128

GRID_X = NP
GRID_Y = MP


@ttl.atom(grid=(GRID_X, GRID_Y))
def atom_summa_mcast(a, w, out):
    a_cb = ttl.make_dataflow_buffer_like(a, shape=(BM, BK), block_count=2)
    b_cb = ttl.make_dataflow_buffer_like(w, shape=(BK, BN), block_count=2)
    # Private staging buffers so the loopback src core does not double-reserve
    # a_cb/b_cb across the send (BRISC) and receive (NCRISC) sides.
    tmp_a_cb = ttl.make_dataflow_buffer_like(a, shape=(BM, BK), block_count=2)
    tmp_b_cb = ttl.make_dataflow_buffer_like(w, shape=(BK, BN), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(BM, BN), block_count=2)

    mcast_a_net = ttl.PipeNet(mcast_rows(MP, NP))
    mcast_b_net = ttl.PipeNet(mcast_cols(MP, NP))

    col_c, row_c = ttl.node(dims=2)
    mr = row_c * BM
    nc = col_c * BN

    p = out_cb.reserve()
    for _ in range(K_BLOCKS):
        a_blk = a_cb.wait()
        b_blk = b_cb.wait()
        p += a_blk @ b_blk

    for kb in range(K_BLOCKS):
        kc = kb * BK
        mcast(mcast_a_net, a[mr : mr + BM, kc : kc + BK], tmp_a_cb, a_cb)
        mcast(mcast_b_net, w[kc : kc + BK, nc : nc + BN], tmp_b_cb, b_cb)

    out_blk = out_cb.wait()
    ttl.copy(out_blk, out[mr : mr + BM, nc : nc + BN])


def test_atom_summa_mcast(device):
    a_torch = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    w_torch = torch.randn(K, N, dtype=torch.bfloat16) * 0.1
    expected = torch.matmul(a_torch.float(), w_torch.float()).to(torch.bfloat16)

    a_dram = to_dram(a_torch, device)
    w_dram = to_dram(w_torch, device)
    out_dram = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    atom_summa_mcast(a_dram, w_dram, out_dram)

    got = ttnn.to_torch(out_dram).reshape(M, N).to(torch.bfloat16)
    assert_pcc(expected, got, threshold=0.99)
