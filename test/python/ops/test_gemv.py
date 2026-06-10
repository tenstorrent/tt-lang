# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Tests for ttl.ops.gemv.make_gemv against torch.matmul.

Shapes mirror decode projections: K is the hidden dim, N the projection
width, M one row tile. Cases cover one block per core and the N loop, plus
the Gemma per-card projection shapes."""

import pytest
import torch

import ttl
from ttl.ops.gemv import make_gemv

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32


def run_gemv(device, K, N, grid_cfg, bn):
    x_t = torch.randn(TILE, K, dtype=torch.bfloat16) * 0.1
    w_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.1
    expected = torch.matmul(x_t.float(), w_t.float()).to(torch.bfloat16)

    x_d = to_dram(x_t, device)
    w_d = to_dram(w_t, device)
    out_d = to_dram(torch.zeros(TILE, N, dtype=torch.bfloat16), device)

    make_gemv(TILE, K, N, grid_cfg, bn)(x_d, w_d, out_d)

    got = ttnn.to_torch(out_d).reshape(TILE, N).to(torch.bfloat16)
    assert_pcc(expected, got, threshold=0.99)


@pytest.mark.parametrize(
    "K, N, grid_cfg, bn",
    [
        # (K, N, (Np,Kp), bn) -- comment: blocks per core
        (256, 256, (4, 2), 2),    # 1 block/core
        (256, 1024, (4, 2), 2),   # 4 blocks/core (N loop)
    ],
)
def test_gemv(device, K, N, grid_cfg, bn):
    run_gemv(device, K, N, grid_cfg, bn)


@pytest.mark.parametrize(
    "K, N, grid_cfg, bn",
    [
        # Gemma 26B-A4B per-card decode projections (H=2816 = 88 tiles).
        (2816, 1536, (12, 2), 2),  # sliding QKV: 4Q + 2K + 2V heads x 256
        (1024, 2816, (11, 2), 4),  # O proj row-shard: [1, H/4] @ [H/4, H]
        (2816, 2816, (11, 2), 4),  # dense-MLP up at H wide
    ],
)
def test_gemv_gemma_shapes(device, K, N, grid_cfg, bn):
    run_gemv(device, K, N, grid_cfg, bn)
