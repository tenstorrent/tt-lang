# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""ttl.ops.indexed_gemv vs torch index_select matmul (the MoE gather)."""

import pytest
import torch

import ttl
from ttl.ops.indexed_gemv import make_indexed_gemv

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32


def run_indexed_gemv(device, E, K, N, ids, grid_cfg, bn):
    topk = len(ids)
    x_t = torch.randn(TILE, K, dtype=torch.bfloat16) * 0.1
    w_t = torch.randn(E * K, N, dtype=torch.bfloat16) * 0.1

    idx_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    idx_t[0, :topk] = torch.tensor(ids, dtype=torch.bfloat16)

    expected = torch.stack(
        [torch.matmul(x_t[:1].float(), w_t[e * K:(e + 1) * K].float())[0] for e in ids]
    ).to(torch.bfloat16)

    x_d = to_dram(x_t, device)
    idx_d = to_dram(idx_t, device)
    w_d = to_dram(w_t, device)
    out_d = to_dram(torch.zeros(topk * TILE, N, dtype=torch.bfloat16), device)

    make_indexed_gemv(E, K, N, topk, grid_cfg, bn)(x_d, idx_d, w_d, out_d)

    got = ttnn.to_torch(out_d).reshape(topk, TILE, N)[:, 0, :].to(torch.bfloat16)
    assert_pcc(expected, got, threshold=0.99)


@pytest.mark.parametrize(
    "E, K, N, ids, grid_cfg, bn",
    [
        (8, 256, 256, [2, 5, 0], (4, 2), 2),       # toy gather incl. id 0
        (32, 2816, 704, [7, 31, 0, 12], (11, 2), 2),  # Gemma per-card expert slice
    ],
)
def test_indexed_gemv(device, E, K, N, ids, grid_cfg, bn):
    run_indexed_gemv(device, E, K, N, ids, grid_cfg, bn)
