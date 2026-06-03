# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Tests for ttl.ops.topk against torch.topk.

``test_topk`` covers a small width; ``test_topk_routing`` runs the MoE
routing shape (top-8 of 256 experts) over many tokens spread across the
full device grid. Values are checked directly; indices are checked by
gathering the input at the returned indices (tolerant of value ties)."""

import pytest
import torch

import ttl
from ttl.ops.topk import make_topk

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = ttnn.TILE_SIZE


def run_topk(device, n_tokens, N, K):
    Wt = N // TILE
    Rt = n_tokens // TILE

    x_t = torch.randn(n_tokens, N, dtype=torch.bfloat16)
    ramp = torch.arange(N, dtype=torch.bfloat16).unsqueeze(0).repeat(TILE, 1)

    tv, ti = torch.topk(x_t.float(), K, dim=-1)

    x_d = to_dram(x_t, device)
    idx_d = to_dram(ramp, device)
    ov_d = to_dram(torch.zeros(n_tokens, K * TILE, dtype=torch.bfloat16), device)
    oi_d = to_dram(torch.zeros(n_tokens, K * TILE, dtype=torch.bfloat16), device)

    make_topk(Rt=Rt, PNt=1, Wt=Wt, K=K, N=N)(x_d, idx_d, ov_d, oi_d)

    got_v = ttnn.to_torch(ov_d).reshape(n_tokens, K * TILE)[:, 0::TILE][:, :K].float()
    got_i = ttnn.to_torch(oi_d).reshape(n_tokens, K * TILE)[:, 0::TILE][:, :K].float()

    assert_pcc(tv, got_v, threshold=0.99)
    # Indices are correct if the input gathered at them matches the top-k
    # values (robust to ties where a different index gives the same value).
    gathered = torch.gather(x_t.float(), -1, got_i.long().clamp(0, N - 1))
    assert_pcc(tv, gathered, threshold=0.99)


def test_topk(device):
    run_topk(device, n_tokens=TILE, N=64, K=4)


# DeepSeek-style MoE routing: top-8 of 256 experts, many tokens over the grid.
def test_topk_routing(device):
    run_topk(device, n_tokens=128 * TILE, N=256, K=8)
