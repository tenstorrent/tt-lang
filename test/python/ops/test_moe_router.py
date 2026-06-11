# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Device router chain (norm -> proj -> softmax -> topk -> renorm*pe) vs torch."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.ops.gemv import make_gemv
from ttl.ops.moe_router import make_moe_weights, make_softmax_row
from ttl.ops.rmsnorm import make_rmsnorm
from ttl.ops.topk import make_topk

TILE, H, E, K = 32, 2816, 128, 8


def to_dev(t, device):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def rowt(t, n, device):
    z = torch.zeros(TILE, n, dtype=torch.bfloat16)
    z[0] = t.to(torch.bfloat16)
    return to_dev(z, device)


def test_router_chain(device):
    torch.manual_seed(0)
    eps, rscale = 1e-6, 1.0 * H ** -0.5
    x = torch.randn(H) * 0.5
    rw = torch.randn(E, H) * 0.1
    pe = 0.5 + torch.rand(E)

    xn = x / torch.sqrt(x.pow(2).mean() + eps) * rscale
    probs = torch.softmax(rw @ xn, dim=-1)
    wts, idx = torch.topk(probs, K)
    want = (wts / wts.sum()) * pe[idx]

    def buf(n):
        return to_dev(torch.zeros(TILE, n, dtype=torch.bfloat16), device)

    hr, rl, pr = buf(H), buf(E), buf(E)
    vals, idxs, w = buf(K * TILE), buf(K * TILE), buf(K * TILE)
    ramp = to_dev(torch.arange(E, dtype=torch.bfloat16).unsqueeze(0).repeat(TILE, 1), device)

    make_rmsnorm(1, 1, H // TILE, 11, H, eps)(
        rowt(x, H, device), rowt(torch.full((H,), rscale), H, device), hr)
    make_gemv(TILE, H, E, (1, 2), 1)(hr, to_dev(rw.T.contiguous().to(torch.bfloat16), device), rl)
    make_softmax_row(E // TILE)(rl, pr)
    make_topk(1, 1, E // TILE, K, E)(pr, ramp, vals, idxs)
    make_moe_weights(K, E // TILE)(vals, idxs, rowt(pe, E, device), buf(K * TILE), w)

    got_i = ttnn.to_torch(idxs).float()[0, 0::TILE][:K]
    got_w = ttnn.to_torch(w).float()[0, 0::TILE][:K]
    assert got_i.long().tolist() == idx.tolist()
    assert (got_w - want).abs().max() < 0.01
