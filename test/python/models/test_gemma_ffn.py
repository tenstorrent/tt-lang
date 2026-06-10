# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Per-card dense MLP + routed experts vs an HF-semantics torch reference.

Runs all four card shards on one device and sums the partials (the host
all-reduce); reference computes the full unsharded layer FFN."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from gemma4.config import TILE, Gemma4Config
from gemma4.layer_test_scaffolding import FFN, from_dev, row


def gelu(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def test_ffn_vs_ref(device):
    torch.manual_seed(1)
    cfg = Gemma4Config()
    H, M, I, E = cfg.hidden, cfg.mlp_inter, cfg.moe_inter, cfg.experts

    w = {
        "mlp.gate_proj": torch.randn(M, H) * 0.02,
        "mlp.up_proj": torch.randn(M, H) * 0.02,
        "mlp.down_proj": torch.randn(H, M) * 0.02,
        "experts.gate_up_proj": torch.randn(E, 2 * I, H) * 0.02,
        "experts.down_proj": torch.randn(E, H, I) * 0.02,
        "router.proj": torch.randn(E, H) * 0.02,
        "router.scale": torch.ones(H),
        "router.per_expert_scale": torch.ones(E),
    }
    cards = [FFN(cfg, w, device, c) for c in range(4)]

    x = torch.randn(H) * 0.5
    x_d = row(x, H, device)

    idx, wts = cards[0].route(x)
    dense = sum(from_dev(c.dense(x_d))[0] for c in cards)
    routed = sum(c.experts(x_d, idx, wts) for c in cards)

    want_dense = w["mlp.down_proj"] @ (gelu(w["mlp.gate_proj"] @ x) * (w["mlp.up_proj"] @ x))
    want_routed = torch.zeros(H)
    for ii, wt in zip(idx.tolist(), wts.tolist()):
        g, u = (w["experts.gate_up_proj"][ii] @ x).chunk(2)
        want_routed += wt * (w["experts.down_proj"][ii] @ (gelu(g) * u))

    for got, want, name in ((dense, want_dense, "dense"), (routed, want_routed, "routed")):
        pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
        assert pcc > 0.985, f"{name}: pcc {pcc}"
