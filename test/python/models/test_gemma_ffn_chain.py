# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Per-card FFN dense path as a dispatch chain vs torch reference.

post-attn norm -> residual -> pre-FFW norm -> gate/up GEMV -> swiglu ->
down GEMV -> post-FFW norm -> residual. All activations stay on device.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.ops.elementwise import make_add
from ttl.ops.gemv import make_gemv
from ttl.ops.rmsnorm import make_rmsnorm
from ttl.ops.swiglu import make_swiglu
from gemma4.layer import MLP_PAD, from_dev, row, to_dev

TILE = 32


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * w


def gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def test_ffn_chain():
    device = ttnn.open_device(device_id=0)
    try:
        _run(device)
    finally:
        ttnn.close_device(device)


def _run(device):
    torch.manual_seed(0)
    H, P, eps = 2816, MLP_PAD, 1e-6
    inter = 528  # 2112 / 4 per card, padded to P

    x = torch.randn(H) * 0.5
    attn = torch.randn(H) * 0.5
    g_postattn = 1 + torch.randn(H) * 0.1
    g_preffw = 1 + torch.randn(H) * 0.1
    g_postffw = 1 + torch.randn(H) * 0.1
    w_gate = torch.zeros(H, P)
    w_up = torch.zeros(H, P)
    w_down = torch.zeros(P, H)
    w_gate[:, :inter] = torch.randn(H, inter) * 0.02
    w_up[:, :inter] = torch.randn(H, inter) * 0.02
    w_down[:inter] = torch.randn(inter, H) * 0.02

    h = x + rms(attn, g_postattn, eps)
    hn = rms(h, g_preffw, eps)
    down = (gelu_tanh(hn @ w_gate) * (hn @ w_up)) @ w_down
    want = h + rms(down, g_postffw, eps)

    norm = make_rmsnorm(1, 1, H // TILE, 11, H, eps)
    add = make_add(1, 1, H // TILE, 11)
    gate = make_gemv(TILE, H, P, (9, 2), 2)
    down_proj = make_gemv(TILE, P, H, (11, 2), 4)
    swiglu = make_swiglu(1, 1, P // TILE, P // TILE)

    def buf(n):
        return to_dev(torch.zeros(TILE, n, dtype=torch.bfloat16), device)

    x_d, attn_d = row(x, H, device), row(attn, H, device)
    attn_n, h_d, hn_d, dn_d, out_d = buf(H), buf(H), buf(H), buf(H), buf(H)
    g_d, u_d, act_d, down_d = buf(P), buf(P), buf(P), buf(H)

    norm(attn_d, row(g_postattn, H, device), attn_n)
    add(x_d, attn_n, h_d)
    norm(h_d, row(g_preffw, H, device), hn_d)
    w_gate_d = to_dev(w_gate.to(torch.bfloat16), device)
    w_up_d = to_dev(w_up.to(torch.bfloat16), device)
    gate(hn_d, w_gate_d, g_d)
    gate(hn_d, w_up_d, u_d)
    swiglu(g_d, u_d, act_d)
    down_proj(act_d, to_dev(w_down.to(torch.bfloat16), device), down_d)
    norm(down_d, row(g_postffw, H, device), dn_d)
    add(h_d, dn_d, out_d)

    got = from_dev(out_d)[0]
    pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
    assert pcc > 0.98, f"ffn pcc {pcc}"
