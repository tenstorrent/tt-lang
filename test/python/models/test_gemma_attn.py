# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Per-card sliding attention vs an HF-semantics torch reference."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from gemma4.config import TILE, Gemma4Config
from gemma4.layer import SlidingLayer, from_dev, row, to_dev


def rope_ref(x, pos, theta, dim):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    f = pos * inv
    cos = torch.cat([f.cos(), f.cos()])
    sin = torch.cat([f.sin(), f.sin()])
    h = dim // 2
    rh = torch.cat([-x[h:], x[:h]])
    return x * cos + rh * sin


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * w


def test_sliding_attn_vs_ref(device):
    torch.manual_seed(0)
    cfg = Gemma4Config()
    H, D = cfg.hidden, cfg.head_dim
    card, qh, kvh = 0, 4, 2

    w = {
        "q_proj": torch.randn(cfg.q_heads * D, H) * 0.02,
        "k_proj": torch.randn(cfg.kv_heads * D, H) * 0.02,
        "v_proj": torch.randn(cfg.kv_heads * D, H) * 0.02,
        "o_proj": torch.randn(H, cfg.q_heads * D) * 0.02,
        "input_layernorm": torch.randn(H) * 0.1,
        "post_attention_layernorm": torch.randn(H) * 0.1,
        "q_norm": torch.randn(D) * 0.1,
        "k_norm": torch.randn(D) * 0.1,
    }
    layer = SlidingLayer(cfg, w, device, card)

    xs = [torch.randn(H) * 0.5 for _ in range(3)]
    ks, vs = [[] for _ in range(kvh)], [[] for _ in range(kvh)]

    for pos, x in enumerate(xs):
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2).float() / D))
        f = pos * inv
        cos = torch.cat([f.cos(), f.cos()]).expand(TILE, D).contiguous()
        sin = torch.cat([f.sin(), f.sin()]).expand(TILE, D).contiguous()
        cos_d = to_dev(cos.to(torch.bfloat16), device)
        sin_d = to_dev(sin.to(torch.bfloat16), device)

        x_d = row(x, H, device)
        got = from_dev(layer.attn(x_d, pos, cos_d, sin_d))[0]

        # reference
        xn = rms(x, 1 + w["input_layernorm"], cfg.eps)
        outs = []
        for kv in range(kvh):
            kk = w["k_proj"][(card * kvh + kv) * D:(card * kvh + kv + 1) * D] @ xn
            kk = rope_ref(rms(kk, 1 + w["k_norm"], cfg.eps), pos, cfg.rope_theta, D)
            vv = w["v_proj"][(card * kvh + kv) * D:(card * kvh + kv + 1) * D] @ xn
            vv = rms(vv, torch.ones(D), cfg.eps)
            ks[kv].append(kk)
            vs[kv].append(vv)
        for h in range(qh):
            q = w["q_proj"][(card * qh + h) * D:(card * qh + h + 1) * D] @ xn
            q = rope_ref(rms(q, 1 + w["q_norm"], cfg.eps), pos, cfg.rope_theta, D)
            kv = h // (qh // kvh)
            K = torch.stack(ks[kv])
            V = torch.stack(vs[kv])
            att = torch.softmax(q @ K.T, dim=-1)
            outs.append(att @ V)
        want = w["o_proj"][:, card * qh * D:(card + 1) * qh * D] @ torch.cat(outs)

        pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
        assert pcc > 0.985, f"pos {pos}: pcc {pcc}"
