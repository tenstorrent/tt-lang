# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""3-layer DecodeChain (sliding, sliding, global) vs torch, two decode steps."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from gemma4.config import Gemma4Config
from gemma4.decode_chain import (DecodeChain, GlobalChain, SlidingChain,
                                 StepState, MLP_PAD)

TILE = 32
CTX = 1024


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * w


def gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def rope_full(h, pos, rot, theta):
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2).float() / rot))
    f = pos * inv
    cos, sin = torch.cat([f.cos(), f.cos()]), torch.cat([f.sin(), f.sin()])
    half = rot // 2
    x1, x2 = h[:half], h[half:rot]
    return torch.cat([x1 * cos[:half] - x2 * sin[:half],
                      x2 * cos[half:] + x1 * sin[half:], h[rot:]])


def rand_w(cfg, kind):
    H, P, I, E = cfg.hidden, MLP_PAD, cfg.moe_inter, cfg.experts // 4
    inter = cfg.mlp_inter // 4
    w = {
        "g_in": 1 + torch.randn(H) * 0.1,
        "g_postattn": 1 + torch.randn(H) * 0.1,
        "g_preffw": 1 + torch.randn(H) * 0.1,
        "g_postffw1": 1 + torch.randn(H) * 0.1,
        "g_preffw2": 1 + torch.randn(H) * 0.1,
        "g_postffw2": 1 + torch.randn(H) * 0.1,
        "g_postffw": 1 + torch.randn(H) * 0.1,
        "w_gate": torch.zeros(H, P), "w_up": torch.zeros(H, P),
        "w_down": torch.zeros(P, H),
        "w_gu": torch.randn(E, H, 2 * I) * 0.02,
        "w_dn": torch.randn(E, I, H) * 0.02,
        "router_w": torch.randn(E, H) * 0.1,
        "router_scale": 1.0,
        "per_expert": torch.ones(E),
        "layer_scalar": 0.9,
    }
    w["w_gate"][:, :inter] = torch.randn(H, inter) * 0.02
    w["w_up"][:, :inter] = torch.randn(H, inter) * 0.02
    w["w_down"][:inter] = torch.randn(inter, H) * 0.02
    if kind == "sliding":
        D = cfg.head_dim
        w["q_norm"] = 1 + torch.randn(D) * 0.1
        w["k_norm"] = 1 + torch.randn(D) * 0.1
        w["v_norm"] = torch.ones(D)
        w["w_qkv"] = torch.randn(H, 8 * D) * 0.02
        w["w_o"] = torch.randn(4 * D, H) * 0.02
    else:
        D = cfg.global_head_dim
        w["q_norm"] = 1 + torch.randn(D) * 0.1
        w["k_norm"] = 1 + torch.randn(D) * 0.1
        w["w_q"] = torch.randn(H, 4 * D) * 0.02
        w["w_k"] = torch.randn(H, cfg.global_kv_heads * D) * 0.02
        w["w_o"] = torch.randn(4 * D, H) * 0.02
    return w


class TorchLayer:
    def __init__(self, w, cfg, kind):
        self.w, self.cfg, self.kind = w, cfg, kind
        D = cfg.head_dim if kind == "sliding" else cfg.global_head_dim
        S = cfg.sliding_window if kind == "sliding" else CTX
        kvh = 2
        self.D, self.S = D, S
        self.kc = [torch.zeros(S, D) for _ in range(kvh)]
        self.vc = [torch.zeros(S, D) for _ in range(kvh)] if kind == "sliding" else self.kc

    def attn(self, x, pos):
        w, cfg, D = self.w, self.cfg, self.D
        eps = cfg.eps
        theta = cfg.rope_theta if self.kind == "sliding" else cfg.global_rope_theta
        rot = D if self.kind == "sliding" else int(D * cfg.global_rot_frac)
        xn = rms(x, w["g_in"], eps)
        if self.kind == "sliding":
            heads = [xn @ w["w_qkv"][:, i * D:(i + 1) * D] for i in range(8)]
            q = [rope_full(rms(h, w["q_norm"], eps), pos, rot, theta) for h in heads[:4]]
            k = [rope_full(rms(h, w["k_norm"], eps), pos, rot, theta) for h in heads[4:6]]
            v = [rms(h, w["v_norm"], eps) for h in heads[6:8]]
            ring = pos % self.S
            for i in range(2):
                self.kc[i][ring], self.vc[i][ring] = k[i], v[i]
        else:
            q = [rope_full(rms(xn @ w["w_q"][:, h * D:(h + 1) * D], w["q_norm"], eps),
                           pos, rot, theta) for h in range(4)]
            k = [rope_full(rms(xn @ w["w_k"][:, h * D:(h + 1) * D], w["k_norm"], eps),
                           pos, rot, theta) for h in range(2)]
            for i in range(2):
                self.kc[i][pos] = k[i]
        mask = torch.full((self.S,), float("-inf"))
        mask[:min(pos + 1, self.S)] = 0.0
        outs = []
        for h in range(4):
            att = torch.softmax(q[h] @ self.kc[h // 2].T + mask, dim=-1)
            outs.append(att @ self.vc[h // 2])
        return torch.cat(outs) @ w["w_o"]

    def step(self, x, pos):
        w, cfg, eps = self.w, self.cfg, self.cfg.eps
        I = cfg.moe_inter
        h = x + rms(self.attn(x, pos), w["g_postattn"], eps)
        hn = rms(h, w["g_preffw"], eps)
        dense = (gelu_tanh(hn @ w["w_gate"]) * (hn @ w["w_up"])) @ w["w_down"]
        h1 = rms(dense, w["g_postffw1"], eps)
        xn = h / torch.sqrt(h.pow(2).mean() + eps) * w["router_scale"] * cfg.hidden ** -0.5
        probs = torch.softmax(w["router_w"] @ xn, dim=-1)
        wts, idx = torch.topk(probs, cfg.top_k)
        wts = wts / wts.sum() * w["per_expert"][idx]
        hn2 = rms(h, w["g_preffw2"], eps)
        exp = torch.zeros(cfg.hidden)
        for t in range(cfg.top_k):
            gu = hn2 @ w["w_gu"][idx[t]]
            exp += (gelu_tanh(gu[:I]) * gu[I:] * wts[t]) @ w["w_dn"][idx[t]]
        h2 = rms(exp, w["g_postffw2"], eps)
        return (h + rms(h1 + h2, w["g_postffw"], eps)) * w["layer_scalar"]


def test_decode_chain():
    device = ttnn.open_device(device_id=0)
    try:
        _run(device)
    finally:
        ttnn.close_device(device)


def _run(device):
    torch.manual_seed(0)
    cfg = Gemma4Config()
    H, V, rows = cfg.hidden, 2048, 4096
    kinds = ["sliding", "sliding", "global"]
    ws = [rand_w(cfg, k) for k in kinds]
    embed = torch.randn(rows, H) * 0.02
    g_final = 1 + torch.randn(H) * 0.1
    lm = embed.to(torch.bfloat16).float()[:V]

    ref_layers = [TorchLayer(w, cfg, k) for w, k in zip(ws, kinds)]
    st = StepState(device, cfg, CTX)
    layers = [SlidingChain([w], device, cfg, st) if k == "sliding"
              else GlobalChain([w], device, cfg, st, CTX)
              for w, k in zip(ws, kinds)]
    chain = DecodeChain(layers, st, embed, g_final, lm.T.contiguous(), device, cfg)

    # Greedy reference; device feeds its own token back, no host in the loop.
    tok = 1234
    chain.prime(tok, 100)
    for pos in (100, 101):
        x = embed.to(torch.bfloat16).float()[tok] * H ** 0.5
        for rl in ref_layers:
            x = rl.step(x, pos)
        want = rms(x, g_final, cfg.eps) @ lm.T
        tok = want.argmax().item()

        chain.step()
        got = chain.read_token()
        assert got == tok, f"pos {pos}: got {got} want {tok}"
