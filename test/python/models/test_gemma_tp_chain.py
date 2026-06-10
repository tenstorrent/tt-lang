# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""TP=4 3-layer DecodeChain vs full-model torch, two greedy decode steps.

Cards shard QKV by head, dense inter/4, experts 32/card with replicated
global router; global attention replicated. Token parity checks the AR
points: any missing partial diverges immediately.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from gemma4.config import Gemma4Config
from gemma4.decode_chain import DecodeChain, GlobalChain, SlidingChain, StepState
from gemma4.host import MLP_PAD

TILE, TP, CTX = 32, 4, 1024


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


def rand_full(cfg, kind):
    H, I, E = cfg.hidden, cfg.moe_inter, cfg.experts
    inter = cfg.mlp_inter
    w = {k: 1 + torch.randn(H) * 0.1 for k in (
        "g_in", "g_postattn", "g_preffw", "g_postffw1", "g_preffw2",
        "g_postffw2", "g_postffw")}
    w.update({
        "w_gate_f": torch.randn(H, inter) * 0.02,
        "w_up_f": torch.randn(H, inter) * 0.02,
        "w_down_f": torch.randn(inter, H) * 0.02,
        "w_gu_f": torch.randn(E, H, 2 * I) * 0.02,
        "w_dn_f": torch.randn(E, I, H) * 0.02,
        "router_w": torch.randn(E, H) * 0.1,
        "router_scale": 1.0,
        "per_expert": torch.ones(E),
        "layer_scalar": 0.9,
    })
    D = cfg.head_dim if kind == "sliding" else cfg.global_head_dim
    w["q_norm"] = 1 + torch.randn(D) * 0.1
    w["k_norm"] = 1 + torch.randn(D) * 0.1
    if kind == "sliding":
        w["v_norm"] = torch.ones(D)
        w["wq_f"] = torch.randn(cfg.q_heads * D, H) * 0.02
        w["wk_f"] = torch.randn(cfg.kv_heads * D, H) * 0.02
        w["wv_f"] = torch.randn(cfg.kv_heads * D, H) * 0.02
        w["wo_f"] = torch.randn(H, cfg.q_heads * D) * 0.02
    else:
        w["wq_f"] = torch.randn(cfg.q_heads * D, H) * 0.02
        w["wk_f"] = torch.randn(cfg.global_kv_heads * D, H) * 0.02
        w["wo_f"] = torch.randn(H, cfg.q_heads * D) * 0.02
    return w


def card_slice(w, cfg, kind, card):
    """Mirror weights.layer_weights sharding on a rand full dict."""
    H, P = cfg.hidden, MLP_PAD
    E, inter = cfg.experts // TP, cfg.mlp_inter // TP
    D = cfg.head_dim if kind == "sliding" else cfg.global_head_dim
    qh, qs = cfg.q_heads // TP, card * (cfg.q_heads // TP)
    c = {k: w[k] for k in w if not k.endswith("_f")}
    if kind == "sliding":
        ks = card * (cfg.kv_heads // TP)
        c["w_qkv"] = torch.cat([
            w["wq_f"][qs * D:(qs + qh) * D],
            w["wk_f"][ks * D:(ks + 2) * D],
            w["wv_f"][ks * D:(ks + 2) * D]]).T.contiguous()
        c["w_o"] = w["wo_f"][:, qs * D:(qs + qh) * D].T.contiguous()
    else:
        kv = qs // (cfg.q_heads // cfg.global_kv_heads)
        c["w_q"] = w["wq_f"][qs * D:(qs + qh) * D].T.contiguous()
        c["w_k"] = w["wk_f"][kv * D:(kv + 1) * D].T.contiguous()
        c["w_o"] = w["wo_f"][:, qs * D:(qs + qh) * D].T.contiguous()
    s = card * inter
    c["w_gate"] = torch.nn.functional.pad(
        w["w_gate_f"][:, s:s + inter], (0, P - inter)).contiguous()
    c["w_up"] = torch.nn.functional.pad(
        w["w_up_f"][:, s:s + inter], (0, P - inter)).contiguous()
    c["w_down"] = torch.nn.functional.pad(
        w["w_down_f"][s:s + inter], (0, 0, 0, P - inter)).contiguous()
    es = card * E
    c["w_gu"] = w["w_gu_f"][es:es + E]
    c["w_dn"] = w["w_dn_f"][es:es + E]
    return c


class TorchLayer:
    def __init__(self, w, cfg, kind):
        self.w, self.cfg, self.kind = w, cfg, kind
        D = cfg.head_dim if kind == "sliding" else cfg.global_head_dim
        S = cfg.sliding_window if kind == "sliding" else CTX
        kvh = cfg.kv_heads if kind == "sliding" else cfg.global_kv_heads
        self.D, self.S, self.kvh = D, S, kvh
        self.kc = [torch.zeros(S, D) for _ in range(kvh)]
        self.vc = [torch.zeros(S, D) for _ in range(kvh)] if kind == "sliding" else self.kc

    def attn(self, x, pos):
        w, cfg, D = self.w, self.cfg, self.D
        eps, qh = cfg.eps, cfg.q_heads
        theta = cfg.rope_theta if self.kind == "sliding" else cfg.global_rope_theta
        rot = D if self.kind == "sliding" else int(D * cfg.global_rot_frac)
        xn = rms(x, w["g_in"], eps)
        q = [rope_full(rms(xn @ w["wq_f"][h * D:(h + 1) * D].T, w["q_norm"], eps),
                       pos, rot, theta) for h in range(qh)]
        k = [rope_full(rms(xn @ w["wk_f"][h * D:(h + 1) * D].T, w["k_norm"], eps),
                       pos, rot, theta) for h in range(self.kvh)]
        slot = pos % self.S if self.kind == "sliding" else pos
        for i in range(self.kvh):
            self.kc[i][slot] = k[i]
        if self.kind == "sliding":
            v = [rms(xn @ w["wv_f"][h * D:(h + 1) * D].T, w["v_norm"], eps)
                 for h in range(self.kvh)]
            for i in range(self.kvh):
                self.vc[i][slot] = v[i]
        mask = torch.full((self.S,), float("-inf"))
        mask[:min(pos + 1, self.S)] = 0.0
        outs = []
        for h in range(qh):
            g = h // (qh // self.kvh)
            att = torch.softmax(q[h] @ self.kc[g].T + mask, dim=-1)
            outs.append(att @ self.vc[g])
        return w["wo_f"] @ torch.cat(outs)

    def step(self, x, pos):
        w, cfg, eps = self.w, self.cfg, self.cfg.eps
        I = cfg.moe_inter
        h = x + rms(self.attn(x, pos), w["g_postattn"], eps)
        hn = rms(h, w["g_preffw"], eps)
        dense = (gelu_tanh(hn @ w["w_gate_f"]) * (hn @ w["w_up_f"])) @ w["w_down_f"]
        h1 = rms(dense, w["g_postffw1"], eps)
        xn = h / torch.sqrt(h.pow(2).mean() + eps) * w["router_scale"] * cfg.hidden ** -0.5
        probs = torch.softmax(w["router_w"] @ xn, dim=-1)
        wts, idx = torch.topk(probs, cfg.top_k)
        wts = wts / wts.sum() * w["per_expert"][idx]
        hn2 = rms(h, w["g_preffw2"], eps)
        exp = torch.zeros(cfg.hidden)
        for t in range(cfg.top_k):
            gu = hn2 @ w["w_gu_f"][idx[t]]
            exp += (gelu_tanh(gu[:I]) * gu[I:] * wts[t]) @ w["w_dn_f"][idx[t]]
        h2 = rms(exp, w["g_postffw2"], eps)
        return (h + rms(h1 + h2, w["g_postffw"], eps)) * w["layer_scalar"]


def test_tp_chain():
    if ttnn.GetNumAvailableDevices() < TP:
        pytest.skip("needs 4 devices")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    try:
        _run(mesh)
    finally:
        ttnn.close_device(mesh)


def _run(mesh):
    torch.manual_seed(0)
    cfg = Gemma4Config()
    H, V, rows = cfg.hidden, 2048, 4096
    kinds = ["sliding", "sliding", "global"]
    ws = [rand_full(cfg, k) for k in kinds]
    embed = torch.randn(rows, H) * 0.02
    g_final = 1 + torch.randn(H) * 0.1
    lm = embed.to(torch.bfloat16).float()[:V]

    ref_layers = [TorchLayer(w, cfg, k) for w, k in zip(ws, kinds)]
    st = StepState(mesh, cfg, CTX)
    layers = []
    for w, k in zip(ws, kinds):
        cards = [card_slice(w, cfg, k, c) for c in range(TP)]
        layers.append(SlidingChain(cards, mesh, cfg, st) if k == "sliding"
                      else GlobalChain(cards, mesh, cfg, st, CTX))
    chain = DecodeChain(layers, st, embed, g_final, lm.T.contiguous(), mesh, cfg)

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
