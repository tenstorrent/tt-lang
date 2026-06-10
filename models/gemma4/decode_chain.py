# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-card decode step as a dispatch chain (bring-up driver).

One atom per op, activations and routing device-resident: no host or ttnn
logic between atoms (CCLs and DRAM round trips are the only sanctioned
cuts). Atom factories are memoized so 30 layers share compiled kernels.
Fusing back into per-cut atoms is the optimization phase; this driver is
the correctness substrate (test: per-layer goldens, greedy match).
"""

import torch

import ttl  # noqa: F401  (kept first for runtime init)
from ttl.ops.elementwise import make_add, make_binary, make_copy, make_row_scale
from ttl.ops.gemv import make_gemv
from ttl.ops.indexed_gemv import make_indexed_gemv
from ttl.ops.flash_decode import make_flash_decode_kev
from ttl.ops.kv_append import make_kv_append
from ttl.ops.moe_router import make_moe_weights, make_softmax_row
from ttl.ops.rmsnorm import make_rmsnorm
from ttl.ops.rope import make_rope
from ttl.ops.swiglu import make_swiglu
from ttl.ops.topk import make_topk
from ttl.ops.embed_gather import make_embed_gather

from .attn_atom import TILE, make_attn_heads_atom, make_flash_atom
from .host import MLP_PAD, from_dev, row, to_dev

_ATOMS = {}


def atom(maker, *args, **kwargs):
    key = (maker.__name__, args, tuple(sorted(kwargs.items())))
    if key not in _ATOMS:
        _ATOMS[key] = maker(*args, **kwargs)
    return _ATOMS[key]


def pos_tile(p):
    t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    t[0, 0], t[0, 1] = p // TILE, p % TILE
    return t


def causal_masks(S, valid, chunk):
    m = torch.full((S,), float("-inf"))
    m[:valid] = 0.0
    n = S // chunk
    return m.reshape(n, 1, chunk).expand(n, TILE, chunk).reshape(n * TILE, chunk).to(torch.bfloat16)


def rope_tables(pos, rot, theta):
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2).float() / rot))
    f = pos * inv
    cos = torch.cat([f.cos(), f.cos()]).expand(TILE, rot).contiguous()
    sin = torch.cat([f.sin(), f.sin()]).expand(TILE, rot).contiguous()
    return cos.to(torch.bfloat16), sin.to(torch.bfloat16)


class FFNChain:
    """Dense MLP + routed experts; shared by both layer kinds."""

    def __init__(self, w, device, cfg):
        self.device, self.cfg = device, cfg
        H, P, I, E = cfg.hidden, MLP_PAD, cfg.moe_inter, cfg.experts // 4
        self.E, self.I = E, I
        self.Ht, self.It = H // TILE, I // TILE
        for k in ("g_preffw", "g_postffw1", "g_preffw2", "g_postffw2", "g_postffw"):
            setattr(self, k, row(w[k], H, device))
        self.w_gate = to_dev(w["w_gate"].to(torch.bfloat16), device)
        self.w_up = to_dev(w["w_up"].to(torch.bfloat16), device)
        self.w_down = to_dev(w["w_down"].to(torch.bfloat16), device)
        self.w_gu = to_dev(w["w_gu"].reshape(E * H, 2 * I).to(torch.bfloat16), device)
        self.w_dn = to_dev(w["w_dn"].reshape(E * I, H).to(torch.bfloat16), device)
        self.layer_scalar = w["layer_scalar"]

        # Step-invariant router tables: scale-less norm weight with
        # router_scale * H^-0.5 folded in; ids ramp for topk; per-expert row.
        rscale = w["router_scale"] * cfg.hidden ** -0.5
        self.g_router = row(torch.full((H,), rscale), H, device)
        self.w_router = to_dev(w["router_w"].T.contiguous().to(torch.bfloat16), device)
        self.ramp = to_dev(torch.arange(E, dtype=torch.bfloat16)
                           .unsqueeze(0).repeat(TILE, 1), device)
        self.pe = row(w["per_expert"], E, device)

        def buf(n, rows=TILE):
            return to_dev(torch.zeros(rows, n, dtype=torch.bfloat16), device)

        K = cfg.top_k
        self.hn = buf(cfg.hidden)
        self.hr, self.rl, self.probs = buf(H), buf(E), buf(E)
        self.vals, self.idx, self.wts = buf(K * TILE), buf(K * TILE), buf(K * TILE)
        self.g, self.u, self.act = buf(P), buf(P), buf(P)
        self.dense, self.h1, self.hn2, self.h2 = buf(H), buf(H), buf(H), buf(H)
        self.gu = buf(2 * I, 8 * TILE)
        self.eact = buf(I, 8 * TILE)
        self.dn = buf(H, 8 * TILE)
        self.comb, self.combn, self.out = buf(H), buf(H), buf(H)

    def step(self, h_d):
        cfg, dev, K, E = self.cfg, self.device, self.cfg.top_k, self.E
        H, P, Ht, It = cfg.hidden, MLP_PAD, self.Ht, self.It
        Et = E // TILE
        norm = atom(make_rmsnorm, 1, 1, Ht, 11, H, cfg.eps)

        norm(h_d, self.g_router, self.hr)
        atom(make_gemv, TILE, H, E, (1, 2), 1)(self.hr, self.w_router, self.rl)
        atom(make_softmax_row, Et)(self.rl, self.probs)
        atom(make_topk, 1, 1, Et, K, E)(self.probs, self.ramp, self.vals, self.idx)
        atom(make_moe_weights, K, Et)(self.vals, self.idx, self.pe, self.wts)
        idx_d, wts_d = self.idx, self.wts

        norm(h_d, self.g_preffw, self.hn)
        gate = atom(make_gemv, TILE, H, P, (9, 2), 2)
        gate(self.hn, self.w_gate, self.g)
        gate(self.hn, self.w_up, self.u)
        atom(make_swiglu, 1, 1, P // TILE, P // TILE)(self.g, self.u, self.act)
        atom(make_gemv, TILE, P, H, (11, 2), 4)(self.act, self.w_down, self.dense)
        norm(self.dense, self.g_postffw1, self.h1)

        norm(h_d, self.g_preffw2, self.hn2)
        atom(make_indexed_gemv, self.E, H, 2 * self.I, K, (11, 2), 4, idx_stride=TILE)(
            self.hn2, idx_d, self.w_gu, self.gu)
        for t in range(K):
            atom(make_swiglu, 1, 1, It, It,
                 a_off=(t, 0), b_off=(t, It), out_off=(t, 0))(self.gu, self.gu, self.eact)
            atom(make_row_scale, self.I, It, a_row=t, s_col=t, out_row=t)(
                self.eact, wts_d, self.eact)
        atom(make_indexed_gemv, self.E, self.I, H, K, (11, 2), 4,
             x_per_t=True, idx_stride=TILE)(self.eact, idx_d, self.w_dn, self.dn)
        for t in range(1, K):
            atom(make_add, 1, 1, Ht, 11, b_off=(t, 0))(self.dn, self.dn, self.dn)
        norm(self.dn, self.g_postffw2, self.h2)

        atom(make_add, 1, 1, Ht, 11)(self.h1, self.h2, self.comb)
        norm(self.comb, self.g_postffw, self.combn)
        scaled_add = atom(make_scaled_add, Ht, self.layer_scalar)
        scaled_add(h_d, self.combn, self.out)
        return self.out


def make_scaled_add(Ht, scalar):
    return make_binary(
        lambda a, b: ttl.mul(ttl.add(a, b), ttl.block.fill(scalar, shape=a.shape)),
        1, 1, Ht, 11)


class SlidingChain:
    """Per-card sliding layer: fused heads atom + kv appends + flash + o proj."""

    def __init__(self, w, device, cfg):
        self.device, self.cfg = device, cfg
        H, D, S = cfg.hidden, cfg.head_dim, cfg.sliding_window
        self.D, self.S = D, S
        self.Ht, self.Dt = H // TILE, D // TILE
        self.g_in = row(w["g_in"], H, device)
        self.qknorm = row(w["qknorm"], D, device)
        self.w_qkv = to_dev(w["w_qkv"].to(torch.bfloat16), device)
        self.w_o = to_dev(w["w_o"].to(torch.bfloat16), device)
        self.g_postattn = row(w["g_postattn"], H, device)
        half = D // 2
        R = torch.zeros(D, D)
        for j in range(half):
            R[half + j, j], R[j, half + j] = -1.0, 1.0
        self.R = to_dev(R.to(torch.bfloat16), device)
        self.caches = [to_dev(torch.zeros(S, D, dtype=torch.bfloat16), device)
                       for _ in range(4)]
        self.heads = to_dev(torch.zeros(8 * TILE, D, dtype=torch.bfloat16), device)
        self.o_row = to_dev(torch.zeros(TILE, 4 * D, dtype=torch.bfloat16), device)
        self.attn = to_dev(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
        self.attn_n = to_dev(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
        self.h = to_dev(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
        self.ffn = FFNChain(w, device, cfg)

    def step(self, x_d, pos):
        cfg, dev = self.cfg, self.device
        H, D, S, Dt, Ht = cfg.hidden, self.D, self.S, self.Dt, self.Ht
        ring = pos % S
        cos, sin = rope_tables(pos, D, cfg.rope_theta)
        masks = causal_masks(S, min(pos + 1, S), S // 4)

        atom(make_attn_heads_atom, Ht, Dt, cfg.eps)(
            x_d, self.g_in, self.w_qkv, to_dev(cos, dev), to_dev(sin, dev),
            self.qknorm, self.R, self.heads)
        pos_d = to_dev(pos_tile(ring), dev)
        for i, cache in enumerate(self.caches):
            atom(make_kv_append, S // TILE, Dt, k_row=4 + i)(cache, self.heads, pos_d, cache)
        atom(make_flash_atom, Dt, S // TILE)(
            self.heads, *self.caches, to_dev(masks, dev), self.o_row)
        atom(make_gemv, TILE, 4 * D, H, (8, 2), 11)(self.o_row, self.w_o, self.attn)

        norm = atom(make_rmsnorm, 1, 1, Ht, 11, H, cfg.eps)
        norm(self.attn, self.g_postattn, self.attn_n)
        atom(make_add, 1, 1, Ht, 11)(x_d, self.attn_n, self.h)
        return self.ffn.step(self.h)


class GlobalChain:
    """Single-card global layer: full ctx local (TP combine comes later)."""

    def __init__(self, w, device, cfg, ctx):
        self.device, self.cfg = device, cfg
        H, D = cfg.hidden, cfg.global_head_dim
        self.D, self.S = D, ctx
        self.qh, self.kvh = 4, cfg.global_kv_heads
        self.Ht, self.Dt = H // TILE, D // TILE
        self.rot = int(D * cfg.global_rot_frac)
        self.g_in = row(w["g_in"], H, device)
        self.qknorm = row(w["qknorm"], D, device)
        self.w_q = to_dev(w["w_q"].to(torch.bfloat16), device)
        self.w_k = to_dev(w["w_k"].to(torch.bfloat16), device)
        self.w_o = to_dev(w["w_o"].to(torch.bfloat16), device)
        self.g_postattn = row(w["g_postattn"], H, device)
        self.caches = [to_dev(torch.zeros(self.S, D, dtype=torch.bfloat16), device)
                       for _ in range(self.kvh)]

        def buf(n, rows=TILE):
            return to_dev(torch.zeros(rows, n, dtype=torch.bfloat16), device)

        self.xn, self.q, self.k = buf(H), buf(self.qh * D), buf(self.kvh * D)
        self.hd, self.hh, self.hr = buf(D), buf(D), buf(D)
        self.o, self.m, self.l, self.of = buf(D), buf(TILE), buf(TILE), buf(D)
        self.a_row, self.attn = buf(self.qh * D), buf(H)
        self.attn_n, self.h = buf(H), buf(H)
        self.ffn = FFNChain(w, device, cfg)

    def head_rot(self, src, hidx, cos_d, sin_d):
        cfg, Dt = self.cfg, self.Dt
        atom(make_copy, 1, 1, Dt, Dt, a_off=(0, hidx * Dt))(src, self.hd)
        atom(make_rmsnorm, 1, 1, Dt, Dt, self.D, cfg.eps)(self.hd, self.qknorm, self.hh)
        atom(make_rope, Dt, self.rot // TILE)(self.hh, cos_d, sin_d, self.hr)
        return self.hr

    def step(self, x_d, pos):
        cfg, dev = self.cfg, self.device
        H, D, S, Dt, Ht = cfg.hidden, self.D, self.S, self.Dt, self.Ht
        chunk = 8 * TILE
        cos, sin = rope_tables(pos, self.rot, cfg.global_rope_theta)
        cos_d, sin_d = to_dev(cos, dev), to_dev(sin, dev)
        masks_d = to_dev(causal_masks(S, pos + 1, chunk), dev)
        pos_d = to_dev(pos_tile(pos), dev)

        atom(make_rmsnorm, 1, 1, Ht, 11, H, cfg.eps)(x_d, self.g_in, self.xn)
        atom(make_gemv, TILE, H, self.qh * D, (8, 2), 4)(self.xn, self.w_q, self.q)
        atom(make_gemv, TILE, H, self.kvh * D, (8, 2), 4)(self.xn, self.w_k, self.k)

        for kv in range(self.kvh):
            kr = self.head_rot(self.k, kv, cos_d, sin_d)
            atom(make_kv_append, S // TILE, Dt)(self.caches[kv], kr, pos_d, self.caches[kv])
        for h in range(self.qh):
            qr = self.head_rot(self.q, h, cos_d, sin_d)
            atom(make_flash_decode_kev, 1, 1, 1, Dt, 8, S // chunk)(
                qr, self.caches[h // 2], masks_d, self.o, self.m, self.l)
            atom(make_row_scale, Dt, 8, fn=ttl.recip)(self.o, self.l, self.of)
            atom(make_copy, 1, 1, Dt, Dt, out_off=(0, h * Dt))(self.of, self.a_row)

        atom(make_gemv, TILE, self.qh * D, H, (8, 2), 11)(self.a_row, self.w_o, self.attn)
        norm = atom(make_rmsnorm, 1, 1, Ht, 11, H, cfg.eps)
        norm(self.attn, self.g_postattn, self.attn_n)
        atom(make_add, 1, 1, Ht, 11)(x_d, self.attn_n, self.h)
        return self.ffn.step(self.h)


class DecodeChain:
    """Embed -> layers -> final norm -> lm_head logits (host argmax)."""

    def __init__(self, layers, embed, g_final, lm_head, device, cfg):
        self.layers, self.device, self.cfg = layers, device, cfg
        H = cfg.hidden
        self.Ht = H // TILE
        self.V = lm_head.shape[1]
        self.table = to_dev(embed.to(torch.bfloat16), device)
        self.g_final = row(g_final, H, device)
        self.lm = to_dev(lm_head.to(torch.bfloat16), device)
        self.x = to_dev(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
        self.xn = to_dev(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
        self.logits = to_dev(torch.zeros(TILE, self.V, dtype=torch.bfloat16), device)

    def step(self, tok, pos):
        cfg, dev, H = self.cfg, self.device, self.cfg.hidden
        atom(make_embed_gather, self.Ht, H ** 0.5)(self.table, to_dev(pos_tile(tok), dev), self.x)
        x = self.x
        for layer in self.layers:
            x = layer.step(x, pos)
        atom(make_rmsnorm, 1, 1, self.Ht, 11, H, cfg.eps)(x, self.g_final, self.xn)
        atom(make_gemv, TILE, H, self.V, (8, 2), 4)(self.xn, self.lm, self.logits)
        return from_dev(self.logits)[0]
