# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-card decode step as a dispatch chain (bring-up driver).

One atom per op, activations and all step state device-resident: no host or
ttnn logic between atoms (CCLs and DRAM round trips via ttl.copy are the
only sanctioned cuts). Host stages step-invariant tables and the prompt
token before generation; the step loop only enqueues dispatches. Atom
factories are memoized so 30 layers share kernels.
"""

import torch
import ttnn

import ttl  # noqa: F401  (kept first for runtime init)
from ttl.ops.argmax import CHUNK, make_collapse, make_restack, make_token_select
from ttl.ops.elementwise import make_add, make_binary, make_copy, make_row_scale
from ttl.ops.embed_gather import make_embed_gather
from ttl.ops.flash_decode import make_flash_decode_kev
from ttl.ops.gemv import make_gemv
from ttl.ops.indexed_gemv import make_indexed_gemv
from ttl.ops.kv_append import make_kv_append
from ttl.ops.moe_router import make_idx_gather, make_moe_weights, make_softmax_row
from ttl.ops.pos_slice import make_pos_slice, make_pos_step
from ttl.ops.rmsnorm import make_rmsnorm
from ttl.ops.rope import make_rope
from ttl.ops.swiglu import make_swiglu
from ttl.ops.topk import make_topk

from .attn_atom import TILE, make_attn_heads_atom, make_flash_atom
from .host import MLP_PAD, from_dev, is_mesh, row, shard_cards, to_dev

_ATOMS = {}


def all_reduce(device, t):
    """Sanctioned CCL cut between atoms; identity off-mesh."""
    return ttnn.all_reduce(t) if is_mesh(device) else t


def atom(maker, *args, **kwargs):
    key = (maker.__name__, args, tuple(sorted(kwargs.items())))
    if key not in _ATOMS:
        _ATOMS[key] = maker(*args, **kwargs)
    return _ATOMS[key]


def make_scaled_add(Ht, scalar):
    return make_binary("scaled_add", 1, 1, Ht, 11, scalar=scalar)


def buf(device, n, rows=TILE):
    return to_dev(torch.zeros(rows, n, dtype=torch.bfloat16), device)


def split_tile(p, ring=None):
    t = torch.zeros(TILE, TILE)
    r = p if ring is None else p % ring
    t[0, :3] = torch.tensor([r // TILE, r % TILE, p]).float()
    return t


def pos_lut(smax, ring=None):
    lut = torch.zeros(smax, TILE)
    for p in range(smax - 1):
        lut[p, :3] = split_tile(p + 1, ring)[0, :3]
    return lut


def rope_table(smax, rot, theta):
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2).float() / rot))
    f = torch.arange(smax).float().unsqueeze(1) * inv
    return torch.cat([f.cos(), f.cos()], 1), torch.cat([f.sin(), f.sin()], 1)


def mask_table(smax, S):
    j = torch.arange(S)
    p = torch.arange(smax).unsqueeze(1)
    m = torch.where(j <= p, 0.0, -1e30)
    m[S:] = 0.0  # ring cache fully valid once wrapped
    return m


class FFNChain:
    """Dense MLP + routed experts; routing fully on device.

    TP: dense inter and experts are card-local; the router is replicated and
    runs the global top-k, with off-card experts killed by zeroed pe staging
    and ids translated to card-local rows. One AR covers both partials.
    """

    def __init__(self, ws, device, cfg):
        self.device, self.cfg = device, cfg
        tp, w = len(ws), ws[0]
        H, P, I, E = cfg.hidden, MLP_PAD, cfg.moe_inter, cfg.experts // 4
        self.E, self.I, self.Etot = E, I, E * tp
        self.Ht, self.It = H // TILE, I // TILE
        for k in ("g_preffw", "g_postffw1", "g_preffw2", "g_postffw2", "g_postffw"):
            setattr(self, k, row(w[k], H, device))
        shard = lambda key, rs=None: shard_cards(
            [c[key] if rs is None else c[key].reshape(*rs) for c in ws], device)
        self.w_gate, self.w_up = shard("w_gate"), shard("w_up")
        self.w_down = shard("w_down")
        self.w_gu = shard("w_gu", (E * H, 2 * I))
        self.w_dn = shard("w_dn", (E * I, H))
        self.layer_scalar = w["layer_scalar"]

        rscale = torch.as_tensor(w["router_scale"]).expand(H) * cfg.hidden ** -0.5
        self.g_router = row(rscale, H, device)
        self.w_router = to_dev(w["router_w"].T.contiguous().to(torch.bfloat16), device)
        self.ramp = to_dev(torch.arange(self.Etot, dtype=torch.bfloat16)
                           .unsqueeze(0).repeat(TILE, 1), device)
        pes, luts = [], []
        for c in range(tp):
            pe = torch.zeros(TILE, self.Etot)
            pe[0, c * E:(c + 1) * E] = w["per_expert"][c * E:(c + 1) * E]
            pes.append(pe)
            lut = (torch.arange(self.Etot) - c * E).clamp(0, E - 1).float()
            luts.append(lut.unsqueeze(0).expand(TILE, self.Etot))
        self.pe, self.lut = shard_cards(pes, device), shard_cards(luts, device)

        K = cfg.top_k
        self.hn = buf(device, H)
        self.hr, self.rl, self.probs = buf(device, H), buf(device, self.Etot), buf(device, self.Etot)
        self.vals, self.idx, self.wts = (buf(device, K * TILE) for _ in range(3))
        self.idx_l = buf(device, K * TILE)
        self.g, self.u, self.act = buf(device, P), buf(device, P), buf(device, P)
        self.dense, self.h1, self.hn2, self.h2 = (buf(device, H) for _ in range(4))
        self.gu = buf(device, 2 * I, K * TILE)
        self.eact = buf(device, I, K * TILE)
        self.dn = buf(device, H, K * TILE)
        self.pack = buf(device, H, 2 * TILE)
        self.comb, self.combn, self.out = (buf(device, H) for _ in range(3))

    def step(self, h_d):
        cfg, K, E = self.cfg, self.cfg.top_k, self.E
        H, P, Ht, It = cfg.hidden, MLP_PAD, self.Ht, self.It
        Et = self.Etot // TILE
        norm = atom(make_rmsnorm, 1, 1, Ht, 11, H, cfg.eps)
        cp = atom(make_copy, 1, 1, Ht, 11)

        norm(h_d, self.g_router, self.hr)
        atom(make_gemv, TILE, H, self.Etot, (1, 2), 1)(self.hr, self.w_router, self.rl)
        atom(make_softmax_row, Et)(self.rl, self.probs)
        atom(make_topk, 1, 1, Et, K, self.Etot)(self.probs, self.ramp, self.vals, self.idx)
        atom(make_moe_weights, K, Et)(self.vals, self.idx, self.pe, self.wts)
        atom(make_idx_gather, K, Et)(self.idx, self.lut, self.idx_l)

        norm(h_d, self.g_preffw, self.hn)
        gate = atom(make_gemv, TILE, H, P, (9, 2), 2)
        gate(self.hn, self.w_gate, self.g)
        gate(self.hn, self.w_up, self.u)
        atom(make_swiglu, 1, 1, P // TILE, P // TILE)(self.g, self.u, self.act)
        atom(make_gemv, TILE, P, H, (11, 2), 4)(self.act, self.w_down, self.dense)

        norm(h_d, self.g_preffw2, self.hn2)
        atom(make_indexed_gemv, E, H, 2 * self.I, K, (11, 2), 4, idx_stride=TILE)(
            self.hn2, self.idx_l, self.w_gu, self.gu)
        for t in range(K):
            atom(make_swiglu, 1, 1, It, It,
                 a_off=(t, 0), b_off=(t, It), out_off=(t, 0))(self.gu, self.gu, self.eact)
            atom(make_row_scale, self.I, It, a_row=t, s_col=t, out_row=t)(
                self.eact, self.wts, self.eact)
        for t in range(K):
            atom(make_indexed_gemv, E, self.I, H, 1, (11, 2), 4,
                 x_row=t, idx_col=t * TILE, out_row=t)(
                self.eact, self.idx_l, self.w_dn, self.dn)
        for t in range(1, K):
            atom(make_add, 1, 1, Ht, 11, b_off=(t, 0))(self.dn, self.dn, self.dn)

        cp(self.dense, self.pack)
        atom(make_copy, 1, 1, Ht, 11, out_off=(1, 0))(self.dn, self.pack)
        pack = all_reduce(self.device, self.pack)
        cp(pack, self.dense)
        atom(make_copy, 1, 1, Ht, 11, a_off=(1, 0))(pack, self.dn)
        if pack is not self.pack:
            ttnn.deallocate(pack)
        norm(self.dense, self.g_postffw1, self.h1)
        norm(self.dn, self.g_postffw2, self.h2)

        atom(make_add, 1, 1, Ht, 11)(self.h1, self.h2, self.comb)
        norm(self.comb, self.g_postffw, self.combn)
        atom(make_scaled_add, Ht, self.layer_scalar)(h_d, self.combn, self.out)
        return self.out


class SlidingChain:
    """Sliding layer; 4 Q + 2 KV heads per card, O row-shard + AR."""

    def __init__(self, ws, device, cfg, st):
        self.device, self.cfg, self.st = device, cfg, st
        w = ws[0]
        H, D, S = cfg.hidden, cfg.head_dim, cfg.sliding_window
        self.D, self.S = D, S
        self.Ht, self.Dt = H // TILE, D // TILE
        self.g_in = row(w["g_in"], H, device)
        # One tile row per norm: slicing is tile-granular.
        qkv_n = torch.zeros(3 * TILE, D, dtype=torch.bfloat16)
        for i, k in enumerate(("q_norm", "k_norm", "v_norm")):
            qkv_n[i * TILE] = w[k].to(torch.bfloat16)
        self.qknorm = to_dev(qkv_n, device)
        self.w_qkv = shard_cards([c["w_qkv"] for c in ws], device)
        self.w_o = shard_cards([c["w_o"] for c in ws], device)
        self.g_postattn = row(w["g_postattn"], H, device)
        half = D // 2
        R = torch.zeros(D, D)
        for j in range(half):
            R[half + j, j], R[j, half + j] = -1.0, 1.0
        self.R = to_dev(R.to(torch.bfloat16), device)
        self.caches = [buf(device, D, S) for _ in range(4)]
        self.heads = buf(device, D, 8 * TILE)
        self.o_row = buf(device, 4 * D)
        self.attn, self.attn_n, self.h = (buf(device, H) for _ in range(3))
        self.ffn = FFNChain(ws, device, cfg)

    def step(self, x_d):
        cfg, st = self.cfg, self.st
        H, D, S, Dt, Ht = cfg.hidden, self.D, self.S, self.Dt, self.Ht

        atom(make_attn_heads_atom, Ht, Dt, cfg.eps)(
            x_d, self.g_in, self.w_qkv, st.cos_sl, st.sin_sl,
            self.qknorm, self.R, self.heads)
        for i, cache in enumerate(self.caches):
            atom(make_kv_append, S // TILE, Dt, k_row=4 + i)(
                cache, self.heads, st.pos_ring, cache)
        atom(make_flash_atom, Dt, S // TILE)(
            self.heads, *self.caches, st.mask_sl, self.o_row)
        atom(make_gemv, TILE, 4 * D, H, (8, 2), 11)(self.o_row, self.w_o, self.attn)
        attn = all_reduce(self.device, self.attn)

        norm = atom(make_rmsnorm, 1, 1, Ht, 11, H, cfg.eps)
        norm(attn, self.g_postattn, self.attn_n)
        if attn is not self.attn:
            ttnn.deallocate(attn)
        atom(make_add, 1, 1, Ht, 11)(x_d, self.attn_n, self.h)
        return self.ffn.step(self.h)


class GlobalChain:
    """Global layer: Q heads shard 4/card; each card stages only the one
    KV head its Q group reads (w_k sliced per card), so the SPMD step is
    uniform. O partials AR. Seq-shard + flash combine is the planned
    optimization."""

    def __init__(self, ws, device, cfg, st, ctx):
        self.device, self.cfg, self.st = device, cfg, st
        w = ws[0]
        H, D = cfg.hidden, cfg.global_head_dim
        self.D, self.S = D, ctx
        self.qh = 4
        self.kvh = cfg.global_kv_heads if len(ws) == 1 else 1
        self.Ht, self.Dt = H // TILE, D // TILE
        self.rot = int(D * cfg.global_rot_frac)
        self.g_in = row(w["g_in"], H, device)
        self.q_norm = row(w["q_norm"], D, device)
        self.k_norm = row(w["k_norm"], D, device)
        self.w_q = shard_cards([c["w_q"] for c in ws], device)
        self.w_k = shard_cards([c["w_k"] for c in ws], device)
        self.w_o = shard_cards([c["w_o"] for c in ws], device)
        self.g_postattn = row(w["g_postattn"], H, device)
        self.caches = [buf(device, D, ctx) for _ in range(self.kvh)]
        self.xn, self.q, self.k = buf(device, H), buf(device, self.qh * D), buf(device, self.kvh * D)
        self.hd, self.hh, self.hr = (buf(device, D) for _ in range(3))
        self.o, self.of = buf(device, D), buf(device, D)
        self.m, self.l = buf(device, TILE), buf(device, TILE)
        self.a_row, self.attn = buf(device, self.qh * D), buf(device, H)
        self.attn_n, self.h = buf(device, H), buf(device, H)
        self.ffn = FFNChain(ws, device, cfg)

    def head_rot(self, src, hidx, nw):
        cfg, st, Dt = self.cfg, self.st, self.Dt
        atom(make_copy, 1, 1, Dt, Dt, a_off=(0, hidx * Dt))(src, self.hd)
        atom(make_rmsnorm, 1, 1, Dt, Dt, self.D, cfg.eps)(self.hd, nw, self.hh)
        atom(make_rope, Dt, self.rot // TILE)(self.hh, st.cos_gl, st.sin_gl, self.hr)
        return self.hr

    def step(self, x_d):
        cfg, st = self.cfg, self.st
        H, D, S, Dt, Ht = cfg.hidden, self.D, self.S, self.Dt, self.Ht
        chunk = 8 * TILE

        atom(make_rmsnorm, 1, 1, Ht, 11, H, cfg.eps)(x_d, self.g_in, self.xn)
        atom(make_gemv, TILE, H, self.qh * D, (8, 2), 2)(self.xn, self.w_q, self.q)
        atom(make_gemv, TILE, H, self.kvh * D, (8, 2), 2)(self.xn, self.w_k, self.k)

        for kv in range(self.kvh):
            kr = self.head_rot(self.k, kv, self.k_norm)
            atom(make_kv_append, S // TILE, Dt)(self.caches[kv], kr, st.pos_abs, self.caches[kv])
        for h in range(self.qh):
            qr = self.head_rot(self.q, h, self.q_norm)
            atom(make_flash_decode_kev, 1, 1, 1, Dt, 8, S // chunk)(
                qr, self.caches[h // (self.qh // self.kvh)], st.mask_gl, self.o, self.m, self.l)
            atom(make_row_scale, Dt, 8, recip=True)(self.o, self.l, self.of)
            atom(make_copy, 1, 1, Dt, Dt, out_off=(0, h * Dt))(self.of, self.a_row)

        atom(make_gemv, TILE, self.qh * D, H, (11, 2), 4)(self.a_row, self.w_o, self.attn)
        attn = all_reduce(self.device, self.attn)
        norm = atom(make_rmsnorm, 1, 1, Ht, 11, H, cfg.eps)
        norm(attn, self.g_postattn, self.attn_n)
        if attn is not self.attn:
            ttnn.deallocate(attn)
        atom(make_add, 1, 1, Ht, 11)(x_d, self.attn_n, self.h)
        return self.ffn.step(self.h)


class StepState:
    """Device-resident step state: pos tiles, LUTs, rope/mask staging."""

    def __init__(self, device, cfg, ctx):
        D, rot = cfg.head_dim, int(cfg.global_head_dim * cfg.global_rot_frac)
        S, smax = cfg.sliding_window, ctx
        self.S, self.ctx = S, ctx
        self.n_sl, self.n_gl = S // CHUNK, ctx // CHUNK
        f32 = ttnn.float32

        self.pos_abs = to_dev(torch.zeros(TILE, TILE), device, f32)
        self.pos_ring = to_dev(torch.zeros(TILE, TILE), device, f32)
        self.lut_abs = to_dev(pos_lut(smax), device, f32)
        self.lut_ring = to_dev(pos_lut(smax, S), device, f32)

        c, s = rope_table(smax, D, cfg.rope_theta)
        self.cos_sl_t, self.sin_sl_t = to_dev(c, device), to_dev(s, device)
        c, s = rope_table(smax, rot, cfg.global_rope_theta)
        self.cos_gl_t, self.sin_gl_t = to_dev(c, device), to_dev(s, device)
        self.mask_sl_t = to_dev(mask_table(smax, S).to(torch.bfloat16), device)
        self.mask_gl_t = to_dev(mask_table(ctx, ctx).to(torch.bfloat16), device)

        self.cos_sl, self.sin_sl = buf(device, D), buf(device, D)
        self.cos_gl, self.sin_gl = buf(device, rot), buf(device, rot)
        self.mask_sl = buf(device, CHUNK, self.n_sl * TILE)
        self.mask_gl = buf(device, CHUNK, self.n_gl * TILE)
        self.D, self.rot = D, rot

    def prime(self, device, pos):
        self.pos_abs = to_dev(split_tile(pos), device, ttnn.float32)
        self.pos_ring = to_dev(split_tile(pos, self.S), device, ttnn.float32)

    def step(self):
        Dt, rt = self.D // TILE, self.rot // TILE
        atom(make_pos_slice, Dt)(self.cos_sl_t, self.pos_abs, self.cos_sl)
        atom(make_pos_slice, Dt)(self.sin_sl_t, self.pos_abs, self.sin_sl)
        atom(make_pos_slice, rt)(self.cos_gl_t, self.pos_abs, self.cos_gl)
        atom(make_pos_slice, rt)(self.sin_gl_t, self.pos_abs, self.sin_gl)
        ct = CHUNK // TILE
        for c in range(self.n_sl):
            atom(make_pos_slice, ct, col_off=c * ct, out_row=c)(
                self.mask_sl_t, self.pos_abs, self.mask_sl)
        for c in range(self.n_gl):
            atom(make_pos_slice, ct, col_off=c * ct, out_row=c)(
                self.mask_gl_t, self.pos_abs, self.mask_gl)

    def advance(self):
        atom(make_pos_step)(self.lut_abs, self.pos_abs, self.pos_abs)
        atom(make_pos_step)(self.lut_ring, self.pos_ring, self.pos_ring)


class DecodeChain:
    """Embed -> layers -> final norm -> lm_head -> argmax -> next token."""

    def __init__(self, layers, st, embed, g_final, lm_head, device, cfg):
        self.layers, self.st, self.device, self.cfg = layers, st, device, cfg
        H = cfg.hidden
        self.Ht, self.V = H // TILE, lm_head.shape[1]
        self.n_chunks = self.V // CHUNK
        nt = (self.n_chunks + TILE - 1) // TILE
        self.nt = nt
        self.table = to_dev(embed.to(torch.bfloat16), device)
        self.g_final = row(g_final, H, device)
        self.lm = to_dev(lm_head.to(torch.bfloat16), device)
        self.x, self.xn = buf(device, H), buf(device, H)
        self.logits = buf(device, self.V)
        self.tok = buf(device, TILE)

        self.tall = buf(device, CHUNK, self.n_chunks * TILE)
        self.vals = buf(device, TILE, self.n_chunks * TILE)
        self.ids = buf(device, TILE, self.n_chunks * TILE)
        cw = torch.zeros(TILE, nt * TILE)
        cw[0] = -1e30
        self.cw = to_dev(cw.to(torch.bfloat16), device)
        self.wv, self.wi = buf(device, TILE), buf(device, TILE)
        self.ramp256 = to_dev(torch.arange(CHUNK, dtype=torch.bfloat16)
                              .unsqueeze(0).repeat(TILE, 1), device)
        self.rampn = to_dev(torch.arange(nt * TILE, dtype=torch.bfloat16)
                            .unsqueeze(0).repeat(TILE, 1), device)
        stage = torch.zeros(3 * TILE, CHUNK)
        stage[0] = torch.arange(CHUNK) // TILE
        stage[1] = torch.arange(CHUNK) % TILE
        stage[2, :self.n_chunks] = torch.arange(self.n_chunks) * (CHUNK // TILE)
        self.stage = to_dev(stage.to(torch.bfloat16), device)
        self.zero = buf(device, TILE)
        self.tok_a, self.tok_b = buf(device, TILE), buf(device, TILE)

    def prime(self, tok, pos):
        self.tok = to_dev(split_tile(tok).to(torch.bfloat16), self.device)
        self.st.prime(self.device, pos)

    def step(self):
        cfg, H, st = self.cfg, self.cfg.hidden, self.st
        st.step()
        atom(make_embed_gather, self.Ht, H ** 0.5)(self.table, self.tok, self.x)
        x = self.x
        for layer in self.layers:
            x = layer.step(x)
        atom(make_rmsnorm, 1, 1, self.Ht, 11, H, cfg.eps)(x, self.g_final, self.xn)
        atom(make_gemv, TILE, H, self.V, (8, 2), 4)(self.xn, self.lm, self.logits)

        n = self.n_chunks
        atom(make_restack, n)(self.logits, self.tall)
        atom(make_topk, n, 1, CHUNK // TILE, 1, CHUNK)(self.tall, self.ramp256, self.vals, self.ids)
        atom(make_collapse, n)(self.vals, self.cw)
        atom(make_collapse, n, out_row=1)(self.ids, self.stage)
        atom(make_topk, 1, 1, self.nt, 1, self.nt * TILE)(self.cw, self.rampn, self.wv, self.wi)
        atom(make_copy, 1, 1, 1, 1, out_off=(2, 0))(self.wi, self.stage)
        atom(make_token_select, n)(self.stage, self.zero, self.tok_a, self.tok_b)
        atom(make_add, 1, 1, 1, 1)(self.tok_a, self.tok_b, self.tok)
        st.advance()

    def read_token(self):
        t = from_dev(self.tok)
        return int(t[0, 0]) * TILE + int(t[0, 1])
