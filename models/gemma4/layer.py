# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-card Gemma 4 decode layer, v1: each matmul/norm/flash is a ttl atom,
glued by host adds and slicing. TP=4 partial outputs (pre all-reduce).
The fused per-layer atom replaces this glue with DFB residency; until
then this defines the sharding and the per-layer compute graph.
"""

import torch
import ttnn

from ttl.ops.gemv import make_gemv
from ttl.ops.rmsnorm import make_rmsnorm
from ttl.ops.rope import make_rope
from ttl.ops.kv_append import make_kv_append
from ttl.ops.flash_decode import make_flash_decode
from ttl.ops.swiglu import make_swiglu
from ttl.ops.indexed_gemv import make_indexed_gemv

from .config import TILE, Gemma4Config


def to_dev(t, device, dtype=ttnn.bfloat16, mem=None):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=mem or ttnn.DRAM_MEMORY_CONFIG)


def from_dev(t):
    return ttnn.to_torch(t).float()


def row(t, D, device):
    """Host [D] -> [TILE, D] tile row tensor on device."""
    z = torch.zeros(TILE, D, dtype=torch.bfloat16)
    z[0] = t.to(torch.bfloat16)
    return to_dev(z, device)


class SlidingLayer:
    """Per-card weights and atoms for one sliding layer at TP=4."""

    def __init__(self, cfg: Gemma4Config, w, device, card):
        self.cfg, self.device = cfg, device
        H, D = cfg.hidden, cfg.head_dim
        qh, kvh = 4, 2
        self.qh, self.kvh, self.D = qh, kvh, D
        Ht, Dt = H // TILE, D // TILE
        self.Ht, self.Dt = Ht, Dt
        S = cfg.sliding_window
        self.S = S

        qs, ks = card * qh, card * kvh
        wq = w["q_proj"][qs * D:(qs + qh) * D]
        wk = w["k_proj"][ks * D:(ks + kvh) * D]
        wv = w["v_proj"][ks * D:(ks + kvh) * D]
        self.w_qkv = to_dev(torch.cat([wq, wk, wv]).T, device)            # [H, 8D]
        self.w_o = to_dev(w["o_proj"][:, qs * D:(qs + qh) * D].T, device)  # [4D, H]

        self.norm_in = row(1 + w["input_layernorm"], H, device)
        self.norm_post = row(1 + w["post_attention_layernorm"], H, device)
        self.q_norm = row(1 + w["q_norm"], D, device)
        self.k_norm = row(1 + w["k_norm"], D, device)
        self.v_norm = row(torch.ones(D), D, device)

        # One tensor per kv head: ttnn slicing copies, an aliasable view
        # would silently drop appends.
        self.k_cache = [to_dev(torch.zeros(S, D, dtype=torch.bfloat16),
                               device) for _ in range(kvh)]
        self.v_cache = [to_dev(torch.zeros(S, D, dtype=torch.bfloat16),
                               device) for _ in range(kvh)]

        n = qh + 2 * kvh
        self.qkv = make_gemv(TILE, H, n * D, (8, 2), 4)
        self.o_proj = make_gemv(TILE, qh * D, H, (11, 2), 4)
        self.norm = make_rmsnorm(1, 1, Ht, 11, H, cfg.eps)
        self.hnorm = make_rmsnorm(1, 1, Dt, Dt, D, cfg.eps)
        self.rope = make_rope(Dt, Dt)
        self.append = make_kv_append(S // TILE, Dt)
        self.flash = make_flash_decode(1, 1, 1, Dt, Dt, 8, S // (8 * TILE))

    def attn(self, x_d, pos, cos, sin):
        """x_d: [TILE, H] device row. Returns o_partial [TILE, H] pre-AR."""
        cfg, dev = self.cfg, self.device
        H, D, S = cfg.hidden, self.D, self.S
        scale = 1.0

        xn = to_dev(torch.zeros(TILE, H, dtype=torch.bfloat16), dev)
        self.norm(x_d, self.norm_in, xn)

        qkv = to_dev(torch.zeros(TILE, (self.qh + 2 * self.kvh) * D, dtype=torch.bfloat16), dev)
        self.qkv(xn, self.w_qkv, qkv)
        qkv_h = from_dev(qkv)[:1]

        ring = pos % S
        pos_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
        pos_t[0, 0], pos_t[0, 1] = ring // TILE, ring % TILE

        valid = min(pos + 1, S)
        mrow = torch.full((S,), float("-inf"))
        mrow[:valid] = 0.0
        n_chunks = S // (8 * TILE)
        masks = mrow.reshape(n_chunks, 1, 8 * TILE).expand(n_chunks, TILE, 8 * TILE)
        masks_d = to_dev(masks.reshape(n_chunks * TILE, 8 * TILE).to(torch.bfloat16), dev)

        heads = []
        for h in range(self.qh):
            q = row(qkv_h[0, h * D:(h + 1) * D], D, dev)
            qn = to_dev(torch.zeros(TILE, D, dtype=torch.bfloat16), dev)
            self.hnorm(q, self.q_norm, qn)
            qr = to_dev(torch.zeros(TILE, D, dtype=torch.bfloat16), dev)
            self.rope(qn, cos, sin, qr)
            heads.append(qr)

        for kv in range(self.kvh):
            base = (self.qh + kv) * D
            k = row(qkv_h[0, base:base + D], D, dev)
            kn = to_dev(torch.zeros(TILE, D, dtype=torch.bfloat16), dev)
            self.hnorm(k, self.k_norm, kn)
            kr = to_dev(torch.zeros(TILE, D, dtype=torch.bfloat16), dev)
            self.rope(kn, cos, sin, kr)
            kcache = self.k_cache[kv]
            self.append(kcache, kr, to_dev(pos_t, dev), kcache)

            vbase = (self.qh + self.kvh + kv) * D
            v = row(qkv_h[0, vbase:vbase + D], D, dev)
            vn = to_dev(torch.zeros(TILE, D, dtype=torch.bfloat16), dev)
            self.hnorm(v, self.v_norm, vn)
            vcache = self.v_cache[kv]
            self.append(vcache, vn, to_dev(pos_t, dev), vcache)

        outs = []
        for h in range(self.qh):
            kv = h // (self.qh // self.kvh)
            o = to_dev(torch.zeros(TILE, D, dtype=torch.bfloat16), dev)
            m = to_dev(torch.zeros(TILE, TILE, dtype=torch.bfloat16), dev)
            l = to_dev(torch.zeros(TILE, TILE, dtype=torch.bfloat16), dev)
            self.flash(heads[h], self.k_cache[kv], self.v_cache[kv],
                       masks_d, o, m, l)
            o_h = from_dev(o)[0] / from_dev(l)[0, 0]
            outs.append(o_h)

        a = row(torch.cat(outs), self.qh * D, dev)
        o_part = to_dev(torch.zeros(TILE, H, dtype=torch.bfloat16), dev)
        self.o_proj(a, self.w_o, o_part)
        return o_part


MLP_PAD = 576  # 2112 / 4 col-shard padded to tile alignment (Nt=18 keeps bands divisible)


class FFN:
    """Per-card dense MLP + routed experts at TP=4 (partial outputs)."""

    def __init__(self, cfg: Gemma4Config, w, device, card):
        self.cfg, self.device, self.card = cfg, device, card
        H, P, E = cfg.hidden, MLP_PAD, cfg.experts // 4
        self.E = E

        def pad_cols(t):
            return torch.nn.functional.pad(t, (0, P - t.shape[1]))

        def pad_rows(t):
            return torch.nn.functional.pad(t, (0, 0, 0, P - t.shape[0]))

        s = card * (cfg.mlp_inter // 4)
        e = s + cfg.mlp_inter // 4
        self.w_gate = to_dev(pad_cols(w["mlp.gate_proj"][s:e].T), device)  # [H, P]
        self.w_up = to_dev(pad_cols(w["mlp.up_proj"][s:e].T), device)
        self.w_down = to_dev(pad_rows(w["mlp.down_proj"][:, s:e].T), device)  # [P, H]

        es = card * E
        gu = w["experts.gate_up_proj"][es:es + E]                # [E, 2I, H]
        self.w_gu = to_dev(gu.transpose(1, 2).reshape(E * H, 2 * cfg.moe_inter), device)
        dn = w["experts.down_proj"][es:es + E]                   # [E, H, I]
        self.w_dn = to_dev(dn.transpose(1, 2).reshape(E * cfg.moe_inter, H), device)

        self.router_w = w["router.proj"]                          # full [128, H]
        self.router_scale = w["router.scale"]
        self.per_expert = w["router.per_expert_scale"]

        self.gate = make_gemv(TILE, H, P, (9, 2), 2)
        self.down = make_gemv(TILE, P, H, (11, 2), 4)
        self.swiglu = make_swiglu(1, 1, P // TILE, P // TILE)
        self.gu_e = make_indexed_gemv(E, H, 2 * cfg.moe_inter, 1, (11, 2), 4)
        self.dn_e = make_indexed_gemv(E, cfg.moe_inter, H, 1, (11, 2), 4)
        self.swiglu_e = make_swiglu(1, 1, cfg.moe_inter // TILE, cfg.moe_inter // TILE)

    def route(self, x):
        """Host router: norm-noscale * scale * H^-0.5, softmax, top-8."""
        cfg = self.cfg
        xn = x / torch.sqrt(x.pow(2).mean() + cfg.eps)
        xn = xn * self.router_scale * cfg.hidden ** -0.5
        probs = torch.softmax(self.router_w.float() @ xn.float(), dim=-1)
        wts, idx = torch.topk(probs, cfg.top_k)
        wts = wts / wts.sum()
        return idx, wts * self.per_expert[idx]

    def dense(self, h_d):
        cfg, dev = self.cfg, self.device
        H, P = cfg.hidden, MLP_PAD
        g = to_dev(torch.zeros(TILE, P, dtype=torch.bfloat16), dev)
        u = to_dev(torch.zeros(TILE, P, dtype=torch.bfloat16), dev)
        self.gate(h_d, self.w_gate, g)
        self.gate(h_d, self.w_up, u)
        act = to_dev(torch.zeros(TILE, P, dtype=torch.bfloat16), dev)
        self.swiglu(g, u, act)
        out = to_dev(torch.zeros(TILE, H, dtype=torch.bfloat16), dev)
        self.down(act, self.w_down, out)
        return out

    def experts(self, h_d, idx, wts):
        """Local experts only; ids outside this card's band are skipped
        (their contribution arrives via the all-reduce)."""
        cfg, dev = self.cfg, self.device
        H, I, E = cfg.hidden, cfg.moe_inter, self.E
        lo = self.card * E
        acc = torch.zeros(H)
        for ii, wt in zip(idx.tolist(), wts.tolist()):
            if not lo <= ii < lo + E:
                continue
            e = ii - lo
            idx_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
            idx_t[0, 0] = e
            idx_d = to_dev(idx_t, dev)
            gu = to_dev(torch.zeros(TILE, 2 * I, dtype=torch.bfloat16), dev)
            self.gu_e(h_d, idx_d, self.w_gu, gu)
            gu_h = from_dev(gu)[0]
            g = row(gu_h[:I], I, dev)
            u = row(gu_h[I:], I, dev)
            act = to_dev(torch.zeros(TILE, I, dtype=torch.bfloat16), dev)
            self.swiglu_e(g, u, act)
            dn = to_dev(torch.zeros(TILE, H, dtype=torch.bfloat16), dev)
            self.dn_e(act, idx_d, self.w_dn, dn)
            acc += from_dev(dn)[0] * wt
        return acc
