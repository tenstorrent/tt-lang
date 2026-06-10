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
