# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Per-card global attention dispatch chain vs torch (card 0 of 4).

V = K, 2 KV heads replicated, K cache sequence-sharded in 32-row granules
(card owns granule % 4 == card), partial RoPE (first 128 of 512, theta 1e6).
Output is the pre-AR o-projection partial; cross-card flash combine comes
with TP.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttl.ops.elementwise import make_copy, make_row_scale
from ttl.ops.flash_decode import make_flash_decode_kev
from ttl.ops.gemv import make_gemv
from ttl.ops.kv_append import make_kv_append
from ttl.ops.rmsnorm import make_rmsnorm
from ttl.ops.rope import make_rope
from gemma4.layer_test_scaffolding import from_dev, row, to_dev

TILE = 32


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * w


def rope_partial(h, cos, sin, rot):
    half = rot // 2
    x1, x2 = h[:half], h[half:rot]
    y1 = x1 * cos[:half] - x2 * sin[:half]
    y2 = x2 * cos[half:] + x1 * sin[half:]
    return torch.cat([y1, y2, h[rot:]])


def test_global_chain():
    device = ttnn.open_device(device_id=0)
    try:
        _run(device)
    finally:
        ttnn.close_device(device)


def _run(device):
    torch.manual_seed(0)
    H, D, rot, eps, theta = 2816, 512, 128, 1e-6, 1e6
    qh, kvh, card = 4, 2, 0
    S = 1024            # local shard of 4096 ctx
    pos = 128           # granule 4 -> card 0, local row 32
    local = 32
    Dt, rot_t, Ht = D // TILE, rot // TILE, H // TILE

    x = torch.randn(H) * 0.5
    g_in = 1 + torch.randn(H) * 0.1
    qknorm = 1 + torch.randn(D) * 0.1
    wq = torch.randn(H, qh * D) * 0.02
    wk = torch.randn(H, kvh * D) * 0.02
    wo = torch.randn(qh * D, H) * 0.02
    k_cache = [torch.randn(S, D) * 0.3 for _ in range(kvh)]

    inv = 1.0 / (theta ** (torch.arange(0, rot, 2).float() / rot))
    f = pos * inv
    cos, sin = torch.cat([f.cos(), f.cos()]), torch.cat([f.sin(), f.sin()])

    # reference over the local shard
    xn = rms(x, g_in, eps)
    q = [rope_partial(rms(xn @ wq[:, h * D:(h + 1) * D], qknorm, eps), cos, sin, rot)
         for h in range(qh)]
    k = [rope_partial(rms(xn @ wk[:, h * D:(h + 1) * D], qknorm, eps), cos, sin, rot)
         for h in range(kvh)]
    kc = [c.clone() for c in k_cache]
    for i in range(kvh):
        kc[i][local] = k[i]
    mask = torch.full((S,), float("-inf"))
    mask[:local + 1] = 0.0
    outs = []
    for h in range(qh):
        att = torch.softmax(q[h] @ kc[h // 2].T + mask, dim=-1)
        outs.append(att @ kc[h // 2])
    want = torch.cat(outs) @ wo

    # device
    pos_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    pos_t[0, 0], pos_t[0, 1] = local // TILE, local % TILE
    chunk = 8 * TILE
    n_chunks = S // chunk
    masks = mask.reshape(n_chunks, 1, chunk).expand(n_chunks, TILE, chunk)
    masks_d = to_dev(masks.reshape(n_chunks * TILE, chunk).to(torch.bfloat16), device)
    cos_d = to_dev(cos.expand(TILE, rot).contiguous().to(torch.bfloat16), device)
    sin_d = to_dev(sin.expand(TILE, rot).contiguous().to(torch.bfloat16), device)

    def buf(n, rows=TILE):
        return to_dev(torch.zeros(rows, n, dtype=torch.bfloat16), device)

    xn_d, q_d, k_d = buf(H), buf(qh * D), buf(kvh * D)
    a_row, o_part = buf(qh * D), buf(H)
    caches = [to_dev(c.to(torch.bfloat16), device) for c in k_cache]

    make_rmsnorm(1, 1, Ht, 11, H, eps)(row(x, H, device), row(g_in, H, device), xn_d)
    make_gemv(TILE, H, qh * D, (8, 2), 2)(xn_d, to_dev(wq.to(torch.bfloat16), device), q_d)
    make_gemv(TILE, H, kvh * D, (8, 2), 2)(xn_d, to_dev(wk.to(torch.bfloat16), device), k_d)

    hnorm = make_rmsnorm(1, 1, Dt, Dt, D, eps)
    rope = make_rope(Dt, rot_t)
    qk_w = row(qknorm, D, device)

    def head_rot(src, h):
        hd, hn, hr = buf(D), buf(D), buf(D)
        make_copy(1, 1, Dt, Dt, a_off=(0, h * Dt))(src, hd)
        hnorm(hd, qk_w, hn)
        rope(hn, cos_d, sin_d, hr)
        return hr

    for kv in range(kvh):
        kr = head_rot(k_d, kv)
        make_kv_append(S // TILE, Dt)(caches[kv], kr, to_dev(pos_t, device), caches[kv])

    flash = make_flash_decode_kev(1, 1, 1, Dt, 8, n_chunks)
    fin = make_row_scale(Dt, 8, recip=True)
    for h in range(qh):
        qr = head_rot(q_d, h)
        o, m, l = buf(D), buf(TILE), buf(TILE)
        flash(qr, caches[h // 2], masks_d, o, m, l)
        of = buf(D)
        fin(o, l, of)
        make_copy(1, 1, Dt, Dt, out_off=(0, h * Dt))(of, a_row)

    make_gemv(TILE, qh * D, H, (11, 2), 4)(a_row, to_dev(wo.to(torch.bfloat16), device), o_part)

    got = from_dev(o_part)[0]
    pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
    assert pcc > 0.98, f"global pcc {pcc}"
