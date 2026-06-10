# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Full per-card sliding layer as a dispatch chain vs torch reference.

Composes the attention, dense-FFN, and experts chains: attn -> residual ->
dense MLP || routed experts (dual norms) -> combine -> post norm -> residual
* layer_scalar. Router is host scaffolding; everything else on device.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttl.ops.elementwise import make_add, make_binary, make_row_scale
from ttl.ops.gemv import make_gemv
from ttl.ops.indexed_gemv import make_indexed_gemv
from ttl.ops.kv_append import make_kv_append
from ttl.ops.rmsnorm import make_rmsnorm
from ttl.ops.swiglu import make_swiglu
from gemma4.attn_atom import TILE, make_attn_heads_atom, make_flash_atom
from gemma4.layer_test_scaffolding import MLP_PAD, from_dev, row, to_dev


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * w


def gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def make_swiglu_t(It, t):
    return make_binary("swiglu", 1, 1, It, It,
                       a_off=(t, 0), b_off=(t, It), out_off=(t, 0))


def test_layer_chain():
    device = ttnn.open_device(device_id=0)
    try:
        _run(device)
    finally:
        ttnn.close_device(device)


def _run(device):
    torch.manual_seed(0)
    H, D, S, eps, theta = 2816, 256, 1024, 1e-6, 10000.0
    P, I, E, K = MLP_PAD, 704, 32, 8
    inter, It, Ht = 528, I // TILE, 2816 // TILE
    pos, half = 100, D // 2
    layer_scalar = 0.9

    x = torch.randn(H) * 0.5
    g_in = 1 + torch.randn(H) * 0.1
    qknorm = 1 + torch.randn(D) * 0.1
    wqkv = torch.randn(H, 8 * D) * 0.02
    wo = torch.randn(4 * D, H) * 0.02
    k_cache = [torch.randn(S, D) * 0.3 for _ in range(2)]
    v_cache = [torch.randn(S, D) * 0.3 for _ in range(2)]
    g_postattn = 1 + torch.randn(H) * 0.1
    g_preffw = 1 + torch.randn(H) * 0.1
    g_postffw1 = 1 + torch.randn(H) * 0.1
    g_preffw2 = 1 + torch.randn(H) * 0.1
    g_postffw2 = 1 + torch.randn(H) * 0.1
    g_postffw = 1 + torch.randn(H) * 0.1
    w_gate = torch.zeros(H, P)
    w_up = torch.zeros(H, P)
    w_down = torch.zeros(P, H)
    w_gate[:, :inter] = torch.randn(H, inter) * 0.02
    w_up[:, :inter] = torch.randn(H, inter) * 0.02
    w_down[:inter] = torch.randn(inter, H) * 0.02
    w_gu = torch.randn(E, H, 2 * I) * 0.02
    w_dn = torch.randn(E, I, H) * 0.02
    idx = torch.randperm(E)[:K]
    wts = torch.softmax(torch.randn(K), dim=0)

    inv = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
    f = pos * inv
    cos, sin = torch.cat([f.cos(), f.cos()]), torch.cat([f.sin(), f.sin()])
    R = torch.zeros(D, D)
    for j in range(half):
        R[half + j, j], R[j, half + j] = -1.0, 1.0

    # reference
    xn = rms(x, g_in, eps)
    heads = [xn @ wqkv[:, i * D:(i + 1) * D] for i in range(8)]
    q = [rms(h, qknorm, eps) for h in heads[:4]]
    q = [h * cos + (h @ R) * sin for h in q]
    k = [rms(h, qknorm, eps) for h in heads[4:6]]
    k = [h * cos + (h @ R) * sin for h in k]
    v = [rms(h, qknorm, eps) for h in heads[6:8]]
    kc, vc = [c.clone() for c in k_cache], [c.clone() for c in v_cache]
    for i in range(2):
        kc[i][pos], vc[i][pos] = k[i], v[i]
    mask = torch.full((S,), float("-inf"))
    mask[:pos + 1] = 0.0
    outs = []
    for hh in range(4):
        att = torch.softmax(q[hh] @ kc[hh // 2].T + mask, dim=-1)
        outs.append(att @ vc[hh // 2])
    attn = torch.cat(outs) @ wo

    h = x + rms(attn, g_postattn, eps)
    hn = rms(h, g_preffw, eps)
    dense = (gelu_tanh(hn @ w_gate) * (hn @ w_up)) @ w_down
    h1 = rms(dense, g_postffw1, eps)
    hn2 = rms(h, g_preffw2, eps)
    exp = torch.zeros(H)
    for t in range(K):
        gu = hn2 @ w_gu[idx[t]]
        exp += (gelu_tanh(gu[:I]) * gu[I:] * wts[t]) @ w_dn[idx[t]]
    h2 = rms(exp, g_postffw2, eps)
    want = (h + rms(h1 + h2, g_postffw, eps)) * layer_scalar

    # device
    pos_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    pos_t[0, 0], pos_t[0, 1] = pos // TILE, pos % TILE
    n_chunks, chunk = 4, S // 4
    masks = mask.reshape(n_chunks, 1, chunk).expand(n_chunks, TILE, chunk)
    masks = masks.reshape(n_chunks * TILE, chunk).to(torch.bfloat16)
    idx_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    idx_t[0, :K] = idx.float()

    def buf(n, rows=TILE):
        return to_dev(torch.zeros(rows, n, dtype=torch.bfloat16), device)

    qkv_n = torch.zeros(3 * TILE, D, dtype=torch.bfloat16)
    for i in range(3):
        qkv_n[i * TILE] = qknorm.to(torch.bfloat16)
    attn_args = [row(x, H, device), row(g_in, H, device),
                 to_dev(wqkv.to(torch.bfloat16), device),
                 to_dev(cos.expand(TILE, D).contiguous().to(torch.bfloat16), device),
                 to_dev(sin.expand(TILE, D).contiguous().to(torch.bfloat16), device),
                 to_dev(qkv_n, device), to_dev(R.to(torch.bfloat16), device)]
    caches = [to_dev(c.to(torch.bfloat16), device)
              for c in (k_cache[0], k_cache[1], v_cache[0], v_cache[1])]
    heads_dev, o_row, attn_d = buf(D, 8 * TILE), buf(4 * D), buf(H)
    attn_n, h_d, hn_d, hn2_d = buf(H), buf(H), buf(H), buf(H)
    g_d, u_d, act_d, dense_d = buf(P), buf(P), buf(P), buf(H)
    h1_d, h2_d, comb_d, combn_d, out_d = buf(H), buf(H), buf(H), buf(H), buf(H)
    gu_d, eact_d, dn_d = buf(2 * I, K * TILE), buf(I, K * TILE), buf(H, K * TILE)

    norm = make_rmsnorm(1, 1, Ht, 11, H, eps)
    add = make_add(1, 1, Ht, 11)

    make_attn_heads_atom(Ht, D // TILE, eps)(*attn_args, heads_dev)
    for i, cache in enumerate(caches):
        make_kv_append(S // TILE, D // TILE, k_row=4 + i)(
            cache, heads_dev, to_dev(pos_t, device), cache)
    make_flash_atom(D // TILE, S // TILE)(
        heads_dev, *caches, to_dev(masks, device), o_row)
    make_gemv(TILE, 4 * D, H, (8, 2), 11)(
        o_row, to_dev(wo.to(torch.bfloat16), device), attn_d)

    norm(attn_d, row(g_postattn, H, device), attn_n)
    add(attn_args[0], attn_n, h_d)

    norm(h_d, row(g_preffw, H, device), hn_d)
    gate = make_gemv(TILE, H, P, (9, 2), 2)
    gate(hn_d, to_dev(w_gate.to(torch.bfloat16), device), g_d)
    gate(hn_d, to_dev(w_up.to(torch.bfloat16), device), u_d)
    make_swiglu(1, 1, P // TILE, P // TILE)(g_d, u_d, act_d)
    make_gemv(TILE, P, H, (11, 2), 4)(
        act_d, to_dev(w_down.to(torch.bfloat16), device), dense_d)
    norm(dense_d, row(g_postffw1, H, device), h1_d)

    norm(h_d, row(g_preffw2, H, device), hn2_d)
    make_indexed_gemv(E, H, 2 * I, K, (11, 2), 4)(
        hn2_d, to_dev(idx_t, device), to_dev(w_gu.reshape(E * H, 2 * I).to(torch.bfloat16), device), gu_d)
    # row_scale s_col selects tile columns: weight t lives at element (0, 32*t).
    wts_t = torch.zeros(TILE, K * TILE, dtype=torch.bfloat16)
    wts_t[0, 0::TILE] = wts
    wts_d = to_dev(wts_t, device)
    for t in range(K):
        make_swiglu_t(It, t)(gu_d, gu_d, eact_d)
        make_row_scale(It, It, a_row=t, s_col=t, out_row=t)(eact_d, wts_d, eact_d)
    w_dn_dev = to_dev(w_dn.reshape(E * I, H).to(torch.bfloat16), device)
    for t in range(K):
        make_indexed_gemv(E, I, H, 1, (11, 2), 4, x_row=t, out_row=t, idx_col=t)(
            eact_d, to_dev(idx_t, device), w_dn_dev, dn_d)
    for t in range(1, K):
        make_add(1, 1, Ht, 11, b_off=(t, 0))(dn_d, dn_d, dn_d)
    norm(dn_d, row(g_postffw2, H, device), h2_d)

    add(h1_d, h2_d, comb_d)
    norm(comb_d, row(g_postffw, H, device), combn_d)
    make_binary("scaled_add", 1, 1, Ht, 11, scalar=layer_scalar)(h_d, combn_d, out_d)

    got = from_dev(out_d)[0]
    pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
    assert pcc > 0.98, f"layer pcc {pcc}"
