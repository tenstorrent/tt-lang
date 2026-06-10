# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Pre-AR sliding attention (patch atom + flash atom) vs torch reference."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.ops.gemv import make_gemv
from ttl.ops.kv_append import make_kv_append
from gemma4.attn_atom import TILE, make_attn_heads_atom, make_flash_atom
from gemma4.layer_test_scaffolding import from_dev, row, to_dev


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * w


def test_attn_full_atom():
    device = ttnn.open_device(device_id=0)
    try:
        _run(device)
    finally:
        ttnn.close_device(device)


def _run(device):
    torch.manual_seed(0)
    H, D, S, eps, theta = 2816, 256, 1024, 1e-6, 10000.0
    pos = 100
    half = D // 2

    x = torch.randn(H) * 0.5
    gamma = 1 + torch.randn(H) * 0.1
    qknorm = 1 + torch.randn(D) * 0.1
    wqkv = torch.randn(H, 8 * D) * 0.02
    wo = torch.randn(4 * D, H) * 0.02
    k_cache = [torch.randn(S, D) * 0.3 for _ in range(2)]
    v_cache = [torch.randn(S, D) * 0.3 for _ in range(2)]

    inv = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
    f = pos * inv
    cos = torch.cat([f.cos(), f.cos()])
    sin = torch.cat([f.sin(), f.sin()])
    R = torch.zeros(D, D)
    for j in range(half):
        R[half + j, j] = -1.0
        R[j, half + j] = 1.0

    # reference
    xn = rms(x, gamma, eps)
    heads = [xn @ wqkv[:, i * D:(i + 1) * D] for i in range(8)]
    q = [rms(h, qknorm, eps) for h in heads[:4]]
    q = [h * cos + (h @ R) * sin for h in q]
    k = [rms(h, qknorm, eps) for h in heads[4:6]]
    k = [h * cos + (h @ R) * sin for h in k]
    v = [rms(h, qknorm, eps) for h in heads[6:8]]
    kc = [c.clone() for c in k_cache]
    vc = [c.clone() for c in v_cache]
    for i in range(2):
        kc[i][pos] = k[i]
        vc[i][pos] = v[i]
    mask = torch.full((S,), float("-inf"))
    mask[:pos + 1] = 0.0
    outs = []
    for h in range(4):
        kv = h // 2
        att = torch.softmax(q[h] @ kc[kv].T + mask, dim=-1)
        outs.append(att @ vc[kv])
    want = torch.cat(outs) @ wo

    # device
    pos_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    pos_t[0, 0], pos_t[0, 1] = pos // TILE, pos % TILE
    n_chunks, chunk = 4, S // 4
    masks = mask.reshape(n_chunks, 1, chunk).expand(n_chunks, TILE, chunk)
    masks = masks.reshape(n_chunks * TILE, chunk).to(torch.bfloat16)

    args = [row(x, H, device), row(gamma, H, device),
            to_dev(wqkv.to(torch.bfloat16), device),
            to_dev(cos.expand(TILE, D).contiguous().to(torch.bfloat16), device),
            to_dev(sin.expand(TILE, D).contiguous().to(torch.bfloat16), device),
            row(qknorm, D, device), to_dev(R.to(torch.bfloat16), device)]
    caches = [to_dev(c.to(torch.bfloat16), device) for c in (k_cache[0], k_cache[1], v_cache[0], v_cache[1])]
    heads_dev = to_dev(torch.zeros(8 * TILE, D, dtype=torch.bfloat16), device)
    o_row = to_dev(torch.zeros(TILE, 4 * D, dtype=torch.bfloat16), device)
    o_proj = to_dev(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
    pos_dev = to_dev(pos_t, device)

    heads_atom = make_attn_heads_atom(H // TILE, D // TILE, eps)
    heads_atom(*args, heads_dev)
    for i, cache in enumerate(caches):
        append = make_kv_append(S // TILE, D // TILE, k_row=4 + i)
        append(cache, heads_dev, pos_dev, cache)
    flash_atom = make_flash_atom(D // TILE, S // TILE)
    flash_atom(heads_dev, *caches, to_dev(masks, device), o_row)
    make_gemv(TILE, 4 * D, H, (8, 2), 11)(o_row, to_dev(wo.to(torch.bfloat16), device), o_proj)

    got = from_dev(o_proj)[0].float()
    pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
    assert pcc > 0.98, f"o pcc {pcc}"
