# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Fused attn heads atom (norm + QKV + QK-norm + RoPE) vs torch."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from gemma4.attn_atom import TILE, make_attn_heads_atom
from gemma4.layer_test_scaffolding import from_dev, row, to_dev


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * w


def test_attn_heads_atom(device):
    torch.manual_seed(0)
    H, D, eps, pos, theta = 2816, 256, 1e-6, 7, 10000.0
    half = D // 2

    x = torch.randn(H) * 0.5
    gamma = 1 + torch.randn(H) * 0.1
    qknorm = 1 + torch.randn(D) * 0.1
    wqkv = torch.randn(H, 8 * D) * 0.02

    inv = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
    f = pos * inv
    cos = torch.cat([f.cos(), f.cos()])
    sin = torch.cat([f.sin(), f.sin()])

    R = torch.zeros(D, D)
    for j in range(half):
        R[half + j, j] = -1.0
        R[j, half + j] = 1.0

    xn = rms(x, gamma, eps)
    heads_ref = []
    for hcol in range(8):
        h = xn @ wqkv[:, hcol * D:(hcol + 1) * D]
        hn = rms(h, qknorm, eps)
        if hcol < 6:
            h = hn * cos + (hn @ R) * sin
        else:
            h = hn
        heads_ref.append(h)

    cos32 = cos.expand(TILE, D).contiguous()
    sin32 = sin.expand(TILE, D).contiguous()

    x_d = row(x, H, device)
    gamma_d = row(gamma, H, device)
    qknorm_d = row(qknorm, D, device)
    wqkv_d = to_dev(wqkv.to(torch.bfloat16), device)
    cos_d = to_dev(cos32.to(torch.bfloat16), device)
    sin_d = to_dev(sin32.to(torch.bfloat16), device)
    rot_d = to_dev(R.to(torch.bfloat16), device)
    heads_d = to_dev(torch.zeros(8 * TILE, D, dtype=torch.bfloat16), device)

    atom = make_attn_heads_atom(H // TILE, D // TILE, eps)
    atom(x_d, gamma_d, wqkv_d, cos_d, sin_d, qknorm_d, rot_d, heads_d)

    got = from_dev(heads_d).reshape(8, TILE, D)[:, 0, :]
    for i in range(8):
        pcc = torch.corrcoef(torch.stack([got[i], heads_ref[i]]))[0, 1].item()
        assert pcc > 0.99, f"head {i}: pcc {pcc}"
