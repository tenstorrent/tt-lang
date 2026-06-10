# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Embed gather -> final norm -> lm_head GEMV (tied weights) vs torch.

Small-vocab stand-in (2048 of the 65536-per-card shard; the GEMV streams
identically at full width). Greedy argmax over the logits readback
(softcap is monotone, so it cannot change the argmax; on-device wide
argmax is a perf-phase op).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.ops.embed_gather import make_embed_gather
from ttl.ops.gemv import make_gemv
from ttl.ops.rmsnorm import make_rmsnorm
from gemma4.layer import from_dev, row, to_dev

TILE = 32


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * w


def test_lm_head_chain():
    device = ttnn.open_device(device_id=0)
    try:
        _run(device)
    finally:
        ttnn.close_device(device)


def _run(device):
    torch.manual_seed(0)
    H, V, eps = 2816, 2048, 1e-6
    rows, tok = 4096, 1234
    scale = H ** 0.5

    table = torch.randn(rows, H) * 0.02
    g_final = 1 + torch.randn(H) * 0.1
    table_bf = table.to(torch.bfloat16).float()
    x = table_bf[tok] * scale
    xn = rms(x, g_final, eps)
    lm = table_bf[:V]
    want = xn @ lm.T

    tok_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    tok_t[0, 0], tok_t[0, 1] = tok // TILE, tok % TILE
    table_d = to_dev(table.to(torch.bfloat16), device)
    lm_d = to_dev(lm.T.contiguous().to(torch.bfloat16), device)

    def buf(n):
        return to_dev(torch.zeros(TILE, n, dtype=torch.bfloat16), device)

    x_d, xn_d, logits_d = buf(H), buf(H), buf(V)
    make_embed_gather(H // TILE, scale)(table_d, to_dev(tok_t, device), x_d)
    make_rmsnorm(1, 1, H // TILE, 11, H, eps)(x_d, row(g_final, H, device), xn_d)
    make_gemv(TILE, H, V, (8, 2), 4)(xn_d, lm_d, logits_d)

    got = from_dev(logits_d)[0]
    pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
    assert pcc > 0.98, f"logits pcc {pcc}"
    assert got.argmax().item() == want.argmax().item()
