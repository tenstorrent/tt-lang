# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Per-card routed experts as a dispatch chain vs torch reference.

Router (top-8 ids + renormed gate weights) is host scaffolding; the chain is
gate_up indexed GEMV -> per-expert swiglu (gate weight folded into the
activation) -> down indexed GEMV (per-expert x rows) -> accumulate rows.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttl.ops.elementwise import make_add, make_binary
from ttl.ops.indexed_gemv import make_indexed_gemv
from gemma4.layer_test_scaffolding import from_dev, row, to_dev

TILE = 32


def gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def make_swiglu_scaled(It, wt, t):
    return make_binary(
        lambda g, u: ttl.mul(ttl.mul(ttl.gelu(g), u),
                             ttl.block.fill(wt, shape=g.shape)),
        1, 1, It, It, a_off=(t, 0), b_off=(t, It), out_off=(t, 0))


def test_experts_chain():
    device = ttnn.open_device(device_id=0)
    try:
        _run(device)
    finally:
        ttnn.close_device(device)


def _run(device):
    torch.manual_seed(0)
    H, I, E, K = 2816, 704, 32, 8
    It = I // TILE

    x = torch.randn(H) * 0.5
    w_gu = torch.randn(E, H, 2 * I) * 0.02
    w_dn = torch.randn(E, I, H) * 0.02
    idx = torch.randperm(E)[:K]
    wts = torch.softmax(torch.randn(K), dim=0)

    want = torch.zeros(H)
    for t in range(K):
        gu = x @ w_gu[idx[t]]
        act = gelu_tanh(gu[:I]) * gu[I:] * wts[t]
        want += act @ w_dn[idx[t]]

    idx_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    idx_t[0, :K] = idx.float()
    x_d = row(x, H, device)
    idx_d = to_dev(idx_t, device)
    w_gu_d = to_dev(w_gu.reshape(E * H, 2 * I).to(torch.bfloat16), device)
    w_dn_d = to_dev(w_dn.reshape(E * I, H).to(torch.bfloat16), device)
    gu_d = to_dev(torch.zeros(K * TILE, 2 * I, dtype=torch.bfloat16), device)
    act_d = to_dev(torch.zeros(K * TILE, I, dtype=torch.bfloat16), device)
    dn_d = to_dev(torch.zeros(K * TILE, H, dtype=torch.bfloat16), device)

    make_indexed_gemv(E, H, 2 * I, K, (11, 2), 4)(x_d, idx_d, w_gu_d, gu_d)
    for t in range(K):
        make_swiglu_scaled(It, wts[t].item(), t)(gu_d, gu_d, act_d)
    make_indexed_gemv(E, I, H, K, (11, 2), 4, x_per_t=True)(act_d, idx_d, w_dn_d, dn_d)
    for t in range(1, K):
        make_add(1, 1, H // TILE, 11, b_off=(t, 0))(dn_d, dn_d, dn_d)

    got = from_dev(dn_d)[0]
    pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
    assert pcc > 0.98, f"experts pcc {pcc}"
