# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Gemma flash decode shapes: sliding ring mask + global k_eq_v."""

import math

import pytest
import torch

import ttl
from ttl.ops.flash_decode import make_flash_decode, make_flash_decode_kev

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32


def run_flash(device, D, S, valid, k_eq_v, q_heads):
    """Single-core shard over a length-S cache with `valid` filled rows."""
    PNHt, DHt = 1, D // TILE
    chunk_t, n_chunks = 8, S // (8 * TILE)
    scale = 1.0

    q_t = torch.zeros(TILE, D, dtype=torch.bfloat16)
    q_t[:q_heads] = torch.randn(q_heads, D, dtype=torch.bfloat16) * 0.2
    k_t = torch.randn(S, D, dtype=torch.bfloat16) * 0.2
    v_t = k_t if k_eq_v else torch.randn(S, D, dtype=torch.bfloat16) * 0.2

    mask_rows = torch.full((S,), float("-inf"))
    mask_rows[:valid] = 0.0
    masks_t = mask_rows.reshape(n_chunks, 1, chunk_t * TILE).expand(
        n_chunks, TILE, chunk_t * TILE).reshape(n_chunks * TILE, chunk_t * TILE)
    masks_t = masks_t.to(torch.bfloat16).contiguous()

    att = (q_t[:q_heads].float() @ k_t.float().T) * scale + mask_rows
    expected = (torch.softmax(att, dim=-1) @ v_t.float()).to(torch.bfloat16)

    q_d, k_d, v_d = to_dram(q_t, device), to_dram(k_t, device), to_dram(v_t, device)
    m_d = to_dram(masks_t, device)
    o_d = to_dram(torch.zeros(TILE, D, dtype=torch.bfloat16), device)
    mm_d = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
    ll_d = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    if k_eq_v:
        make_flash_decode_kev(1, 1, PNHt, DHt, chunk_t, n_chunks, scale)(
            q_d, k_d, m_d, o_d, mm_d, ll_d)
    else:
        make_flash_decode(1, 1, PNHt, DHt, DHt, chunk_t, n_chunks, scale)(
            q_d, k_d, v_d, m_d, o_d, mm_d, ll_d)

    o = ttnn.to_torch(o_d).reshape(TILE, D).float()[:q_heads]
    l = ttnn.to_torch(ll_d).reshape(TILE, TILE).float()[:q_heads, 0:1]
    got = (o / l).to(torch.bfloat16)
    assert_pcc(expected, got, threshold=0.985)


@pytest.mark.parametrize(
    "D, S, valid, k_eq_v, q_heads",
    [
        (256, 1024, 700, False, 4),   # sliding: ring partially filled, 4 q heads
        (512, 2048, 1500, True, 4),   # global: V = K, partial fill
    ],
)
def test_flash_gemma(device, D, S, valid, k_eq_v, q_heads):
    run_flash(device, D, S, valid, k_eq_v, q_heads)
