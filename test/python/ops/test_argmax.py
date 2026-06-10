# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Wide greedy argmax chain vs torch; token emitted as (row, intra) split."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.ops.argmax import CHUNK, make_collapse, make_restack, make_token_select
from ttl.ops.topk import make_topk

TILE, V = 32, 2048
n_chunks = V // CHUNK
nt = (n_chunks + TILE - 1) // TILE


def to_dev(t, device):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_argmax_chain(device):
    torch.manual_seed(0)
    for _ in range(3):
        logits = torch.randn(TILE, V).to(torch.bfloat16)
        want = logits[0].float().argmax().item()

        z = torch.zeros
        tall = to_dev(z(n_chunks * TILE, CHUNK, dtype=torch.bfloat16), device)
        vals = to_dev(z(n_chunks * TILE, TILE, dtype=torch.bfloat16), device)
        ids = to_dev(z(n_chunks * TILE, TILE, dtype=torch.bfloat16), device)
        cw_init = z(TILE, nt * TILE)
        cw_init[0] = -1e30
        cw = to_dev(cw_init.to(torch.bfloat16), device)
        wv = to_dev(z(TILE, TILE, dtype=torch.bfloat16), device)
        wi = to_dev(z(TILE, TILE, dtype=torch.bfloat16), device)
        tok = to_dev(z(TILE, TILE, dtype=torch.bfloat16), device)
        ramp256 = to_dev(torch.arange(CHUNK, dtype=torch.bfloat16).unsqueeze(0).repeat(TILE, 1), device)
        rampn = to_dev(torch.arange(nt * TILE, dtype=torch.bfloat16).unsqueeze(0).repeat(TILE, 1), device)
        lut = z(TILE, CHUNK)
        lut[0] = torch.arange(CHUNK) // TILE
        lut[1] = torch.arange(CHUNK) % TILE
        lut[2, :n_chunks] = torch.arange(n_chunks) * (CHUNK // TILE)
        lut_d = to_dev(lut.to(torch.bfloat16), device)
        zero = to_dev(z(TILE, TILE, dtype=torch.bfloat16), device)

        make_restack(n_chunks)(to_dev(logits, device), tall)
        make_topk(n_chunks, 1, CHUNK // TILE, 1, CHUNK)(tall, ramp256, vals, ids)
        make_collapse(n_chunks)(vals, ids, cw)
        make_topk(1, 1, nt, 1, nt * TILE)(cw, rampn, wv, wi)
        make_token_select(n_chunks)(cw, wi, lut_d, zero, tok)

        t = ttnn.to_torch(tok).float()
        assert int(t[0, 0]) * TILE + int(t[0, 1]) == want
