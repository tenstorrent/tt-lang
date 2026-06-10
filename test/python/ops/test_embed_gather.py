# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""ttl.ops.embed_gather pulls one scaled table row by runtime token id."""

import pytest
import torch

import ttl
from ttl.ops.embed_gather import make_embed_gather

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32


@pytest.mark.parametrize("V, D, tok", [(4096, 2816, 1234), (4096, 2816, 31)])
def test_embed_gather(device, V, D, tok):
    scale = D ** 0.5
    table_t = torch.randn(V, D, dtype=torch.bfloat16) * 0.05
    tok_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    tok_t[0, 0] = tok // TILE
    tok_t[0, 1] = tok % TILE
    expected = (table_t[tok].float() * scale).to(torch.bfloat16)

    table_d = to_dram(table_t, device)
    tok_d = to_dram(tok_t, device)
    out_d = to_dram(torch.zeros(TILE, D, dtype=torch.bfloat16), device)

    make_embed_gather(D // TILE, scale)(table_d, tok_d, out_d)

    got = ttnn.to_torch(out_d).reshape(TILE, D)[0].to(torch.bfloat16)
    assert_pcc(expected.unsqueeze(0), got.unsqueeze(0), threshold=0.999)
