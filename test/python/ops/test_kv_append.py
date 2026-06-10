# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""ttl.ops.kv_append patches one cache row in-place at a runtime position."""

import pytest
import torch

import ttl
from ttl.ops.kv_append import make_kv_append

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram

TILE = 32


@pytest.mark.parametrize("S, D, pos", [(1024, 256, 42), (1024, 512, 1023)])
def test_kv_append(device, S, D, pos):
    cache_t = torch.randn(S, D, dtype=torch.bfloat16)
    k_t = torch.randn(TILE, D, dtype=torch.bfloat16)
    pos_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    pos_t[0, 0] = pos // TILE
    pos_t[0, 1] = pos % TILE

    expected = cache_t.clone()
    expected[pos] = k_t[0]

    cache_d = to_dram(cache_t, device)
    k_d = to_dram(k_t, device)
    pos_d = to_dram(pos_t, device)

    make_kv_append(S // TILE, D // TILE)(cache_d, k_d, pos_d, cache_d)

    got = ttnn.to_torch(cache_d).reshape(S, D).to(torch.bfloat16)
    assert torch.equal(got, expected)
