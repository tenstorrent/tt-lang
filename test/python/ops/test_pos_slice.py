# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""pos_slice (runtime table row extract) and pos_step (pos counter) ops."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.ops.pos_slice import make_pos_slice, make_pos_step

TILE, SMAX, D, RING = 32, 1024, 256, 128


def to_dev(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def pos_tile(p, ring=None):
    t = torch.zeros(TILE, TILE)
    r = p if ring is None else p % ring
    t[0, :3] = torch.tensor([r // TILE, r % TILE, p]).float()
    return t


def test_pos_slice_and_step(device):
    torch.manual_seed(0)
    table = torch.randn(SMAX, D)
    pos = 100

    table_d = to_dev(table, device)
    pos_d = to_dev(pos_tile(pos), device, ttnn.float32)
    out = to_dev(torch.zeros(TILE, D), device)
    make_pos_slice(D // TILE)(table_d, pos_d, out)
    got = ttnn.to_torch(out).float()[0]
    want = table[pos].to(torch.bfloat16).float()
    assert (got - want).abs().max() < 0.06

    lut = torch.zeros(SMAX, TILE)
    for p in range(SMAX - 1):
        lut[p, :3] = pos_tile(p + 1, RING)[0, :3]
    lut_d = to_dev(lut, device, ttnn.float32)
    pos_d = to_dev(pos_tile(pos, RING), device, ttnn.float32)
    nxt = to_dev(torch.zeros(TILE, TILE), device, ttnn.float32)
    make_pos_step()(lut_d, pos_d, nxt)
    got = ttnn.to_torch(nxt).float()[0, :3]
    assert torch.equal(got, pos_tile(pos + 1, RING)[0, :3])
