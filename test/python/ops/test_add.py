# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Elementwise residual add vs torch."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.ops.elementwise import make_add

TILE = 32


def to_dev(t, device):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_add():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        D = 2816
        a = torch.randn(TILE, D)
        b = torch.randn(TILE, D)
        out = to_dev(torch.zeros(TILE, D), device)
        make_add(1, 1, D // TILE, 11)(to_dev(a, device), to_dev(b, device), out)
        got = ttnn.to_torch(out).float()
        want = (a.to(torch.bfloat16).float() + b.to(torch.bfloat16).float())
        assert torch.allclose(got, want, atol=0.06), (got - want).abs().max()
    finally:
        ttnn.close_device(device)
