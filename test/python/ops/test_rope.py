# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""ttl.ops.rope vs HF rotate-half, full and partial rotary."""

import pytest
import torch

import ttl
from ttl.ops.rope import make_rope

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32


def hf_rope(x, cos, sin, rot):
    xr, xp = x[:, :rot], x[:, rot:]
    half = rot // 2
    rh = torch.cat([-xr[:, half:], xr[:, :half]], dim=-1)
    return torch.cat([xr * cos + rh * sin, xp], dim=-1)


def run_rope(device, head_dim, rot, theta, pos):
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2).float() / rot))
    freqs = pos * inv
    cos = torch.cat([freqs.cos(), freqs.cos()]).expand(TILE, rot).contiguous()
    sin = torch.cat([freqs.sin(), freqs.sin()]).expand(TILE, rot).contiguous()

    x_t = torch.randn(TILE, head_dim, dtype=torch.bfloat16)
    expected = hf_rope(x_t.float(), cos, sin, rot).to(torch.bfloat16)

    x_d = to_dram(x_t, device)
    cos_d = to_dram(cos.to(torch.bfloat16), device)
    sin_d = to_dram(sin.to(torch.bfloat16), device)
    out_d = to_dram(torch.zeros(TILE, head_dim, dtype=torch.bfloat16), device)

    make_rope(head_dim // TILE, rot // TILE)(x_d, cos_d, sin_d, out_d)

    got = ttnn.to_torch(out_d).reshape(TILE, head_dim).to(torch.bfloat16)
    assert_pcc(expected, got, threshold=0.99)


@pytest.mark.parametrize(
    "head_dim, rot, theta, pos",
    [
        (256, 256, 10000.0, 511),    # sliding: full rotary
        (512, 128, 1000000.0, 4096), # global: partial 0.25
    ],
)
def test_rope(device, head_dim, rot, theta, pos):
    run_rope(device, head_dim, rot, theta, pos)
