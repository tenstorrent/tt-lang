# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Tests for ttl.ops.rmsnorm against torch RMSNorm.

``test_rmsnorm`` covers small shapes: fewer blocks than cores (idle cores)
and a block count that does not divide the core count (tail guard).
``test_rmsnorm_wide`` runs the production decoder widths (7168/1536/512, eps
1e-6) with the feature dimension streamed in width-tile chunks and several
row blocks per core. The op always fills the device grid, so no grid is
passed; row count is independent of the core count."""

import pytest
import torch

import ttl
from ttl.ops.rmsnorm import make_rmsnorm

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = ttnn.TILE_SIZE

EPS = 1e-6


def rmsnorm_golden(x, weight, eps):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * weight


def run_rmsnorm(device, n_rows, PNt, Dt, WCt, D):
    Rt = n_rows // TILE

    x_t = torch.randn(n_rows, D, dtype=torch.bfloat16)
    w_t = torch.randn(1, D, dtype=torch.bfloat16) * 0.1 + 1.0

    expected = rmsnorm_golden(x_t, w_t, EPS)

    x_d = to_dram(x_t, device)
    w_d = to_dram(w_t, device)
    out_d = to_dram(torch.zeros(n_rows, D, dtype=torch.bfloat16), device)

    rmsnorm = make_rmsnorm(Rt=Rt, PNt=PNt, Dt=Dt, WCt=WCt, D=D, eps=EPS)
    rmsnorm(x_d, w_d, out_d)

    got = ttnn.to_torch(out_d).reshape(n_rows, D).to(torch.bfloat16)
    assert_pcc(expected, got, threshold=0.99)


@pytest.mark.parametrize(
    "n_rows, PNt, Dt, WCt, D",
    [
        (2 * TILE, 1, 2, 2, 64),  # fewer blocks than cores: idle cores
        (261 * TILE, 1, 2, 2, 64),  # block count indivisible by cores: tail guard
    ],
)
def test_rmsnorm(device, n_rows, PNt, Dt, WCt, D):
    run_rmsnorm(device, n_rows, PNt, Dt, WCt, D)


# Production decoder norm widths: hidden 7168, q-proj 1536, kv-proj 512. A
# fixed row count (~4 blocks per core on a full Blackhole grid); the op spreads
# it over whatever grid it lands on, so the constant needs no device query.
N_ROWS_WIDE = 16384


@pytest.mark.parametrize(
    "D, Dt",
    [(7168, 224), (1536, 48), (512, 16)],
)
def test_rmsnorm_wide(device, D, Dt):
    run_rmsnorm(device, n_rows=N_ROWS_WIDE, PNt=1, Dt=Dt, WCt=8, D=D)
