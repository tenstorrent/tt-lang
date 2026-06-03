# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Tests for ttl.ops.matmul.make_ksplit against torch.matmul.

Cases cover one block per core (the original atom shape) and several blocks
per core (the loop), at small sizes and at benchmark-sweep shapes."""

import pytest
import torch

import ttl
from ttl.ops.matmul import make_ksplit

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32


def _pad_up(n, mult):
    return ((n + mult - 1) // mult) * mult


def run_ksplit(device, M, K, N, block_cfg, part_cfg):
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    # The op requires block*part to divide the dims exactly; pad M/N up to the
    # grid's full coverage (zeros add nothing to the real output, sliced below).
    # K must divide exactly: the K-partition partials have to line up.
    Mpad = _pad_up(M, bm * Mp * TILE)
    Npad = _pad_up(N, bn * Np * TILE)

    a_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    w_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.1
    expected = torch.matmul(a_t.float(), w_t.float()).to(torch.bfloat16)

    a_p = torch.nn.functional.pad(a_t, (0, 0, 0, Mpad - M))
    w_p = torch.nn.functional.pad(w_t, (0, Npad - N))

    a_d = to_dram(a_p, device)
    w_d = to_dram(w_p, device)
    out_d = to_dram(torch.zeros(Mpad, Npad, dtype=torch.bfloat16), device)

    make_ksplit(Mpad, K, Npad, block_cfg, part_cfg)(a_d, w_d, out_d)

    got = ttnn.to_torch(out_d).reshape(Mpad, Npad)[:M, :N].to(torch.bfloat16)
    assert_pcc(expected, got, threshold=0.99)


@pytest.mark.parametrize(
    "M, K, N, block_cfg, part_cfg",
    [
        # (M, K, N, (bm,bn,bk), (Mp,Np,Kp))  -- comment: blocks per core
        (128, 256, 128, (2, 2, 2), (2, 2, 2)),   # 1 block/core (atom baseline)
        (256, 256, 256, (2, 2, 2), (2, 2, 2)),   # 4 blocks/core (loop)
    ],
)
def test_ksplit(device, M, K, N, block_cfg, part_cfg):
    run_ksplit(device, M, K, N, block_cfg, part_cfg)


# The benchmark-winner plans use bm/bn/bk up to 8; their unrolled matmul blows
# past the default kernel-config buffer, so trim worker L1 to enlarge it.
@pytest.mark.parametrize("ttnn_device", [{"worker_l1_size": 1374544}], indirect=True)
@pytest.mark.parametrize(
    "M, K, N, block_cfg, part_cfg",
    [
        # Benchmark-sweep shapes at their empirically-best Kp=2 plans.
        (1024, 1024, 1024, (4, 8, 8), (8, 4, 2)),  # 64c, 1 block/core
        (2048, 2048, 2048, (8, 4, 8), (8, 6, 2)),  # 96c, 3 blocks/core (N pad)
        (2048, 4096, 2048, (8, 4, 8), (8, 6, 2)),  # 96c, deeper K (K_BPN=8)
        (2048, 8192, 2048, (8, 4, 8), (8, 6, 2)),  # 96c, long K (K_BPN=16)
    ],
)
def test_ksplit_bench(device, M, K, N, block_cfg, part_cfg):
    run_ksplit(device, M, K, N, block_cfg, part_cfg)
