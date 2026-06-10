# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""ttl.ops.swiglu vs torch gelu_pytorch_tanh * up."""

import pytest
import torch

import ttl
from ttl.ops.swiglu import make_swiglu

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32


def run_swiglu(device, R, D, PNt, WCt):
    g_t = torch.randn(R, D, dtype=torch.bfloat16)
    u_t = torch.randn(R, D, dtype=torch.bfloat16)
    expected = (
        torch.nn.functional.gelu(g_t.float(), approximate="tanh") * u_t.float()
    ).to(torch.bfloat16)

    g_d, u_d = to_dram(g_t, device), to_dram(u_t, device)
    out_d = to_dram(torch.zeros(R, D, dtype=torch.bfloat16), device)

    make_swiglu(R // TILE, PNt, D // TILE, WCt)(g_d, u_d, out_d)

    got = ttnn.to_torch(out_d).reshape(R, D).to(torch.bfloat16)
    assert_pcc(expected, got, threshold=0.99)


@pytest.mark.parametrize(
    "R, D, PNt, WCt",
    [
        (32, 704, 1, 22),     # Gemma expert inter (1 row tile)
        (32, 2112, 1, 22),    # Gemma dense-MLP inter
        (128, 2048, 2, 16),   # multi-row blocks
    ],
)
def test_swiglu(device, R, D, PNt, WCt):
    run_swiglu(device, R, D, PNt, WCt)
