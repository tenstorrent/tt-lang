# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LayerNorm kernel under investigation for a hardware hang.

Uses the factory `make_layernorm_kernel` from `_examples/layernorm.py`.
Three-pass streaming: mean -> variance -> normalize+affine.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v --tb=short

import os
import sys

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "_examples",
)
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from layernorm import make_layernorm_kernel  # noqa: E402
from layernorm_explicit import (  # noqa: E402
    make_layernorm_kernel as make_layernorm_kernel_explicit,
)
from layernorm_minimal_dfbs import (  # noqa: E402
    make_layernorm_kernel as make_layernorm_kernel_minimal_dfbs,
)
from layernorm_ssa_intermediates import (  # noqa: E402
    make_layernorm_kernel as make_layernorm_kernel_ssa,
)

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_pcc  # noqa: E402

TILE = 32


def _torch_layernorm(x, weight, bias, eps=1e-6):
    """Row-wise layernorm with per-element weight/bias (matching the kernel)."""
    xf = x.float()
    mean = xf.mean(dim=1, keepdim=True)
    var = ((xf - mean) ** 2).mean(dim=1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    return ((xf - mean) * inv_std) * weight.float() + bias.float()


def _run_layernorm(kernel_factory, seq_tiles, dim_tiles, device):
    M, N = seq_tiles * TILE, dim_tiles * TILE
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.randn(M, N, dtype=torch.bfloat16)
    bias = torch.randn(M, N, dtype=torch.bfloat16)
    scaler = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    mean_scale = torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16)

    golden = _torch_layernorm(x, weight, bias)

    x_dev = to_dram(x, device)
    weight_dev = to_dram(weight, device)
    bias_dev = to_dram(bias, device)
    scaler_dev = to_dram(scaler, device)
    mean_scale_dev = to_dram(mean_scale, device)
    out_dev = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = kernel_factory(dim_tiles)
    kernel(x_dev, weight_dev, bias_dev, scaler_dev, mean_scale_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result, threshold=0.99)


@pytest.mark.parametrize("seq_tiles,dim_tiles", [(2, 2)], ids=["2x2"])
@pytest.mark.requires_device
def test_layernorm(seq_tiles, dim_tiles, device):
    """Hangs on hardware — compute thread deadlocks on its own cb_wait_front
    because `r.store(...)` does not auto-emit cb_push_back."""
    _run_layernorm(make_layernorm_kernel, seq_tiles, dim_tiles, device)


@pytest.mark.parametrize("seq_tiles,dim_tiles", [(2, 2)], ids=["2x2"])
@pytest.mark.requires_device
def test_layernorm_explicit(seq_tiles, dim_tiles, device):
    """Same kernel with explicit .push()/.pop() — runs to completion."""
    _run_layernorm(make_layernorm_kernel_explicit, seq_tiles, dim_tiles, device)


@pytest.mark.parametrize("seq_tiles,dim_tiles", [(2, 2)], ids=["2x2"])
@pytest.mark.requires_device
def test_layernorm_minimal_dfbs(seq_tiles, dim_tiles, device):
    """Cross-thread user DFBs only (inputs/outputs + mean/inv_std carry).
    Compiler allocates the rest via TTLInsertIntermediateDFBs."""
    _run_layernorm(
        make_layernorm_kernel_minimal_dfbs, seq_tiles, dim_tiles, device
    )


@pytest.mark.parametrize("seq_tiles,dim_tiles", [(2, 2)], ids=["2x2"])
@pytest.mark.requires_device
def test_layernorm_ssa(seq_tiles, dim_tiles, device):
    """No user DFBs for mean/inv_std; SSA tile accumulation.
    TTLInsertIntermediateDFBs materializes mean/var where ttl.bcast
    requires CB-attached input."""
    _run_layernorm(make_layernorm_kernel_ssa, seq_tiles, dim_tiles, device)
