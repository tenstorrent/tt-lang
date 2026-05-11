# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end correctness tests for ttl.math.typecast.

Each parametrized case uses a different SFPU LLK kernel in tt-metal
(a distinct typecast_tile<> template instantiation), so the suite provides
runtime coverage that the compile-only lit test in simple_typecast.py cannot.

Supported floating-point pairs:
  bf16 -> f32   (lossless widening, hardware result must be exact)
  f32  -> bf16  (lossy narrowing, hardware rounding must match torch)

Integer pairs (int32 <-> float) are omitted: the SFPU typecast_tile LLK is
defined only for floating-point data formats; issuing an integer typecast
would require a different hardware path not currently exposed by ttl.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttlang_test_utils import assert_pcc, to_l1


# =============================================================================
# Kernel definitions
#
# Each kernel must be defined at module scope so that the TTL source inspector
# can locate the decorated function's source text. The target dtype is encoded
# as a literal in the kernel body because the TTL AST compiler resolves
# torch.float32 / torch.bfloat16 as compile-time constants.
# =============================================================================


@ttl.operation(grid=(1, 1))
def _typecast_bf16_to_f32(inp, out):
    """Elementwise typecast: bf16 -> f32."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(ttl.math.typecast(x, torch.float32))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


@ttl.operation(grid=(1, 1))
def _typecast_f32_to_bf16(inp, out):
    """Elementwise typecast: f32 -> bf16."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(ttl.math.typecast(x, torch.bfloat16))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# =============================================================================
# Parametrized configurations
# =============================================================================

TILE = 32

_CONFIGS = [
    pytest.param(
        _typecast_bf16_to_f32,
        torch.bfloat16,
        ttnn.bfloat16,
        torch.float32,
        ttnn.float32,
        id="bf16_to_f32",
    ),
    pytest.param(
        _typecast_f32_to_bf16,
        torch.float32,
        ttnn.float32,
        torch.bfloat16,
        ttnn.bfloat16,
        id="f32_to_bf16",
    ),
]


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize(
    "kernel, in_torch_dtype, in_ttnn_dtype, out_torch_dtype, out_ttnn_dtype",
    _CONFIGS,
)
def test_typecast_correctness(
    device, kernel, in_torch_dtype, in_ttnn_dtype, out_torch_dtype, out_ttnn_dtype
):
    """Verify that the hardware typecast result matches the torch reference cast.

    Uses random input to ensure the comparison has sufficient variance for PCC.
    The PCC threshold is set to 0.9999 (consistent with tt-metal numerics
    conventions); in practice both pairs should be bit-exact.
    """
    in_name = str(in_torch_dtype).split(".")[-1]
    out_name = str(out_torch_dtype).split(".")[-1]
    label = f"{in_name}->{out_name}"

    inp_torch = torch.rand((TILE, TILE), dtype=in_torch_dtype)
    out_torch = torch.zeros((TILE, TILE), dtype=out_torch_dtype)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, out)

    result = ttnn.to_torch(out)
    expected = inp_torch.to(out_torch_dtype)

    pcc = assert_pcc(expected.float(), result.float())
    max_diff = (result.float() - expected.float()).abs().max().item()
    print(f"  [{label}] pcc={pcc:.6f}  max_diff={max_diff:.2e}  PASSED")
