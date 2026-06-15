# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Scaled additive matmul recurrence: out = scale * acc + (a @ b).

`scale * acc` is a body SSA value that lowering precomputes into the matmul
output DST slots, then `matmul_block` accumulates `a @ b` onto them (the add
vanishes). Numerically verifies that prefill-then-accumulate produces the same
result as the reference expression, for bf16 and fp32.

The flash variant broadcasts a per-row scale across the V/output dimension with
two V tiles, exercising the multi-output-tile expansion.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram

TILE = 32

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
# bf16 accumulates more rounding across the mul + matmul-accumulate chain; fp32
# stays tight. Do not share tolerances across dtypes.
ALLCLOSE_TOL = {
    "bf16": {"rtol": 5e-2, "atol": 1e-1},
    "fp32": {"rtol": 1e-3, "atol": 1e-3},
}


@ttl.operation(grid=(1, 1))
def scaled_acc_matmul_kernel(scale, acc, a, b, out):
    """out = scale * acc + (a @ b) in one fused compute body."""
    Mt = a.shape[0] // TILE
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(Mt, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, Nt), block_count=2)
    scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(Mt, Nt), block_count=2)
    acc_dfb = ttl.make_dataflow_buffer_like(acc, shape=(Mt, Nt), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), block_count=2)

    @ttl.compute()
    def mm_compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            scale_dfb.wait() as scale_blk,
            acc_dfb.wait() as acc_blk,
        ):
            with out_dfb.reserve() as o:
                o.store(scale_blk * acc_blk + a_blk @ b_blk)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:Mt, 0:1], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:1, 0:Nt], blk)
            tx.wait()
        with scale_dfb.reserve() as blk:
            tx = ttl.copy(scale[0:Mt, 0:Nt], blk)
            tx.wait()
        with acc_dfb.reserve() as blk:
            tx = ttl.copy(acc[0:Mt, 0:Nt], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:Mt, 0:Nt])
            tx.wait()


@ttl.operation(grid=(1, 1))
def flash_scaled_acc_kernel(alpha, o_old, scores, v, out):
    """out = broadcast_col(alpha) * o_old + (scores @ v).

    alpha is a single tile broadcast across the V/output columns; V has Nt
    tiles, so the lowering prefills Nt output slots before one matmul_block.
    """
    Nt = v.shape[1] // TILE

    alpha_dfb = ttl.make_dataflow_buffer_like(alpha, shape=(1, 1), block_count=2)
    old_dfb = ttl.make_dataflow_buffer_like(o_old, shape=(1, Nt), block_count=2)
    scores_dfb = ttl.make_dataflow_buffer_like(scores, shape=(1, 1), block_count=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, Nt), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, Nt), block_count=2)

    @ttl.compute()
    def mm_compute():
        with (
            alpha_dfb.wait() as alpha_blk,
            old_dfb.wait() as old_blk,
            scores_dfb.wait() as scores_blk,
            v_dfb.wait() as v_blk,
        ):
            with out_dfb.reserve() as o:
                # alpha is a per-row scalar: broadcast its column across the
                # tile and across the Nt V tiles, matching flash-attention's
                # row-wise rescale.
                alpha_b = ttl.block.broadcast(alpha_blk, dims=[-1], shape=(1, Nt))
                o.store(alpha_b * old_blk + scores_blk @ v_blk)

    @ttl.datamovement()
    def dm_read():
        with alpha_dfb.reserve() as blk:
            tx = ttl.copy(alpha[0:1, 0:1], blk)
            tx.wait()
        with old_dfb.reserve() as blk:
            tx = ttl.copy(o_old[0:1, 0:Nt], blk)
            tx.wait()
        with scores_dfb.reserve() as blk:
            tx = ttl.copy(scores[0:1, 0:1], blk)
            tx.wait()
        with v_dfb.reserve() as blk:
            tx = ttl.copy(v[0:1, 0:Nt], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:1, 0:Nt])
            tx.wait()


# Each output tile uses 1 output slot + 2 scratch copies = 3 DST slots, so
# Mt*Nt is bounded by capacity (8 for bf16, 4 for fp32).
SCALED_CASES = [
    ("bf16", 1, 1),
    ("bf16", 1, 2),
    ("bf16", 2, 1),
    ("fp32", 1, 1),
]
SCALED_IDS = [f"{d}_{m}x{n}" for d, m, n in SCALED_CASES]


def binary_fraction_grid(rows, cols, center, step, dtype, period=32):
    offsets = torch.arange(rows * cols, dtype=torch.float32).remainder(period)
    offsets = offsets.reshape(rows, cols) - period // 2
    values = center + offsets * step
    return values.to(dtype)


@pytest.mark.parametrize("dtype_name,Mt,Nt", SCALED_CASES, ids=SCALED_IDS)
@pytest.mark.requires_device
def test_scaled_acc_matmul(dtype_name, Mt, Nt, device):
    """out = scale * acc + (a @ b): scaled accumulator folded into matmul."""
    dtype = DTYPES[dtype_name]
    M, K, N = Mt * TILE, TILE, Nt * TILE

    scale_torch = binary_fraction_grid(M, N, 1.0, 1.0 / 64.0, dtype)
    acc_torch = binary_fraction_grid(M, N, 0.0, 1.0 / 64.0, dtype)
    a_torch = torch.eye(M, K, dtype=torch.float32).to(dtype)
    b_torch = binary_fraction_grid(K, N, 0.0, 1.0 / 128.0, dtype)

    scale = to_dram(scale_torch, device)
    acc = to_dram(acc_torch, device)
    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(M, N, dtype=dtype), device)

    scaled_acc_matmul_kernel(scale, acc, a, b, out)

    result = ttnn.to_torch(out)
    golden = scale_torch.float() * acc_torch.float() + a_torch.float() @ b_torch.float()
    assert_allclose(result.float(), golden, **ALLCLOSE_TOL[dtype_name])


# Flash-based recurrence cases. bf16 includes 16 V/output tiles to cover the
# wide-V regression; fp32 remains restricted to one V/output tile because each
# output tile uses three DST slots and fp32 capacity is four.
FLASH_CASES = [
    ("bf16", 1),
    ("bf16", 2),
    ("bf16", 16),
    ("fp32", 1),
]
FLASH_IDS = [f"{dtype_name}_v{num_v_tiles}" for dtype_name, num_v_tiles in FLASH_CASES]


def make_flash_scaled_acc_inputs(dtype, num_v_tiles):
    """Use bounded inputs that expose a dropped scaled-accumulator term."""
    alpha_torch = binary_fraction_grid(TILE, 1, 1.0, 1.0 / 64.0, torch.float32)
    old_torch = binary_fraction_grid(
        TILE, num_v_tiles * TILE, 0.0, 1.0 / 4.0, torch.float32
    )
    scores_torch = torch.eye(TILE, dtype=torch.float32)
    v_torch = binary_fraction_grid(
        TILE, num_v_tiles * TILE, 0.0, 1.0 / 128.0, torch.float32
    )

    return (
        alpha_torch.to(dtype),
        old_torch.to(dtype),
        scores_torch.to(dtype),
        v_torch.to(dtype),
    )


@pytest.mark.parametrize("dtype_name,num_v_tiles", FLASH_CASES, ids=FLASH_IDS)
@pytest.mark.requires_device
def test_flash_broadcast_scaled_acc(dtype_name, num_v_tiles, device):
    """out = broadcast(alpha) * o_old + (scores @ v)."""
    dtype = DTYPES[dtype_name]
    output_cols = num_v_tiles * TILE

    alpha_torch, old_torch, scores_torch, v_torch = make_flash_scaled_acc_inputs(
        dtype, num_v_tiles
    )

    alpha = to_dram(alpha_torch, device)
    o_old = to_dram(old_torch, device)
    scores = to_dram(scores_torch, device)
    v = to_dram(v_torch, device)
    out = to_dram(torch.zeros(TILE, output_cols, dtype=dtype), device)

    flash_scaled_acc_kernel(alpha, o_old, scores, v, out)

    result = ttnn.to_torch(out)
    golden_torch = (
        alpha_torch.float() * old_torch.float() + scores_torch.float() @ v_torch.float()
    )
    matmul_only_torch = scores_torch.float() @ v_torch.float()
    tolerance = ALLCLOSE_TOL[dtype_name]
    scaled_term_delta = (golden_torch - matmul_only_torch).abs().max()
    required_delta = 10 * max(
        tolerance["atol"], tolerance["rtol"] * golden_torch.abs().max()
    )
    assert scaled_term_delta > required_delta
    assert_allclose(result.float(), golden_torch, **ALLCLOSE_TOL[dtype_name])
