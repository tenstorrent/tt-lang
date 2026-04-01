# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul with K-accumulation via explicit and fused patterns.

Explicit: matmul to intermediate CB per K-step, then elementwise add with
accumulator. Requires partial_dfb + acc_dfb.

Fused: `prev + a @ b` lowers to copy_tile(prev) + matmul_block(DST += A*B).
The add is eliminated. Requires only acc_dfb.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram

TILE = 32


@ttl.operation(grid=(1, 1))
def matmul_acc_kernel(a, b, out):
    Mt = a.shape[0] // TILE
    Kt = a.shape[1] // TILE
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(Mt, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, Nt), buffer_factor=2)
    partial_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)
    # Compute-local accumulator. DM writer does NOT touch this.
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)
    # Output DFB: only written once after accumulation completes.
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        # First K-step: matmul directly to accumulator.
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        with acc_dfb.reserve() as acc:
            acc.store(a_blk @ b_blk)
        a_blk.pop()
        b_blk.pop()

        # Remaining K-steps: matmul to partial, add with accumulator.
        for _ in range(Kt - 1):
            a_blk = a_dfb.wait()
            b_blk = b_dfb.wait()
            with partial_dfb.reserve() as p:
                p.store(a_blk @ b_blk)
            a_blk.pop()
            b_blk.pop()

            with partial_dfb.wait() as new, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + new)

        # Copy final accumulator to output (single push, DM writer sees this).
        with acc_dfb.wait() as final:
            with out_dfb.reserve() as o:
                o.store(final)

    @ttl.datamovement()
    def dm_read():
        for kt in range(Kt):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:Mt, kt : kt + 1], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[kt : kt + 1, 0:Nt], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:Mt, 0:Nt])
            tx.wait()


@ttl.operation(grid=(1, 1))
def matmul_fused_acc_kernel(a, b, out):
    """Fused accumulation: prev + a @ b folds into copy_tile + matmul_block."""
    Mt = a.shape[0] // TILE
    Kt = a.shape[1] // TILE
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(Mt, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, Nt), buffer_factor=2)
    # Compute-local accumulator. No partial_dfb needed — fusion eliminates it.
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        # First K-step: matmul directly to accumulator.
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        with acc_dfb.reserve() as acc:
            acc.store(a_blk @ b_blk)
        a_blk.pop()
        b_blk.pop()

        # Remaining K-steps: fused prev + a @ b (add vanishes).
        for _ in range(Kt - 1):
            with (
                a_dfb.wait() as a_blk,
                b_dfb.wait() as b_blk,
                acc_dfb.wait() as prev,
            ):
                with acc_dfb.reserve() as acc:
                    acc.store(prev + a_blk @ b_blk)

        # Copy final accumulator to output.
        with acc_dfb.wait() as final:
            with out_dfb.reserve() as o:
                o.store(final)

    @ttl.datamovement()
    def dm_read():
        for kt in range(Kt):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:Mt, kt : kt + 1], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[kt : kt + 1, 0:Nt], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:Mt, 0:Nt])
            tx.wait()


SHAPE_PARAMS = [
    (1, 1, 1),
    (1, 2, 1),
    (1, 4, 1),
    (2, 1, 2),
    (2, 2, 2),
    (2, 4, 2),
    (1, 3, 1),  # Odd K.
    (2, 2, 4),  # 2x4 output = 8 tiles at max DST, K=2.
    (1, 4, 4),  # Wide output with K accumulation.
    (1, 2, 2),  # Non-square: A[1,2] @ B[2,2] = C[1,2].
    (4, 8, 2),  # Large K with tall output.
    (2, 8, 4),  # Large K with wide output.
]

SHAPE_IDS = [f"{m}x{k}x{n}" for m, k, n in SHAPE_PARAMS]


def _run_matmul_acc(kernel_fn, Mt, Kt, Nt, device, options=None):
    """Shared test harness: run kernel, compare against torch golden."""
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kwargs = {"options": options} if options else {}
    kernel_fn(a, b, out, **kwargs)

    result = ttnn.to_torch(out)
    golden = a_torch @ b_torch

    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.99, (
        f"PCC {pcc:.6f} < 0.99 for {Mt}x{Kt}x{Nt} matmul. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )


@pytest.mark.parametrize("Mt,Kt,Nt", SHAPE_PARAMS, ids=SHAPE_IDS)
@pytest.mark.requires_device
def test_matmul_accumulate_explicit(Mt, Kt, Nt, device):
    """Explicit accumulation: matmul to partial CB, then add with accumulator."""
    _run_matmul_acc(matmul_acc_kernel, Mt, Kt, Nt, device)


@pytest.mark.parametrize("Mt,Kt,Nt", SHAPE_PARAMS, ids=SHAPE_IDS)
@pytest.mark.parametrize(
    "pack_opts",
    [None, "--ttl-combine-pack-tiles"],
    ids=["pack_tile", "pack_tile_block"],
)
@pytest.mark.requires_device
def test_matmul_accumulate_fused(Mt, Kt, Nt, pack_opts, device):
    """Fused accumulation: prev + a @ b compiles to copy_tile + matmul_block."""
    _run_matmul_acc(matmul_fused_acc_kernel, Mt, Kt, Nt, device, options=pack_opts)


# =============================================================================
# Matmul + broadcast bias: Y = (A @ B) + broadcast_col(bias)
# =============================================================================


@ttl.operation(grid=(1, 1))
def matmul_bcast_bias_kernel(a, b, bias, out):
    """Matmul with column-broadcast bias: Y[Mt,Nt] = (A @ B) + bcast(bias[1,Nt])."""
    Mt = a.shape[0] // TILE
    Kt = a.shape[1] // TILE
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(Mt, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, Nt), buffer_factor=2)
    partial_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, Nt), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        # K-accumulation (same as matmul_acc_kernel).
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        with acc_dfb.reserve() as acc:
            acc.store(a_blk @ b_blk)
        a_blk.pop()
        b_blk.pop()

        for _ in range(Kt - 1):
            a_blk = a_dfb.wait()
            b_blk = b_dfb.wait()
            with partial_dfb.reserve() as p:
                p.store(a_blk @ b_blk)
            a_blk.pop()
            b_blk.pop()
            with partial_dfb.wait() as new, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + new)

        # Add broadcast bias: bcast(bias[1,Nt]) -> (Mt,Nt), then add with acc.
        with bias_dfb.wait() as bias_blk, acc_dfb.wait() as acc_blk:
            with out_dfb.reserve() as o:
                bias_expanded = ttl.math.broadcast(bias_blk, o, dims=[0])
                o.store(bias_expanded + acc_blk)

    @ttl.datamovement()
    def dm_read():
        for kt in range(Kt):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:Mt, kt : kt + 1], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[kt : kt + 1, 0:Nt], blk)
                tx.wait()
        with bias_dfb.reserve() as blk:
            tx = ttl.copy(bias[0:1, 0:Nt], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:Mt, 0:Nt])
            tx.wait()


@pytest.mark.parametrize(
    "Mt,Kt,Nt",
    [
        (1, 1, 1),
        (2, 1, 2),
        (2, 2, 2),
    ],
    ids=["1x1x1", "2x1x2", "2x2x2"],
)
@pytest.mark.requires_device
def test_matmul_bcast_bias(Mt, Kt, Nt, device):
    """Matmul + column-broadcast bias addition."""
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    bias_torch = torch.randn(TILE, N, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    bias = to_dram(bias_torch, device)
    out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    matmul_bcast_bias_kernel(a, b, bias, out)

    result = ttnn.to_torch(out)
    # Broadcast bias: (1 tile row, Nt cols) -> (Mt tile rows, Nt cols).
    bias_expanded = bias_torch.repeat(Mt, 1)
    golden = a_torch @ b_torch + bias_expanded

    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    # Relaxed threshold: matmul + broadcast + add chain in bf16 accumulates
    # more rounding error than matmul alone.
    assert pcc > 0.95, (
        f"PCC {pcc:.6f} < 0.95 for {Mt}x{Kt}x{Nt} matmul+bcast_bias. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )


# =============================================================================
# Distinct per-tile values: verifies tile indexing in multi-tile matmul
# =============================================================================


@pytest.mark.requires_device
def test_matmul_distinct_tiles(device):
    """[2x2] @ [2x2] = [2x2] with distinct random values per tile.

    Each tile is filled with random values from a different range so that
    a tile-indexing bug would produce visibly wrong per-tile results.
    """
    a_torch = torch.zeros(2 * TILE, 2 * TILE, dtype=torch.bfloat16)
    a_torch[:TILE, :TILE] = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5 + 1.0
    a_torch[:TILE, TILE:] = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5 - 1.0
    a_torch[TILE:, :TILE] = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5 + 2.0
    a_torch[TILE:, TILE:] = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.5 - 2.0
    b_torch = torch.randn(2 * TILE, 2 * TILE, dtype=torch.bfloat16) * 0.5

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(2 * TILE, 2 * TILE, dtype=torch.bfloat16), device)

    matmul_acc_kernel(a, b, out)

    result = ttnn.to_torch(out)
    golden = a_torch @ b_torch
    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.99, (
        f"PCC {pcc:.6f} < 0.99 for 2x2x2 distinct-tile matmul. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )


# =============================================================================
# Fused accumulation with outer block loop: block sizes smaller than the full
# output, so the compute iterates over output blocks. Exercises the
# lower-matmul-block input index mapping (the accumulator shifts lhs/rhs
# positions in the compute's input list).
# =============================================================================


@ttl.operation(grid=(1, 1))
def matmul_fused_acc_blocked_kernel(a, b, out):
    """Fused accumulation with 1x1 output blocks and outer M/N/K loops."""
    Mt = a.shape[0] // TILE
    Kt = a.shape[1] // TILE
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        for _ in range(Mt):
            for _ in range(Nt):
                # First K-step: matmul directly to accumulator.
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                with acc_dfb.reserve() as acc:
                    acc.store(a_blk @ b_blk)
                a_blk.pop()
                b_blk.pop()

                # Remaining K-steps: fused prev + a @ b.
                for _ in range(Kt - 1):
                    with (
                        a_dfb.wait() as a_blk,
                        b_dfb.wait() as b_blk,
                        acc_dfb.wait() as prev,
                    ):
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + a_blk @ b_blk)

                with acc_dfb.wait() as final:
                    with out_dfb.reserve() as o:
                        o.store(final)

    @ttl.datamovement()
    def dm_read():
        for mt in range(Mt):
            for nt in range(Nt):
                for kt in range(Kt):
                    with a_dfb.reserve() as blk:
                        tx = ttl.copy(a[mt, kt], blk)
                        tx.wait()
                    with b_dfb.reserve() as blk:
                        tx = ttl.copy(b[kt, nt], blk)
                        tx.wait()

    @ttl.datamovement()
    def dm_write():
        for mt in range(Mt):
            for nt in range(Nt):
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[mt, nt])
                    tx.wait()


BLOCKED_PARAMS = [
    (2, 1, 2),  # K=1: fused path not exercised, only standalone matmul.
    (2, 2, 2),  # Square output, basic K-accumulation.
    (4, 2, 2),  # Tall output, M-loop dominant.
    (1, 3, 4),  # Single M row, wide output, odd K.
    (4, 2, 1),  # Single N column, tall output.
    (2, 3, 4),  # Non-square with odd K.
    (2, 8, 2),  # Long K-accumulation chain.
]

BLOCKED_IDS = [f"{m}x{k}x{n}_blocked" for m, k, n in BLOCKED_PARAMS]


@pytest.mark.parametrize("Mt,Kt,Nt", BLOCKED_PARAMS, ids=BLOCKED_IDS)
@pytest.mark.parametrize(
    "pack_opts",
    [None, "--ttl-combine-pack-tiles"],
    ids=["pack_tile", "pack_tile_block"],
)
@pytest.mark.requires_device
def test_matmul_fused_acc_blocked(Mt, Kt, Nt, pack_opts, device):
    """Fused accumulation with outer block loop (1x1 blocks)."""
    _run_matmul_acc(
        matmul_fused_acc_blocked_kernel, Mt, Kt, Nt, device, options=pack_opts
    )


# =============================================================================
# Post-matmul unary fusion: relu(a @ b) and relu(a @ b + prev)
# =============================================================================


@ttl.operation(grid=(1, 1))
def matmul_relu_kernel(a, b, out):
    """Post-matmul relu: relu(a @ b) fuses into matmul_block + relu_tile."""
    Mt = a.shape[0] // TILE
    Kt = a.shape[1] // TILE
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(Mt, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, Nt), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        # First K-step: matmul to accumulator (no relu on intermediates).
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        with acc_dfb.reserve() as acc:
            acc.store(a_blk @ b_blk)
        a_blk.pop()
        b_blk.pop()

        # Remaining K-steps: fused accumulation.
        for _ in range(Kt - 1):
            with (
                a_dfb.wait() as a_blk,
                b_dfb.wait() as b_blk,
                acc_dfb.wait() as prev,
            ):
                with acc_dfb.reserve() as acc:
                    acc.store(prev + a_blk @ b_blk)

        # Final step: relu applied after full K-accumulation.
        with acc_dfb.wait() as final:
            with out_dfb.reserve() as o:
                o.store(ttl.math.relu(final))

    @ttl.datamovement()
    def dm_read():
        for kt in range(Kt):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:Mt, kt : kt + 1], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[kt : kt + 1, 0:Nt], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:Mt, 0:Nt])
            tx.wait()


@pytest.mark.parametrize(
    "Mt,Kt,Nt",
    [
        (1, 1, 1),
        (2, 2, 2),
        (1, 4, 2),
    ],
    ids=[f"{m}x{k}x{n}" for m, k, n in [(1, 1, 1), (2, 2, 2), (1, 4, 2)]],
)
@pytest.mark.parametrize(
    "pack_opts",
    [None, "--ttl-combine-pack-tiles"],
    ids=["pack_tile", "pack_tile_block"],
)
@pytest.mark.requires_device
def test_matmul_relu(Mt, Kt, Nt, pack_opts, device):
    """relu(A @ B): relu applied after full K-accumulation."""
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kwargs = {"options": pack_opts} if pack_opts else {}
    matmul_relu_kernel(a, b, out, **kwargs)

    result = ttnn.to_torch(out)
    golden = torch.relu(a_torch @ b_torch)

    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.99, (
        f"PCC {pcc:.6f} < 0.99 for {Mt}x{Kt}x{Nt} matmul+relu. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )


# =============================================================================
# Fused matmul + add + unary: relu(a @ b + prev) in a single expression.
# The add folds into the 3-operand matmul and relu is applied in-place,
# all within one compute body.
# =============================================================================


@ttl.operation(grid=(1, 1))
def matmul_add_relu_kernel(a, b, c, out):
    """relu((a @ b) + c) as a single fused expression."""
    Mt = a.shape[0] // TILE
    Nt = b.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(Mt, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, Nt), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(Mt, Nt), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            c_dfb.wait() as c_blk,
        ):
            with out_dfb.reserve() as o:
                o.store(ttl.math.relu(a_blk @ b_blk + c_blk))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:Mt, 0:1], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:1, 0:Nt], blk)
            tx.wait()
        with c_dfb.reserve() as blk:
            tx = ttl.copy(c[0:Mt, 0:Nt], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:Mt, 0:Nt])
            tx.wait()


@pytest.mark.parametrize(
    "Mt,Nt",
    [(1, 1), (2, 2), (1, 4)],
    ids=[f"{m}x{n}" for m, n in [(1, 1), (2, 2), (1, 4)]],
)
@pytest.mark.parametrize(
    "pack_opts",
    [None, "--ttl-combine-pack-tiles"],
    ids=["pack_tile", "pack_tile_block"],
)
@pytest.mark.requires_device
def test_matmul_add_relu(Mt, Nt, pack_opts, device):
    """relu((A @ B) + C): add folded + relu in-place, single fused compute."""
    M, K, N = Mt * TILE, TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    c_torch = torch.randn(M, N, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    c = to_dram(c_torch, device)
    out = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kwargs = {"options": pack_opts} if pack_opts else {}
    matmul_add_relu_kernel(a, b, c, out, **kwargs)

    result = ttnn.to_torch(out)
    golden = torch.relu(a_torch @ b_torch + c_torch)

    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), golden.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.99, (
        f"PCC {pcc:.6f} < 0.99 for {Mt}x1x{Nt} matmul+add+relu. "
        f"Max diff: {(result - golden).abs().max().item()}"
    )
