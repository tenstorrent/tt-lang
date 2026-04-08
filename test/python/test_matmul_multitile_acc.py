# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul with bias and multi-tile blocks: Y = A @ B + C.

Tests the fused prev + a @ b accumulation pattern with varying block sizes.
Multi-tile blocks (e.g., 4x2 @ 2x4 -> 4x4) require the matmul+add fusion
in ConvertTTLToCompute to handle the matmul's contraction dimension correctly.

See tenstorrent/tt-lang#460.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32


def _make_matmul_bias_kernel(block_m, block_k, block_n):
    """Create a matmul+bias kernel with the given block sizes."""

    @ttl.operation(grid=(1, 1))
    def matmul_bias(a_tensor, b_tensor, c_tensor, out_tensor):
        Mt = a_tensor.shape[0] // TILE
        Kt = a_tensor.shape[1] // TILE
        Nt = b_tensor.shape[1] // TILE

        a_dfb = ttl.make_dataflow_buffer_like(
            a_tensor, shape=(block_m, block_k), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b_tensor, shape=(block_k, block_n), block_count=2
        )
        c_dfb = ttl.make_dataflow_buffer_like(
            c_tensor, shape=(block_m, block_n), block_count=2
        )
        acc_dfb = ttl.make_dataflow_buffer_like(
            out_tensor, shape=(block_m, block_n), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out_tensor, shape=(block_m, block_n), block_count=2
        )

        m_blocks = Mt // block_m
        n_blocks = Nt // block_n
        k_blocks = Kt // block_k

        @ttl.compute()
        def mm_compute():
            for _ in range(m_blocks):
                for _ in range(n_blocks):
                    with c_dfb.wait() as c_blk, acc_dfb.reserve() as acc:
                        acc.store(c_blk)

                    for _ in range(k_blocks):
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            acc_dfb.wait() as prev,
                        ):
                            with acc_dfb.reserve() as acc:
                                acc.store(prev + a_blk @ b_blk)

                    with acc_dfb.wait() as acc_blk:
                        with out_dfb.reserve() as o:
                            o.store(acc_blk)

        @ttl.datamovement()
        def dm_read():
            for mi in range(m_blocks):
                for ni in range(n_blocks):
                    row = mi * block_m
                    col = ni * block_n
                    with c_dfb.reserve() as blk:
                        tx = ttl.copy(
                            c_tensor[row : row + block_m, col : col + block_n],
                            blk,
                        )
                        tx.wait()

                    for ki in range(k_blocks):
                        kcol = ki * block_k
                        with a_dfb.reserve() as blk:
                            tx = ttl.copy(
                                a_tensor[row : row + block_m, kcol : kcol + block_k],
                                blk,
                            )
                            tx.wait()
                        with b_dfb.reserve() as blk:
                            tx = ttl.copy(
                                b_tensor[kcol : kcol + block_k, col : col + block_n],
                                blk,
                            )
                            tx.wait()

        @ttl.datamovement()
        def dm_write():
            for mi in range(m_blocks):
                for ni in range(n_blocks):
                    row = mi * block_m
                    col = ni * block_n
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(
                            blk,
                            out_tensor[row : row + block_m, col : col + block_n],
                        )
                        tx.wait()

    return matmul_bias


# (block_m, block_k, block_n, Mt, Kt, Nt) -- block sizes and tile counts.
# Tile counts must be divisible by the corresponding block size.
PARAMS = [
    (1, 1, 1, 2, 2, 2),
    (2, 1, 2, 2, 2, 2),
    (2, 2, 2, 4, 4, 4),
    (4, 2, 4, 4, 4, 4),
]

PARAM_IDS = [
    f"blk{bm}x{bk}x{bn}_tiles{mt}x{kt}x{nt}" for bm, bk, bn, mt, kt, nt in PARAMS
]


@pytest.mark.parametrize("block_m,block_k,block_n,Mt,Kt,Nt", PARAMS, ids=PARAM_IDS)
@pytest.mark.requires_device
def test_matmul_multitile_acc(block_m, block_k, block_n, Mt, Kt, Nt, device):
    """Matmul+bias with multi-tile blocks: prev + a @ b accumulation."""
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    c_torch = torch.randn(M, N, dtype=torch.bfloat16)

    a_dev = to_dram(a_torch, device)
    b_dev = to_dram(b_torch, device)
    c_dev = to_dram(c_torch, device)
    out_dev = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = _make_matmul_bias_kernel(block_m, block_k, block_n)
    kernel(a_dev, b_dev, c_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    golden = (a_torch @ b_torch + c_torch).float()

    assert_pcc(golden, result, threshold=0.99)


# =============================================================================
# Multi-tile c + a @ b with full K in one DFB block (no user K-loop).
# The matmul's 3D [M, N, K] iteration handles the reduction internally.
# =============================================================================


def _make_matmul_bias_full_k_kernel(block_m, block_n):
    """Create a matmul+bias kernel with full K in one block (no K-loop)."""

    @ttl.operation(grid=(1, 1))
    def matmul_bias_full_k(a_tensor, b_tensor, c_tensor, out_tensor):
        Mt = a_tensor.shape[0] // TILE
        Kt = a_tensor.shape[1] // TILE
        Nt = b_tensor.shape[1] // TILE

        a_dfb = ttl.make_dataflow_buffer_like(
            a_tensor, shape=(block_m, Kt), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b_tensor, shape=(Kt, block_n), block_count=2
        )
        c_dfb = ttl.make_dataflow_buffer_like(
            c_tensor, shape=(block_m, block_n), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out_tensor, shape=(block_m, block_n), block_count=2
        )

        m_blocks = Mt // block_m
        n_blocks = Nt // block_n

        @ttl.compute()
        def mm_compute():
            for _ in range(m_blocks):
                for _ in range(n_blocks):
                    with (
                        a_dfb.wait() as a_blk,
                        b_dfb.wait() as b_blk,
                        c_dfb.wait() as c_blk,
                    ):
                        with out_dfb.reserve() as o:
                            o.store(c_blk + a_blk @ b_blk)

        @ttl.datamovement()
        def dm_read():
            for mi in range(m_blocks):
                for ni in range(n_blocks):
                    row = mi * block_m
                    col = ni * block_n
                    with c_dfb.reserve() as blk:
                        tx = ttl.copy(
                            c_tensor[row : row + block_m, col : col + block_n],
                            blk,
                        )
                        tx.wait()
                    with a_dfb.reserve() as blk:
                        tx = ttl.copy(a_tensor[row : row + block_m, 0:Kt], blk)
                        tx.wait()
                    with b_dfb.reserve() as blk:
                        tx = ttl.copy(b_tensor[0:Kt, col : col + block_n], blk)
                        tx.wait()

        @ttl.datamovement()
        def dm_write():
            for mi in range(m_blocks):
                for ni in range(n_blocks):
                    row = mi * block_m
                    col = ni * block_n
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(
                            blk,
                            out_tensor[row : row + block_m, col : col + block_n],
                        )
                        tx.wait()

    return matmul_bias_full_k


FULL_K_PARAMS = [
    (1, 1, 2, 2),
    (2, 2, 4, 4),
    (4, 4, 4, 4),
]

FULL_K_IDS = [f"blk{bm}x{bn}_tiles{mt}x{nt}" for bm, bn, mt, nt in FULL_K_PARAMS]


@pytest.mark.parametrize(
    "block_m,block_n,Mt,Nt",
    FULL_K_PARAMS,
    ids=FULL_K_IDS,
)
@pytest.mark.requires_device
def test_matmul_multitile_full_k(block_m, block_n, Mt, Nt, device):
    """c + a @ b with full K in one DFB block (no user K-loop)."""
    Kt = 2
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    c_torch = torch.randn(M, N, dtype=torch.bfloat16)

    a_dev = to_dram(a_torch, device)
    b_dev = to_dram(b_torch, device)
    c_dev = to_dram(c_torch, device)
    out_dev = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = _make_matmul_bias_full_k_kernel(block_m, block_n)
    kernel(a_dev, b_dev, c_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    golden = (a_torch @ b_torch + c_torch).float()

    assert_pcc(golden, result, threshold=0.99)


# =============================================================================
# Multi-tile relu(prev + a @ b): matmul+add+relu fused in one compute body.
# Exercises buildFusedCompute's matmul-aware 3D iteration space with a
# post-matmul unary op.
# =============================================================================


def _make_matmul_relu_kernel(block_m, block_k, block_n):
    """Create a matmul+relu kernel: relu(prev + a @ b) with K-accumulation."""

    @ttl.operation(grid=(1, 1))
    def matmul_relu(a_tensor, b_tensor, out_tensor):
        Mt = a_tensor.shape[0] // TILE
        Kt = a_tensor.shape[1] // TILE
        Nt = b_tensor.shape[1] // TILE

        a_dfb = ttl.make_dataflow_buffer_like(
            a_tensor, shape=(block_m, block_k), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b_tensor, shape=(block_k, block_n), block_count=2
        )
        acc_dfb = ttl.make_dataflow_buffer_like(
            out_tensor, shape=(block_m, block_n), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out_tensor, shape=(block_m, block_n), block_count=2
        )

        m_blocks = Mt // block_m
        n_blocks = Nt // block_n
        k_blocks = Kt // block_k

        @ttl.compute()
        def mm_compute():
            for _ in range(m_blocks):
                for _ in range(n_blocks):
                    # First K-step: standalone matmul.
                    a_blk = a_dfb.wait()
                    b_blk = b_dfb.wait()
                    with acc_dfb.reserve() as acc:
                        acc.store(a_blk @ b_blk)
                    a_blk.pop()
                    b_blk.pop()

                    # Remaining K-steps: prev + a @ b.
                    for _ in range(k_blocks - 1):
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            acc_dfb.wait() as prev,
                        ):
                            with acc_dfb.reserve() as acc:
                                acc.store(prev + a_blk @ b_blk)

                    # relu applied after full K-accumulation.
                    with acc_dfb.wait() as final:
                        with out_dfb.reserve() as o:
                            o.store(ttl.math.relu(final))

        @ttl.datamovement()
        def dm_read():
            for mi in range(m_blocks):
                for ni in range(n_blocks):
                    row = mi * block_m
                    col = ni * block_n
                    for ki in range(k_blocks):
                        kcol = ki * block_k
                        with a_dfb.reserve() as blk:
                            tx = ttl.copy(
                                a_tensor[row : row + block_m, kcol : kcol + block_k],
                                blk,
                            )
                            tx.wait()
                        with b_dfb.reserve() as blk:
                            tx = ttl.copy(
                                b_tensor[kcol : kcol + block_k, col : col + block_n],
                                blk,
                            )
                            tx.wait()

        @ttl.datamovement()
        def dm_write():
            for mi in range(m_blocks):
                for ni in range(n_blocks):
                    row = mi * block_m
                    col = ni * block_n
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(
                            blk,
                            out_tensor[row : row + block_m, col : col + block_n],
                        )
                        tx.wait()

    return matmul_relu


RELU_PARAMS = [
    (1, 1, 1, 2, 2, 2),
    (2, 2, 2, 4, 4, 4),
    (4, 2, 4, 4, 4, 4),
]

RELU_IDS = [
    f"blk{bm}x{bk}x{bn}_tiles{mt}x{kt}x{nt}" for bm, bk, bn, mt, kt, nt in RELU_PARAMS
]


@pytest.mark.parametrize("block_m,block_k,block_n,Mt,Kt,Nt", RELU_PARAMS, ids=RELU_IDS)
@pytest.mark.requires_device
def test_matmul_multitile_relu(block_m, block_k, block_n, Mt, Kt, Nt, device):
    """relu(A @ B) with multi-tile blocks and K-accumulation."""
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)

    a_dev = to_dram(a_torch, device)
    b_dev = to_dram(b_torch, device)
    out_dev = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = _make_matmul_relu_kernel(block_m, block_k, block_n)
    kernel(a_dev, b_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    golden = torch.relu((a_torch @ b_torch).float())

    assert_pcc(golden, result, threshold=0.99)
