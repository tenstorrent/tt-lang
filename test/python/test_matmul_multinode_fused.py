# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multinode fused matmul+bias: Y = A @ B + C with grid="auto".

Each parametrized shape (M_blk, K_blk, N_blk) defines the DFB block
dimensions in tiles. The outer M/N/K loops distribute work across the
grid. The fused compute expression `pre_acc + a @ b` lowers to
copy_tile + matmul_block with 3D [M, N, K] indexing maps so that DST
subblocking slices operands correctly for any block shape.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32


def _make_kernel(m_blk, k_blk, n_blk):
    @ttl.operation(grid="auto")
    def matmul_bias(
        a_tensor: ttnn.Tensor,
        b_tensor: ttnn.Tensor,
        c_tensor: ttnn.Tensor,
        y_tensor: ttnn.Tensor,
    ) -> None:
        grid_n, grid_m = ttl.grid_size(dims=2)

        m_blocks = a_tensor.shape[0] // TILE // m_blk
        n_blocks = b_tensor.shape[1] // TILE // n_blk
        k_blocks = a_tensor.shape[1] // TILE // k_blk

        m_blocks_per_node = -(-m_blocks // grid_m)
        n_blocks_per_node = -(-n_blocks // grid_n)

        a_dfb = ttl.make_dataflow_buffer_like(
            a_tensor, shape=(m_blk, k_blk), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b_tensor, shape=(k_blk, n_blk), block_count=2
        )
        c_dfb = ttl.make_dataflow_buffer_like(
            c_tensor, shape=(m_blk, n_blk), block_count=2
        )
        acc_dfb = ttl.make_dataflow_buffer_like(
            y_tensor, shape=(m_blk, n_blk), block_count=2
        )
        y_dfb = ttl.make_dataflow_buffer_like(
            y_tensor, shape=(m_blk, n_blk), block_count=2
        )

        @ttl.datamovement()
        def read():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_m
                if mb < m_blocks:
                    sm = mb * m_blk
                    em = (mb + 1) * m_blk
                    for local_n in range(n_blocks_per_node):
                        nb = node_n * n_blocks_per_node + local_n
                        if nb < n_blocks:
                            sn = nb * n_blk
                            en = (nb + 1) * n_blk
                            with c_dfb.reserve() as c_blk:
                                tx = ttl.copy(c_tensor[sm:em, sn:en], c_blk)
                                tx.wait()
                            for kb in range(k_blocks):
                                sk = kb * k_blk
                                ek = (kb + 1) * k_blk
                                with (
                                    a_dfb.reserve() as a_blk,
                                    b_dfb.reserve() as b_blk,
                                ):
                                    tx_a = ttl.copy(a_tensor[sm:em, sk:ek], a_blk)
                                    tx_b = ttl.copy(b_tensor[sk:ek, sn:en], b_blk)
                                    tx_a.wait()
                                    tx_b.wait()

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_m
                if mb < m_blocks:
                    for local_n in range(n_blocks_per_node):
                        nb = node_n * n_blocks_per_node + local_n
                        if nb < n_blocks:
                            with (
                                c_dfb.wait() as c_blk,
                                acc_dfb.reserve() as acc_blk,
                            ):
                                acc_blk.store(c_blk)
                            for _ in range(k_blocks):
                                with (
                                    a_dfb.wait() as a_blk,
                                    b_dfb.wait() as b_blk,
                                    acc_dfb.wait() as pre_acc,
                                ):
                                    with acc_dfb.reserve() as acc_blk:
                                        acc_blk.store(pre_acc + a_blk @ b_blk)
                            with acc_dfb.wait() as acc_blk:
                                with y_dfb.reserve() as y_blk:
                                    y_blk.store(acc_blk)

        @ttl.datamovement()
        def write():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_m
                if mb < m_blocks:
                    sm = mb * m_blk
                    em = (mb + 1) * m_blk
                    for local_n in range(n_blocks_per_node):
                        nb = node_n * n_blocks_per_node + local_n
                        if nb < n_blocks:
                            sn = nb * n_blk
                            en = (nb + 1) * n_blk
                            with y_dfb.wait() as y_blk:
                                tx = ttl.copy(y_blk, y_tensor[sm:em, sn:en])
                                tx.wait()

    return matmul_bias


# (M_blk, K_blk, N_blk) — block dimensions in tiles.
# Total tensor size is chosen so that each shape has at least 2 K-blocks
# and multiple M/N blocks to exercise multinode distribution.
SHAPES = [
    pytest.param((2, 2, 2), id="2x2x2"),
    pytest.param((4, 2, 4), id="4x2x4"),
    pytest.param((4, 4, 4), id="4x4x4"),
    pytest.param((2, 4, 2), id="2x4x2"),
    pytest.param((1, 2, 4), id="1x2x4"),
    pytest.param((4, 2, 1), id="4x2x1"),
    pytest.param((2, 1, 2), id="2x1x2"),
]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("block_shape", SHAPES)
def test_matmul_bias_multinode(device, block_shape):
    m_blk, k_blk, n_blk = block_shape
    # Total size: 8 blocks in each dimension (except K gets 4 blocks
    # to keep runtime reasonable).
    total_m = m_blk * 8 * TILE
    total_k = k_blk * 4 * TILE
    total_n = n_blk * 8 * TILE

    a_torch = torch.randn((total_m, total_k), dtype=torch.bfloat16)
    b_torch = torch.randn((total_k, total_n), dtype=torch.bfloat16)
    c_torch = torch.randn((total_m, total_n), dtype=torch.bfloat16)

    a_dev = to_dram(a_torch, device)
    b_dev = to_dram(b_torch, device)
    c_dev = to_dram(c_torch, device)
    y_dev = to_dram(torch.zeros((total_m, total_n), dtype=torch.bfloat16), device)

    kernel = _make_kernel(m_blk, k_blk, n_blk)
    kernel(a_dev, b_dev, c_dev, y_dev)

    result = ttnn.to_torch(y_dev).float()
    expected = (a_torch @ b_torch + c_torch).float()
    assert_pcc(expected, result, threshold=0.99)
