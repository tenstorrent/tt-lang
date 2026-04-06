# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SPMD matmul+bias test across multiple devices.

Shards A, C, Y on dim=0 across all available devices; replicates B.
Each device computes Y_shard = A_shard @ B + C_shard independently.
Requires >=2 devices.
"""

# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
import ttl

TILE_SIZE = 32
M_GRANULARITY = 4
N_GRANULARITY = 4
K_GRANULARITY = 1


@ttl.operation(grid="auto")
def matmul_with_bias(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    c: ttnn.Tensor,
    y: ttnn.Tensor,
) -> None:
    m_tiles_per_block = M_GRANULARITY
    n_tiles_per_block = N_GRANULARITY
    k_tiles_per_block = K_GRANULARITY

    grid_n, grid_m = ttl.grid_size(dims=2)

    m_blocks = a.shape[0] // TILE_SIZE // m_tiles_per_block
    n_blocks = b.shape[1] // TILE_SIZE // n_tiles_per_block
    k_blocks = a.shape[1] // TILE_SIZE // k_tiles_per_block

    m_blocks_per_node = -(-m_blocks // grid_m)  # divceil
    n_blocks_per_node = -(-n_blocks // grid_n)  # divceil

    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(m_tiles_per_block, k_tiles_per_block), block_count=2
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(k_tiles_per_block, n_tiles_per_block), block_count=2
    )
    c_dfb = ttl.make_dataflow_buffer_like(
        c, shape=(m_tiles_per_block, n_tiles_per_block), block_count=2
    )
    acc_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(m_tiles_per_block, n_tiles_per_block), block_count=2
    )
    y_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(m_tiles_per_block, n_tiles_per_block), block_count=2
    )

    @ttl.datamovement()
    def read():
        node_n, node_m = ttl.node(dims=2)

        for local_m_block in range(m_blocks_per_node):
            m_block = node_m * m_blocks_per_node + local_m_block
            if m_block < m_blocks:
                start_m_tile = m_block * m_tiles_per_block
                end_m_tile = (m_block + 1) * m_tiles_per_block

                for local_n_block in range(n_blocks_per_node):
                    n_block = node_n * n_blocks_per_node + local_n_block
                    if n_block < n_blocks:
                        start_n_tile = n_block * n_tiles_per_block
                        end_n_tile = (n_block + 1) * n_tiles_per_block

                        with c_dfb.reserve() as c_blk:
                            tx_c = ttl.copy(
                                c[
                                    start_m_tile:end_m_tile,
                                    start_n_tile:end_n_tile,
                                ],
                                c_blk,
                            )

                            tx_c.wait()

                        for k_block in range(k_blocks):
                            start_k_tile = k_block * k_tiles_per_block
                            end_k_tile = (k_block + 1) * k_tiles_per_block
                            with (
                                a_dfb.reserve() as a_blk,
                                b_dfb.reserve() as b_blk,
                            ):
                                tx_a = ttl.copy(
                                    a[
                                        start_m_tile:end_m_tile,
                                        start_k_tile:end_k_tile,
                                    ],
                                    a_blk,
                                )
                                tx_b = ttl.copy(
                                    b[
                                        start_k_tile:end_k_tile,
                                        start_n_tile:end_n_tile,
                                    ],
                                    b_blk,
                                )

                                tx_a.wait()
                                tx_b.wait()

    @ttl.compute()
    def compute():
        node_n, node_m = ttl.node(dims=2)

        for local_m_block in range(m_blocks_per_node):
            m_block = node_m * m_blocks_per_node + local_m_block
            if m_block < m_blocks:
                for local_n_block in range(n_blocks_per_node):
                    n_block = node_n * n_blocks_per_node + local_n_block
                    if n_block < n_blocks:
                        with acc_dfb.reserve() as acc_blk:
                            acc_blk.store(ttl.math.fill(acc_blk, 0))

                        for _ in range(k_blocks):
                            with (
                                a_dfb.wait() as a_blk,
                                b_dfb.wait() as b_blk,
                                acc_dfb.wait() as pre_acc_blk,
                            ):
                                with acc_dfb.reserve() as acc_blk:
                                    acc_blk.store(pre_acc_blk + a_blk @ b_blk)

                        with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                            with y_dfb.reserve() as y_blk:
                                y_blk.store(c_blk + acc_blk)

    @ttl.datamovement()
    def write():
        node_n, node_m = ttl.node(dims=2)

        for local_m_block in range(m_blocks_per_node):
            m_block = node_m * m_blocks_per_node + local_m_block
            if m_block < m_blocks:
                start_m_tile = m_block * m_tiles_per_block
                end_m_tile = (m_block + 1) * m_tiles_per_block

                for local_n_block in range(n_blocks_per_node):
                    n_block = node_n * n_blocks_per_node + local_n_block
                    if n_block < n_blocks:
                        start_n_tile = n_block * n_tiles_per_block
                        end_n_tile = (n_block + 1) * n_tiles_per_block

                        with y_dfb.wait() as y_blk:
                            tx = ttl.copy(
                                y_blk,
                                y[
                                    start_m_tile:end_m_tile,
                                    start_n_tile:end_n_tile,
                                ],
                            )
                            tx.wait()


MIN_DEVICES = 2


@pytest.fixture
def mesh_device():
    n_devices = ttnn.GetNumAvailableDevices()
    if n_devices < MIN_DEVICES:
        pytest.skip(f"need >={MIN_DEVICES} devices, have {n_devices}")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, n_devices))
    yield mesh, n_devices
    ttnn.close_mesh_device(mesh)


def test_matmul_with_bias_spmd(mesh_device):
    mesh, n_devices = mesh_device

    M, K, N = 8192, 8192, 8192

    A_torch = torch.randn((M, K), dtype=torch.bfloat16)
    B_torch = torch.randn((K, N), dtype=torch.bfloat16)
    C_torch = torch.randn((M, N), dtype=torch.bfloat16)

    A = ttnn.from_torch(
        A_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    B = ttnn.from_torch(
        B_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    C = ttnn.from_torch(
        C_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    Y = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )

    matmul_with_bias(A, B, C, Y)

    result = ttnn.to_torch(
        Y,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    )

    expected = A_torch @ B_torch + C_torch

    pcc = torch.corrcoef(
        torch.stack([result.flatten().float(), expected.flatten().float()])
    )[0, 1].item()
    assert pcc > 0.99, (
        f"PCC {pcc:.6f} < 0.99 for matmul+bias. "
        f"Max diff: {(result - expected).abs().max().item()}"
    )
