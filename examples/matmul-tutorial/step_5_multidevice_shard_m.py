# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#
# Tutorial Step 5: Multi-Device, Shard M
# =======================================
# Extends Step 4 to run across multiple devices using SPMD (Single-Program
# Multiple-Data) mode.  The TT-Lang operation itself is unchanged; only the
# tensor distribution across devices differs.
#
# New concepts introduced:
#   - ttnn.MeshShape / ttnn.open_mesh_device — open a 1D mesh of all available
#     devices
#   - ttnn.ShardTensorToMesh(dim=0) — split a tensor along the M dimension so
#     each device receives M/n_devices rows
#   - ttnn.ReplicateTensorToMesh   — send the same tensor to every device
#   - ttnn.ConcatMeshToTensor(dim=0) — gather per-device output tensors back to
#     the host by concatenating along M
#
# Sharding strategy: a and c are sharded along M (rows), b is replicated.
# Each device computes its portion of the M×N output independently with no
# inter-device communication required.  The host concatenates the results.

import ttnn
import torch


def from_torch(tensor: torch.Tensor, mesh_mapper):

    # Upload a bfloat16 torch tensor to DRAM on all mesh devices, applying the
    # given mapper to determine how the tensor is distributed.

    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


import ttl

TILE_SIZE = 32
M_GRANULARITY = 4
N_GRANULARITY = 4
K_GRANULARITY = 4


# The TT-Lang operation body is identical to Step 4.  grid="auto" applies
# independently to each device in SPMD mode; each device fills its own grid.


@ttl.operation(grid="auto")
def tutorial_operation(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    c: ttnn.Tensor,
    y: ttnn.Tensor,
) -> None:
    m_tiles_per_block = M_GRANULARITY
    n_tiles_per_block = N_GRANULARITY
    k_tiles_per_block = K_GRANULARITY

    m_blocks = a.shape[0] // TILE_SIZE // m_tiles_per_block
    n_blocks = b.shape[1] // TILE_SIZE // n_tiles_per_block
    k_blocks = a.shape[1] // TILE_SIZE // k_tiles_per_block

    grid_n, grid_m = ttl.grid_size(dims=2)

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
                                y_blk.store(ttl.math.relu(c_blk + acc_blk))

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


torch.manual_seed(42)

n_devices = ttnn.GetNumAvailableDevices()
assert n_devices > 0 and (
    n_devices & (n_devices - 1) == 0
), "Number of available devices must be power of 2 "
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

# Open a 1D mesh of all available devices.  Each device will process an
# independent M/n_devices slice of the output rows.

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, n_devices))

try:
    M, K, N = 8192, 8192, 8192

    a = torch.randn((M, K), dtype=torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.bfloat16)
    c = torch.randn((M, N), dtype=torch.bfloat16)

    expected_y = torch.relu(a @ b + c)

    # Distribute tensors across devices:
    #   a: sharded along M (each device gets M/n_devices rows)
    #   b: replicated on every device (all devices need the full K×N matrix)
    #   c: sharded along M to match the corresponding rows of a

    a = from_torch(a, ttnn.ShardTensorToMesh(mesh_device, dim=0))
    b = from_torch(b, ttnn.ReplicateTensorToMesh(mesh_device))
    c = from_torch(c, ttnn.ShardTensorToMesh(mesh_device, dim=0))

    y = torch.zeros((M, N), dtype=torch.bfloat16)
    y = from_torch(y, ttnn.ShardTensorToMesh(mesh_device, dim=0))

    tutorial_operation(a, b, c, y)

    # Gather per-device output shards back to the host by concatenating along M.

    y = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    pcc = torch.corrcoef(
        torch.stack([y.flatten().float(), expected_y.flatten().float()])
    )[0, 1].item()

    print(f"PCC {pcc:.6f}")

    assert pcc > 0.99

finally:
    ttnn.close_device(mesh_device)
