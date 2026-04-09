# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#
# Tutorial Step 7: Multi-Device, Shard K with All-Reduce
# =======================================================
# Replaces the host-side manual reduction from Step 6 with an on-device
# all-reduce, keeping the final result on the mesh rather than pulling it
# to the host for summation.
#
# New concepts introduced:
#   - ttnn.all_reduce — sums partial_ys across all devices in-place using the
#     TT-Fabric interconnect; each device ends up with the fully reduced M×N
#     result (the result is replicated across all devices)
#   - Post-reduce activation: relu is applied on-device after all_reduce,
#     replacing the host-side relu from Step 6
#
# The TT-Lang operation body and the K-sharding setup are identical to Step 6.
# The only change is in the host code: ttnn.all_reduce + ttnn.relu replace the
# manual Python loop that summed partial outputs.

import ttnn
import torch


def from_torch(tensor: torch.Tensor, mesh_mapper):
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


torch.manual_seed(42)

n_devices = ttnn.GetNumAvailableDevices()
assert n_devices > 1 and (
    n_devices & (n_devices - 1) == 0
), "Number of available devices must be >1 and be power of 2 "
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, n_devices))

try:
    M, K, N = 8192, 8192, 8192

    a = torch.randn((M, K), dtype=torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.bfloat16)
    c = torch.randn((M, N), dtype=torch.bfloat16)

    expected_y = torch.relu(a @ b + c)

    # K-sharding setup is identical to Step 6.

    a = from_torch(a, ttnn.ShardTensorToMesh(mesh_device, dim=1))
    b = from_torch(b, ttnn.ShardTensorToMesh(mesh_device, dim=0))

    replicated_cs = torch.zeros((M * n_devices, N), dtype=torch.bfloat16)
    replicated_cs[:M, :] = c
    replicated_cs = from_torch(
        replicated_cs, ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )

    partial_ys = torch.zeros((M * n_devices, N), dtype=torch.bfloat16)
    partial_ys = from_torch(partial_ys, ttnn.ShardTensorToMesh(mesh_device, dim=0))

    tutorial_operation(a, b, replicated_cs, partial_ys)

    # ttnn.all_reduce sums partial_ys across all devices using TT-Fabric,
    # producing a fully reduced M×N result replicated on every device.
    # ttnn.relu is then applied on-device, replacing the host-side loop from
    # Step 6.

    replicated_ys = ttnn.all_reduce(partial_ys)
    replicated_ys = ttnn.relu(replicated_ys)

    # Because the result is replicated, all devices hold the correct answer.
    # Verify each device's copy against the expected output.

    replicated_ys = ttnn.to_torch(
        replicated_ys, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )

    for i in range(n_devices):
        y = replicated_ys[i * M : (i + 1) * M, :]

        pcc = torch.corrcoef(
            torch.stack([y.flatten().float(), expected_y.flatten().float()])
        )[0, 1].item()

        print(f"PCC {pcc:.6f}")

        assert pcc > 0.99

finally:
    ttnn.close_device(mesh_device)
