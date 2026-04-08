# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

"""
Test SPMD mesh tensor compilation and execution.

Logical shape: 128x32 (4x1 tiles)
Shard shape:    32x32 (1x1 tile) -- dim-0 sharded across 4 devices

The kernel processes a single tile (1x1). With the full logical shape the
kernel would only touch the first tile and produce incorrect results for the
rest. Correct output for all elements proves the tensor was properly sharded
so each device sees its own 32x32 slice.

Requires >=4 devices for real mesh sharding. On single-card, the test is
skipped (single-device execution is covered by other tests).
"""

import torch
import ttnn
import ttl

TILE = 32
N_DEVICES = 4
LOGICAL_ROWS = TILE * N_DEVICES  # 128
LOGICAL_COLS = TILE  # 32
SHARD_ROWS = LOGICAL_ROWS // N_DEVICES  # 32


@ttl.operation(grid=(1, 1))
def add_kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as l, b_dfb.wait() as r, out_dfb.reserve() as o:
            o.store(l + r)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()

        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# CHECK: Mesh Tensor SPMD Test
print("=== Mesh Tensor SPMD Test ===")

n_available = ttnn.GetNumAvailableDevices()
print("Available devices: %d" % n_available)

if n_available < N_DEVICES:
    # CHECK: PASS
    print("PASS: skipped (need %d devices, have %d)" % (N_DEVICES, n_available))
else:
    print(
        "Multi-card path: sharding [%d, %d] across %d devices"
        % (LOGICAL_ROWS, LOGICAL_COLS, N_DEVICES)
    )

    a_torch = torch.full((SHARD_ROWS, LOGICAL_COLS), 2.0, dtype=torch.bfloat16)
    b_torch = torch.full((SHARD_ROWS, LOGICAL_COLS), 3.0, dtype=torch.bfloat16)
    expected = a_torch + b_torch

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_DEVICES))

    a_logical = torch.full((LOGICAL_ROWS, LOGICAL_COLS), 2.0, dtype=torch.bfloat16)
    b_logical = torch.full((LOGICAL_ROWS, LOGICAL_COLS), 3.0, dtype=torch.bfloat16)
    out_logical = torch.zeros(LOGICAL_ROWS, LOGICAL_COLS, dtype=torch.bfloat16)

    a = ttnn.from_torch(
        a_logical,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    b = ttnn.from_torch(
        b_logical,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    out = ttnn.from_torch(
        out_logical,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    add_kernel(a, b, out)

    result = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    # Each device should have computed 2+3=5 on its 32x32 shard.
    # Without proper sharding the kernel would only write the first tile,
    # leaving the rest as zeros.
    for i in range(N_DEVICES):
        shard = result[i * SHARD_ROWS : (i + 1) * SHARD_ROWS]
        assert torch.allclose(
            shard.float(), expected.float(), rtol=1e-2
        ), "Device %d shard incorrect: max error %.4f" % (
            i,
            (shard.float() - expected.float()).abs().max().item(),
        )
    print("PASS: all %d shards correct (2 + 3 = 5)" % N_DEVICES)

    ttnn.close_mesh_device(mesh_device)

# CHECK: Mesh Tensor SPMD Test Passed
print("=== Mesh Tensor SPMD Test Passed ===")
