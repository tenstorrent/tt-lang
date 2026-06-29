# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device, multi-device
# RUN: env -u TT_VISIBLE_DEVICES %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

"""
Test SPMD mesh tensor compilation and execution.

Logical shape: (32 * num_devices) x 32 -- one 32x32 tile per device.
Shard shape:    32x32 (1x1 tile) -- dim-0 sharded across all mesh devices.

The kernel processes a single tile (1x1). With the full logical shape the
kernel would only touch the first tile and produce incorrect results for the
rest. Correct output for all elements proves the tensor was properly sharded
so each device sees its own 32x32 slice.

Requires multiple devices for real mesh sharding. The `multi-device` lit
feature restricts the test so lit reports it unsupported on single-card hosts
(single-device execution is covered by other tests).
"""

import torch
import ttnn
import ttl
from ttlang_test_utils import open_fabric_mesh

TILE = 32
LOGICAL_COLS = TILE  # 32
SHARD_ROWS = TILE  # one 32x32 tile per device
MIN_DEVICES = 2


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


# CHECKs assert the multi-card path ran; the Python asserts below check shards.
# CHECK: === Mesh Tensor SPMD Test ===
# CHECK: Available devices: {{[0-9]+}}
# CHECK: Multi-card path: sharding [{{[0-9]+}}, 32] across {{[0-9]+}} devices
# CHECK: PASS: all {{[0-9]+}} shards correct (2 + 3 = 5)
# CHECK: === Mesh Tensor SPMD Test Passed ===
print("=== Mesh Tensor SPMD Test ===")

n_devices = ttnn.get_num_devices()
print("Available devices: %d" % n_devices)
assert (
    n_devices >= MIN_DEVICES
), "multi-device gating guarantees >= %d cards, but ttnn reports %d" % (
    MIN_DEVICES,
    n_devices,
)

logical_rows = TILE * n_devices
print(
    "Multi-card path: sharding [%d, %d] across %d devices"
    % (logical_rows, LOGICAL_COLS, n_devices)
)

a_torch = torch.full((SHARD_ROWS, LOGICAL_COLS), 2.0, dtype=torch.bfloat16)
b_torch = torch.full((SHARD_ROWS, LOGICAL_COLS), 3.0, dtype=torch.bfloat16)
expected = a_torch + b_torch

with open_fabric_mesh() as mesh_device:
    a_logical = torch.full((logical_rows, LOGICAL_COLS), 2.0, dtype=torch.bfloat16)
    b_logical = torch.full((logical_rows, LOGICAL_COLS), 3.0, dtype=torch.bfloat16)
    out_logical = torch.zeros(logical_rows, LOGICAL_COLS, dtype=torch.bfloat16)

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
    # Each device should have computed 2+3=5 on its 32x32 shard. Without proper
    # sharding the kernel would only write the first tile, leaving the rest as
    # zeros.
    for i in range(n_devices):
        shard = result[i * SHARD_ROWS : (i + 1) * SHARD_ROWS]
        assert torch.allclose(
            shard.float(), expected.float(), rtol=1e-2
        ), "Device %d shard incorrect: max error %.4f" % (
            i,
            (shard.float() - expected.float()).abs().max().item(),
        )
    print("PASS: all %d shards correct (2 + 3 = 5)" % n_devices)

print("=== Mesh Tensor SPMD Test Passed ===")
