# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SPMD mesh tensor coverage for PipeNet lowering.

Each mesh device receives one row tile shard. Within each shard, core (0, 0)
sends tile (0, 0) to core (1, 0) through a true unicast pipe. The destination
core computes abs and writes tile (0, 1). Composing the mesh result proves that
the same pipe schedule executes correctly on every device shard.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, open_fabric_mesh

# Opens a fabric mesh across all chips; run serially, not in the per-chip pool.
pytestmark = pytest.mark.multi_device

TILE = 32
GRID_X = 2
GRID_Y = 1
MIN_DEVICES = 2


@ttl.operation(grid=(GRID_X, GRID_Y))
def mesh_unicast_pipe_kernel(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])

    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_received_tile():
        if net.is_dst():
            with recv_dfb.wait() as recv_blk, out_dfb.reserve() as out_blk:
                out_blk.store(ttl.math.abs(recv_blk))

    @ttl.datamovement()
    def read_and_pipe_tile():
        def send(pipe):
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 0], send_blk).wait()
                ttl.copy(send_blk, pipe).wait()

        net.if_src(send)

        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()

        net.if_dst(recv)

    @ttl.datamovement()
    def write_received_tile():
        if net.is_dst():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0, 1]).wait()


@pytest.fixture
def mesh_device():
    num_devices = ttnn.get_num_devices()
    if num_devices < MIN_DEVICES:
        pytest.skip(f"need >={MIN_DEVICES} devices, have {num_devices}")

    with open_fabric_mesh() as mesh:
        yield mesh, num_devices


def test_mesh_unicast_pipe_spmd(mesh_device):
    mesh, num_devices = mesh_device

    logical_shape = (num_devices * TILE, GRID_X * TILE)
    inp_torch = torch.randn(logical_shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(logical_shape, dtype=torch.bfloat16)

    inp = ttnn.from_torch(
        inp_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    out = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )

    mesh_unicast_pipe_kernel(inp, out)

    result = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    )

    expected = torch.zeros_like(inp_torch)
    for device_idx in range(num_devices):
        row_start = device_idx * TILE
        row_end = row_start + TILE
        expected[row_start:row_end, TILE : 2 * TILE] = torch.abs(
            inp_torch[row_start:row_end, 0:TILE]
        )

    assert_allclose(result.float(), expected.float(), rtol=0.02, atol=0.5)
