# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_HARDWARE_CI: skip-compiler
# TTLANG_TUTORIAL_CI: requires-multi-device
# type: ignore

"""Ring all-gather of a column-sharded tensor over the device fabric.

Implements the relay lowering described in
docs/development/AllGatherLowering.md. Every device owns a K shard of a
row-major tile matrix. Each shard is split into a left and a right half; right
halves travel forward around the ring and left halves backward, so every
incoming link direction carries (D - 1) / 2 shards for D devices, the
all-gather lower bound.

Nodes with y = 0 carry the forward direction and nodes with y = 1 the
backward direction; x selects one of the lanes per direction, and every node
sends in one ring direction only. On each node, thread `send_local` sends the
node's chunks of the local half one hop over the seed net, and thread `relay`
receives each remote chunk directly into its DRAM destination slot, reads it
back, and forwards it one more hop over the relay net until it has reached
every device.

The destination holds only remote data, indexed by source distance relative
to the receiving device d: slot s - 1 holds the left half of device d + s and
the right half of device d - s.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from math import prod

import torch
import ttl
import ttnn

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh, to_dram
from utils.correctness import assert_allclose

TILE_SIZE = 32


def make_ring_all_gather_operation(
    device_count: int,
    *,
    m_tiles: int,
    k_shard_tiles: int,
    lanes: int,
    chunk_shape: tuple[int, int],
) -> Callable[[ttnn.Tensor, ttnn.Tensor], None]:
    chunk_rows, chunk_cols = chunk_shape
    half_tiles = k_shard_tiles // 2
    if device_count < 3:
        raise ValueError("the ring all-gather requires at least three devices")
    if min(m_tiles, k_shard_tiles, lanes, chunk_rows, chunk_cols) < 1:
        raise ValueError("tile counts, lanes, and chunk shape must be positive")
    if k_shard_tiles % 2:
        raise ValueError("the K shard must have an even number of tiles")
    if half_tiles % chunk_cols:
        raise ValueError("chunk columns must evenly divide each half of the shard")
    if m_tiles % (chunk_rows * lanes):
        raise ValueError("chunk rows times lanes must evenly divide M")
    m_chunks_per_lane = m_tiles // (chunk_rows * lanes)
    col_chunks = half_tiles // chunk_cols

    device_domain = ttl.DeviceDomain((device_count, 1))

    def ring_net(direction: int, offset: int) -> ttl.PipeNet:
        return ttl.PipeNet(
            graph=ttl.TransferGraph.axis_neighbor(
                device_domain, axis=0, offset=offset, wrap=True
            ),
            pipes=[
                ttl.Pipe(src=(lane, direction), dst=(lane, direction))
                for lane in range(lanes)
            ],
        )

    forward_seed_net = ring_net(0, 1)
    forward_relay_net = ring_net(0, 1)
    backward_seed_net = ring_net(1, device_count - 1)
    backward_relay_net = ring_net(1, device_count - 1)

    @ttl.operation(grid=(lanes, 2), device_domain=device_domain)
    def ring_all_gather(source: ttnn.Tensor, destination: ttnn.Tensor) -> None:
        local_dfb = ttl.make_dataflow_buffer_like(
            source, shape=(chunk_rows, chunk_cols), block_count=2
        )
        relay_dfb = ttl.make_dataflow_buffer_like(
            source, shape=(chunk_rows, chunk_cols), block_count=2
        )

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def send_local():
            lane, direction = ttl.node(dims=2)
            for m_chunk in range(m_chunks_per_lane):
                m_begin = (m_chunk * lanes + lane) * chunk_rows
                for col_chunk in range(col_chunks):
                    k_begin = (1 - direction) * half_tiles + col_chunk * chunk_cols
                    with local_dfb.reserve() as local_blk:
                        ttl.copy(
                            source[
                                m_begin : m_begin + chunk_rows,
                                k_begin : k_begin + chunk_cols,
                            ],
                            local_blk,
                        ).wait()
                    with local_dfb.wait() as local_blk:

                        def send_seed(pipe):
                            ttl.copy(local_blk, pipe).wait()

                        if direction == 0:
                            forward_seed_net.if_src(send_seed)
                        else:
                            backward_seed_net.if_src(send_seed)

        # Only this thread accesses `destination`: every read reuses the slice
        # of the receive that wrote it, which keeps the ownership proof local.
        @ttl.datamovement()
        def relay():
            lane, direction = ttl.node(dims=2)
            for source_distance in range(1, device_count):
                slot_begin = (source_distance - 1) * k_shard_tiles
                for m_chunk in range(m_chunks_per_lane):
                    m_begin = (m_chunk * lanes + lane) * chunk_rows
                    for col_chunk in range(col_chunks):
                        k_begin = (
                            slot_begin
                            + (1 - direction) * half_tiles
                            + col_chunk * chunk_cols
                        )
                        region = destination[
                            m_begin : m_begin + chunk_rows,
                            k_begin : k_begin + chunk_cols,
                        ]

                        def receive_chunk(pipe):
                            ttl.copy(
                                pipe, region, shape=(chunk_rows, chunk_cols)
                            ).wait()

                        if direction == 0:
                            if source_distance == 1:
                                forward_seed_net.if_dst(receive_chunk)
                            else:
                                forward_relay_net.if_dst(receive_chunk)
                        else:
                            if source_distance == 1:
                                backward_seed_net.if_dst(receive_chunk)
                            else:
                                backward_relay_net.if_dst(receive_chunk)
                        if source_distance < device_count - 1:
                            with relay_dfb.reserve() as relay_blk:
                                ttl.copy(region, relay_blk).wait()
                            with relay_dfb.wait() as relay_blk:

                                def send_relay(pipe):
                                    ttl.copy(relay_blk, pipe).wait()

                                if direction == 0:
                                    forward_relay_net.if_src(send_relay)
                                else:
                                    backward_relay_net.if_src(send_relay)

    return ring_all_gather


def expected_destination(
    full: torch.Tensor, device: int, device_count: int, k_shard: int
) -> torch.Tensor:
    half = k_shard // 2
    expected = torch.empty(
        full.shape[0], (device_count - 1) * k_shard, dtype=full.dtype
    )
    for source_distance in range(1, device_count):
        slot_begin = (source_distance - 1) * k_shard
        left_source = (device + source_distance) % device_count
        right_source = (device - source_distance) % device_count
        expected[:, slot_begin : slot_begin + half] = full[
            :, left_source * k_shard : left_source * k_shard + half
        ]
        expected[:, slot_begin + half : slot_begin + k_shard] = full[
            :, right_source * k_shard + half : (right_source + 1) * k_shard
        ]
    return expected


# Discovered mesh shapes whose reshaped (N, 1) line has linked ends, so the
# line is a ring and every hop, including the wrap, is one fabric hop.
RING_MESH_SHAPES = ((2, 2), (2, 4), (4, 2))


class UnsupportedRingMesh(ValueError):
    """The discovered mesh shape is not in RING_MESH_SHAPES."""


@contextmanager
def open_ring_mesh(fabric_config=ttnn.FabricConfig.FABRIC_2D):
    """Open every discovered device as an (N, 1) mesh whose line is a ring.

    Raises UnsupportedRingMesh for a discovered shape outside RING_MESH_SHAPES.
    """
    discovered_shape = tuple(get_fabric_mesh_shape(fabric_config=fabric_config))
    if discovered_shape not in RING_MESH_SHAPES:
        supported = ", ".join("x".join(map(str, shape)) for shape in RING_MESH_SHAPES)
        raise UnsupportedRingMesh(
            f"the ring all-gather needs a mesh of shape {supported}; "
            f"found {discovered_shape}"
        )
    with open_fabric_mesh(fabric_config=fabric_config) as mesh_device:
        mesh_device.reshape(ttnn.MeshShape((prod(discovered_shape), 1)))
        yield mesh_device


def make_ring_all_gather_workload(
    mesh_device,
    full: torch.Tensor,
    *,
    m_tiles: int,
    k_shard_tiles: int,
    lanes: int,
    chunk_shape: tuple[int, int],
) -> tuple[Callable[[], None], Callable[[], list[tuple[torch.Tensor, torch.Tensor]]]]:
    """Place `full`, sharded along K, on `mesh_device` and return a function
    that runs the ring all-gather and a function that returns each device's
    destination paired with its expected value."""
    device_count = mesh_device.get_num_devices()
    m = m_tiles * TILE_SIZE
    k_shard = k_shard_tiles * TILE_SIZE
    source = to_dram(
        full, mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1)
    )
    destination = to_dram(
        torch.zeros(device_count * m, (device_count - 1) * k_shard, dtype=full.dtype),
        mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    operation = make_ring_all_gather_operation(
        device_count,
        m_tiles=m_tiles,
        k_shard_tiles=k_shard_tiles,
        lanes=lanes,
        chunk_shape=chunk_shape,
    )

    def run():
        operation(source, destination)

    def destinations():
        result = ttnn.to_torch(
            destination, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )
        return [
            (
                result[device * m : (device + 1) * m],
                expected_destination(full, device, device_count, k_shard),
            )
            for device in range(device_count)
        ]

    return run, destinations


def run_ring_all_gather(
    mesh_device,
    torch_dtype: torch.dtype,
    *,
    m_tiles: int,
    k_shard_tiles: int,
    lanes: int,
    chunk_shape: tuple[int, int],
) -> None:
    full = torch.randn(
        m_tiles * TILE_SIZE,
        mesh_device.get_num_devices() * k_shard_tiles * TILE_SIZE,
        dtype=torch_dtype,
    )
    run, destinations = make_ring_all_gather_workload(
        mesh_device,
        full,
        m_tiles=m_tiles,
        k_shard_tiles=k_shard_tiles,
        lanes=lanes,
        chunk_shape=chunk_shape,
    )
    run()
    for actual, expected in destinations():
        assert_allclose(actual.float(), expected.float(), rtol=0, atol=0)


def main() -> None:
    with open_ring_mesh() as mesh_device:
        run_ring_all_gather(
            mesh_device,
            torch.bfloat16,
            m_tiles=16,
            k_shard_tiles=8,
            lanes=2,
            chunk_shape=(8, 2),
        )
    print("ring all-gather matches the expected destination on every device")


if __name__ == "__main__":
    main()
