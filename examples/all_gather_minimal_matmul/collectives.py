# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TILE/DRAM column all-gather, preserving device-order shards exactly."""

from math import prod
from itertools import product

import ttl


def make_all_gather_graph(device_domain, mesh_shape, algorithm):
    """Select direct peer transfers or forwarding to the next device in index order."""
    if algorithm not in ("all_to_all", "ring"):
        raise ValueError(f"unsupported all-gather algorithm: {algorithm}")
    device_count = prod(mesh_shape)
    if algorithm == "all_to_all" or device_count == 1:
        return ttl.TransferGraph.all_to_all(device_domain)
    coordinates = tuple(product(*(range(extent) for extent in mesh_shape)))
    return ttl.TransferGraph.edges(
        device_domain,
        edges=tuple(
            (coordinates[source_index], coordinates[(source_index + 1) % device_count])
            for source_index in range(device_count)
        ),
    )


def make_column_all_gather(
    mesh_shape: tuple[int, ...],
    *,
    m_tiles: int,
    n_tiles_per_device: int,
    worker_count: int,
    block_tiles: int = 1,
    m_block_tiles: int = 1,
    algorithm: str = "all_to_all",
):
    """Copy each M x N/D shard into every device's M x N output.

    Workers partition rows; each fabric message contains a rectangular tile block.
    No arithmetic is applied, including to FP32 payloads.
    """
    if not mesh_shape or any(extent <= 0 for extent in mesh_shape):
        raise ValueError("mesh extents must be positive")
    if min(m_tiles, n_tiles_per_device, worker_count, block_tiles, m_block_tiles) <= 0:
        raise ValueError("tile counts and worker_count must be positive")
    if m_tiles % (worker_count * m_block_tiles):
        raise ValueError("m_tiles must be divisible by worker_count * m_block_tiles")
    if n_tiles_per_device % block_tiles:
        raise ValueError("n_tiles_per_device must be divisible by block_tiles")
    device_domain = ttl.DeviceDomain(mesh_shape)
    device_count = prod(mesh_shape)
    ring = algorithm == "ring" and device_count > 1
    direct = algorithm == "all_to_all" and device_count > 1
    graph = make_all_gather_graph(device_domain, mesh_shape, algorithm)
    output_all_gather_net = ttl.PipeNet(graph=graph)
    row_rounds = m_tiles // (worker_count * m_block_tiles)
    column_rounds = n_tiles_per_device // block_tiles

    @ttl.operation(grid=(worker_count, 1), device_domain=device_domain)
    def gather_output(output_shard, replicated_output):
        block_bytes = (
            m_block_tiles
            * block_tiles
            * output_shard.get_tile().get_tile_size(output_shard.dtype)
        )
        send_dfb = ttl.make_dataflow_buffer_like(
            output_shard, shape=(m_block_tiles, block_tiles), block_count=1
        )
        receive_dfb = ttl.make_dataflow_buffer_like(
            output_shard, shape=(m_block_tiles, block_tiles), block_count=1
        )
        local_dfb = ttl.make_dataflow_buffer_like(
            output_shard, shape=(m_block_tiles, block_tiles), block_count=1
        )

        # TTNN generic_op requires a compute kernel even for data movement.
        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def receive_output_shards():
            worker_index, _worker_row = ttl.node(dims=2)
            local_device_index = device_domain.current_index()
            for row_round in range(row_rounds):
                row_index = (row_round * worker_count + worker_index) * m_block_tiles
                for column_round in range(column_rounds):
                    column_begin = column_round * block_tiles
                    local_column = (
                        local_device_index * n_tiles_per_device + column_begin
                    )
                    local_block = local_dfb.reserve()
                    ttl.copy(
                        output_shard[
                            row_index : row_index + m_block_tiles,
                            column_begin : column_begin + block_tiles,
                        ],
                        local_block,
                    ).wait()
                    local_block = local_dfb.wait()
                    if ring:
                        initial_send_block = send_dfb.reserve()
                        ttl.copy(
                            local_block, initial_send_block, byte_count=block_bytes
                        ).wait()
                    ttl.copy(
                        local_block,
                        replicated_output[
                            row_index : row_index + m_block_tiles,
                            local_column : local_column + block_tiles,
                        ],
                    ).wait()

                    def receive_from_device(pipe):
                        remote_column = (
                            pipe.source_device_index * n_tiles_per_device + column_begin
                        )
                        receive_block = receive_dfb.reserve()
                        ttl.copy(pipe, receive_block).wait()
                        receive_block = receive_dfb.wait()
                        ttl.copy(
                            receive_block,
                            replicated_output[
                                row_index : row_index + m_block_tiles,
                                remote_column : remote_column + block_tiles,
                            ],
                        ).wait()

                    def receive_from_previous_device(pipe):
                        for ring_round in range(device_count - 1):
                            source_index = (
                                local_device_index + device_count - ring_round - 1
                            ) % device_count
                            remote_column = (
                                source_index * n_tiles_per_device + column_begin
                            )
                            receive_block = receive_dfb.reserve()
                            ttl.copy(pipe, receive_block).wait()
                            receive_block = receive_dfb.wait()
                            ttl.copy(
                                receive_block,
                                replicated_output[
                                    row_index : row_index + m_block_tiles,
                                    remote_column : remote_column + block_tiles,
                                ],
                            ).wait()
                            if ring_round < device_count - 2:
                                relay_send_block = send_dfb.reserve()
                                ttl.copy(
                                    receive_block,
                                    relay_send_block,
                                    byte_count=block_bytes,
                                ).wait()

                    if direct:
                        output_all_gather_net.if_dst(receive_from_device)
                    if ring:
                        output_all_gather_net.if_dst(receive_from_previous_device)

        @ttl.datamovement()
        def send_output_shards():
            worker_index, _worker_row = ttl.node(dims=2)
            for row_round in range(row_rounds):
                row_index = (row_round * worker_count + worker_index) * m_block_tiles
                for column_round in range(column_rounds):
                    column_begin = column_round * block_tiles
                    if direct:
                        send_block = send_dfb.reserve()
                        ttl.copy(
                            output_shard[
                                row_index : row_index + m_block_tiles,
                                column_begin : column_begin + block_tiles,
                            ],
                            send_block,
                        ).wait()
                        send_block = send_dfb.wait()

                        def send_to_device(pipe):
                            ttl.copy(send_block, pipe).wait()

                        output_all_gather_net.if_src(send_to_device)
                    if ring:

                        def forward_to_next_device(pipe):
                            for ring_round in range(device_count - 1):
                                forward_block = send_dfb.wait()
                                ttl.copy(forward_block, pipe).wait()

                        output_all_gather_net.if_src(forward_to_next_device)

    return gather_output
