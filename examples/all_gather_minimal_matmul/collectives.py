# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TILE/DRAM output replication, preserving device-order N shards exactly."""

from math import prod

import ttl


def make_output_all_gather(
    mesh_shape: tuple[int, ...],
    *,
    m_tiles: int,
    n_tiles_per_device: int,
    worker_count: int,
    block_tiles: int = 1,
):
    """Copy each M x N/D shard into every device's M x N output.

    Workers partition rows; each fabric message contains one row of tiles.
    No arithmetic is applied, including to FP32 payloads.
    """
    if not mesh_shape or any(extent <= 0 for extent in mesh_shape):
        raise ValueError("mesh extents must be positive")
    if min(m_tiles, n_tiles_per_device, worker_count, block_tiles) <= 0:
        raise ValueError("tile counts and worker_count must be positive")
    if m_tiles % worker_count:
        raise ValueError("m_tiles must be divisible by worker_count")
    if n_tiles_per_device % block_tiles:
        raise ValueError("n_tiles_per_device must be divisible by block_tiles")
    device_domain = ttl.DeviceDomain(mesh_shape)
    device_count = prod(mesh_shape)
    output_all_gather_net = ttl.PipeNet(
        graph=ttl.TransferGraph.all_to_all(device_domain)
    )
    row_rounds = m_tiles // worker_count
    column_rounds = n_tiles_per_device // block_tiles

    @ttl.operation(grid=(worker_count, 1), device_domain=device_domain)
    def gather_output(output_shard, replicated_output):
        send_dfb = ttl.make_dataflow_buffer_like(
            output_shard, shape=(1, block_tiles), block_count=2
        )
        receive_dfb = ttl.make_dataflow_buffer_like(
            output_shard, shape=(1, block_tiles), block_count=2
        )
        local_dfb = ttl.make_dataflow_buffer_like(
            output_shard, shape=(1, block_tiles), block_count=2
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
                row_index = row_round * worker_count + worker_index
                for column_round in range(column_rounds):
                    column_begin = column_round * block_tiles
                    local_column = (
                        local_device_index * n_tiles_per_device + column_begin
                    )
                    local_block = local_dfb.reserve()
                    ttl.copy(
                        output_shard[
                            row_index : row_index + 1,
                            column_begin : column_begin + block_tiles,
                        ],
                        local_block,
                    ).wait()
                    local_block = local_dfb.wait()
                    ttl.copy(
                        local_block,
                        replicated_output[
                            row_index : row_index + 1,
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
                                row_index : row_index + 1,
                                remote_column : remote_column + block_tiles,
                            ],
                        ).wait()

                    if device_count > 1:
                        output_all_gather_net.if_dst(receive_from_device)

        @ttl.datamovement()
        def send_output_shards():
            worker_index, _worker_row = ttl.node(dims=2)
            for row_round in range(row_rounds):
                row_index = row_round * worker_count + worker_index
                for column_round in range(column_rounds):
                    column_begin = column_round * block_tiles
                    if device_count > 1:
                        send_block = send_dfb.reserve()
                        ttl.copy(
                            output_shard[
                                row_index : row_index + 1,
                                column_begin : column_begin + block_tiles,
                            ],
                            send_block,
                        ).wait()
                        send_block = send_dfb.wait()

                        def send_to_device(pipe):
                            ttl.copy(send_block, pipe).wait()

                        output_all_gather_net.if_src(send_to_device)

    return gather_output
