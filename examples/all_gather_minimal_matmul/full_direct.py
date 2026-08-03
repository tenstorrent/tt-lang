# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_HARDWARE_CI: skip-compiler

"""Full direct-style fabric all-gather fused with a block matmul.

This mirrors the core data dependence of the tt-metal example under
`third-party/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/`
`all_gather_minimal_matmul_async`:

* activations are sharded across the K dimension;
* each device gathers the remote activation shards over fabric;
* weights are sharded across the output N dimension;
* each device computes and stores its local N shard;
* an optional row-broadcast bias is added to the local output shard.

This variant accumulates the optional features listed in the package README.
The smaller `direct.py` variant remains the stable baseline.
"""

import torch
import ttl
import ttnn

from .common import (
    ExampleConfig,
    make_arg_parser,
    num_devices,
    open_configured_mesh,
    require_device_count,
)
from .full_common import (
    full_config_from_args,
    make_full_host_tensors,
    make_full_mesh_tensors,
    print_full_config,
    validate_full_results,
)


def make_all_gather_minimal_matmul_operation(config: ExampleConfig):
    mesh_shape = config.mesh_shape
    m_tiles = config.m_tiles
    k_tiles_per_device = config.k_tiles_per_device
    n_tiles_per_device = config.n_tiles_per_device
    k_tiles_per_transfer = config.k_tiles_per_transfer
    device_count = num_devices(mesh_shape)
    num_k_transfers = k_tiles_per_device // k_tiles_per_transfer

    device_domain = ttl.DeviceDomain(mesh_shape)
    activation_all_gather_net = ttl.PipeNet(
        graph=ttl.TransferGraph.all_to_all(device_domain)
    )

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def all_gather_minimal_matmul(
        activation_shard: ttnn.Tensor,
        weight_shard: ttnn.Tensor,
        bias_shard: ttnn.Tensor,
        gathered_activation: ttnn.Tensor,
        output_shard: ttnn.Tensor,
    ) -> None:
        send_activation_dfb = ttl.make_dataflow_buffer_like(
            activation_shard, shape=(m_tiles, k_tiles_per_transfer), block_count=2
        )
        local_activation_dfb = ttl.make_dataflow_buffer_like(
            activation_shard, shape=(m_tiles, k_tiles_per_transfer), block_count=2
        )
        receive_activation_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_tiles, k_tiles_per_transfer),
            block_count=device_count,
        )
        activation_compute_dfb = ttl.make_dataflow_buffer_like(
            activation_shard, shape=(m_tiles, k_tiles_per_transfer), block_count=2
        )
        weight_dfb = ttl.make_dataflow_buffer_like(
            weight_shard,
            shape=(k_tiles_per_transfer, n_tiles_per_device),
            block_count=2,
        )
        bias_dfb = ttl.make_dataflow_buffer_like(
            bias_shard, shape=(1, n_tiles_per_device), block_count=2
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output_shard, shape=(m_tiles, n_tiles_per_device), block_count=2
        )

        @ttl.datamovement()
        def send_activation():
            def send_activation_tile(pipe):
                for k_transfer_index in range(num_k_transfers):
                    k_begin = k_transfer_index * k_tiles_per_transfer
                    k_end = k_begin + k_tiles_per_transfer
                    activation_block = send_activation_dfb.reserve()
                    ttl.copy(
                        activation_shard[0:m_tiles, k_begin:k_end],
                        activation_block,
                    ).wait()

                    activation_block = send_activation_dfb.wait()
                    ttl.copy(activation_block, pipe).wait()

            activation_all_gather_net.if_src(send_activation_tile)

            for source_index in range(device_count):
                source_begin = source_index * k_tiles_per_device
                for k_transfer_index in range(num_k_transfers):
                    k_begin = source_begin + k_transfer_index * k_tiles_per_transfer
                    k_end = k_begin + k_tiles_per_transfer
                    weight_block = weight_dfb.reserve()
                    ttl.copy(
                        weight_shard[k_begin:k_end, 0:n_tiles_per_device],
                        weight_block,
                    ).wait()

            bias_block = bias_dfb.reserve()
            ttl.copy(
                bias_shard[0:1, 0:n_tiles_per_device],
                bias_block,
            ).wait()

        @ttl.datamovement()
        def receive_activation():
            device_index = device_domain.current_index()
            device_begin = device_index * k_tiles_per_device
            for k_transfer_index in range(num_k_transfers):
                local_k_begin = k_transfer_index * k_tiles_per_transfer
                local_k_end = local_k_begin + k_tiles_per_transfer
                gather_k_begin = device_begin + local_k_begin
                gather_k_end = gather_k_begin + k_tiles_per_transfer
                activation_block = local_activation_dfb.reserve()
                ttl.copy(
                    activation_shard[0:m_tiles, local_k_begin:local_k_end],
                    activation_block,
                ).wait()

                activation_block = local_activation_dfb.wait()
                ttl.copy(
                    activation_block,
                    gathered_activation[0:m_tiles, gather_k_begin:gather_k_end],
                ).wait()

            def receive_activation_tile(pipe):
                source_index = pipe.source_device_index
                source_begin = source_index * k_tiles_per_device
                k_begin = source_begin + k_transfer_index * k_tiles_per_transfer
                k_end = k_begin + k_tiles_per_transfer
                activation_block = receive_activation_dfb.reserve()
                ttl.copy(pipe, activation_block).wait()

                activation_block = receive_activation_dfb.wait()
                ttl.copy(
                    activation_block,
                    gathered_activation[0:m_tiles, k_begin:k_end],
                ).wait()

            for k_transfer_index in range(num_k_transfers):
                activation_all_gather_net.if_dst(receive_activation_tile)

            for source_index in range(device_count):
                source_begin = source_index * k_tiles_per_device
                for k_transfer_index in range(num_k_transfers):
                    k_begin = source_begin + k_transfer_index * k_tiles_per_transfer
                    k_end = k_begin + k_tiles_per_transfer
                    activation_block = activation_compute_dfb.reserve()
                    ttl.copy(
                        gathered_activation[0:m_tiles, k_begin:k_end],
                        activation_block,
                    ).wait()

            output_block = output_dfb.wait()
            ttl.copy(output_block, output_shard[0:m_tiles, 0:n_tiles_per_device]).wait()

        @ttl.compute()
        def compute():
            output_block = output_dfb.reserve()
            accumulator = ttl.block.fill(0.0, shape=output_block.shape)
            for _chunk_index in range(device_count * num_k_transfers):
                activation_block = activation_compute_dfb.wait()
                weight_block = weight_dfb.wait()
                accumulator += activation_block @ weight_block
            bias_block = bias_dfb.wait()
            accumulator += ttl.block.broadcast(
                bias_block,
                dims=[0],
                shape=(m_tiles, n_tiles_per_device),
            )
            output_block.store(accumulator)

    return all_gather_minimal_matmul


def main() -> None:
    parser = make_arg_parser(__doc__ or "")
    config = full_config_from_args(parser.parse_args())
    torch.manual_seed(config.seed)
    require_device_count(config.mesh_shape)

    all_gather_minimal_matmul = make_all_gather_minimal_matmul_operation(config)
    print_full_config("all_gather_minimal_matmul_full_direct", config)

    with open_configured_mesh(config) as mesh_device:
        host_tensors = make_full_host_tensors(config)
        mesh_tensors = make_full_mesh_tensors(config, mesh_device, host_tensors)
        all_gather_minimal_matmul(
            mesh_tensors.activation_shard,
            mesh_tensors.weight_shard,
            mesh_tensors.bias_shard,
            mesh_tensors.gathered_activation,
            mesh_tensors.output_shard,
        )
        validate_full_results(config, mesh_device, host_tensors, mesh_tensors)


if __name__ == "__main__":
    main()
