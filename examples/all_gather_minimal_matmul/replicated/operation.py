# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Replicated output from K-sharded activations and replicated weights/bias.

Per device:
    gathered_activation[DRAM] = all_gather(activation_shard[DRAM])
    for output block:
        accumulator[L1] = broadcast(bias)
        for K block:
            activation[DRAM] -> row multicast -> activation block[L1]
            weight[DRAM] -> column multicast -> weight block[L1]
            accumulator += activation block @ weight block
        output[DRAM] = cast(accumulator)

The collective completes before matmul starts. Streaming receives directly into
the compute DFB; cached activation blocks use a separate multicast receive DFB.

Run on one, two, or four devices:
    python -m examples.all_gather_minimal_matmul.replicated --mesh-shape 1x1
    python -m examples.all_gather_minimal_matmul.replicated --mesh-shape 1x2
    python -m examples.all_gather_minimal_matmul.replicated --mesh-shape 2x2

The TT-Lang kernels are in make_replicated_matmul_operation below.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import gcd

import ttl
import ttnn

from ..operation import AllGatherMinimalMatmulConfig
from ..collectives import make_column_all_gather


@dataclass(frozen=True)
class ActivationAllGatherConfig:
    """Transfer geometry shared by the full operation and CCL measurements."""

    mesh_shape: tuple[int, ...]
    m_tiles: int
    n_tiles_per_device: int
    worker_count: int
    block_tiles: int
    m_block_tiles: int
    algorithm: str


@dataclass(frozen=True)
class ReplicatedAllGatherMatmul:
    activation_gather_config: ActivationAllGatherConfig | None
    activation_all_gather: Callable[..., None] | None
    matmul: Callable[..., None]

    @property
    def program_count(self) -> int:
        return 1 if self.activation_all_gather is None else 2

    def __call__(self, activation, weight, bias, output, gathered_activation=None):
        if self.activation_all_gather is not None:
            if gathered_activation is None:
                raise ValueError(
                    "multi-device replicated matmul requires gathered activation storage"
                )
            self.activation_all_gather(activation, gathered_activation)
            activation = gathered_activation
        self.matmul(activation, weight, bias, output)


def make_replicated_all_gather_matmul_operation(
    config: AllGatherMinimalMatmulConfig,
    *,
    math_fidelity: str | None = None,
    fp32_dest_acc_en: bool | None = None,
    all_gather_algorithm: str = "all_to_all",
) -> ReplicatedAllGatherMatmul:
    gather_config = None
    collective = None
    if config.device_count > 1:
        worker_count = gcd(2, config.padded_m_tiles)
        gather_config = ActivationAllGatherConfig(
            mesh_shape=config.mesh_shape,
            m_tiles=config.padded_m_tiles,
            n_tiles_per_device=config.k_tiles_per_device,
            worker_count=worker_count,
            block_tiles=gcd(40, config.k_tiles_per_device),
            m_block_tiles=gcd(2, config.padded_m_tiles // worker_count),
            algorithm=all_gather_algorithm,
        )
        collective = make_column_all_gather(**asdict(gather_config))
    return ReplicatedAllGatherMatmul(
        gather_config,
        collective,
        make_replicated_matmul_operation(
            config, math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_dest_acc_en
        ),
    )


def make_replicated_matmul_operation(
    config: AllGatherMinimalMatmulConfig,
    *,
    math_fidelity=None,
    fp32_dest_acc_en=None,
):
    device_domain = ttl.DeviceDomain(config.mesh_shape)
    m_workers, n_workers = config.m_workers, config.n_workers
    m_tiles, k_tiles, n_tiles = (
        config.m_block_tiles,
        config.k_block_tiles,
        config.n_block_tiles,
    )
    m_rounds = config.padded_m_tiles // (m_tiles * m_workers)
    n_rounds = config.n_tiles_per_device // (n_tiles * n_workers)
    k_rounds = config.device_count * config.k_tiles_per_device // k_tiles
    logical_m_tiles = config.m_tiles
    m_axis = 0 if config.transpose else 1
    n_axis = 1 if config.transpose else 0
    activation_rounds = 1 if config.reuse_activation else n_rounds
    activation_capacity = config.activation_block_count
    reuse_activation = config.reuse_activation

    def coordinates(n_worker, m_worker):
        return (m_worker, n_worker) if config.transpose else (n_worker, m_worker)

    activation_rows = ttl.PipeNet(
        [
            ttl.Pipe(
                src=coordinates(0, m_worker),
                dst=coordinates(slice(1, n_workers), m_worker),
            )
            for m_worker in range(m_workers)
        ]
    )
    weight_columns = ttl.PipeNet(
        [
            ttl.Pipe(
                src=coordinates(n_worker, 0),
                dst=coordinates(n_worker, slice(1, m_workers)),
            )
            for n_worker in range(n_workers)
        ]
    )

    @ttl.operation(
        grid=config.grid,
        device_domain=device_domain,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    def replicated_matmul(activation, weight, bias, output):
        activation_dfb = ttl.make_dataflow_buffer_like(
            activation, shape=(m_tiles, k_tiles), block_count=activation_capacity
        )
        received_activation_dfb = (
            ttl.make_dataflow_buffer_like(
                activation, shape=(m_tiles, k_tiles), block_count=1
            )
            if reuse_activation
            else activation_dfb
        )
        activation_bytes = (
            m_tiles * k_tiles * activation.get_tile().get_tile_size(activation.dtype)
        )
        weight_dfb = ttl.make_dataflow_buffer_like(
            weight, shape=(k_tiles, n_tiles), block_count=2
        )
        bias_dfb = ttl.make_dataflow_buffer_like(
            bias, shape=(1, n_tiles), block_count=2
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(m_tiles, n_tiles), block_count=2
        )
        accumulation_dtype = ttnn.float32 if fp32_dest_acc_en else output.dtype
        accumulator_dfb = ttl.make_dfb(
            accumulation_dtype, shape=(m_tiles, n_tiles), block_count=1
        )
        converted_bias_dfb = ttl.make_dfb(
            accumulation_dtype, shape=(1, n_tiles), block_count=1
        )

        @ttl.datamovement()
        def read_activation_and_write_output():
            physical_column, physical_row = ttl.node(dims=2)
            m_worker = physical_column * (1 - m_axis) + physical_row * m_axis
            n_worker = physical_column * (1 - n_axis) + physical_row * n_axis
            for m_round in range(m_rounds):
                m_begin = (m_round * m_workers + m_worker) * m_tiles
                for n_round in range(n_rounds):
                    for k_round in range(k_rounds):
                        if n_round < activation_rounds:
                            activation_block = activation_dfb.reserve()
                            if n_worker == 0:
                                k_begin = k_round * k_tiles
                                ttl.copy(
                                    activation[
                                        m_begin : m_begin + m_tiles,
                                        k_begin : k_begin + k_tiles,
                                    ],
                                    activation_block,
                                ).wait()

                                def broadcast_activation(pipe):
                                    ttl.copy(activation_block, pipe).wait()

                                activation_rows.if_src(broadcast_activation)
                            else:

                                def receive_activation(pipe):
                                    if reuse_activation:
                                        received_block = (
                                            received_activation_dfb.reserve()
                                        )
                                        ttl.copy(pipe, received_block).wait()
                                        received_block = received_activation_dfb.wait()
                                        ttl.copy(
                                            received_block,
                                            activation_block,
                                            byte_count=activation_bytes,
                                        ).wait()
                                    else:
                                        ttl.copy(pipe, activation_block).wait()

                                activation_rows.if_dst(receive_activation)
                        else:
                            activation_block = activation_dfb.reserve()
                    n_begin = (n_round * n_workers + n_worker) * n_tiles
                    output_block = output_dfb.wait()
                    if m_begin < logical_m_tiles:
                        ttl.copy(
                            output_block,
                            output[
                                m_begin : m_begin + m_tiles, n_begin : n_begin + n_tiles
                            ],
                        ).wait()

        @ttl.datamovement()
        def read_and_distribute_weights():
            physical_column, physical_row = ttl.node(dims=2)
            m_worker = physical_column * (1 - m_axis) + physical_row * m_axis
            n_worker = physical_column * (1 - n_axis) + physical_row * n_axis
            for m_round in range(m_rounds):
                for n_round in range(n_rounds):
                    n_begin = (n_round * n_workers + n_worker) * n_tiles
                    bias_block = bias_dfb.reserve()
                    ttl.copy(bias[0:1, n_begin : n_begin + n_tiles], bias_block).wait()
                    for k_round in range(k_rounds):
                        weight_block = weight_dfb.reserve()
                        if m_worker == 0:
                            k_begin = k_round * k_tiles
                            ttl.copy(
                                weight[
                                    k_begin : k_begin + k_tiles,
                                    n_begin : n_begin + n_tiles,
                                ],
                                weight_block,
                            ).wait()

                            def broadcast_weight(pipe):
                                ttl.copy(weight_block, pipe).wait()

                            weight_columns.if_src(broadcast_weight)
                        else:

                            def receive_weight(pipe):
                                ttl.copy(pipe, weight_block).wait()

                            weight_columns.if_dst(receive_weight)

        @ttl.compute()
        def compute_matmul_and_bias():
            for m_round in range(m_rounds):
                for n_round in range(n_rounds):
                    bias_block = bias_dfb.wait()
                    converted_bias = converted_bias_dfb.reserve()
                    converted_bias.store(
                        ttl.math.typecast(bias_block, converted_bias.dtype)
                    )
                    converted_bias = converted_bias_dfb.wait()
                    accumulator = accumulator_dfb.reserve()
                    if k_rounds == 1:
                        activation_block = activation_dfb.wait()
                        weight_block = weight_dfb.wait()
                        accumulator.store(
                            ttl.math.typecast(
                                activation_block @ weight_block, accumulator.dtype
                            )
                            + ttl.block.broadcast(
                                converted_bias, dims=[0], shape=(m_tiles, n_tiles)
                            )
                        )
                    else:
                        accumulator.store(
                            ttl.block.broadcast(
                                converted_bias, dims=[0], shape=(m_tiles, n_tiles)
                            )
                        )
                        for k_round in range(k_rounds):
                            activation_block = activation_dfb.wait()
                            weight_block = weight_dfb.wait()
                            accumulator += ttl.math.typecast(
                                activation_block @ weight_block, accumulator.dtype
                            )
                    accumulator = accumulator_dfb.wait()
                    output_block = output_dfb.reserve()
                    output_block.store(
                        ttl.math.typecast(accumulator, output_block.dtype)
                    )

    return replicated_matmul
