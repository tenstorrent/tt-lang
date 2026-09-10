# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run the all-gather minimal matmul example on a fabric mesh."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from math import prod

import torch
import ttnn

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh, to_dram
from utils.correctness import assert_pcc

from .collectives import make_column_all_gather
from .operation import (
    AllGatherMinimalMatmulConfig,
    make_all_gather_minimal_matmul_operation,
)

TILE_SIZE = 32


def _parse_mesh_shape(value: str) -> tuple[int, ...]:
    try:
        mesh_shape = tuple(int(extent) for extent in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mesh shape must contain integer extents separated by x"
        ) from error
    if not mesh_shape or any(extent <= 0 for extent in mesh_shape):
        raise argparse.ArgumentTypeError("mesh shape extents must be positive")
    return mesh_shape


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-shape", type=_parse_mesh_shape)
    parser.add_argument("--m-tiles", type=int, default=2)
    parser.add_argument("--k-tiles-per-device", type=int, default=1)
    parser.add_argument("--n-tiles-per-device", type=int, default=2)
    parser.add_argument("--n-tiles", type=int, help="complete output width in tiles")
    parser.add_argument("--m-block-tiles", type=int, default=1)
    parser.add_argument("--k-block-tiles", type=int, default=1)
    parser.add_argument("--n-block-tiles", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--gather-output", action="store_true")
    parser.add_argument(
        "--activation-all-gather", choices=("all_to_all", "ring"), default="all_to_all"
    )
    parser.add_argument(
        "--output-all-gather", choices=("all_to_all", "ring"), default="all_to_all"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _resolve_mesh_shapes(
    requested_mesh_shape: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if requested_mesh_shape is not None and prod(requested_mesh_shape) == 1:
        return requested_mesh_shape, requested_mesh_shape

    discovered_mesh_shape = get_fabric_mesh_shape(
        fabric_config=ttnn.FabricConfig.FABRIC_2D
    )
    participant_mesh_shape = requested_mesh_shape or discovered_mesh_shape
    if len(participant_mesh_shape) != len(discovered_mesh_shape) or any(
        requested > available
        for requested, available in zip(
            participant_mesh_shape, discovered_mesh_shape, strict=True
        )
    ):
        raise ValueError(
            f"requested mesh {participant_mesh_shape} does not fit discovered mesh "
            f"{discovered_mesh_shape}"
        )
    return participant_mesh_shape, discovered_mesh_shape


@contextmanager
def _open_operation_mesh(
    participant_mesh_shape: tuple[int, ...],
    parent_mesh_shape: tuple[int, ...],
):
    single_device = prod(participant_mesh_shape) == 1
    with open_fabric_mesh(
        requested_mesh_shape=parent_mesh_shape,
        fabric_config=(
            ttnn.FabricConfig.DISABLED if single_device else ttnn.FabricConfig.FABRIC_2D
        ),
        reliability_mode=(
            None if single_device else ttnn.FabricReliabilityMode.RELAXED_INIT
        ),
    ) as parent_mesh:
        if participant_mesh_shape == parent_mesh_shape:
            yield parent_mesh
            return

        participant_mesh = parent_mesh.create_submesh(
            ttnn.MeshShape(participant_mesh_shape)
        )
        try:
            yield participant_mesh
        finally:
            ttnn.close_mesh_device(participant_mesh)


def main(*, variant="n_sharded") -> None:
    arguments = _make_parser().parse_args()
    replicated = variant == "replicated"
    if variant not in ("n_sharded", "replicated"):
        raise ValueError(f"unknown variant: {variant}")
    if replicated and arguments.gather_output:
        raise ValueError("replicated weights already produce replicated output")
    mesh_shape, parent_mesh_shape = _resolve_mesh_shapes(arguments.mesh_shape)
    n_partitions = 1 if replicated else prod(mesh_shape)
    n_tiles_per_device = arguments.n_tiles_per_device
    if arguments.n_tiles is not None:
        if arguments.n_tiles <= 0 or arguments.n_tiles % n_partitions:
            raise ValueError(
                "complete N tile count must be positive and divisible by N partitions"
            )
        n_tiles_per_device = arguments.n_tiles // n_partitions
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=mesh_shape,
        m_tiles=arguments.m_tiles,
        k_tiles_per_device=arguments.k_tiles_per_device,
        n_tiles_per_device=n_tiles_per_device,
        m_block_tiles=arguments.m_block_tiles,
        k_block_tiles=arguments.k_block_tiles,
        n_block_tiles=arguments.n_block_tiles,
    )
    operation = make_all_gather_minimal_matmul_operation(
        config, all_gather_algorithm=arguments.activation_all_gather
    )

    torch.manual_seed(arguments.seed)
    torch_dtype = torch.bfloat16 if arguments.dtype == "bf16" else torch.float32
    m_elements = config.m_tiles * TILE_SIZE
    k_elements = config.device_count * config.k_tiles_per_device * TILE_SIZE
    n_elements = n_partitions * config.n_tiles_per_device * TILE_SIZE

    activation_torch = torch.randn((m_elements, k_elements), dtype=torch_dtype)
    weight_torch = torch.randn((k_elements, n_elements), dtype=torch_dtype)
    weight_torch /= float(k_elements)
    bias_torch = torch.zeros((1, n_elements), dtype=torch_dtype)
    if not arguments.no_bias:
        bias_torch = torch.randn((1, n_elements), dtype=torch_dtype)

    with _open_operation_mesh(mesh_shape, parent_mesh_shape) as mesh_device:
        output_mapper = (
            ttnn.ReplicateTensorToMesh(mesh_device)
            if replicated
            else ttnn.ShardTensorToMesh(mesh_device, dim=1)
        )
        activation_shard = to_dram(
            activation_torch,
            mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        weight_shard = to_dram(
            weight_torch,
            mesh_device,
            mesh_mapper=output_mapper,
        )
        bias_shard = to_dram(
            bias_torch,
            mesh_device,
            mesh_mapper=output_mapper,
        )
        output_shard = to_dram(
            torch.zeros((m_elements, n_elements), dtype=torch_dtype),
            mesh_device,
            mesh_mapper=output_mapper,
        )

        operation(
            activation_shard,
            weight_shard,
            bias_shard,
            output_shard,
        )

        if arguments.gather_output:
            replicated_output = to_dram(
                torch.zeros((m_elements, n_elements), dtype=torch_dtype),
                mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            gather_output = make_column_all_gather(
                mesh_shape,
                m_tiles=config.m_tiles,
                n_tiles_per_device=config.n_tiles_per_device,
                worker_count=config.m_workers,
                block_tiles=config.n_block_tiles,
                algorithm=arguments.output_all_gather,
            )
            gather_output(output_shard, replicated_output)

        output_result = ttnn.to_torch(
            replicated_output if arguments.gather_output else output_shard,
            mesh_composer=ttnn.ConcatMeshToTensor(
                mesh_device, dim=0 if arguments.gather_output or replicated else 1
            ),
        )

    expected_output = activation_torch.float() @ weight_torch.float()
    expected_output += bias_torch.float()
    threshold = 0.99 if torch_dtype == torch.bfloat16 else 0.999
    if arguments.gather_output or replicated:
        for device_output in output_result.split(m_elements, dim=0):
            assert_pcc(expected_output, device_output.float(), threshold=threshold)
    else:
        assert_pcc(expected_output, output_result.float(), threshold=threshold)


if __name__ == "__main__":
    main()
