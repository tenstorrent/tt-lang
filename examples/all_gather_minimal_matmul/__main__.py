# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run the all-gather minimal matmul example on a fabric mesh."""

from __future__ import annotations

import argparse
from math import prod

import torch
import ttnn

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh, to_dram
from utils.correctness import assert_allclose, assert_pcc

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
    parser.add_argument("--m-block-tiles", type=int, default=1)
    parser.add_argument("--k-tiles-per-transfer", type=int, default=1)
    parser.add_argument("--n-block-tiles", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    arguments = _make_parser().parse_args()
    mesh_shape = arguments.mesh_shape or get_fabric_mesh_shape(
        fabric_config=ttnn.FabricConfig.FABRIC_2D
    )
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=mesh_shape,
        m_tiles=arguments.m_tiles,
        k_tiles_per_device=arguments.k_tiles_per_device,
        n_tiles_per_device=arguments.n_tiles_per_device,
        m_block_tiles=arguments.m_block_tiles,
        k_tiles_per_transfer=arguments.k_tiles_per_transfer,
        n_block_tiles=arguments.n_block_tiles,
    )
    operation = make_all_gather_minimal_matmul_operation(config)

    torch.manual_seed(arguments.seed)
    torch_dtype = torch.bfloat16 if arguments.dtype == "bf16" else torch.float32
    m_elements = config.m_tiles * TILE_SIZE
    k_elements = config.device_count * config.k_tiles_per_device * TILE_SIZE
    n_elements = config.device_count * config.n_tiles_per_device * TILE_SIZE

    activation_torch = torch.randn((m_elements, k_elements), dtype=torch_dtype)
    weight_torch = torch.randn((k_elements, n_elements), dtype=torch_dtype)
    weight_torch /= float(k_elements)
    bias_torch = torch.zeros((1, n_elements), dtype=torch_dtype)
    if not arguments.no_bias:
        bias_torch = torch.randn((1, n_elements), dtype=torch_dtype)

    with open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ) as mesh_device:
        activation_shard = to_dram(
            activation_torch,
            mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        weight_shard = to_dram(
            weight_torch,
            mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        bias_shard = to_dram(
            bias_torch,
            mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        gathered_activation = to_dram(
            torch.zeros_like(activation_torch),
            mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        output_shard = to_dram(
            torch.zeros((m_elements, n_elements), dtype=torch_dtype),
            mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )

        operation(
            activation_shard,
            weight_shard,
            bias_shard,
            gathered_activation,
            output_shard,
        )

        gathered_result = ttnn.to_torch(
            gathered_activation,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        output_result = ttnn.to_torch(
            output_shard,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
        )

    expected_gather = activation_torch.repeat(prod(mesh_shape), 1)
    assert_allclose(
        gathered_result.float(), expected_gather.float(), rtol=0.0, atol=0.0
    )
    expected_output = activation_torch.float() @ weight_torch.float()
    expected_output += bias_torch.float()
    threshold = 0.99 if torch_dtype == torch.bfloat16 else 0.999
    assert_pcc(expected_output, output_result.float(), threshold=threshold)


if __name__ == "__main__":
    main()
