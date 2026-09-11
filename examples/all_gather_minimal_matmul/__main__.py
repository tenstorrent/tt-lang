# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run column-parallel all-gather matmul on a fabric-connected device line."""

import argparse
from contextlib import contextmanager
from math import prod

import torch
import ttnn

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh, to_dram
from utils.correctness import assert_allclose, assert_pcc

from .config import AllGatherMinimalMatmulConfig
from .operation import make_all_gather_minimal_matmul_operation


def parse_mesh_shape(value: str) -> tuple[int, int]:
    try:
        mesh_shape = tuple(int(extent) for extent in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("mesh shape must be ROWSxCOLS") from error
    if len(mesh_shape) != 2 or min(mesh_shape) < 1 or prod(mesh_shape) < 2:
        raise argparse.ArgumentTypeError(
            "mesh shape must describe at least two devices, such as 4x1"
        )
    return mesh_shape


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-shape", type=parse_mesh_shape, required=True)
    parser.add_argument("--compute-grid", type=positive_int, nargs=2, default=(5, 4))
    parser.add_argument("--communication-workers", type=positive_int, default=4)
    parser.add_argument("--m-tiles", type=positive_int, default=8)
    parser.add_argument("--k-tiles-per-device", type=positive_int, default=4)
    parser.add_argument("--n-tiles-per-device", type=positive_int, default=4)
    parser.add_argument("--n-tiles", type=positive_int)
    parser.add_argument("--m-block-tiles", type=positive_int, default=2)
    parser.add_argument("--k-block-tiles", type=positive_int, default=2)
    parser.add_argument("--n-block-tiles", type=positive_int, default=1)
    parser.add_argument(
        "--reuse-activation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@contextmanager
def open_participant_mesh(mesh_shape: tuple[int, int]):
    discovered_shape = get_fabric_mesh_shape(
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
        reliability_mode=ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    if prod(mesh_shape) > prod(discovered_shape):
        raise ValueError(
            f"requested mesh {mesh_shape} exceeds discovered mesh {discovered_shape}"
        )
    parent_shape = tuple(discovered_shape)
    if any(
        requested > available
        for requested, available in zip(mesh_shape, discovered_shape, strict=True)
    ):
        parent_shape = (
            (prod(discovered_shape), 1)
            if mesh_shape[0] > 1
            else (1, prod(discovered_shape))
        )
    with open_fabric_mesh(
        requested_mesh_shape=discovered_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
        reliability_mode=ttnn.FabricReliabilityMode.STRICT_INIT,
    ) as parent_mesh:
        if parent_shape != tuple(discovered_shape):
            parent_mesh.reshape(ttnn.MeshShape(parent_shape))
        if mesh_shape == parent_shape:
            yield parent_mesh
            return
        participant_mesh = parent_mesh.create_submesh(ttnn.MeshShape(mesh_shape))
        try:
            yield participant_mesh
        finally:
            ttnn.close_mesh_device(participant_mesh)


def main() -> None:
    arguments = parse_args()
    device_count = prod(arguments.mesh_shape)
    n_tiles_per_device = arguments.n_tiles_per_device
    if arguments.n_tiles is not None:
        if arguments.n_tiles % device_count:
            raise ValueError("complete N tile count must be divisible by device count")
        n_tiles_per_device = arguments.n_tiles // device_count

    config = AllGatherMinimalMatmulConfig(
        mesh_shape=arguments.mesh_shape,
        m_tiles=arguments.m_tiles,
        k_tiles_per_device=arguments.k_tiles_per_device,
        n_tiles_per_device=n_tiles_per_device,
        compute_grid=arguments.compute_grid,
        m_block_tiles=arguments.m_block_tiles,
        k_block_tiles=arguments.k_block_tiles,
        n_block_tiles=arguments.n_block_tiles,
        reuse_activation=arguments.reuse_activation,
    )
    operation = make_all_gather_minimal_matmul_operation(
        config,
        math_fidelity="HiFi2" if arguments.dtype == "bf16" else "HiFi4",
        fp32_dest_acc_en=True,
        communication_worker_count=arguments.communication_workers,
    )

    torch.manual_seed(arguments.seed)
    torch_dtype = torch.bfloat16 if arguments.dtype == "bf16" else torch.float32
    m_elements = config.m_tiles * 32
    k_elements = config.device_count * config.k_tiles_per_device * 32
    n_elements = config.device_count * config.n_tiles_per_device * 32
    activation = torch.randn((m_elements, k_elements), dtype=torch_dtype)
    weight = torch.randn((k_elements, n_elements), dtype=torch_dtype) / k_elements**0.5
    bias = torch.randn((1, n_elements), dtype=torch_dtype) * 0.1
    expected = activation.float() @ weight.float() + bias.float()

    with open_participant_mesh(arguments.mesh_shape) as mesh:
        n_shard_mapper = ttnn.ShardTensorToMesh(mesh, dim=1)
        activation_storage = torch.nn.functional.pad(
            activation, (0, 0, 0, config.padded_m_tiles * 32 - m_elements)
        )
        activation_device = to_dram(
            activation_storage, mesh, mesh_mapper=n_shard_mapper
        )
        weight_device = to_dram(weight, mesh, mesh_mapper=n_shard_mapper)
        bias_device = to_dram(bias, mesh, mesh_mapper=n_shard_mapper)
        output_device = to_dram(
            torch.zeros((m_elements, n_elements), dtype=torch_dtype),
            mesh,
            mesh_mapper=n_shard_mapper,
        )
        operation(activation_device, weight_device, bias_device, output_device)
        output = ttnn.to_torch(
            output_device, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)
        ).float()

    assert_pcc(expected, output, threshold=0.99 if arguments.dtype == "bf16" else 0.999)
    tolerance = 0.05 if arguments.dtype == "bf16" else 0.005
    assert_allclose(output, expected, rtol=tolerance, atol=tolerance)


if __name__ == "__main__":
    main()
