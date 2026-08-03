# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration, tensors, and validation for the full variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .common import (
    ExampleConfig,
    HostTensors,
    MeshTensors,
    assert_gathered_activation,
    assert_pcc,
    config_from_args,
    from_torch,
    make_host_tensors,
    make_mesh_tensors,
)


@dataclass(frozen=True)
class FullHostTensors(HostTensors):
    bias: torch.Tensor


@dataclass(frozen=True)
class FullMeshTensors(MeshTensors):
    bias_shard: ttnn.Tensor


def full_config_from_args(args) -> ExampleConfig:
    return config_from_args(args)


def make_full_host_tensors(config: ExampleConfig) -> FullHostTensors:
    base_tensors = make_host_tensors(config)
    bias = torch.randn((1, config.n_dim), dtype=torch.bfloat16)
    return FullHostTensors(
        activation=base_tensors.activation,
        weight=base_tensors.weight,
        bias=bias,
    )


def make_full_mesh_tensors(
    config: ExampleConfig,
    mesh_device,
    host_tensors: FullHostTensors,
) -> FullMeshTensors:
    base_tensors = make_mesh_tensors(config, mesh_device, host_tensors)
    bias_shard = from_torch(
        host_tensors.bias,
        mesh_device,
        ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    return FullMeshTensors(
        activation_shard=base_tensors.activation_shard,
        weight_shard=base_tensors.weight_shard,
        gathered_activation=base_tensors.gathered_activation,
        output_shard=base_tensors.output_shard,
        bias_shard=bias_shard,
    )


def validate_full_results(
    config: ExampleConfig,
    mesh_device,
    host_tensors: FullHostTensors,
    mesh_tensors: FullMeshTensors,
) -> None:
    gathered_result = ttnn.to_torch(
        mesh_tensors.gathered_activation,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    assert_gathered_activation(host_tensors.activation, gathered_result, config)

    output_result = ttnn.to_torch(
        mesh_tensors.output_shard,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
    )
    expected_output = host_tensors.activation.float() @ host_tensors.weight.float()
    expected_output += host_tensors.bias.float()
    assert_pcc(expected_output, output_result.float(), threshold=config.pcc_threshold)


def print_full_config(label: str, config: ExampleConfig) -> None:
    print(
        f"{label}: "
        f"mesh={config.mesh_shape} devices={config.num_devices} "
        f"M={config.m_dim} K={config.k_dim} N={config.n_dim} "
        f"K_transfer_tiles={config.k_tiles_per_transfer} "
        "features=bias "
        f"fabric_reliability={config.fabric_reliability_mode}"
    )
