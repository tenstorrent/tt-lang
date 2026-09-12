# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Column-parallel all-gather matmul correctness tests."""

import torch
import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from examples.all_gather_minimal_matmul import (
    AllGatherMinimalMatmulConfig,
    make_all_gather_minimal_matmul_operation,
)
from examples.all_gather_minimal_matmul.__main__ import open_participant_mesh
from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose, assert_pcc

pytestmark = pytest.mark.multi_device


def run_case(mesh, config, communication_workers, torch_dtype):
    torch.manual_seed(19)
    m_elements = config.m_tiles * 32
    k_elements = config.device_count * config.k_tiles_per_device * 32
    n_elements = config.device_count * config.n_tiles_per_device * 32
    activation = torch.randn((m_elements, k_elements), dtype=torch_dtype)
    weight = torch.randn((k_elements, n_elements), dtype=torch_dtype) / k_elements**0.5
    bias = torch.randn((1, n_elements), dtype=torch_dtype) * 0.1
    expected = activation.float() @ weight.float() + bias.float()
    shard_mapper = ttnn.ShardTensorToMesh(mesh, dim=1)
    activation_device = to_dram(
        torch.nn.functional.pad(
            activation, (0, 0, 0, config.padded_m_tiles * 32 - m_elements)
        ),
        mesh,
        mesh_mapper=shard_mapper,
    )
    weight_device = to_dram(weight, mesh, mesh_mapper=shard_mapper)
    bias_device = to_dram(bias, mesh, mesh_mapper=shard_mapper)
    output_device = to_dram(
        torch.zeros((m_elements, n_elements), dtype=torch_dtype),
        mesh,
        mesh_mapper=shard_mapper,
    )
    operation = make_all_gather_minimal_matmul_operation(
        config,
        math_fidelity="HiFi2" if torch_dtype == torch.bfloat16 else "HiFi4",
        fp32_dest_acc_en=True,
        communication_worker_count=communication_workers,
    )

    operation(activation_device, weight_device, bias_device, output_device)
    actual = ttnn.to_torch(
        output_device, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)
    ).float()
    assert_pcc(
        expected,
        actual,
        threshold=0.99 if torch_dtype == torch.bfloat16 else 0.999,
    )
    tolerance = 0.05 if torch_dtype == torch.bfloat16 else 0.005
    assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "mesh_shape,communication_workers",
    [((2, 1), 2), ((4, 1), 4)],
    ids=["two-devices", "four-devices"],
)
@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
@pytest.mark.parametrize("reuse_activation", [False, True], ids=["stream", "cache"])
def test_all_gather_minimal_matmul(
    mesh_shape, communication_workers, torch_dtype, reuse_activation
):
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=mesh_shape,
        m_tiles=10,
        k_tiles_per_device=4,
        n_tiles_per_device=8,
        compute_grid=(5, 4),
        m_block_tiles=2,
        k_block_tiles=2,
        n_block_tiles=1,
        reuse_activation=reuse_activation,
    )
    with open_participant_mesh(mesh_shape) as mesh:
        run_case(mesh, config, communication_workers, torch_dtype)


@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
def test_all_gather_minimal_matmul_partial_m_block(torch_dtype):
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=(2, 1),
        m_tiles=9,
        k_tiles_per_device=4,
        n_tiles_per_device=8,
        compute_grid=(5, 4),
        m_block_tiles=2,
        k_block_tiles=2,
        n_block_tiles=1,
        reuse_activation=False,
    )
    with open_participant_mesh(config.mesh_shape) as mesh:
        run_case(mesh, config, 2, torch_dtype)


def test_all_gather_minimal_matmul_full_grid():
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=(4, 1),
        m_tiles=24,
        k_tiles_per_device=4,
        n_tiles_per_device=20,
        compute_grid=(12, 10),
        m_block_tiles=2,
        k_block_tiles=2,
        n_block_tiles=1,
        reuse_activation=False,
    )
    with open_participant_mesh(config.mesh_shape) as mesh:
        run_case(mesh, config, 4, torch.bfloat16)
