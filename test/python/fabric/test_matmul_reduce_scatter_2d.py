# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Two-dimensional matmul reduce-scatter correctness tests."""

from math import prod

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from examples.all_gather_minimal_matmul.__main__ import open_participant_mesh
from examples.matmul_reduce_scatter_2d import (
    MatmulReduceScatter2DConfig,
    make_matmul_reduce_scatter_2d_operation,
)
from ttlang_test_utils import get_fabric_mesh_shape, to_dram
from utils.correctness import assert_allclose, assert_pcc

pytestmark = pytest.mark.multi_device


@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
@pytest.mark.parametrize("m_tiles", [16, 32], ids=["one-block", "two-block"])
def test_matmul_reduce_scatter_2d(torch_dtype, m_tiles):
    mesh_shape = (2, 2)
    discovered_shape = get_fabric_mesh_shape(
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
        reliability_mode=ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    if prod(mesh_shape) > prod(discovered_shape):
        pytest.skip(
            f"test requires mesh {mesh_shape}; discovered mesh is {discovered_shape}"
        )

    config = MatmulReduceScatter2DConfig(
        mesh_shape=mesh_shape,
        m_tiles=m_tiles,
        k_tiles_per_group=4,
        n_tiles_per_group=8,
        compute_grid=(4, 4),
        m_block_tiles=2,
        k_block_tiles=2,
        n_block_tiles=2,
    )
    torch.manual_seed(23)
    m_elements = config.m_tiles * 32
    k_elements = config.k_group_count * config.k_tiles_per_group * 32
    n_elements = config.n_group_count * config.n_tiles_per_group * 32
    activation = torch.randn((m_elements, k_elements), dtype=torch_dtype)
    weight = torch.randn((k_elements, n_elements), dtype=torch_dtype)
    weight /= k_elements**0.5
    bias = torch.randn((1, n_elements), dtype=torch_dtype) * 0.1
    expected = activation.float() @ weight.float() + bias.float()

    with open_participant_mesh(mesh_shape) as mesh:
        activation_mapper = ttnn.ShardTensor2dMesh(
            mesh, mesh_shape=mesh_shape, dims=(1, None)
        )
        weight_mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh_shape, dims=(0, 1))
        bias_mapper = ttnn.ShardTensor2dMesh(
            mesh, mesh_shape=mesh_shape, dims=(None, 1)
        )
        output_mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh_shape, dims=(0, 1))
        activation_device = to_dram(
            torch.nn.functional.pad(
                activation,
                (0, 0, 0, config.padded_m_tiles * 32 - m_elements),
            ),
            mesh,
            mesh_mapper=activation_mapper,
        )
        weight_device = to_dram(weight, mesh, mesh_mapper=weight_mapper)
        bias_device = to_dram(bias, mesh, mesh_mapper=bias_mapper)
        output_device = to_dram(
            torch.zeros((config.padded_m_tiles * 32, n_elements), dtype=torch_dtype),
            mesh,
            mesh_mapper=output_mapper,
        )
        operation = make_matmul_reduce_scatter_2d_operation(
            config,
            math_fidelity="HiFi2" if torch_dtype == torch.bfloat16 else "HiFi4",
            fp32_dest_acc_en=True,
        )

        operation(activation_device, weight_device, bias_device, output_device)
        actual = ttnn.to_torch(
            output_device,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh, mesh_shape=mesh_shape, dims=(0, 1)
            ),
        )[:m_elements, :n_elements].float()

    assert_pcc(
        expected,
        actual,
        threshold=0.99 if torch_dtype == torch.bfloat16 else 0.999,
    )
    tolerance = 0.05 if torch_dtype == torch.bfloat16 else 0.005
    assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
