# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware coverage for full-grid all-gather minimal matmul."""

from math import prod

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from examples.all_gather_minimal_matmul import (
    AllGatherMinimalMatmulConfig,
    make_all_gather_minimal_matmul_operation,
)
from ttlang_test_utils import (
    get_fabric_mesh_shape,
    open_fabric_mesh,
    requires_forwarding_link_indices,
    to_dram,
)
from utils.correctness import assert_allclose, assert_pcc

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32
MATMUL_DTYPES = [
    pytest.param(torch.bfloat16, 0.99, False, id="bf16-dst16"),
    pytest.param(torch.bfloat16, 0.99, True, id="bf16-dst32"),
    pytest.param(torch.float32, 0.999, True, id="fp32"),
]


@pytest.fixture(scope="module")
def fabric_mesh_shape() -> tuple[int, ...]:
    if ttnn.get_num_devices() < 2:
        pytest.skip("requires at least two devices")
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    if prod(mesh_shape) < 2:
        pytest.skip("requires a multi-device fabric mesh")
    return mesh_shape


@pytest.fixture(scope="module")
def participant_mesh_shape(fabric_mesh_shape) -> tuple[int, ...]:
    participant_axis = next(
        axis for axis, extent in enumerate(fabric_mesh_shape) if extent > 1
    )
    return tuple(
        2 if axis == participant_axis else 1 for axis in range(len(fabric_mesh_shape))
    )


@pytest.fixture(scope="module")
def participant_mesh(fabric_mesh_shape, participant_mesh_shape):
    with open_fabric_mesh(
        requested_mesh_shape=fabric_mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ) as parent_mesh:
        owns_participant_mesh = participant_mesh_shape != fabric_mesh_shape
        mesh_device = (
            parent_mesh.create_submesh(ttnn.MeshShape(participant_mesh_shape))
            if owns_participant_mesh
            else parent_mesh
        )
        try:
            yield mesh_device
        finally:
            if owns_participant_mesh:
                ttnn.close_mesh_device(mesh_device)


@requires_forwarding_link_indices(ttnn)
@pytest.mark.parametrize("torch_dtype,pcc_threshold,fp32_dest_acc_en", MATMUL_DTYPES)
@pytest.mark.parametrize(
    "k_tiles_per_device,k_block_tiles",
    [
        pytest.param(1, 1, id="one-transfer"),
        pytest.param(4, 1, id="four-transfers"),
        pytest.param(4, 4, id="full-shard"),
        pytest.param(6, 3, id="two-transfers"),
    ],
)
@pytest.mark.parametrize("reuse_activation", [False, True], ids=["stream", "reuse"])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.parametrize(
    "m_tiles,n_tiles,worker_grid,transpose,output_block_tiles",
    [
        pytest.param(2, 2, None, False, 1, id="one-block"),
        pytest.param(4, 6, (3, 2), False, 1, id="repeated-blocks"),
        pytest.param(4, 6, (2, 3), True, 1, id="transposed-repeated-blocks"),
        pytest.param(8, 12, (2, 3), True, 2, id="multi-tile-blocks"),
    ],
)
def test_all_gather_minimal_matmul(
    participant_mesh_shape,
    participant_mesh,
    torch_dtype,
    pcc_threshold,
    fp32_dest_acc_en,
    k_tiles_per_device,
    reuse_activation,
    k_block_tiles,
    with_bias,
    m_tiles,
    n_tiles,
    worker_grid,
    transpose,
    output_block_tiles,
):
    """Cover TILE/DRAM tensors for every supported numeric dtype."""

    config = AllGatherMinimalMatmulConfig(
        mesh_shape=participant_mesh_shape,
        m_tiles=m_tiles,
        k_tiles_per_device=k_tiles_per_device,
        reuse_activation=reuse_activation,
        k_block_tiles=k_block_tiles,
        m_block_tiles=output_block_tiles,
        n_block_tiles=output_block_tiles,
        n_tiles_per_device=n_tiles,
        worker_grid=worker_grid,
        transpose=transpose,
    )
    operation = make_all_gather_minimal_matmul_operation(
        config,
        math_fidelity="HiFi2" if torch_dtype == torch.bfloat16 else "HiFi4",
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    torch.manual_seed(0)

    m_elements = config.m_tiles * TILE_SIZE
    k_elements = config.device_count * config.k_tiles_per_device * TILE_SIZE
    n_elements = config.device_count * config.n_tiles_per_device * TILE_SIZE
    activation_torch = torch.randn((m_elements, k_elements), dtype=torch_dtype)
    weight_torch = torch.randn((k_elements, n_elements), dtype=torch_dtype)
    weight_torch /= k_elements**0.5
    bias_torch = torch.randn((1, n_elements), dtype=torch_dtype) * 0.1
    if not with_bias:
        bias_torch.zero_()

    activation_shard = to_dram(
        activation_torch,
        participant_mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(participant_mesh, dim=1),
    )
    weight_shard = to_dram(
        weight_torch,
        participant_mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(participant_mesh, dim=1),
    )
    bias_shard = to_dram(
        bias_torch,
        participant_mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(participant_mesh, dim=1),
    )
    output_shard = to_dram(
        torch.zeros((m_elements, n_elements), dtype=torch_dtype),
        participant_mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(participant_mesh, dim=1),
    )

    operation(
        activation_shard,
        weight_shard,
        bias_shard,
        output_shard,
    )

    output_result = ttnn.to_torch(
        output_shard,
        mesh_composer=ttnn.ConcatMeshToTensor(participant_mesh, dim=1),
    )

    expected_output = activation_torch.float() @ weight_torch.float()
    expected_output += bias_torch.float()
    assert_pcc(expected_output, output_result.float(), threshold=pcc_threshold)
    # FP32 FPU sources retain TF32 precision, including accumulator reloads.
    tolerance = 0.005 if torch_dtype == torch.float32 else 0.05
    assert_allclose(
        output_result.float(), expected_output, rtol=tolerance, atol=tolerance
    )
