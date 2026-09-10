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
from examples.all_gather_minimal_matmul.collectives import make_column_all_gather
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


@pytest.fixture(scope="module", params=[2, 4], ids=["two-devices", "four-devices"])
def participant_mesh_shape(fabric_mesh_shape, request) -> tuple[int, ...]:
    remaining = request.param
    extents = []
    for available in fabric_mesh_shape:
        extent = max(
            divisor
            for divisor in range(1, min(available, remaining) + 1)
            if remaining % divisor == 0
        )
        extents.append(extent)
        remaining //= extent
    if remaining != 1:
        pytest.skip(f"requires a {request.param}-device submesh")
    return tuple(extents)


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
@pytest.mark.parametrize("torch_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block_tiles", [1, 2])
@pytest.mark.parametrize("m_block_tiles", [1, 2])
@pytest.mark.parametrize("algorithm", ["all_to_all", "ring"])
def test_output_all_gather(
    participant_mesh,
    participant_mesh_shape,
    torch_dtype,
    block_tiles,
    m_block_tiles,
    algorithm,
):
    """TILE/DRAM gather preserves every payload bit and device-order N placement."""
    device_count = prod(participant_mesh_shape)
    torch.manual_seed(17)
    expected = torch.randn((128, 128 * device_count), dtype=torch_dtype)
    output_shard = to_dram(
        expected,
        participant_mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(participant_mesh, dim=1),
    )
    replicated_output = to_dram(
        torch.zeros_like(expected),
        participant_mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(participant_mesh),
    )
    gather = make_column_all_gather(
        participant_mesh_shape,
        m_tiles=4,
        n_tiles_per_device=4,
        worker_count=2,
        block_tiles=block_tiles,
        m_block_tiles=m_block_tiles,
        algorithm=algorithm,
    )
    for _invocation in range(2):
        gather(output_shard, replicated_output)
        actual = ttnn.to_torch(
            replicated_output,
            mesh_composer=ttnn.ConcatMeshToTensor(participant_mesh, dim=0),
        ).float()
        for replica in actual.split(128, dim=0):
            assert_allclose(replica, expected.float(), rtol=0, atol=0)


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
@pytest.mark.parametrize("algorithm", ["all_to_all", "ring"])
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
    algorithm,
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
        all_gather_algorithm=algorithm,
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


@requires_forwarding_link_indices(ttnn)
@pytest.mark.parametrize("torch_dtype,pcc_threshold,fp32_dest_acc_en", MATMUL_DTYPES)
@pytest.mark.parametrize("reuse_activation", [False, True], ids=["stream", "reuse"])
@pytest.mark.parametrize("m_tiles", [25, 26], ids=["padded-rows", "aligned-rows"])
@pytest.mark.parametrize("replicated", [False, True], ids=["n-sharded", "replicated"])
def test_all_gather_matmul_130_workers(
    participant_mesh,
    participant_mesh_shape,
    torch_dtype,
    pcc_threshold,
    fp32_dest_acc_en,
    reuse_activation,
    m_tiles,
    replicated,
):
    available_grid = participant_mesh.compute_with_storage_grid_size()
    if available_grid.x < 13 or available_grid.y < 10:
        pytest.skip("requires a 13x10 compute grid")
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=participant_mesh_shape,
        m_tiles=m_tiles,
        k_tiles_per_device=2,
        n_tiles_per_device=20,
        worker_grid=(13, 10),
        transpose=True,
        reuse_activation=reuse_activation,
    )
    operation = make_all_gather_minimal_matmul_operation(
        config,
        math_fidelity="HiFi4" if torch_dtype == torch.float32 else "HiFi2",
        fp32_dest_acc_en=fp32_dest_acc_en,
        all_gather_algorithm="ring",
    )
    torch.manual_seed(31)
    m_elements = config.m_tiles * TILE_SIZE
    k_elements = config.device_count * config.k_tiles_per_device * TILE_SIZE
    n_elements = (
        config.n_tiles_per_device
        * TILE_SIZE
        * (1 if replicated else config.device_count)
    )
    activation = torch.randn((m_elements, k_elements), dtype=torch_dtype)
    weight = torch.randn((k_elements, n_elements), dtype=torch_dtype) / k_elements**0.5
    bias = torch.randn((1, n_elements), dtype=torch_dtype) * 0.1
    shard_mapper = ttnn.ShardTensorToMesh(participant_mesh, dim=1)
    output_mapper = (
        ttnn.ReplicateTensorToMesh(participant_mesh) if replicated else shard_mapper
    )
    activation_shard = to_dram(
        torch.nn.functional.pad(
            activation, (0, 0, 0, config.padded_m_tiles * TILE_SIZE - m_elements)
        ),
        participant_mesh,
        mesh_mapper=shard_mapper,
    )
    weight_shard = to_dram(weight, participant_mesh, mesh_mapper=output_mapper)
    bias_shard = to_dram(bias, participant_mesh, mesh_mapper=output_mapper)
    output = to_dram(
        torch.zeros((m_elements, n_elements), dtype=torch_dtype),
        participant_mesh,
        mesh_mapper=output_mapper,
    )
    expected = activation.float() @ weight.float() + bias.float()
    for _invocation in range(2):
        operation(activation_shard, weight_shard, bias_shard, output)
        actual = ttnn.to_torch(
            output,
            mesh_composer=ttnn.ConcatMeshToTensor(
                participant_mesh, dim=0 if replicated else 1
            ),
        ).float()
        replicas = actual.split(m_elements, dim=0) if replicated else (actual,)
        for replica in replicas:
            assert_pcc(expected, replica, threshold=pcc_threshold)
            tolerance = 0.005 if torch_dtype == torch.float32 else 0.05
            assert_allclose(replica, expected, rtol=tolerance, atol=tolerance)
