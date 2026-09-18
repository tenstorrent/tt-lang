# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware coverage for the explicit four-device tree all-reduce."""

from math import prod

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from examples.multidevice_tree_all_reduce_4_devices import (
    NUM_DEVICES,
    TILE_SIZE,
    _select_participant_mesh_shape,
    make_tree_all_reduce_operation,
)
from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TREE_REDUCTION_DTYPES = [
    pytest.param(torch.bfloat16, ttnn.bfloat16, 0.05, 1.0, id="bf16"),
    pytest.param(torch.float32, ttnn.float32, 5e-3, 5e-2, id="fp32"),
]


def _mesh_tensor(mesh, tensor, dtype):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


@pytest.fixture(scope="module")
def participant_mesh_shape():
    if ttnn.get_num_devices() < NUM_DEVICES:
        pytest.skip(f"requires at least {NUM_DEVICES} devices")
    parent_mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    if prod(parent_mesh_shape) < NUM_DEVICES:
        pytest.skip(f"requires at least {NUM_DEVICES} devices")
    return _select_participant_mesh_shape(parent_mesh_shape)


@pytest.fixture(scope="module")
def participant_mesh(participant_mesh_shape):
    parent_mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    with open_fabric_mesh(
        requested_mesh_shape=parent_mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as parent_mesh:
        owns_participant_mesh = participant_mesh_shape != parent_mesh_shape
        mesh = (
            parent_mesh.create_submesh(ttnn.MeshShape(participant_mesh_shape))
            if owns_participant_mesh
            else parent_mesh
        )
        try:
            yield mesh
        finally:
            if owns_participant_mesh:
                ttnn.close_mesh_device(mesh)


# Verify that the explicit pairwise tree reduces four input tiles and
# broadcasts the result to every participating device.
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", TREE_REDUCTION_DTYPES)
def test_tree_all_reduce(
    participant_mesh_shape,
    participant_mesh,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    tree_all_reduce = make_tree_all_reduce_operation(participant_mesh_shape)
    logical_shape = (NUM_DEVICES * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    inp = _mesh_tensor(participant_mesh, inp_torch, ttnn_dtype)
    out = _mesh_tensor(participant_mesh, out_torch, ttnn_dtype)

    tree_all_reduce(inp, out)

    result = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMeshToTensor(participant_mesh, dim=0),
    )

    input_tiles = inp_torch.reshape(NUM_DEVICES, TILE_SIZE, TILE_SIZE)
    reduced = input_tiles.float().sum(dim=0).to(torch_dtype)
    expected = reduced.repeat(NUM_DEVICES, 1)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
