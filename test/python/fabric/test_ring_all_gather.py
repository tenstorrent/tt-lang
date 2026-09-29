# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device correctness of the ring all-gather relay lowering example."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from examples.multidevice_ring_all_gather import (
    RING_MESH_SHAPES,
    open_ring_mesh,
    run_ring_all_gather,
)
from ttlang_test_utils import get_fabric_mesh_shape

pytestmark = pytest.mark.multi_device


@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
@pytest.mark.parametrize(
    ("m_tiles", "k_shard_tiles", "lanes", "chunk_shape"),
    [(16, 8, 2, (8, 4)), (32, 16, 2, (8, 4)), (16, 8, 1, (4, 4))],
    ids=["one-chunk-per-lane", "two-by-two-chunks-per-lane", "four-chunks-one-lane"],
)
def test_ring_all_gather(torch_dtype, m_tiles, k_shard_tiles, lanes, chunk_shape):
    mesh_shape = tuple(get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D))
    if mesh_shape not in RING_MESH_SHAPES:
        pytest.skip("the ring all-gather needs a 2x2 or 2x4 mesh")
    with open_ring_mesh() as mesh_device:
        run_ring_all_gather(
            mesh_device,
            torch_dtype,
            m_tiles=m_tiles,
            k_shard_tiles=k_shard_tiles,
            lanes=lanes,
            chunk_shape=chunk_shape,
        )
