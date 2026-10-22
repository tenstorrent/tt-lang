# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device correctness of the ring all-gather relay lowering example."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from examples.multidevice_ring_all_gather import (
    UnsupportedRingMesh,
    open_ring_mesh,
    run_ring_all_gather,
)

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
    try:
        with open_ring_mesh() as mesh_device:
            run_ring_all_gather(
                mesh_device,
                torch_dtype,
                m_tiles=m_tiles,
                k_shard_tiles=k_shard_tiles,
                lanes=lanes,
                chunk_shape=chunk_shape,
            )
    except UnsupportedRingMesh as error:
        pytest.skip(str(error))
