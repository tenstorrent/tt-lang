# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Device coverage for composed operations using indexed DFB collections."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose


@ttl.operation()
def _copy_collection_stage(source: ttl.DFB, destination: ttl.DFB):
    source_block = source.wait()
    destination_block = destination.reserve()
    destination_block.store(source_block)


@ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
def _copy_through_dfb_collection(input_tensor, output_tensor):
    buffers = [
        ttl.make_dataflow_buffer_like(
            input_tensor,
            shape=(1, 1),
            block_count=2,
        ),
        ttl.make_dataflow_buffer_like(
            output_tensor,
            shape=(1, 1),
            block_count=2,
        ),
    ]

    ttl.copy(input_tensor[0:1, 0:1], buffers[0].reserve()).wait()
    _copy_collection_stage(buffers[0], buffers[1])
    ttl.copy(buffers[1].wait(), output_tensor[0:1, 0:1]).wait()


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        pytest.param(torch.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, 1.0e-3, 1.0e-3, id="fp32"),
    ],
)
def test_composed_operation_executes_with_indexed_dfb_collection(
    device, dtype, rtol, atol
):
    tile_size = ttnn.TILE_SIZE
    input_data = torch.randn(tile_size, tile_size, dtype=dtype) * 8.0
    input_tensor = to_dram(input_data, device)
    output_tensor = to_dram(
        torch.zeros(tile_size, tile_size, dtype=dtype),
        device,
    )

    _copy_through_dfb_collection(input_tensor, output_tensor)

    result = ttnn.to_torch(output_tensor).reshape(tile_size, tile_size)
    assert_allclose(result.float(), input_data.float(), rtol=rtol, atol=atol)
