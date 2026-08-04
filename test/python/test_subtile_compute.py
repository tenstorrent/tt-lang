# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for compute operations using LLK-supported tile dimensions."""

import pytest
import torch

import ttl
from utils.correctness import assert_allclose

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)


@ttl.operation(grid=(1, 1))
def materialized_subtile_exp(inp, out):
    input_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        input_block = input_dfb.wait()
        result = ttl.math.exp(input_block)
        input_block.pop()
        output_block = output_dfb.reserve()
        output_block.store(result)

    @ttl.datamovement()
    def reader():
        ttl.copy(inp[0:1, 0:1], input_dfb.reserve())

    @ttl.datamovement()
    def writer():
        ttl.copy(output_dfb.wait(), out[0:1, 0:1])


COMPUTE_TILE_SIZES = [(16, 16), (16, 32), (32, 16), (32, 32)]
DTYPES = [
    (torch.bfloat16, ttnn.bfloat16, 5e-2, 1.0),
    (torch.float32, ttnn.float32, 1e-3, 1e-3),
]


def _to_device(torch_tensor, device, tile_hw, dtype):
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=ttnn.Tile(tile_hw),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize(
    "tile_hw", COMPUTE_TILE_SIZES, ids=lambda tile: f"{tile[0]}x{tile[1]}"
)
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    DTYPES,
    ids=["bf16", "fp32"],
)
def test_materialized_subtile_exp(device, tile_hw, torch_dtype, ttnn_dtype, rtol, atol):
    tile_height, tile_width = tile_hw
    source = torch.linspace(-0.5, 0.5, tile_height * tile_width).reshape(tile_hw)
    source = source.to(torch_dtype)
    expected = torch.exp(source.float())
    input_tensor = _to_device(source, device, tile_hw, ttnn_dtype)
    output_tensor = _to_device(
        torch.zeros(tile_hw, dtype=torch_dtype), device, tile_hw, ttnn_dtype
    )

    materialized_subtile_exp(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()
    assert_allclose(actual, expected.float(), rtol=rtol, atol=atol)
