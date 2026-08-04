# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for subtile elementwise, matmul, and reduce operations."""

import pytest
import torch

import ttl
from utils.correctness import assert_allclose, assert_pcc

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device


@ttl.operation(grid=(1, 1))
def subtile_exp(inp, out):
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


@ttl.operation(grid=(1, 1))
def subtile_add(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as lhs_block, rhs_dfb.wait() as rhs_block:
            with output_dfb.reserve() as output_block:
                output_block.store(lhs_block + rhs_block)

    @ttl.datamovement()
    def reader():
        with lhs_dfb.reserve() as lhs_block:
            ttl.copy(lhs[0:1, 0:1], lhs_block).wait()
        with rhs_dfb.reserve() as rhs_block:
            ttl.copy(rhs[0:1, 0:1], rhs_block).wait()

    @ttl.datamovement()
    def writer():
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, out[0:1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def subtile_matmul(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 2), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(2, 2), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 2), block_count=2)

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as lhs_block, rhs_dfb.wait() as rhs_block:
            with output_dfb.reserve() as output_block:
                output_block.store(lhs_block @ rhs_block)

    @ttl.datamovement()
    def reader():
        with lhs_dfb.reserve() as lhs_block:
            ttl.copy(lhs[0:1, 0:2], lhs_block).wait()
        with rhs_dfb.reserve() as rhs_block:
            ttl.copy(rhs[0:2, 0:2], rhs_block).wait()

    @ttl.datamovement()
    def writer():
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, out[0:1, 0:2]).wait()


@ttl.operation(grid=(1, 1))
def subtile_matmul_transposed(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 2), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 2), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as lhs_block, rhs_dfb.wait() as rhs_block:
            result = ttl.matmul(lhs_block, rhs_block, transpose_rhs=True)
            with output_dfb.reserve() as output_block:
                output_block.store(result)

    @ttl.datamovement()
    def reader():
        with lhs_dfb.reserve() as lhs_block:
            ttl.copy(lhs[0:1, 0:2], lhs_block).wait()
        with rhs_dfb.reserve() as rhs_block:
            ttl.copy(rhs[0:1, 0:2], rhs_block).wait()

    @ttl.datamovement()
    def writer():
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, out[0:1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def subtile_reduce(inp, out):
    input_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with input_dfb.wait() as input_block:
            with output_dfb.reserve() as output_block:
                output_block.store(ttl.math.reduce_sum(input_block, dims=[0, 1]))

    @ttl.datamovement()
    def reader():
        with input_dfb.reserve() as input_block:
            ttl.copy(inp[0:1, 0:1], input_block).wait()

    @ttl.datamovement()
    def writer():
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, out[0:1, 0:1]).wait()


COMPUTE_TILE_SIZES = [(16, 16), (16, 32), (32, 16), (32, 32)]
MATMUL_TILE_CONFIGS = [
    ((height, 32), (32, 32), (height, 32)) for height in [1, 2, 4, 8, 16, 32]
] + [
    ((16, 32), (32, 16), (16, 16)),
    ((32, 32), (32, 16), (32, 16)),
    ((32, 16), (16, 32), (32, 32)),
]
MATMUL_TRANSPOSE_TILE_CONFIGS = [
    ((height, 32), (32, 32), (height, 32)) for height in [1, 2, 4, 8, 16, 32]
] + [
    ((16, 32), (16, 32), (16, 16)),
]
DTYPES = [
    (torch.bfloat16, ttnn.bfloat16, 5e-2, 1.0),
    (torch.float32, ttnn.float32, 1e-3, 1e-3),
]
MATMUL_DTYPES = [
    (torch.bfloat16, ttnn.bfloat16, 0.999),
    (torch.float32, ttnn.float32, 0.9999),
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
def test_subtile_exp(device, tile_hw, torch_dtype, ttnn_dtype, rtol, atol):
    tile_height, tile_width = tile_hw
    source = torch.linspace(-0.5, 0.5, tile_height * tile_width).reshape(tile_hw)
    source = source.to(torch_dtype)
    expected = torch.exp(source.float())
    input_tensor = _to_device(source, device, tile_hw, ttnn_dtype)
    output_tensor = _to_device(
        torch.zeros(tile_hw, dtype=torch_dtype), device, tile_hw, ttnn_dtype
    )

    subtile_exp(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()
    assert_allclose(actual, expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "tile_hw", COMPUTE_TILE_SIZES, ids=lambda tile: f"{tile[0]}x{tile[1]}"
)
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    DTYPES,
    ids=["bf16", "fp32"],
)
def test_subtile_add(device, tile_hw, torch_dtype, ttnn_dtype, rtol, atol):
    lhs_source = torch.ones(tile_hw, dtype=torch_dtype)
    rhs_source = torch.full(tile_hw, 2.0, dtype=torch_dtype)
    output_source = torch.zeros(tile_hw, dtype=torch_dtype)

    lhs_tensor = _to_device(lhs_source, device, tile_hw, ttnn_dtype)
    rhs_tensor = _to_device(rhs_source, device, tile_hw, ttnn_dtype)
    output_tensor = _to_device(output_source, device, tile_hw, ttnn_dtype)

    subtile_add(lhs_tensor, rhs_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()
    expected = lhs_source.float() + rhs_source.float()
    assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "lhs_tile,rhs_tile,output_tile",
    MATMUL_TILE_CONFIGS,
    ids=lambda config: f"{config[0]}x{config[1]}",
)
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,pcc_threshold",
    MATMUL_DTYPES,
    ids=["bf16", "fp32"],
)
def test_subtile_matmul(
    device, lhs_tile, rhs_tile, output_tile, torch_dtype, ttnn_dtype, pcc_threshold
):
    lhs_shape = (lhs_tile[0], 2 * lhs_tile[1])
    rhs_shape = (2 * rhs_tile[0], 2 * rhs_tile[1])
    output_shape = (output_tile[0], 2 * output_tile[1])
    torch.manual_seed(0)
    lhs_source = torch.randn(lhs_shape).to(torch_dtype)
    rhs_source = torch.randn(rhs_shape).to(torch_dtype)
    output_source = torch.zeros(output_shape, dtype=torch_dtype)

    lhs_tensor = _to_device(lhs_source, device, lhs_tile, ttnn_dtype)
    rhs_tensor = _to_device(rhs_source, device, rhs_tile, ttnn_dtype)
    output_tensor = _to_device(output_source, device, output_tile, ttnn_dtype)

    subtile_matmul(lhs_tensor, rhs_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(output_shape).float()
    expected = lhs_source.float() @ rhs_source.float()
    assert_pcc(expected, actual, threshold=pcc_threshold)


@pytest.mark.parametrize(
    "lhs_tile,rhs_tile,output_tile",
    MATMUL_TRANSPOSE_TILE_CONFIGS,
    ids=lambda config: f"{config[0]}x{config[1]}",
)
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,pcc_threshold",
    MATMUL_DTYPES,
    ids=["bf16", "fp32"],
)
def test_subtile_matmul_transposed(
    device, lhs_tile, rhs_tile, output_tile, torch_dtype, ttnn_dtype, pcc_threshold
):
    lhs_shape = (lhs_tile[0], 2 * lhs_tile[1])
    rhs_shape = (rhs_tile[0], 2 * rhs_tile[1])
    output_shape = output_tile
    torch.manual_seed(0)
    lhs_source = torch.randn(lhs_shape).to(torch_dtype)
    rhs_source = torch.randn(rhs_shape).to(torch_dtype)
    output_source = torch.zeros(output_shape, dtype=torch_dtype)

    lhs_tensor = _to_device(lhs_source, device, lhs_tile, ttnn_dtype)
    rhs_tensor = _to_device(rhs_source, device, rhs_tile, ttnn_dtype)
    output_tensor = _to_device(output_source, device, output_tile, ttnn_dtype)

    subtile_matmul_transposed(lhs_tensor, rhs_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(output_shape).float()
    expected = lhs_source.float() @ rhs_source.float().t()
    assert_pcc(expected, actual, threshold=pcc_threshold)


@pytest.mark.parametrize(
    "tile_hw", COMPUTE_TILE_SIZES, ids=lambda tile: f"{tile[0]}x{tile[1]}"
)
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    DTYPES,
    ids=["bf16", "fp32"],
)
def test_subtile_reduce(device, tile_hw, torch_dtype, ttnn_dtype, rtol, atol):
    source = torch.ones(tile_hw, dtype=torch_dtype)
    output_source = torch.zeros(tile_hw, dtype=torch_dtype)
    input_tensor = _to_device(source, device, tile_hw, ttnn_dtype)
    output_tensor = _to_device(output_source, device, tile_hw, ttnn_dtype)

    subtile_reduce(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()[0, 0]
    expected = source.float().sum()
    assert_allclose(actual, expected, rtol=rtol, atol=atol)
