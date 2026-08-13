# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for subtile compute operations."""

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
def short_height_situ(gate, up, out):
    gate_dfb = ttl.make_tensor_backed_dfb(gate, shape=(1, 3), block_count=1)
    up_dfb = ttl.make_tensor_backed_dfb(up, shape=(1, 3), block_count=1)
    output_dfb = ttl.make_tensor_backed_dfb(out, shape=(1, 3), block_count=1)

    @ttl.compute()
    def compute():
        gate_block = gate_dfb.wait()
        up_block = up_dfb.wait()
        output_block = output_dfb.reserve()
        quarter = ttl.block.fill(
            0.25,
            shape=gate_block.shape,
            dtype=gate_block.dtype,
            tile=gate_block.tile,
        )
        four = ttl.block.fill(
            4.0,
            shape=gate_block.shape,
            dtype=gate_block.dtype,
            tile=gate_block.tile,
        )
        one_twenty_fifth = ttl.block.fill(
            0.04,
            shape=up_block.shape,
            dtype=up_block.dtype,
            tile=up_block.tile,
        )
        twenty_five = ttl.block.fill(
            25.0,
            shape=up_block.shape,
            dtype=up_block.dtype,
            tile=up_block.tile,
        )
        gate_result = (
            four * ttl.math.tanh(gate_block * quarter) * ttl.math.sigmoid(gate_block)
        )
        up_result = twenty_five * ttl.math.tanh(up_block * one_twenty_fifth)
        output_block.store(gate_result * up_result)
        output_block.push()
        up_block.pop()
        gate_block.pop()

    @ttl.datamovement()
    def publish_inputs():
        gate_dfb.publish()
        up_dfb.publish()

    @ttl.datamovement()
    def consume_output():
        output_block = output_dfb.wait()
        output_block.pop()


@ttl.operation(grid=(1, 1))
def subtile_sub(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as lhs_block, rhs_dfb.wait() as rhs_block:
            with output_dfb.reserve() as output_block:
                output_block.store(lhs_block - rhs_block)

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
def subtile_mul(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as lhs_block, rhs_dfb.wait() as rhs_block:
            with output_dfb.reserve() as output_block:
                output_block.store(lhs_block * rhs_block)

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
def subtile_broadcast_row(inp, out):
    input_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with input_dfb.wait() as input_block, output_dfb.reserve() as output_block:
            output_block.store(ttl.block.broadcast(input_block, dims=[0], shape=(1, 1)))

    @ttl.datamovement()
    def reader():
        with input_dfb.reserve() as input_block:
            ttl.copy(inp[0:1, 0:1], input_block).wait()

    @ttl.datamovement()
    def writer():
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, out[0:1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def subtile_broadcast_col(inp, out):
    input_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with input_dfb.wait() as input_block, output_dfb.reserve() as output_block:
            output_block.store(ttl.block.broadcast(input_block, dims=[1], shape=(1, 1)))

    @ttl.datamovement()
    def reader():
        with input_dfb.reserve() as input_block:
            ttl.copy(inp[0:1, 0:1], input_block).wait()

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
def subtile_matmul_relu(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 2), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(2, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as lhs_block, rhs_dfb.wait() as rhs_block:
            with output_dfb.reserve() as output_block:
                output_block.store(ttl.math.relu(lhs_block @ rhs_block))

    @ttl.datamovement()
    def reader():
        with lhs_dfb.reserve() as lhs_block:
            ttl.copy(lhs[0:1, 0:2], lhs_block).wait()
        with rhs_dfb.reserve() as rhs_block:
            ttl.copy(rhs[0:2, 0:1], rhs_block).wait()

    @ttl.datamovement()
    def writer():
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, out[0:1, 0:1]).wait()


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


@ttl.operation(grid=(1, 1))
def subtile_reduce_sum_row(inp, out):
    input_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with input_dfb.wait() as input_block:
            with output_dfb.reserve() as output_block:
                output_block.store(ttl.math.reduce_sum(input_block, dims=[1]))

    @ttl.datamovement()
    def reader():
        with input_dfb.reserve() as input_block:
            ttl.copy(inp[0:1, 0:1], input_block).wait()

    @ttl.datamovement()
    def writer():
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, out[0:1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def subtile_reduce_max_row(inp, out):
    input_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with input_dfb.wait() as input_block:
            with output_dfb.reserve() as output_block:
                output_block.store(ttl.math.reduce_max(input_block, dims=[1]))

    @ttl.datamovement()
    def reader():
        with input_dfb.reserve() as input_block:
            ttl.copy(inp[0:1, 0:1], input_block).wait()

    @ttl.datamovement()
    def writer():
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, out[0:1, 0:1]).wait()


SHORT_HEIGHT_TILE_SIZES = [(1, 32), (2, 32), (4, 32), (8, 32)]
COMPUTE_TILE_SIZES = [(16, 16), (16, 32), (32, 16), (32, 32)]
ELEMENTWISE_TILE_SIZES = SHORT_HEIGHT_TILE_SIZES + COMPUTE_TILE_SIZES
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
MATMUL_FUSED_TILE_CONFIGS = [
    ((8, 32), (32, 32), (8, 32)),
    ((16, 32), (32, 16), (16, 16)),
]
DTYPES = [
    (torch.bfloat16, ttnn.bfloat16, 5e-2, 1.0),
    (torch.float32, ttnn.float32, 1e-3, 1e-3),
]
REDUCE_DTYPES = [
    (torch.bfloat16, ttnn.bfloat16, 5e-2, 1.0),
    (torch.float32, ttnn.float32, 5e-3, 1e-2),
]
MATMUL_DTYPES = [
    (torch.bfloat16, ttnn.bfloat16, 0.999),
    (torch.float32, ttnn.float32, 0.9999),
]
INTEGER_DTYPES = [
    (torch.int32, ttnn.int32),
    (torch.uint32, ttnn.uint32),
    (torch.uint16, ttnn.uint16),
]
MEMORY_CONFIGS = [
    pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="dram"),
    pytest.param(ttnn.L1_MEMORY_CONFIG, id="l1"),
]
INTEGER_OPERATIONS = [
    pytest.param(subtile_add, 10, id="add"),
    pytest.param(subtile_sub, 4, id="sub"),
    pytest.param(subtile_mul, 21, id="mul"),
]
ROW_REDUCTION_OPERATIONS = [
    pytest.param(subtile_reduce_sum_row, torch.sum, id="sum"),
    pytest.param(subtile_reduce_max_row, torch.amax, id="max"),
]


def _to_device(torch_tensor, device, tile_hw, dtype, memory_config):
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=ttnn.Tile(tile_hw),
        memory_config=memory_config,
    )


def _to_height_sharded_l1(torch_tensor, device, tile_hw):
    tensor_height, tensor_width = torch_tensor.shape
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(0, 0),
                )
            }
        ),
        (tensor_height, tensor_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    return _to_device(
        torch_tensor,
        device,
        tile_hw,
        ttnn.bfloat16,
        memory_config,
    )


@pytest.mark.parametrize(
    "tile_hw", ELEMENTWISE_TILE_SIZES, ids=lambda tile: f"{tile[0]}x{tile[1]}"
)
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    DTYPES,
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_subtile_exp(
    device, tile_hw, torch_dtype, ttnn_dtype, rtol, atol, memory_config
):
    tile_height, tile_width = tile_hw
    source = torch.linspace(-0.5, 0.5, tile_height * tile_width).reshape(tile_hw)
    source = source.to(torch_dtype)
    expected = torch.exp(source.float())
    input_tensor = _to_device(source, device, tile_hw, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        torch.zeros(tile_hw, dtype=torch_dtype),
        device,
        tile_hw,
        ttnn_dtype,
        memory_config,
    )

    subtile_exp(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()
    assert_allclose(actual, expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "tile_hw", ELEMENTWISE_TILE_SIZES, ids=lambda tile: f"{tile[0]}x{tile[1]}"
)
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    DTYPES,
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_subtile_add(
    device, tile_hw, torch_dtype, ttnn_dtype, rtol, atol, memory_config
):
    lhs_source = torch.ones(tile_hw, dtype=torch_dtype)
    rhs_source = torch.full(tile_hw, 2.0, dtype=torch_dtype)
    output_source = torch.zeros(tile_hw, dtype=torch_dtype)

    lhs_tensor = _to_device(lhs_source, device, tile_hw, ttnn_dtype, memory_config)
    rhs_tensor = _to_device(rhs_source, device, tile_hw, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        output_source, device, tile_hw, ttnn_dtype, memory_config
    )

    subtile_add(lhs_tensor, rhs_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()
    expected = lhs_source.float() + rhs_source.float()
    assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "tile_hw", SHORT_HEIGHT_TILE_SIZES, ids=lambda tile: f"{tile[0]}x{tile[1]}"
)
def test_short_height_tensor_backed_situ(device, tile_hw):
    """The K3 SiTU expression runs over three direct sharded-L1 pages."""
    tile_height, tile_width = tile_hw
    tensor_shape = (tile_height, 3 * tile_width)
    torch.manual_seed(0)
    gate = torch.randn(tensor_shape, dtype=torch.bfloat16)
    up = torch.randn(tensor_shape, dtype=torch.bfloat16)
    output = torch.zeros(tensor_shape, dtype=torch.bfloat16)
    expected = (4.0 * torch.tanh(gate.float() / 4.0) * torch.sigmoid(gate.float())) * (
        25.0 * torch.tanh(up.float() / 25.0)
    )

    gate_tensor = _to_height_sharded_l1(gate, device, tile_hw)
    up_tensor = _to_height_sharded_l1(up, device, tile_hw)
    output_tensor = _to_height_sharded_l1(output, device, tile_hw)

    short_height_situ(gate_tensor, up_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tensor_shape).float()
    assert_allclose(actual, expected, rtol=5e-2, atol=1e-1)


@pytest.mark.parametrize("kernel,expected_value", INTEGER_OPERATIONS)
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    INTEGER_DTYPES,
    ids=["int32", "uint32", "uint16"],
)
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_subtile_integer_binary(
    device, kernel, expected_value, torch_dtype, ttnn_dtype, memory_config
):
    tile_hw = (16, 32)
    lhs_source = torch.full(tile_hw, 7, dtype=torch.int64).to(torch_dtype)
    rhs_source = torch.full(tile_hw, 3, dtype=torch.int64).to(torch_dtype)
    output_source = torch.zeros(tile_hw, dtype=torch_dtype)

    lhs_tensor = _to_device(lhs_source, device, tile_hw, ttnn_dtype, memory_config)
    rhs_tensor = _to_device(rhs_source, device, tile_hw, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        output_source, device, tile_hw, ttnn_dtype, memory_config
    )

    kernel(lhs_tensor, rhs_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()
    expected = torch.full(tile_hw, expected_value, dtype=torch.float32)
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    INTEGER_DTYPES,
    ids=["int32", "uint32", "uint16"],
)
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_subtile_integer_broadcast(device, torch_dtype, ttnn_dtype, memory_config):
    tile_hw = (16, 32)
    source = torch.zeros(tile_hw, dtype=torch_dtype)
    source[0, :] = torch.arange(tile_hw[1], dtype=torch.int64).to(torch_dtype)
    output_source = torch.zeros(tile_hw, dtype=torch_dtype)

    input_tensor = _to_device(source, device, tile_hw, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        output_source, device, tile_hw, ttnn_dtype, memory_config
    )

    subtile_broadcast_row(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()
    expected = source[0:1, :].expand(tile_hw).float()
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    DTYPES,
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_subtile_broadcast_col(
    device, torch_dtype, ttnn_dtype, rtol, atol, memory_config
):
    tile_hw = (8, 32)
    source = torch.zeros(tile_hw, dtype=torch_dtype)
    source[:, 0] = torch.linspace(-3.0, 4.0, tile_hw[0]).to(torch_dtype)
    input_tensor = _to_device(source, device, tile_hw, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        torch.zeros(tile_hw, dtype=torch_dtype),
        device,
        tile_hw,
        ttnn_dtype,
        memory_config,
    )

    subtile_broadcast_col(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()
    expected = source[:, :1].expand(tile_hw).float()
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
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_subtile_matmul(
    device,
    lhs_tile,
    rhs_tile,
    output_tile,
    torch_dtype,
    ttnn_dtype,
    pcc_threshold,
    memory_config,
):
    lhs_shape = (lhs_tile[0], 2 * lhs_tile[1])
    rhs_shape = (2 * rhs_tile[0], 2 * rhs_tile[1])
    output_shape = (output_tile[0], 2 * output_tile[1])
    torch.manual_seed(0)
    lhs_source = torch.randn(lhs_shape).to(torch_dtype)
    rhs_source = torch.randn(rhs_shape).to(torch_dtype)
    output_source = torch.zeros(output_shape, dtype=torch_dtype)

    lhs_tensor = _to_device(lhs_source, device, lhs_tile, ttnn_dtype, memory_config)
    rhs_tensor = _to_device(rhs_source, device, rhs_tile, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        output_source, device, output_tile, ttnn_dtype, memory_config
    )

    subtile_matmul(lhs_tensor, rhs_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(output_shape).float()
    expected = lhs_source.float() @ rhs_source.float()
    assert_pcc(expected, actual, threshold=pcc_threshold)


@pytest.mark.parametrize(
    "lhs_tile,rhs_tile,output_tile",
    MATMUL_FUSED_TILE_CONFIGS,
    ids=["short-height", "short-width"],
)
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,pcc_threshold",
    MATMUL_DTYPES,
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_subtile_matmul_relu(
    device,
    lhs_tile,
    rhs_tile,
    output_tile,
    torch_dtype,
    ttnn_dtype,
    pcc_threshold,
    memory_config,
):
    lhs_shape = (lhs_tile[0], 2 * lhs_tile[1])
    rhs_shape = (2 * rhs_tile[0], rhs_tile[1])
    output_shape = output_tile
    torch.manual_seed(0)
    lhs_source = torch.randn(lhs_shape).to(torch_dtype)
    rhs_source = torch.randn(rhs_shape).to(torch_dtype)
    output_source = torch.zeros(output_shape, dtype=torch_dtype)

    lhs_tensor = _to_device(lhs_source, device, lhs_tile, ttnn_dtype, memory_config)
    rhs_tensor = _to_device(rhs_source, device, rhs_tile, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        output_source, device, output_tile, ttnn_dtype, memory_config
    )

    subtile_matmul_relu(lhs_tensor, rhs_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(output_shape).float()
    expected = torch.relu(lhs_source.float() @ rhs_source.float())
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
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_subtile_matmul_transposed(
    device,
    lhs_tile,
    rhs_tile,
    output_tile,
    torch_dtype,
    ttnn_dtype,
    pcc_threshold,
    memory_config,
):
    lhs_shape = (lhs_tile[0], 2 * lhs_tile[1])
    rhs_shape = (rhs_tile[0], 2 * rhs_tile[1])
    output_shape = output_tile
    torch.manual_seed(0)
    lhs_source = torch.randn(lhs_shape).to(torch_dtype)
    rhs_source = torch.randn(rhs_shape).to(torch_dtype)
    output_source = torch.zeros(output_shape, dtype=torch_dtype)

    lhs_tensor = _to_device(lhs_source, device, lhs_tile, ttnn_dtype, memory_config)
    rhs_tensor = _to_device(rhs_source, device, rhs_tile, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        output_source, device, output_tile, ttnn_dtype, memory_config
    )

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
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_subtile_reduce(
    device, tile_hw, torch_dtype, ttnn_dtype, rtol, atol, memory_config
):
    source = torch.ones(tile_hw, dtype=torch_dtype)
    output_source = torch.zeros(tile_hw, dtype=torch_dtype)
    input_tensor = _to_device(source, device, tile_hw, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        output_source, device, tile_hw, ttnn_dtype, memory_config
    )

    subtile_reduce(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()[0, 0]
    expected = source.float().sum()
    assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("kernel,reducer", ROW_REDUCTION_OPERATIONS)
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    REDUCE_DTYPES,
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_subtile_reduce_row(
    device,
    kernel,
    reducer,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
    memory_config,
):
    tile_hw = (8, 32)
    source = torch.linspace(-3.0, 5.0, tile_hw[0] * tile_hw[1]).reshape(tile_hw)
    source += torch.arange(tile_hw[0]).reshape(-1, 1) * 0.125
    source = source.to(torch_dtype)
    input_tensor = _to_device(source, device, tile_hw, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        torch.zeros(tile_hw, dtype=torch_dtype),
        device,
        tile_hw,
        ttnn_dtype,
        memory_config,
    )

    kernel(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(tile_hw).float()[:, 0]
    expected = reducer(source.float(), dim=1)
    assert_allclose(actual, expected, rtol=rtol, atol=atol)
