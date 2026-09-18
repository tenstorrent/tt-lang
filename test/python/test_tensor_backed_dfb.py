# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device correctness for scratch and direct sharded-L1 DFB storage."""

import importlib.util
import tempfile

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from conftest import temp_kernel_files
from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose


KERNEL_TEMPLATES = {
    "scratch": """
import ttl

@ttl.operation(grid=({grid_x}, {grid_y}))
def dfb_storage_mul(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(
        lhs, shape=(1, {tile_count}), block_count=2
    )
    rhs_dfb = ttl.make_dataflow_buffer_like(
        rhs, shape=(1, {tile_count}), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, {tile_count}), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with (
            lhs_dfb.wait() as lhs_block,
            rhs_dfb.wait() as rhs_block,
            out_dfb.reserve() as out_block,
        ):
            out_block.store(lhs_block * rhs_block)

    @ttl.datamovement()
    def load_inputs():
        node_x, _node_y = ttl.node(dims=2)
        with lhs_dfb.reserve() as lhs_block, rhs_dfb.reserve() as rhs_block:
            lhs_copy = ttl.copy(
                lhs[node_x:node_x + 1, 0:{tile_count}], lhs_block
            )
            rhs_copy = ttl.copy(
                rhs[node_x:node_x + 1, 0:{tile_count}], rhs_block
            )
            lhs_copy.wait()
            rhs_copy.wait()

    @ttl.datamovement()
    def store_output():
        node_x, _node_y = ttl.node(dims=2)
        with out_dfb.wait() as out_block:
            output_copy = ttl.copy(
                out_block, out[node_x:node_x + 1, 0:{tile_count}]
            )
            output_copy.wait()
""",
    "tensor_backed": """
import ttl

@ttl.operation(grid=({grid_x}, {grid_y}))
def dfb_storage_mul(lhs, rhs, out):
    lhs_dfb = ttl.make_tensor_backed_dfb(
        lhs,
        shape=({tile_rows}, {tile_count}),
        block_count={block_count},
        byte_offset={byte_offset},
    )
    rhs_dfb = ttl.make_tensor_backed_dfb(
        rhs,
        shape=({tile_rows}, {tile_count}),
        block_count={block_count},
        byte_offset={byte_offset},
    )
    out_dfb = ttl.make_tensor_backed_dfb(
        out,
        shape=({tile_rows}, {tile_count}),
        block_count={block_count},
        byte_offset={byte_offset},
    )

    @ttl.compute()
    def compute_fn():
        for _ in range({block_count}):
            lhs_block = lhs_dfb.wait()
            rhs_block = rhs_dfb.wait()
            out_block = out_dfb.reserve()
            out_block.store(lhs_block * rhs_block)
            out_block.push()
            rhs_block.pop()
            lhs_block.pop()

    @ttl.datamovement()
    def publish_inputs():
        lhs_dfb.publish()
        rhs_dfb.publish()

    @ttl.datamovement()
    def consume_output():
        for _ in range({block_count}):
            out_block = out_dfb.wait()
            out_block.pop()
""",
}

_SHARD_DIMENSION_BY_MEMORY_LAYOUT = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: 0,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: 1,
}


def _make_kernel(
    storage_kind,
    tile_count,
    grid_x,
    grid_y=1,
    byte_offset=0,
    block_count=1,
    tile_rows=1,
):
    """Create source with a compile-time DFB capacity."""
    assert (
        storage_kind != "scratch" or tile_rows == 1
    ), "scratch DFB template supports one tile row"
    assert (
        storage_kind != "scratch" or grid_y == 1
    ), "scratch DFB template supports one grid row"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"{storage_kind}_dfb_"
    ) as source_file:
        source_file.write(
            KERNEL_TEMPLATES[storage_kind].format(
                tile_count=tile_count,
                grid_x=grid_x,
                grid_y=grid_y,
                byte_offset=byte_offset,
                block_count=block_count,
                tile_rows=tile_rows,
            )
        )
        source_name = source_file.name
    temp_kernel_files.append(source_name)
    spec = importlib.util.spec_from_file_location(
        f"{storage_kind}_dfb_{tile_rows}_{tile_count}_{grid_x}_{grid_y}_"
        f"{byte_offset}_{block_count}",
        source_name,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dfb_storage_mul


def _to_sharded(torch_tensor, device, node_count, memory_layout):
    dram_tensor = to_dram(torch_tensor, device)
    tensor_rows, tensor_columns = torch_tensor.shape[-2:]
    shard_shape = [tensor_rows, tensor_columns]
    shard_dimension = _SHARD_DIMENSION_BY_MEMORY_LAYOUT[memory_layout]
    shard_shape[shard_dimension] //= node_count
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(node_count - 1, 0),
                )
            }
        ),
        tuple(shard_shape),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.L1,
        shard_spec,
    )
    return ttnn.to_memory_config(dram_tensor, memory_config=memory_config)


def _to_height_sharded(torch_tensor, device, node_count):
    return _to_sharded(
        torch_tensor,
        device,
        node_count,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )


def _to_block_sharded(
    torch_tensor,
    device,
    grid_x,
    grid_y,
    shard_shape,
    shard_orientation,
):
    dram_tensor = to_dram(torch_tensor, device)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_x - 1, grid_y - 1),
                )
            }
        ),
        shard_shape,
        shard_orientation,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    return ttnn.to_memory_config(dram_tensor, memory_config=memory_config)


def _assert_dtype_aware_allclose(actual, expected, torch_dtype):
    if torch_dtype == torch.bfloat16:
        assert_allclose(actual.float(), expected.float(), rtol=0.05, atol=1e-2)
    else:
        assert_allclose(actual.float(), expected.float(), rtol=5e-3, atol=1e-4)


@ttl.operation(grid=(1, 1))
def _replace_waited_tensor_backed_state(state, output):
    state_dfb = ttl.make_tensor_backed_dfb(state, shape=(1, 1), block_count=1)
    output_dfb = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def replace_state():
        with state_dfb.wait() as state_block:
            increment = ttl.block.fill(
                1,
                shape=state_block.shape,
                dtype=state_block.dtype,
            )
            state_block.store(state_block + increment)
            with output_dfb.reserve() as output_block:
                output_block.store(state_block)

    @ttl.datamovement()
    def publish_state():
        state_dfb.publish()

    @ttl.datamovement()
    def write_output():
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, output[0, 0]).wait()


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
@pytest.mark.parametrize("tile_count", [1, 8], ids=["one_tile", "eight_tiles"])
@pytest.mark.parametrize("node_count", [1, 2], ids=["one_node", "two_nodes"])
@pytest.mark.parametrize(
    "storage_kind", ["scratch", "tensor_backed"], ids=["scratch", "tensor-backed"]
)
def test_dfb_storage_eltwise_mul(
    device, torch_dtype, tile_count, node_count, storage_kind
):
    """The same computation is correct with staged and direct DFB storage."""
    tensor_shape = (32 * node_count, 32 * tile_count)
    torch.manual_seed(0)
    lhs_torch = torch.rand(tensor_shape, dtype=torch_dtype)
    rhs_torch = torch.rand(tensor_shape, dtype=torch_dtype)
    expected = lhs_torch * rhs_torch

    lhs = _to_height_sharded(lhs_torch, device, node_count)
    rhs = _to_height_sharded(rhs_torch, device, node_count)
    out = _to_height_sharded(torch.zeros_like(expected), device, node_count)

    _make_kernel(storage_kind, tile_count, node_count)(lhs, rhs, out)

    actual = ttnn.to_torch(out)
    _assert_dtype_aware_allclose(actual, expected, torch_dtype)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
@pytest.mark.parametrize("tile_count", [1, 8], ids=["one_tile", "eight_tiles"])
@pytest.mark.parametrize("node_count", [1, 2], ids=["one_node", "two_nodes"])
@pytest.mark.parametrize(
    "shard_tile_rows", [1, 2], ids=["one_tile_row", "two_tile_rows"]
)
def test_tensor_backed_dfb_width_sharded_storage(
    device, torch_dtype, tile_count, node_count, shard_tile_rows
):
    """Tensor-backed DFBs bind multi-row width shards on every launch node."""
    memory_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    tensor_shape = (32 * shard_tile_rows, 32 * tile_count * node_count)
    torch.manual_seed(0)
    lhs_torch = torch.rand(tensor_shape, dtype=torch_dtype)
    rhs_torch = torch.rand(tensor_shape, dtype=torch_dtype)
    expected = lhs_torch * rhs_torch

    lhs = _to_sharded(lhs_torch, device, node_count, memory_layout)
    rhs = _to_sharded(rhs_torch, device, node_count, memory_layout)
    out = _to_sharded(torch.zeros_like(expected), device, node_count, memory_layout)

    _make_kernel("tensor_backed", tile_count, node_count, tile_rows=shard_tile_rows)(
        lhs, rhs, out
    )

    actual = ttnn.to_torch(out)
    _assert_dtype_aware_allclose(actual, expected, torch_dtype)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    ids=["row-major", "column-major"],
)
@pytest.mark.parametrize(
    ("shard_tile_rows", "shard_tile_columns"),
    [(1, 1), (2, 2)],
    ids=["one-tile-shard", "four-tile-shard"],
)
def test_tensor_backed_dfb_block_sharded_storage(
    device,
    torch_dtype,
    shard_orientation,
    shard_tile_rows,
    shard_tile_columns,
):
    """Tensor-backed DFBs bind each block shard on its launch node."""
    grid_x = 2
    grid_y = 2
    shard_shape = (32 * shard_tile_rows, 32 * shard_tile_columns)
    tensor_shape = (shard_shape[0] * grid_y, shard_shape[1] * grid_x)
    torch.manual_seed(0)
    lhs_torch = torch.rand(tensor_shape, dtype=torch_dtype)
    rhs_torch = torch.rand(tensor_shape, dtype=torch_dtype)
    expected = lhs_torch * rhs_torch

    lhs = _to_block_sharded(
        lhs_torch, device, grid_x, grid_y, shard_shape, shard_orientation
    )
    rhs = _to_block_sharded(
        rhs_torch, device, grid_x, grid_y, shard_shape, shard_orientation
    )
    out = _to_block_sharded(
        torch.zeros_like(expected),
        device,
        grid_x,
        grid_y,
        shard_shape,
        shard_orientation,
    )

    _make_kernel(
        "tensor_backed",
        tile_count=shard_tile_columns,
        grid_x=grid_x,
        grid_y=grid_y,
        tile_rows=shard_tile_rows,
    )(lhs, rhs, out)

    actual = ttnn.to_torch(out)
    _assert_dtype_aware_allclose(actual, expected, torch_dtype)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_tensor_backed_dfb_block_count_two(device, torch_dtype):
    """One publication supplies two FIFO blocks and leaves the DFB reusable."""
    block_count = 2
    tile_count = 1
    tensor_shape = (32, 32 * tile_count * block_count)
    operation = _make_kernel(
        "tensor_backed", tile_count=tile_count, grid_x=1, block_count=block_count
    )

    for seed in range(2):
        torch.manual_seed(seed)
        lhs_torch = torch.rand(tensor_shape, dtype=torch_dtype)
        rhs_torch = torch.rand(tensor_shape, dtype=torch_dtype)
        expected = lhs_torch * rhs_torch
        lhs = _to_height_sharded(lhs_torch, device, node_count=1)
        rhs = _to_height_sharded(rhs_torch, device, node_count=1)
        out = _to_height_sharded(torch.zeros_like(expected), device, node_count=1)

        operation(lhs, rhs, out)

        actual = ttnn.to_torch(out)
        _assert_dtype_aware_allclose(actual, expected, torch_dtype)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_tensor_backed_waited_block_replacement_persists(device, torch_dtype):
    """Replacement updates tensor storage and remains reusable across dispatches."""
    element_indices = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    state_host = ((element_indices.remainder(257) - 128) / 64).to(torch_dtype)
    state = _to_height_sharded(state_host, device, node_count=1)
    output = to_dram(torch.zeros_like(state_host), device)

    _replace_waited_tensor_backed_state(state, output)
    _replace_waited_tensor_backed_state(state, output)

    actual_state = ttnn.to_torch(state).float()
    actual_output = ttnn.to_torch(output).float()
    expected = state_host.float() + 2
    if torch_dtype == torch.bfloat16:
        assert_allclose(actual_state, expected, rtol=0.05, atol=1.0)
        assert_allclose(actual_output, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual_state, expected, rtol=1e-5, atol=1e-6)
        assert_allclose(actual_output, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_tensor_backed_dfb_nonzero_byte_offset(device, torch_dtype):
    """A page-aligned view reads and writes only its logical shard range."""
    tile_width = 32
    torch.manual_seed(0)
    lhs_torch = torch.full((32, 3 * tile_width), -2.0, dtype=torch_dtype)
    rhs_torch = torch.full((32, 3 * tile_width), 3.0, dtype=torch_dtype)
    out_torch = torch.full((32, 3 * tile_width), 7.0, dtype=torch_dtype)
    lhs_torch[:, tile_width : 2 * tile_width] = torch.rand(
        (32, tile_width), dtype=torch_dtype
    )
    rhs_torch[:, tile_width : 2 * tile_width] = torch.rand(
        (32, tile_width), dtype=torch_dtype
    )
    expected = out_torch.clone()
    expected[:, tile_width : 2 * tile_width] = (
        lhs_torch[:, tile_width : 2 * tile_width]
        * rhs_torch[:, tile_width : 2 * tile_width]
    )

    lhs = _to_height_sharded(lhs_torch, device, node_count=1)
    rhs = _to_height_sharded(rhs_torch, device, node_count=1)
    out = _to_height_sharded(out_torch, device, node_count=1)
    byte_offset = int(lhs.get_tile().get_tile_size(lhs.dtype))

    _make_kernel("tensor_backed", tile_count=1, grid_x=1, byte_offset=byte_offset)(
        lhs, rhs, out
    )

    actual = ttnn.to_torch(out)
    _assert_dtype_aware_allclose(actual, expected, torch_dtype)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_tensor_backed_dfb_rejects_range_past_shard_boundary(device, torch_dtype):
    """Invalid ranges fail before TTNN descriptor construction."""
    tensor_shape = (32, 64)
    lhs = _to_height_sharded(torch.ones(tensor_shape, dtype=torch_dtype), device, 1)
    rhs = _to_height_sharded(torch.ones(tensor_shape, dtype=torch_dtype), device, 1)
    out = _to_height_sharded(torch.zeros(tensor_shape, dtype=torch_dtype), device, 1)
    byte_offset = int(lhs.get_tile().get_tile_size(lhs.dtype))

    operation = _make_kernel(
        "tensor_backed", tile_count=2, grid_x=1, byte_offset=byte_offset
    )
    with pytest.raises(ValueError, match="exceeds logical per-shard size"):
        operation(lhs, rhs, out)
