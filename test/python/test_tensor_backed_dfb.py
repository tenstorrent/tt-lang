# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device correctness for scratch and direct sharded-L1 DFB storage."""

import importlib.util
import tempfile

import pytest
import torch

import ttl
from ttl.dataflow_buffer import DFBStorageSegment, PhysicalDFBConfig
from ttl.kernel_runner import build_cb_descriptors

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from conftest import temp_kernel_files
from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose, assert_pcc


KERNEL_TEMPLATES = {
    "scratch": """
import ttl

@ttl.operation(grid=({node_count}, 1))
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

@ttl.operation(grid=({node_count}, 1))
def dfb_storage_mul(lhs, rhs, out):
    lhs_dfb = ttl.make_tensor_backed_dfb(
        lhs,
        shape=(1, {tile_count}),
        block_count={block_count},
        byte_offset={byte_offset},
    )
    rhs_dfb = ttl.make_tensor_backed_dfb(
        rhs,
        shape=(1, {tile_count}),
        block_count={block_count},
        byte_offset={byte_offset},
    )
    out_dfb = ttl.make_tensor_backed_dfb(
        out,
        shape=(1, {tile_count}),
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


def _make_tensor_backed_subtile_view_mul(*, fp32_dest_acc_en):
    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=fp32_dest_acc_en)
    def tensor_backed_subtile_view_mul(lhs, rhs, out):
        """Multiply directly in 1x32 storage interpreted as one 16x32 tile."""
        lhs_dfb = ttl.make_tensor_backed_dfb(lhs, shape=(1, 1), tile=(16, 32))
        rhs_dfb = ttl.make_tensor_backed_dfb(rhs, shape=(1, 1), tile=(16, 32))
        out_dfb = ttl.make_tensor_backed_dfb(out, shape=(1, 1), tile=(16, 32))

        @ttl.compute()
        def compute():
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
            out_block = out_dfb.wait()
            out_block.pop()

    return tensor_backed_subtile_view_mul


TENSOR_BACKED_SUBTILE_VIEW_MUL = {
    torch.bfloat16: _make_tensor_backed_subtile_view_mul(fp32_dest_acc_en=False),
    torch.float32: _make_tensor_backed_subtile_view_mul(fp32_dest_acc_en=True),
    torch.uint16: _make_tensor_backed_subtile_view_mul(fp32_dest_acc_en=False),
}


def _make_kernel(storage_kind, tile_count, node_count, byte_offset=0, block_count=1):
    """Create source with a compile-time DFB capacity."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"{storage_kind}_dfb_"
    ) as source_file:
        source_file.write(
            KERNEL_TEMPLATES[storage_kind].format(
                tile_count=tile_count,
                node_count=node_count,
                byte_offset=byte_offset,
                block_count=block_count,
            )
        )
        source_name = source_file.name
    temp_kernel_files.append(source_name)
    spec = importlib.util.spec_from_file_location(
        f"{storage_kind}_dfb_{tile_count}_{node_count}_{byte_offset}_{block_count}",
        source_name,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dfb_storage_mul


def _to_height_sharded(torch_tensor, device, node_count):
    dram_tensor = to_dram(torch_tensor, device)
    tensor_rows, tensor_columns = torch_tensor.shape[-2:]
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(node_count - 1, 0),
                )
            }
        ),
        (tensor_rows // node_count, tensor_columns),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    return ttnn.to_memory_config(dram_tensor, memory_config=memory_config)


def _to_compact_height_sharded(torch_tensor, device, ttnn_dtype):
    """Store one 512-element row as sixteen contiguous 1x32 L1 pages."""
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(0, 0),
                )
            }
        ),
        (1, 512),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=ttnn.Tile((1, 32)),
        memory_config=memory_config,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "torch_dtype",
    [torch.bfloat16, torch.float32, torch.uint16],
    ids=["bf16", "f32", "uint16"],
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
    if torch_dtype == torch.uint16:
        lhs_torch = torch.randint(0, 32, tensor_shape, dtype=torch.int32).to(
            torch.uint16
        )
        rhs_torch = torch.randint(0, 32, tensor_shape, dtype=torch.int32).to(
            torch.uint16
        )
        expected = (lhs_torch.to(torch.int32) * rhs_torch.to(torch.int32)).to(
            torch.uint16
        )
    else:
        lhs_torch = torch.rand(tensor_shape, dtype=torch_dtype)
        rhs_torch = torch.rand(tensor_shape, dtype=torch_dtype)
        expected = lhs_torch * rhs_torch

    lhs = _to_height_sharded(lhs_torch, device, node_count)
    rhs = _to_height_sharded(rhs_torch, device, node_count)
    out = _to_height_sharded(torch.zeros_like(expected), device, node_count)

    _make_kernel(storage_kind, tile_count, node_count)(lhs, rhs, out)

    actual = ttnn.to_torch(out)
    if torch_dtype == torch.uint16:
        assert_allclose(actual.float(), expected.float(), rtol=0.0, atol=0.0)
    else:
        assert_pcc(expected.float(), actual.float(), threshold=0.999)


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
        "tensor_backed",
        tile_count=tile_count,
        node_count=1,
        block_count=block_count,
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
        assert_pcc(expected.float(), actual.float(), threshold=0.999)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    [
        (torch.bfloat16, ttnn.bfloat16, 0.05, 1.0),
        (torch.float32, ttnn.float32, 1e-3, 1e-3),
        (torch.uint16, ttnn.uint16, 0.0, 0.0),
    ],
    ids=["bf16", "fp32", "uint16"],
)
def test_tensor_backed_subtile_view(device, torch_dtype, ttnn_dtype, rtol, atol):
    """A compute-page view must preserve the compact tensor's byte order."""
    torch.manual_seed(0)
    if torch_dtype == torch.uint16:
        lhs_source = torch.arange(512, dtype=torch.int32).to(torch.uint16)[None, :]
        rhs_source = torch.full((1, 512), 2, dtype=torch.uint16)
    else:
        lhs_source = torch.rand((1, 512), dtype=torch_dtype)
        rhs_source = torch.full((1, 512), 2.0, dtype=torch_dtype)
    expected = lhs_source.float() * rhs_source.float()

    lhs = _to_compact_height_sharded(lhs_source, device, ttnn_dtype)
    rhs = _to_compact_height_sharded(rhs_source, device, ttnn_dtype)
    out = _to_compact_height_sharded(torch.zeros_like(lhs_source), device, ttnn_dtype)

    TENSOR_BACKED_SUBTILE_VIEW_MUL[torch_dtype](lhs, rhs, out)

    actual = ttnn.to_torch(out).reshape(1, 512).float()
    assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.requires_device
def test_tensor_backed_subtile_view_rejects_out_of_bounds_range(device):
    source = torch.ones((1, 512), dtype=torch.bfloat16)
    tensor = _to_compact_height_sharded(source, device, ttnn.bfloat16)
    node_ranges = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(0, 0),
            )
        }
    )
    config = PhysicalDFBConfig(
        dfb_index=0,
        num_tiles=2,
        data_format="bfloat16",
        block_count=1,
        page_size=1024,
        tile=(16, 32),
        storage_segments=(
            DFBStorageSegment(
                nodes=((0, 0),),
                tensor_index=0,
                byte_size=2048,
            ),
        ),
    )

    with pytest.raises(ValueError, match="exceeds logical per-shard size 1024"):
        build_cb_descriptors([tensor], [config], node_ranges)


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

    _make_kernel("tensor_backed", tile_count=1, node_count=1, byte_offset=byte_offset)(
        lhs, rhs, out
    )

    actual = ttnn.to_torch(out).float()
    if torch_dtype == torch.bfloat16:
        assert_allclose(actual, expected.float(), rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected.float(), rtol=5e-3, atol=1e-4)


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
        "tensor_backed", tile_count=2, node_count=1, byte_offset=byte_offset
    )
    with pytest.raises(ValueError, match="exceeds logical per-shard size"):
        operation(lhs, rhs, out)
