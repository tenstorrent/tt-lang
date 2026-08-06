# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device correctness for scratch and direct sharded-L1 DFB storage."""

import importlib.util
import tempfile

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from conftest import temp_kernel_files
from ttlang_test_utils import assert_pcc, to_dram


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
    lhs_dfb = ttl.make_tensor_backed_dfb(lhs, shape=(1, {tile_count}))
    rhs_dfb = ttl.make_tensor_backed_dfb(rhs, shape=(1, {tile_count}))
    out_dfb = ttl.make_tensor_backed_dfb(out, shape=(1, {tile_count}))

    @ttl.compute()
    def compute_fn():
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
""",
}


def _make_kernel(storage_kind, tile_count, node_count):
    """Create source with a compile-time DFB capacity."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"{storage_kind}_dfb_"
    ) as source_file:
        source_file.write(
            KERNEL_TEMPLATES[storage_kind].format(
                tile_count=tile_count, node_count=node_count
            )
        )
        source_name = source_file.name
    temp_kernel_files.append(source_name)
    spec = importlib.util.spec_from_file_location(
        f"{storage_kind}_dfb_{tile_count}_{node_count}", source_name
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
    assert_pcc(expected.float(), actual.float(), threshold=0.999)
