# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""A waited source releases each published DFB block before reuse."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl

from utils.correctness import assert_allclose


@ttl.operation(grid=(1, 1))
def copy_waited_dfb_source(input_tensor, output_tensor):
    scratch_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=1
    )
    for tile_index in range(2):
        with scratch_dfb.reserve() as destination_block:
            ttl.copy(
                input_tensor[0:1, tile_index : tile_index + 1], destination_block
            ).wait()
        with scratch_dfb.wait() as source_block:
            ttl.copy(
                source_block, output_tensor[0:1, tile_index : tile_index + 1]
            ).wait()


DTYPES = [
    pytest.param(ttnn.bfloat16, torch.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, torch.float32, id="fp32"),
    pytest.param(ttnn.bfloat8_b, torch.bfloat16, id="bfp8"),
    pytest.param(ttnn.bfloat4_b, torch.bfloat16, id="bfp4"),
    pytest.param(ttnn.int32, torch.int32, id="i32"),
    pytest.param(ttnn.uint32, torch.uint32, id="u32"),
    pytest.param(ttnn.uint16, torch.uint16, id="u16"),
    pytest.param(ttnn.uint8, torch.uint8, id="u8"),
]


def _to_device(host_tensor, device, dtype, memory_config):
    tensor = ttnn.from_torch(
        host_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if memory_config == "dram":
        return tensor
    if memory_config == "l1":
        return ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        (32, 64),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    shard_layouts = {
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }
    sharded_config = ttnn.MemoryConfig(
        shard_layouts[memory_config], ttnn.BufferType.L1, shard_spec
    )
    return ttnn.to_memory_config(tensor, sharded_config)


@pytest.mark.parametrize(("ttnn_dtype", "torch_dtype"), DTYPES)
@pytest.mark.parametrize("memory_config", ["dram", "l1", "height", "width", "block"])
@pytest.mark.requires_device
def test_copy_waited_dfb_source(device, ttnn_dtype, torch_dtype, memory_config):
    values = torch.arange(32 * 64, dtype=torch.int32).reshape(32, 64).remainder(97)
    input_host = values.to(torch_dtype)
    input_tensor = _to_device(input_host, device, ttnn_dtype, memory_config)
    output_tensor = _to_device(
        torch.zeros_like(input_host), device, ttnn_dtype, memory_config
    )

    copy_waited_dfb_source(input_tensor, output_tensor)
    ttnn.synchronize_device(device)

    expected = ttnn.to_torch(input_tensor).float()
    actual = ttnn.to_torch(output_tensor).float()
    if ttnn_dtype == ttnn.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    elif ttnn_dtype == ttnn.float32:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(actual, expected, rtol=0.0, atol=0.0)
