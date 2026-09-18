# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BF16 activation by block-float weight matmul coverage."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE_SIZE = 32
KIMI_MATMUL_TILE_SHAPE = (1, 48, 2)
KIMI_ACTIVATION_TILE = (1, TILE_SIZE)
KIMI_WEIGHT_TILE = (TILE_SIZE, TILE_SIZE)


def _make_staged_weight_matmul(math_fidelity):
    @ttl.operation(grid=(1, 1), math_fidelity=math_fidelity)
    def mixed_dtype_matmul(activation, weights, output):
        activation_tiles = activation.shape[1] // TILE_SIZE
        output_tiles = output.shape[1] // TILE_SIZE
        activation_dfb = ttl.make_dataflow_buffer_like(
            activation, shape=(1, activation_tiles), block_count=2
        )
        weights_dfb = ttl.make_dataflow_buffer_like(
            weights, shape=(activation_tiles, output_tiles), block_count=2
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(1, output_tiles), block_count=2
        )

        @ttl.compute()
        def compute_matmul():
            activation_block = activation_dfb.wait()
            weights_block = weights_dfb.wait()
            output_block = output_dfb.reserve()
            output_block.store(activation_block @ weights_block)
            activation_block.pop()
            weights_block.pop()
            output_block.push()

        @ttl.datamovement()
        def read_inputs():
            with activation_dfb.reserve() as activation_block:
                activation_copy = ttl.copy(
                    activation[0:1, 0:activation_tiles], activation_block
                )
                activation_copy.wait()
            with weights_dfb.reserve() as weights_block:
                weights_copy = ttl.copy(
                    weights[0:activation_tiles, 0:output_tiles], weights_block
                )
                weights_copy.wait()

        @ttl.datamovement()
        def write_output():
            with output_dfb.wait() as output_block:
                output_copy = ttl.copy(output_block, output[0:1, 0:output_tiles])
                output_copy.wait()

    return mixed_dtype_matmul


def _make_tensor_backed_weight_matmul(math_fidelity):
    @ttl.operation(grid=(1, 1), math_fidelity=math_fidelity)
    def mixed_dtype_matmul(activation, weights, output):
        activation_tiles = activation.shape[1] // TILE_SIZE
        output_tiles = output.shape[1] // TILE_SIZE
        activation_dfb = ttl.make_dataflow_buffer_like(
            activation, shape=(1, activation_tiles), block_count=2
        )
        weights_dfb = ttl.make_tensor_backed_dfb(
            weights, shape=(activation_tiles, output_tiles), block_count=1
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(1, output_tiles), block_count=2
        )

        @ttl.compute()
        def compute_matmul():
            activation_block = activation_dfb.wait()
            weights_block = weights_dfb.wait()
            output_block = output_dfb.reserve()
            output_block.store(activation_block @ weights_block)
            activation_block.pop()
            weights_block.pop()
            output_block.push()

        @ttl.datamovement()
        def publish_inputs():
            with activation_dfb.reserve() as activation_block:
                activation_copy = ttl.copy(
                    activation[0:1, 0:activation_tiles], activation_block
                )
                activation_copy.wait()
            weights_dfb.publish()

        @ttl.datamovement()
        def write_output():
            with output_dfb.wait() as output_block:
                output_copy = ttl.copy(output_block, output[0:1, 0:output_tiles])
                output_copy.wait()

    return mixed_dtype_matmul


STAGED_WEIGHT_MATMULS = {
    fidelity: _make_staged_weight_matmul(fidelity) for fidelity in ("LoFi", "HiFi4")
}
TENSOR_BACKED_WEIGHT_MATMULS = {
    fidelity: _make_tensor_backed_weight_matmul(fidelity)
    for fidelity in ("LoFi", "HiFi4")
}


@pytest.fixture(
    params=(
        (ttnn.bfloat4_b, "bfp_bf4", 576),
        (ttnn.bfloat8_b, "bfp_bf8", 1088),
    ),
    ids=("bfp4", "bfp8"),
)
def weight_format(request):
    return request.param


def _host_tensors(weight_dtype):
    row_tiles, contraction_tiles, column_tiles = KIMI_MATMUL_TILE_SHAPE
    activation = ttnn.from_torch(
        torch.randn(
            row_tiles * KIMI_ACTIVATION_TILE[0],
            contraction_tiles * TILE_SIZE,
            dtype=torch.bfloat16,
        ),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile(KIMI_ACTIVATION_TILE),
    )
    weights = ttnn.from_torch(
        torch.randn(
            contraction_tiles * TILE_SIZE,
            column_tiles * TILE_SIZE,
            dtype=torch.bfloat16,
        ),
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile(KIMI_WEIGHT_TILE),
    )
    output = ttnn.from_torch(
        torch.zeros(
            row_tiles * KIMI_ACTIVATION_TILE[0],
            column_tiles * TILE_SIZE,
            dtype=torch.bfloat16,
        ),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile(KIMI_ACTIVATION_TILE),
    )
    return activation, weights, output


def test_mixed_dtype_matmul_compile_only(tmp_path, monkeypatch, weight_format):
    weight_dtype, mlir_dtype, page_size = weight_format
    initial_mlir = tmp_path / "initial.mlir"
    final_mlir = tmp_path / "final.mlir"
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    monkeypatch.setenv("TTLANG_INITIAL_MLIR", str(initial_mlir))
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))

    STAGED_WEIGHT_MATMULS["LoFi"](*_host_tensors(weight_dtype))

    initial_text = initial_mlir.read_text()
    final_text = final_mlir.read_text()
    assert "ttl.matmul" in initial_text
    assert "!ttcore.tile<1x32, bf16>" in initial_text
    assert f"!ttcore.tile<{TILE_SIZE}x{TILE_SIZE}, {mlir_dtype}>" in initial_text
    assert "ttl.matmul" not in final_text
    assert "!ttcore.tile<1x32, bf16>" in final_text
    assert f"!ttcore.tile<{TILE_SIZE}x{TILE_SIZE}, {mlir_dtype}>" in final_text
    assert f"page_size = {page_size}" in final_text


def _make_weight_tensor(weights_torch, weight_dtype, device, memory_config):
    # Block-float dtype must be explicit because its Torch source is BF16.
    return ttnn.from_torch(
        weights_torch,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile(KIMI_WEIGHT_TILE),
        device=device,
        memory_config=memory_config,
    )


def _make_sharded_l1_memory_config(tensor_shape, memory_layout):
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(0, 0),
                )
            }
        ),
        tensor_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.L1,
        shard_spec,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize(
    ("weight_storage", "tensor_backing_layout"),
    (
        ("dram", None),
        ("interleaved_l1", None),
        ("tensor_backed_l1", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        ("tensor_backed_l1", ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ("tensor_backed_l1", ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ),
    ids=(
        "dram",
        "l1",
        "tensor-backed-height-sharded-l1",
        "tensor-backed-width-sharded-l1",
        "tensor-backed-block-sharded-l1",
    ),
)
@pytest.mark.parametrize("math_fidelity", ("LoFi", "HiFi4"), ids=("lofi", "hifi4"))
def test_mixed_dtype_matmul_device(
    device, weight_storage, tensor_backing_layout, math_fidelity, weight_format
):
    weight_dtype, _mlir_dtype, _page_size = weight_format
    row_tiles, contraction_tiles, column_tiles = KIMI_MATMUL_TILE_SHAPE
    torch.manual_seed(0)
    activation_torch = torch.randn(
        row_tiles * KIMI_ACTIVATION_TILE[0],
        contraction_tiles * TILE_SIZE,
        dtype=torch.bfloat16,
    )
    weights_torch = torch.randn(
        contraction_tiles * TILE_SIZE,
        column_tiles * TILE_SIZE,
        dtype=torch.bfloat16,
    )
    output_torch = torch.zeros(
        row_tiles * KIMI_ACTIVATION_TILE[0],
        column_tiles * TILE_SIZE,
        dtype=torch.bfloat16,
    )

    if weight_storage == "dram":
        weight_memory_config = ttnn.DRAM_MEMORY_CONFIG
        matmul = STAGED_WEIGHT_MATMULS[math_fidelity]
    elif weight_storage == "interleaved_l1":
        weight_memory_config = ttnn.L1_MEMORY_CONFIG
        matmul = STAGED_WEIGHT_MATMULS[math_fidelity]
    elif weight_storage == "tensor_backed_l1":
        weight_memory_config = _make_sharded_l1_memory_config(
            tuple(weights_torch.shape), tensor_backing_layout
        )
        matmul = TENSOR_BACKED_WEIGHT_MATMULS[math_fidelity]
    else:
        raise ValueError(f"unsupported weight storage {weight_storage}")

    activation = to_dram(activation_torch, device, tile=KIMI_ACTIVATION_TILE)
    weights = _make_weight_tensor(
        weights_torch, weight_dtype, device, weight_memory_config
    )
    output = to_dram(output_torch, device, tile=KIMI_ACTIVATION_TILE)
    activation_before = ttnn.to_torch(activation).clone()
    weights_before = ttnn.to_torch(weights).clone()

    device.enable_program_cache()
    matmul(activation, weights, output)
    ttnn.synchronize_device(device)
    first_output = ttnn.to_torch(output).float()
    first_cache_entries = device.num_program_cache_entries()

    matmul(activation, weights, output)
    ttnn.synchronize_device(device)
    second_output = ttnn.to_torch(output).float()
    second_cache_entries = device.num_program_cache_entries()

    assert first_cache_entries > 0
    assert first_cache_entries == second_cache_entries
    assert torch.equal(first_output, second_output)
    assert torch.equal(activation_before, ttnn.to_torch(activation))
    assert torch.equal(weights_before, ttnn.to_torch(weights))
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    assert tuple(activation.get_tile().tile_shape) == KIMI_ACTIVATION_TILE
    assert tuple(weights.get_tile().tile_shape) == KIMI_WEIGHT_TILE
    assert tuple(output.get_tile().tile_shape) == KIMI_ACTIVATION_TILE
    if tensor_backing_layout is not None:
        assert weights.memory_config().memory_layout == tensor_backing_layout

    expected = activation_before.float() @ weights_before.float()
    assert_pcc(expected.float(), first_output.float(), threshold=0.999)
