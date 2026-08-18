# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BF16 activation by BFP4_B weight matmul coverage."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_pcc

TILE_SIZE = 32
MATMUL_TILE_SHAPE = (1, 2, 2)


@ttl.operation(grid=(1, 1))
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


def _host_tensors():
    row_tiles, contraction_tiles, column_tiles = MATMUL_TILE_SHAPE
    activation = ttnn.from_torch(
        torch.randn(
            row_tiles * TILE_SIZE,
            contraction_tiles * TILE_SIZE,
            dtype=torch.bfloat16,
        ),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    weights = ttnn.from_torch(
        torch.randn(
            contraction_tiles * TILE_SIZE,
            column_tiles * TILE_SIZE,
            dtype=torch.bfloat16,
        ),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
    )
    output = ttnn.from_torch(
        torch.zeros(
            row_tiles * TILE_SIZE,
            column_tiles * TILE_SIZE,
            dtype=torch.bfloat16,
        ),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    return activation, weights, output


def test_mixed_dtype_matmul_compile_only(tmp_path, monkeypatch, capsys):
    initial_mlir = tmp_path / "initial.mlir"
    final_mlir = tmp_path / "final.mlir"
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    monkeypatch.setenv("TTLANG_INITIAL_MLIR", str(initial_mlir))
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))

    mixed_dtype_matmul(*_host_tensors())

    initial_text = initial_mlir.read_text()
    final_text = final_mlir.read_text()
    capsys.readouterr()
    assert "ttl.matmul" in initial_text
    assert "!ttcore.tile<32x32, bf16>" in initial_text
    assert "!ttcore.tile<32x32, bfp_bf4>" in initial_text
    assert "ttl.matmul" not in final_text
    assert "!ttcore.tile<32x32, bfp_bf4>" in final_text
    assert "page_size = 576" in final_text


def _make_weight_tensor(weights_torch, device, memory_config):
    # BFP4_B must be requested explicitly because its Torch source is BF16.
    return ttnn.from_torch(
        weights_torch,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_mixed_dtype_matmul_device(device, memory):
    row_tiles, contraction_tiles, column_tiles = MATMUL_TILE_SHAPE
    activation_torch = torch.randn(
        row_tiles * TILE_SIZE,
        contraction_tiles * TILE_SIZE,
        dtype=torch.bfloat16,
    )
    weights_torch = torch.randn(
        contraction_tiles * TILE_SIZE,
        column_tiles * TILE_SIZE,
        dtype=torch.bfloat16,
    )
    output_torch = torch.zeros(
        row_tiles * TILE_SIZE,
        column_tiles * TILE_SIZE,
        dtype=torch.bfloat16,
    )

    tensor_factory = to_l1 if memory == "l1" else to_dram
    memory_config = ttnn.L1_MEMORY_CONFIG if memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
    activation = tensor_factory(activation_torch, device)
    weights = _make_weight_tensor(weights_torch, device, memory_config)
    output = tensor_factory(output_torch, device)
    quantized_weights = ttnn.to_torch(weights).float()

    mixed_dtype_matmul(activation, weights, output)

    actual = ttnn.to_torch(output).float()
    expected = activation_torch.float() @ quantized_weights
    assert_pcc(expected.float(), actual.float(), threshold=0.999)
