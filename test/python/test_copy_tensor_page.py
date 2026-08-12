# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Public row-major tensor page copy coverage."""

from __future__ import annotations

import pytest
import torch

import ttl
from utils.correctness import assert_allclose

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)


HIDDEN_SIZE = 7168
HIDDEN_PAGES = HIDDEN_SIZE // 32
ROW_COUNT = 5
BOUNDARY_ROWS = (0, 2, ROW_COUNT - 1)


@ttl.operation(grid=(1, 1))
def copy_selected_page(table, index_tensor, output):
    index_dfb = ttl.make_dataflow_buffer_like(index_tensor, shape=(1, 1), block_count=2)
    row_dfb = ttl.make_dataflow_buffer_like(
        output, shape=(1, HIDDEN_PAGES), block_count=2
    )

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def read():
        with index_dfb.reserve() as index_block:
            ttl.copy(index_tensor[0, 0], index_block).wait()

        with index_dfb.wait() as index_block:
            page_id = ttl.read_index(index_block, 0, 0)
            with row_dfb.reserve() as row_block:
                ttl.copy_tensor_page(table, page_id, row_block).wait()

    @ttl.datamovement()
    def write():
        with row_dfb.wait() as row_block:
            ttl.copy(row_block, output[0:1, 0:HIDDEN_PAGES]).wait()


@ttl.operation()
def _composed_copy_selected_page(table, index_tensor, output):
    index_dfb = ttl.make_dataflow_buffer_like(index_tensor, shape=(1, 1), block_count=2)
    row_dfb = ttl.make_dataflow_buffer_like(
        output, shape=(1, HIDDEN_PAGES), block_count=2
    )

    with index_dfb.reserve() as index_block:
        ttl.copy(index_tensor[0, 0], index_block).wait()

    with index_dfb.wait() as index_block:
        page_id = ttl.read_index(index_block, 0, 0)
        with row_dfb.reserve() as row_block:
            ttl.copy_tensor_page(table, page_id, row_block).wait()

    with row_dfb.wait() as row_block:
        ttl.copy(row_block, output[0:1, 0:HIDDEN_PAGES]).wait()


@ttl.operation(grid=(1, 1))
def composed_copy_selected_page(table, index_tensor, output):
    _composed_copy_selected_page(table, index_tensor, output)


@ttl.operation(grid=(1, 1))
def copy_selected_page_auto_wait(table, index_tensor, output):
    index_dfb = ttl.make_dataflow_buffer_like(index_tensor, shape=(1, 1), block_count=2)
    row_dfb = ttl.make_dataflow_buffer_like(
        output, shape=(1, HIDDEN_PAGES), block_count=2
    )

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def read():
        with index_dfb.reserve() as index_block:
            ttl.copy(index_tensor[0, 0], index_block).wait()

        with index_dfb.wait() as index_block:
            page_id = ttl.read_index(index_block, 0, 0)
            with row_dfb.reserve() as row_block:
                ttl.copy_tensor_page(table, page_id, row_block)

    @ttl.datamovement()
    def write():
        with row_dfb.wait() as row_block:
            ttl.copy(row_block, output[0:1, 0:HIDDEN_PAGES]).wait()


def _torch_dtype_to_ttnn(dtype):
    if dtype == torch.bfloat16:
        return ttnn.bfloat16
    if dtype == torch.float32:
        return ttnn.float32
    raise ValueError(f"unsupported dtype {dtype}")


def _host_inputs(dtype, row_id):
    table = (
        torch.arange(ROW_COUNT * HIDDEN_SIZE, dtype=torch.float32)
        .remainder(251)
        .reshape(ROW_COUNT, HIDDEN_SIZE)
        .to(dtype)
    )
    index = torch.full((32, 32), row_id, dtype=torch.float32)
    output = torch.zeros((1, HIDDEN_SIZE), dtype=dtype)
    return table, index, output


def _to_row_major(table, device, memory_config):
    return ttnn.from_torch(
        table,
        dtype=_torch_dtype_to_ttnn(table.dtype),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _to_compact_tile(tensor, device, memory_config):
    return ttnn.from_torch(
        tensor,
        dtype=_torch_dtype_to_ttnn(tensor.dtype),
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((1, 32)),
        device=device,
        memory_config=memory_config,
    )


def _compile_only_tensors():
    table, index, output = _host_inputs(torch.bfloat16, BOUNDARY_ROWS[1])
    table_tensor = ttnn.from_torch(
        table,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    index_tensor = ttnn.from_torch(index, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_torch(
        output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((1, 32)),
    )
    return table_tensor, index_tensor, output_tensor


@pytest.mark.parametrize(
    "operation",
    [
        copy_selected_page,
        composed_copy_selected_page,
        copy_selected_page_auto_wait,
    ],
    ids=["direct", "composed", "automatic_wait"],
)
def test_copy_tensor_page_compile_only_purity(operation, tmp_path, monkeypatch, capsys):
    """Frontend and composed pipelines lower to one native page read."""
    initial_mlir = tmp_path / "initial.mlir"
    final_mlir = tmp_path / "final.mlir"
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    monkeypatch.setenv("TTLANG_INITIAL_MLIR", str(initial_mlir))
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))

    operation(*_compile_only_tensors())

    generated_output = capsys.readouterr().out
    assert "ttl.copy_tensor_page" in initial_mlir.read_text()
    assert "ttl.copy_tensor_page" not in final_mlir.read_text()
    assert ".async_read(" in generated_output
    assert "get_aligned_page_size()" in generated_output
    assert "14336" in generated_output
    assert "call_extern_func" not in generated_output
    assert "opaque_call" not in generated_output


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "operation",
    [copy_selected_page, composed_copy_selected_page],
    ids=["direct", "composed"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "source_memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["source_dram", "source_l1"],
)
@pytest.mark.parametrize(
    "output_memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["output_dram", "output_l1"],
)
@pytest.mark.parametrize("row_id", BOUNDARY_ROWS, ids=["first", "interior", "last"])
def test_copy_tensor_page_device(
    device,
    operation,
    dtype,
    source_memory_config,
    output_memory_config,
    row_id,
):
    """Every supported dtype and placement returns the exact selected row."""
    table, index, output = _host_inputs(dtype, row_id)
    table_tensor = _to_row_major(table, device, source_memory_config)
    index_tensor = ttnn.from_torch(
        index,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = _to_compact_tile(output, device, output_memory_config)

    operation(table_tensor, index_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).reshape(1, HIDDEN_SIZE).float()
    expected = table[row_id : row_id + 1].float()
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)
