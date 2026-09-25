# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reject copy endpoints acquired for the wrong dataflow buffer role."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttl.diagnostics import TTLangCompileError


@ttl.operation(grid=(1, 1))
def reserved_tensor_copy_source(input_tensor, output_tensor):
    scratch_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=1
    )
    for tile_index in range(2):
        reserved_block = scratch_dfb.reserve()
        ttl.copy(input_tensor[0:1, tile_index : tile_index + 1], reserved_block)
        ttl.copy(reserved_block, output_tensor[0:1, tile_index : tile_index + 1])


@ttl.operation(grid=(1, 1))
def waited_tensor_copy_destination(input_tensor):
    scratch_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=1
    )
    waited_block = scratch_dfb.wait()
    ttl.copy(input_tensor[0, 0], waited_block)


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        pytest.param(
            reserved_tensor_copy_source,
            r"from a DFB block to a tensor requires.*wait\(\), not reserve\(\)",
            id="reserved-source",
        ),
        pytest.param(
            waited_tensor_copy_destination,
            r"from a tensor to a DFB block requires.*reserve\(\), not wait\(\)",
            id="waited-destination",
        ),
    ],
)
def test_invalid_tensor_copy_acquisition(monkeypatch, operation, error):
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    input_tensor = ttnn.from_torch(
        torch.arange(32 * 64, dtype=torch.bfloat16).reshape(32, 64),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    output_tensor = ttnn.from_torch(
        torch.zeros((32, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    with pytest.raises(TTLangCompileError, match=error):
        if operation is reserved_tensor_copy_source:
            operation(input_tensor, output_tensor)
        else:
            operation(input_tensor)
