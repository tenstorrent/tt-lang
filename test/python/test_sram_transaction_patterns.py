# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Repeated block transactions across unequal DFB capacities."""

import pytest
import torch
import ttl
from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_allclose

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device


def _make_block_add(block_rows, block_columns, capacities, transaction_count):
    lhs_capacity, rhs_capacity, output_capacity = capacities

    @ttl.operation(grid=(1, 1))
    def block_add(lhs, rhs, output):
        lhs_storage = ttl.make_dataflow_buffer_like(
            lhs, shape=(block_rows, block_columns), block_count=lhs_capacity
        )
        rhs_storage = ttl.make_dataflow_buffer_like(
            rhs, shape=(block_rows, block_columns), block_count=rhs_capacity
        )
        output_storage = ttl.make_dataflow_buffer_like(
            output, shape=(block_rows, block_columns), block_count=output_capacity
        )

        @ttl.compute()
        def compute():
            for transaction in range(transaction_count):
                with (
                    lhs_storage.wait() as lhs_block,
                    rhs_storage.wait() as rhs_block,
                    output_storage.reserve() as output_block,
                ):
                    output_block.store(lhs_block + rhs_block)

        @ttl.datamovement()
        def reader():
            for transaction in range(transaction_count):
                first_row = transaction * block_rows
                last_row = first_row + block_rows
                with lhs_storage.reserve() as lhs_block:
                    ttl.copy(lhs[first_row:last_row, 0:block_columns], lhs_block).wait()
                with rhs_storage.reserve() as rhs_block:
                    ttl.copy(rhs[first_row:last_row, 0:block_columns], rhs_block).wait()

        @ttl.datamovement()
        def writer():
            for transaction in range(transaction_count):
                first_row = transaction * block_rows
                last_row = first_row + block_rows
                with output_storage.wait() as output_block:
                    ttl.copy(
                        output_block,
                        output[first_row:last_row, 0:block_columns],
                    ).wait()

    return block_add


TRANSACTION_PATTERNS = (
    pytest.param(1, 1, (1, 1, 1), 5, id="one-page-single-buffer"),
    pytest.param(1, 4, (1, 3, 2), 7, id="row-block-asymmetric-buffers"),
    pytest.param(4, 4, (3, 2, 1), 7, id="square-block-output-backpressure"),
)


@pytest.mark.parametrize(
    "block_rows,block_columns,capacities,transaction_count", TRANSACTION_PATTERNS
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "sram"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
def test_repeated_block_transactions(
    device,
    block_rows,
    block_columns,
    capacities,
    transaction_count,
    dtype,
    to_device,
    memory_model,
):
    """Validate repeated full-block compute across DFB sequence wrap."""
    operation = _make_block_add(
        block_rows, block_columns, capacities, transaction_count
    )
    tensor_shape = (transaction_count * block_rows * 32, block_columns * 32)
    for invocation in range(3):
        lhs_reference = torch.randint(-64, 65, tensor_shape).to(dtype) / 32
        rhs_reference = torch.randint(-64, 65, tensor_shape).to(dtype) / 32
        expected = lhs_reference + rhs_reference
        lhs = to_device(lhs_reference, device)
        rhs = to_device(rhs_reference, device)
        output = to_device(torch.zeros_like(expected), device)
        operation(lhs, rhs, output, options=f"--ttl-memory-model={memory_model}")
        assert_allclose(ttnn.to_torch(output).float(), expected.float(), rtol=0, atol=0)
