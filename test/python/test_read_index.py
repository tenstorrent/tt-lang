# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device tests for tensor-provided dynamic slice indices."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_allclose

TILE_SIZE = 32
SLOT_TILES = 2
SLOT_COUNT = 4


def _make_compute_controlled_arithmetic(expected_index, read_column):
    @ttl.operation(grid=(1, 1))
    def compute_controlled_arithmetic(index_tensor, activation, output):
        index_dfb = ttl.make_dataflow_buffer_like(
            index_tensor, shape=(1, 1), block_count=1
        )
        activation_dfb = ttl.make_dataflow_buffer_like(
            activation, shape=(1, 1), block_count=1
        )
        output_dfb = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=1)

        with index_dfb.reserve() as index_destination:
            ttl.copy(index_tensor[0, 0], index_destination).wait()
        with activation_dfb.reserve() as activation_destination:
            ttl.copy(activation[0, 0], activation_destination).wait()

        with (
            index_dfb.wait() as index_block,
            activation_dfb.wait() as activation_block,
            output_dfb.reserve() as output_block,
        ):
            runtime_index = ttl.read_index(index_block, 0, read_column)
            if runtime_index == expected_index:
                output_block.store(activation_block + activation_block)
            else:
                output_block.store(activation_block * activation_block)

        with output_dfb.wait() as published_output_block:
            ttl.copy(published_output_block, output[0, 0]).wait()

    return compute_controlled_arithmetic


@pytest.mark.parametrize(
    "index_dtype,stored_index,expected_index,read_column",
    [
        (torch.bfloat16, 130, 130, 3),
        (torch.bfloat16, 0, 130, 3),
        (torch.float32, 8388610, 8388610, 5),
        (torch.float32, 0, 8388610, 5),
        (torch.uint8, 255, 255, 7),
        (torch.uint8, 0, 255, 7),
        (torch.uint16, 65535, 65535, 7),
        (torch.uint16, 0, 65535, 7),
        (torch.uint32, 2147483647, 2147483647, 9),
        (torch.uint32, 0, 2147483647, 9),
    ],
    ids=[
        "bf16-match",
        "bf16-mismatch",
        "fp32-match",
        "fp32-mismatch",
        "ui8-match",
        "ui8-mismatch",
        "ui16-match",
        "ui16-mismatch",
        "ui32-match",
        "ui32-mismatch",
    ],
)
@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_read_index_controls_compute(
    device,
    index_dtype,
    stored_index,
    expected_index,
    read_column,
    memory,
):
    """A DFB-provided integer predicates compute-thread tile arithmetic."""

    tensor_factory = to_dram if memory == "dram" else to_l1
    background_index = expected_index if stored_index != expected_index else 0
    index_host = torch.full((TILE_SIZE, TILE_SIZE), background_index, dtype=index_dtype)
    index_host[0, read_column] = stored_index
    activation_host = (
        torch.arange(TILE_SIZE * TILE_SIZE, dtype=torch.float32)
        .remainder(17)
        .reshape(TILE_SIZE, TILE_SIZE)
        .to(torch.bfloat16)
    )
    output_host = torch.zeros_like(activation_host)

    index_tensor = tensor_factory(index_host, device)
    activation = tensor_factory(activation_host, device)
    output = tensor_factory(output_host, device)

    _make_compute_controlled_arithmetic(expected_index, read_column)(
        index_tensor, activation, output
    )

    actual = ttnn.to_torch(output).float()
    if stored_index == expected_index:
        expected = activation_host.float() + activation_host.float()
    else:
        expected = activation_host.float() * activation_host.float()
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def _make_gather_slot(index_bias):
    @ttl.operation(grid=(1, 1))
    def gather_slot(index_tensor, weights, output):
        index_dfb = ttl.make_dataflow_buffer_like(
            index_tensor, shape=(1, 1), block_count=2
        )
        weights_dfb = ttl.make_dataflow_buffer_like(
            weights, shape=(SLOT_TILES, 1), block_count=2
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def read():
            with index_dfb.reserve() as index_block:
                ttl.copy(index_tensor[0, 0], index_block).wait()

            with index_dfb.wait() as index_block:
                raw_index = ttl.read_index(index_block, 0, 0)
                slot = raw_index - index_bias
                with weights_dfb.reserve() as weights_block:
                    ttl.copy(
                        weights[
                            slot * SLOT_TILES : (slot + 1) * SLOT_TILES,
                            0:1,
                        ],
                        weights_block,
                    ).wait()

        @ttl.datamovement()
        def write():
            with weights_dfb.wait() as weights_block:
                ttl.copy(weights_block, output[0:SLOT_TILES, 0:1]).wait()

    return gather_slot


@pytest.mark.parametrize(
    "dtype,index_value,index_bias",
    [
        (torch.bfloat16, 0.5, 0),
        (torch.bfloat16, 2.75, 0),
        (torch.bfloat16, 130.0, 128),
        (torch.float32, 0.9999, 0),
        (torch.float32, 2.75, 0),
        (torch.float32, 8388610.0, 8388608),
    ],
    ids=[
        "bf16-below-one",
        "bf16-right-shift",
        "bf16-left-shift",
        "fp32-below-one",
        "fp32-right-shift",
        "fp32-left-shift",
    ],
)
def test_read_index_selects_dram_slice(device, dtype, index_value, index_bias):
    """Each float-decoding branch selects the expected DRAM slot."""
    expected_slot = int(index_value) - index_bias
    assert 0 <= expected_slot < SLOT_COUNT

    index_host = torch.full((TILE_SIZE, TILE_SIZE), index_value, dtype=dtype)
    weight_elements = SLOT_COUNT * SLOT_TILES * TILE_SIZE * TILE_SIZE
    weights_host = (
        torch.arange(weight_elements, dtype=torch.float32)
        .remainder(31)
        .reshape(SLOT_COUNT * SLOT_TILES * TILE_SIZE, TILE_SIZE)
        .to(dtype)
    )
    output_host = torch.zeros(SLOT_TILES * TILE_SIZE, TILE_SIZE, dtype=dtype)

    index_tensor = to_dram(index_host, device)
    weights = to_dram(weights_host, device)
    output = to_dram(output_host, device)

    _make_gather_slot(index_bias)(index_tensor, weights, output)

    actual = ttnn.to_torch(output).float()
    slot_elements = SLOT_TILES * TILE_SIZE
    expected = weights_host[
        expected_slot * slot_elements : (expected_slot + 1) * slot_elements
    ].float()
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "index_dtype,index_value,index_bias",
    [
        (torch.uint8, 255, 253),
        (torch.uint16, 65535, 65533),
        (torch.uint32, 2147483647, 2147483645),
    ],
    ids=["ui8-zero-extension", "ui16-zero-extension", "ui32"],
)
def test_read_unsigned_index_selects_dram_slice(
    device, index_dtype, index_value, index_bias
):
    """Unsigned index values use their full positive storage range."""
    index_host = torch.full(
        (TILE_SIZE, TILE_SIZE),
        index_value,
        dtype=index_dtype,
    )
    weight_elements = SLOT_COUNT * SLOT_TILES * TILE_SIZE * TILE_SIZE
    weights_host = (
        torch.arange(weight_elements, dtype=torch.float32)
        .remainder(31)
        .reshape(SLOT_COUNT * SLOT_TILES * TILE_SIZE, TILE_SIZE)
        .to(torch.bfloat16)
    )
    output_host = torch.zeros(
        SLOT_TILES * TILE_SIZE,
        TILE_SIZE,
        dtype=torch.bfloat16,
    )

    index_tensor = to_dram(index_host, device)
    weights = to_dram(weights_host, device)
    output = to_dram(output_host, device)

    _make_gather_slot(index_bias)(index_tensor, weights, output)

    actual = ttnn.to_torch(output).float()
    slot_elements = SLOT_TILES * TILE_SIZE
    expected = weights_host[2 * slot_elements : 3 * slot_elements].float()
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)
