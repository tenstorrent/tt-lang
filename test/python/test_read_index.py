# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device tests for tensor-provided dynamic slice indices."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose

TILE_SIZE = 32
SLOT_TILES = 2
SLOT_COUNT = 4


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
