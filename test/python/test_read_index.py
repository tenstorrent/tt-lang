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


@ttl.operation(grid=(1, 1))
def gather_slot(index_tensor, weights, output):
    index_dfb = ttl.make_dataflow_buffer_like(index_tensor, shape=(1, 1), block_count=2)
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
            slot = ttl.read_index(index_block, 0, 0)
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


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "fp32"],
)
def test_read_index_selects_dram_slice(device, dtype):
    """A fractional tensor value truncates to the selected DRAM slot."""
    index_host = torch.full((TILE_SIZE, TILE_SIZE), 2.75, dtype=dtype)
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

    gather_slot(index_tensor, weights, output)

    actual = ttnn.to_torch(output).float()
    expected = weights_host[
        2 * SLOT_TILES * TILE_SIZE : 3 * SLOT_TILES * TILE_SIZE
    ].float()
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)
