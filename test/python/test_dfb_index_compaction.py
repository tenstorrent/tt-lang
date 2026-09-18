# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for compacting unused frontend DFB indices."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

TILE = 32


@ttl.operation(grid=(1, 1))
def _add_with_unused_dfb(lhs, rhs, result):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    _unused_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    result_dfb = ttl.make_dataflow_buffer_like(result, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        lhs_block = lhs_dfb.wait()
        rhs_block = rhs_dfb.wait()
        result_block = result_dfb.reserve()
        result_block.store(lhs_block + rhs_block)
        lhs_block.pop()
        rhs_block.pop()
        result_block.push()

    @ttl.datamovement()
    def read_inputs():
        lhs_block = lhs_dfb.reserve()
        ttl.copy(lhs[0, 0], lhs_block).wait()
        lhs_block.push()
        rhs_block = rhs_dfb.reserve()
        ttl.copy(rhs[0, 0], rhs_block).wait()
        rhs_block.push()

    @ttl.datamovement()
    def write_result():
        result_block = result_dfb.wait()
        ttl.copy(result_block, result[0, 0]).wait()
        result_block.pop()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_unused_dfb_index_is_compacted(device, dtype, memory_config, to_device):
    # The unused second declaration leaves frontend indices 0, 2, and 3 in the
    # kernel IR; successful execution requires finalization to compact them.
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    lhs_host = ((element_indices.remainder(41) - 20) / 16).to(dtype)
    rhs_host = (((3 * element_indices).remainder(37) - 18) / 16).to(dtype)
    lhs = to_device(lhs_host, device)
    rhs = to_device(rhs_host, device)
    result = to_device(torch.zeros_like(lhs_host), device)

    _add_with_unused_dfb(lhs, rhs, result)

    actual = ttnn.to_torch(result).float()
    expected = lhs_host.float() + rhs_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
