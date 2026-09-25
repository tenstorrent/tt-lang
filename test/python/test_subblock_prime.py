# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Numeric correctness for DST remainder peeling on prime tile counts.

A prime dimension larger than the DST budget cannot be subdivided by the
divisor heuristic in ttl-subblock-compute-for-dst, so the rescue raises it to
the budget and the pass peels the leftover tiles into a separate region. Both
dtypes are covered because the budget differs by element type (8 tiles for
bf16, 4 for fp32), which changes both the rescued subblock size and the
remainder extent.

Structural assertions on the peeled IR live in
test/ttlang/Dialect/TTL/Transforms/subblock_prime.mlir.
"""

import os
import sys

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

TILE = 32
pytestmark = pytest.mark.requires_device


def _make_add_operation(row_tiles, column_tiles):
    @ttl.operation(grid=(1, 1))
    def add_prime(lhs, rhs, out):
        lhs_dfb = ttl.make_dataflow_buffer_like(
            lhs, shape=(row_tiles, column_tiles), block_count=2
        )
        rhs_dfb = ttl.make_dataflow_buffer_like(
            rhs, shape=(row_tiles, column_tiles), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(row_tiles, column_tiles), block_count=2
        )

        @ttl.compute()
        def add_compute():
            lhs_blk = lhs_dfb.wait()
            rhs_blk = rhs_dfb.wait()
            out_blk = out_dfb.reserve()
            out_blk.store(lhs_blk + rhs_blk)
            lhs_blk.pop()
            rhs_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def dm_read():
            lhs_blk = lhs_dfb.reserve()
            tx_lhs = ttl.copy(lhs[0:row_tiles, 0:column_tiles], lhs_blk)
            tx_lhs.wait()
            lhs_blk.push()

            rhs_blk = rhs_dfb.reserve()
            tx_rhs = ttl.copy(rhs[0:row_tiles, 0:column_tiles], rhs_blk)
            tx_rhs.wait()
            rhs_blk.push()

        @ttl.datamovement()
        def dm_write():
            out_blk = out_dfb.wait()
            tx_out = ttl.copy(out_blk, out[0:row_tiles, 0:column_tiles])
            tx_out.wait()
            out_blk.pop()

    return add_prime


# 11 and 17 are prime and exceed both the bf16 (8) and fp32 (4) DST budgets, so
# each peels into a main region plus a remainder. 7 exceeds only the fp32
# budget; at bf16 it fits the budget outright and no peel occurs, which keeps
# the non-peeled result covered by the same assertions.
@pytest.mark.parametrize(
    "tile_extents", [(7, 1), (11, 1), (17, 1), (1, 17), (2, 17), (3, 11), (7, 7)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("subblock_sync", [False, True], ids=["coarse", "refined"])
def test_prime_block_add(device, dtype, tile_extents, to_device, subblock_sync):
    torch.manual_seed(0)
    row_tiles, column_tiles = tile_extents
    shape = (row_tiles * TILE, column_tiles * TILE)
    # Exactly representable inputs isolate indexing and synchronization from
    # SRCA/SRCB mantissa truncation, allowing exact checks of every tail element.
    lhs_torch = torch.randint(-32, 33, shape).to(dtype)
    rhs_torch = torch.randint(-32, 33, shape).to(dtype)

    lhs = to_device(lhs_torch, device)
    rhs = to_device(rhs_torch, device)
    out = to_device(torch.zeros(shape, dtype=dtype), device)

    _make_add_operation(row_tiles, column_tiles)(
        lhs,
        rhs,
        out,
        options="--ttl-subblock-sync" if subblock_sync else "--no-ttl-subblock-sync",
    )

    result = ttnn.to_torch(out)
    expected = lhs_torch + rhs_torch
    assert_allclose(result.float(), expected.float(), rtol=0, atol=0)
