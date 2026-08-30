# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for table-planned local PipeNet role predicates."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_allclose

pytestmark = pytest.mark.requires_device

TILE = 32
GRID_X = 5


@ttl.operation(grid=(GRID_X, 1))
def local_role_predicates(inp, out):
    net = ttl.PipeNet(
        [
            ttl.Pipe(src=(0, 0), dst=(3, 0)),
            ttl.Pipe(src=(0, 0), dst=(3, 0)),
            ttl.Pipe(src=(1, 0), dst=(2, 0)),
        ]
    )
    staging_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)
        with staging_dfb.reserve() as staging_block:
            ttl.copy(inp[0, node_x], staging_block).wait()
        with staging_dfb.wait() as staging_block:
            if net.is_src():
                ttl.copy(staging_block, out[0, node_x]).wait()
            if net.is_dst():
                ttl.copy(staging_block, out[1, node_x]).wait()
            if net.is_active():
                ttl.copy(staging_block, out[2, node_x]).wait()

    @ttl.datamovement()
    def dm_brisc():
        pass


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_local_role_predicates(device, dtype, to_device):
    torch.manual_seed(0)
    inp_torch = torch.randn(TILE, GRID_X * TILE, dtype=dtype)
    inp = to_device(inp_torch, device)
    out = to_device(torch.zeros(3 * TILE, GRID_X * TILE, dtype=dtype), device)

    local_role_predicates(inp, out)
    ttnn.synchronize_device(device)

    expected = torch.zeros(3 * TILE, GRID_X * TILE, dtype=dtype)
    expected[0:TILE, 0 : 2 * TILE] = inp_torch[:, 0 : 2 * TILE]
    expected[TILE : 2 * TILE, 2 * TILE : 4 * TILE] = inp_torch[:, 2 * TILE : 4 * TILE]
    expected[2 * TILE : 3 * TILE, 0 : 4 * TILE] = inp_torch[:, 0 : 4 * TILE]

    assert_allclose(expected.float(), ttnn.to_torch(out).float(), rtol=0.0, atol=0.0)
