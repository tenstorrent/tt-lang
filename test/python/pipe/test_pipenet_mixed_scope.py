# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multiple PipeNets in one operation with mixed declaration scopes.

Pins per-operation behavior when one PipeNet is declared at module
scope and another inside the operation body: both must contribute to
the active set, both must run their handshakes independently, and the
output must match the all-body-local form on hardware and simulator.

Pattern mirrors `test_overlapping_pipenets` but with `net_a` moved to
module scope.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32

# Module-scope PipeNet referenced by name inside the operation's threads.
MODULE_MCAST_A = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0))])


@ttl.operation(grid=(4, 1))
def mixed_scope_pipenets_kernel(inp, out):
    # Body-local PipeNet captured by each nested thread.
    body_mcast_b = ttl.PipeNet([ttl.Pipe(src=(3, 0), dst=(slice(1, 3), 0))])

    a_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    b_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        x, _ = ttl.node(dims=2)
        if 1 <= x and x <= 2:
            with a_cb.wait() as a, b_cb.wait() as b, out_cb.reserve() as o:
                o.store(a + b)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with a_cb.reserve() as ablk:

                def src_a(pipe):
                    ttl.copy(inp[0, 0], ablk).wait()
                    ttl.copy(ablk, pipe).wait()

                MODULE_MCAST_A.if_src(src_a)
        elif x == 3:
            with b_cb.reserve() as bblk:

                def src_b(pipe):
                    ttl.copy(inp[0, 3], bblk).wait()
                    ttl.copy(bblk, pipe).wait()

                body_mcast_b.if_src(src_b)
        elif 1 <= x and x <= 2:
            with a_cb.reserve() as ablk:

                def dst_a(pipe):
                    ttl.copy(pipe, ablk).wait()

                MODULE_MCAST_A.if_dst(dst_a)
            with b_cb.reserve() as bblk:

                def dst_b(pipe):
                    ttl.copy(pipe, bblk).wait()

                body_mcast_b.if_dst(dst_b)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        if 1 <= x and x <= 2:
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0, x]).wait()


def test_mixed_scope_pipenets(device):
    """Active set is {(0,0), (1,0), (2,0), (3,0)}. Nodes 1 and 2 receive
    from both PipeNets and sum the two tiles; nodes 0 and 3 are
    pure sources. Each PipeNet runs its handshake independently of the
    other; output on the receiving nodes must match the sum."""
    inp_torch = torch.randn(TILE, 4 * TILE, dtype=torch.bfloat16) * 0.1
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, 4 * TILE, dtype=torch.bfloat16), device)

    mixed_scope_pipenets_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    expected_mid = (
        inp_torch[:, 0:TILE].float() + inp_torch[:, 3 * TILE : 4 * TILE].float()
    )
    for col in (1, 2):
        actual = result[:, col * TILE : (col + 1) * TILE].float()
        diff = (expected_mid - actual).abs().max().item()
        assert diff < 0.05, f"node {col} diff {diff}"
