# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""True-unicast pipe receive uses the user's reserved DFB block.

Regression test for issue #608.

The sender and receiver pipe callbacks live in the same NCRISC function and
are predicated by node coordinate. The receiver reserves the destination DFB in
the pipe callback. Repeating more times than the DFB depth catches sender-side
synthetic producer operations for the receiver DFB.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

N_ITERS = 20


@ttl.operation(grid=(2, 1))
def unicast_loop_split(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])
    send_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    in_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        if node_x == 0:
            for _iter_idx in range(N_ITERS):
                with in_cb.wait() as in_blk, out_cb.reserve() as out_blk:
                    out_blk.store(in_blk)

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)
        for _iter_idx in range(N_ITERS):
            if node_x == 1:
                with send_cb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_cb.wait() as send_blk:

                    def send(pipe):
                        ttl.copy(send_blk, pipe).wait()

                    net.if_src(send)

            if node_x == 0:

                def recv(pipe):
                    with in_cb.reserve() as in_blk:
                        ttl.copy(pipe, in_blk).wait()

                net.if_dst(recv)
                with out_cb.wait() as out_blk:
                    ttl.copy(out_blk, out[0, 0]).wait()

    @ttl.datamovement()
    def dm_brisc():
        pass


def test_unicast_loop_split_multi_iter(device):
    inp_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    unicast_loop_split(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out)
    assert_pcc(inp_torch.float(), result.float())
