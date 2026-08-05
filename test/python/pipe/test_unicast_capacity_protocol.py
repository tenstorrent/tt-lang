# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unicast pipe capacity release from a dataflow-thread receiver pop."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose

N_ITERS = 8


def _capacity_loop_op(recv_block_count, options):
    # recv_block_count > 1 keeps multiple sends outstanding, so the sender's
    # capacity acquire runs concurrently with the receiver's remote release.
    # A lost update on the shared capacity semaphore corrupts that interleaving;
    # block_count == 1 fully serializes acquire/release and hides it.
    @ttl.operation(grid=(2, 1), options=options)
    def unicast_dataflow_capacity_loop(inp, out):
        net = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])
        send_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        recv_cb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=recv_block_count
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm():
            for _iter_idx in range(N_ITERS):

                def send(pipe):
                    with send_cb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_cb.wait() as send_blk:
                        ttl.copy(send_blk, pipe).wait()

                net.if_src(send)

                def recv(pipe):
                    with recv_cb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()
                    with recv_cb.wait() as recv_blk:
                        ttl.copy(recv_blk, out[0, 0]).wait()

                net.if_dst(recv)

        @ttl.datamovement()
        def dm_brisc():
            pass

    return unicast_dataflow_capacity_loop


# Concurrent sender acquires and receiver releases must preserve local-first
# and global-only capacity counters for every supported DFB depth.
@pytest.mark.parametrize("recv_block_count", [1, 2, 3], ids=["bc1", "bc2", "bc3"])
@pytest.mark.parametrize(
    "counter_storage_options",
    [None, "--ttl-pipe-global-semaphores-only"],
    ids=["local-first", "global-only"],
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        pytest.param(torch.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
def test_unicast_dataflow_capacity_loop(
    device, dtype, rtol, atol, counter_storage_options, recv_block_count
):
    inp_torch = torch.randn(32, 32, dtype=dtype)
    out_torch = torch.zeros(32, 32, dtype=dtype)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    _capacity_loop_op(recv_block_count, counter_storage_options)(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out)
    assert_allclose(result.float(), inp_torch.float(), rtol=rtol, atol=atol)
