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
GROUPED_TRANSFERS = 10
GROUP_SIZE_LIMIT = 4
TILE = 32


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


@ttl.operation(grid=(2, 1), options=f"--ttl-pipe-batch-tiles {GROUP_SIZE_LIMIT}")
def unicast_grouped_transport_storage(inp, out):
    """Copy distinct tiles through grouped transport storage and a residual."""
    net = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        for transfer_index in range(GROUPED_TRANSFERS):

            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, transfer_index], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            net.if_src(send)

            def recv(pipe):
                with recv_dfb.reserve() as recv_block:
                    ttl.copy(pipe, recv_block).wait()
                with recv_dfb.wait() as recv_block:
                    ttl.copy(recv_block, out[0, transfer_index]).wait()

            net.if_dst(recv)

    @ttl.datamovement()
    def dm_brisc():
        pass


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


# Distinct tiles prove that transport-owned slots advance in sender/receiver
# order. N=10 with an R<=4 limit also executes the scalar residual through the
# original DFB allocation.
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        pytest.param(torch.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
def test_unicast_grouped_transport_storage(device, dtype, rtol, atol):
    input_torch = torch.randn(TILE, GROUPED_TRANSFERS * TILE, dtype=dtype)
    output_torch = torch.zeros_like(input_torch)

    input_tensor = to_dram(input_torch, device)
    output_tensor = to_dram(output_torch, device)

    unicast_grouped_transport_storage(input_tensor, output_tensor)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(output_tensor)
    assert_allclose(result.float(), input_torch.float(), rtol=rtol, atol=atol)
