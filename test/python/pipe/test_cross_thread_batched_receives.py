# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for receiver DFB reuse across kernel threads."""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32
DTYPES = [torch.bfloat16, torch.float32]
DTYPE_IDS = ["bf16", "fp32"]


@ttl.operation(grid=(3, 1))
def gather_cross_thread_single_slot(inp, out):
    pipe1 = ttl.Pipe(src=(1, 0), dst=(0, 0))
    pipe2 = ttl.Pipe(src=(2, 0), dst=(0, 0))
    net = ttl.PipeNet([pipe1, pipe2])

    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        if node_x == 0:
            with recv_dfb.wait() as recv_block, out_dfb.reserve() as out_block:
                out_block.store(recv_block)
            with recv_dfb.wait() as recv_block, out_dfb.reserve() as out_block:
                out_block.store(recv_block)

    @ttl.datamovement()
    def dm_read():
        node_x, _node_y = ttl.node(dims=2)
        if node_x > 0:
            with send_dfb.reserve() as send_block:
                ttl.copy(inp[0, node_x - 1], send_block).wait()

            def send(pipe):
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            net.if_src(send)

        def receive(pipe):
            with recv_dfb.reserve() as recv_block:
                ttl.copy(pipe, recv_block).wait()

        net.if_dst(receive)

    @ttl.datamovement()
    def dm_write():
        node_x, _node_y = ttl.node(dims=2)
        if node_x == 0:
            with out_dfb.wait() as out_block:
                ttl.copy(out_block, out[0, 0]).wait()
            with out_dfb.wait() as out_block:
                ttl.copy(out_block, out[0, 1]).wait()


# Two receives may reuse one receiver DFB block because the second reserve
# waits until the consumer thread pops the first block.
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_gather_cross_thread_single_slot(device, dtype):
    inp_torch = torch.arange(2 * TILE * TILE, dtype=torch.float32).reshape(
        TILE, 2 * TILE
    )
    inp_torch = inp_torch.to(dtype)
    out_torch = torch.zeros_like(inp_torch)
    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    gather_cross_thread_single_slot(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out)
    assert_pcc(inp_torch.float(), result.float())
