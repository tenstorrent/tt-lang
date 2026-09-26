# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Reject equal Pipe-source coverage with unequal DFB transaction counts."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


@ttl.operation(grid=(2, 1))
def copy_unbalanced_pipe_source_count(input_tensor):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(
        input_tensor,
        shape=(1, 1),
        block_count=2,
    )
    recv_dfb = ttl.make_dataflow_buffer_like(
        input_tensor,
        shape=(1, 1),
        block_count=2,
    )

    @ttl.compute()
    def compute():
        if net.is_src():
            with send_dfb.wait():
                pass

    @ttl.datamovement()
    def reader():
        def send(pipe):
            for iteration in range(5):
                with send_dfb.reserve() as send_block:
                    ttl.copy(input_tensor[0, iteration], send_block).wait()
                    ttl.copy(send_block, pipe).wait()

        net.if_src(send)

        def receive(pipe):
            for receive_iteration in range(5):
                with recv_dfb.reserve() as recv_block:
                    ttl.copy(pipe, recv_block).wait()

        net.if_dst(receive)

    @ttl.datamovement()
    def writer():
        if net.is_dst():
            for consume_iteration in range(5):
                with recv_dfb.wait():
                    pass


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 5 * 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    # CHECK: error: logical DFB {{[0-9]+}} has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0
    # CHECK: the producer pushes 5 block(s) per launch and the consumer pops 1, leaving 4 outstanding block(s) for capacity 2
    copy_unbalanced_pipe_source_count(input_tensor)
