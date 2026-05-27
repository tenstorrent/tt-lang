# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: %python -m pytest %s -v

"""Runtime and generated-code coverage for PipeNet foreach callback lowering."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32
PIPE_COUNT = 7


class BFloat16Tensor:
    dtype = torch.bfloat16


UNICAST_NET = ttl.PipeNet(
    [ttl.Pipe(src=(node, 0), dst=(node, 1)) for node in range(PIPE_COUNT)]
)

# Singleton slice destinations are still multicast PipeNet records. This
# prevents foreach lowering from losing the user-level PipeNet kind when a
# large multicast PipeNet is split into per-record selected pipes.
SINGLETON_MULTICAST_NET = ttl.PipeNet(
    [
        ttl.Pipe(src=(node, 0), dst=(slice(node, node + 1), 1))
        for node in range(PIPE_COUNT)
    ]
)


@ttl.operation(grid=(PIPE_COUNT, 2))
def compile_pipenet_foreach_unicast_iteration():
    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if UNICAST_NET.is_dst():
            with recv_dfb.wait() as _recv_blk:
                pass

    @ttl.datamovement()
    def send_dm():
        if UNICAST_NET.is_src():
            with send_dfb.reserve() as send_blk:
                UNICAST_NET.if_src(lambda pipe: ttl.copy(send_blk, pipe).wait())

    @ttl.datamovement()
    def recv_dm():
        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()

        UNICAST_NET.if_dst(recv)


@ttl.operation(grid=(PIPE_COUNT, 2))
def compile_pipenet_foreach_multicast_iteration():
    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if SINGLETON_MULTICAST_NET.is_dst():
            with recv_dfb.wait() as _recv_blk:
                pass

    @ttl.datamovement()
    def send_dm():
        if SINGLETON_MULTICAST_NET.is_src():
            with send_dfb.reserve() as send_blk:
                SINGLETON_MULTICAST_NET.if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )

    @ttl.datamovement()
    def recv_dm():
        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()

        SINGLETON_MULTICAST_NET.if_dst(recv)


@ttl.operation(grid=(PIPE_COUNT, 2))
def pipenet_foreach_unicast_runtime(input_tensor, output_tensor):
    send_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        node_col, _node_row = ttl.node(dims=2)
        if UNICAST_NET.is_src():
            with send_dfb.reserve() as send_block:
                transfer = ttl.copy(
                    input_tensor[0:1, node_col : node_col + 1], send_block
                )
                transfer.wait()
        if UNICAST_NET.is_dst():

            def recv(pipe):
                with recv_dfb.reserve() as recv_block:
                    transfer = ttl.copy(pipe, recv_block)
                    transfer.wait()

            UNICAST_NET.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        node_col, _node_row = ttl.node(dims=2)
        if UNICAST_NET.is_src():
            with send_dfb.wait() as send_block:

                def send(pipe):
                    transfer = ttl.copy(send_block, pipe)
                    transfer.wait()

                UNICAST_NET.if_src(send)
        if UNICAST_NET.is_dst():
            with recv_dfb.wait() as recv_block:
                transfer = ttl.copy(
                    recv_block, output_tensor[0:1, node_col : node_col + 1]
                )
                transfer.wait()


@ttl.operation(grid=(PIPE_COUNT, 2))
def pipenet_foreach_multicast_runtime(input_tensor, output_tensor):
    send_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        node_col, _node_row = ttl.node(dims=2)
        if SINGLETON_MULTICAST_NET.is_src():
            with send_dfb.reserve() as send_block:
                transfer = ttl.copy(
                    input_tensor[0:1, node_col : node_col + 1], send_block
                )
                transfer.wait()
        if SINGLETON_MULTICAST_NET.is_dst():

            def recv(pipe):
                with recv_dfb.reserve() as recv_block:
                    transfer = ttl.copy(pipe, recv_block)
                    transfer.wait()

            SINGLETON_MULTICAST_NET.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        node_col, _node_row = ttl.node(dims=2)
        if SINGLETON_MULTICAST_NET.is_src():
            with send_dfb.wait() as send_block:

                def send(pipe):
                    transfer = ttl.copy(send_block, pipe)
                    transfer.wait()

                SINGLETON_MULTICAST_NET.if_src(send)
        if SINGLETON_MULTICAST_NET.is_dst():
            with recv_dfb.wait() as recv_block:
                transfer = ttl.copy(
                    recv_block, output_tensor[0:1, node_col : node_col + 1]
                )
                transfer.wait()


def test_pipenet_foreach_iteration_runtime(device):
    input_torch = torch.arange(PIPE_COUNT * TILE * TILE, dtype=torch.float32).reshape(
        TILE, PIPE_COUNT * TILE
    )
    input_torch = (input_torch * 0.001).to(torch.bfloat16)

    input_tt = to_dram(input_torch, device)
    unicast_output_tt = to_dram(torch.zeros_like(input_torch), device)
    multicast_output_tt = to_dram(torch.zeros_like(input_torch), device)

    pipenet_foreach_unicast_runtime(input_tt, unicast_output_tt)
    pipenet_foreach_multicast_runtime(input_tt, multicast_output_tt)

    unicast_result = ttnn.to_torch(unicast_output_tt)
    multicast_result = ttnn.to_torch(multicast_output_tt)

    assert_pcc(input_torch.float(), unicast_result.float())
    assert_pcc(input_torch.float(), multicast_result.float())


if __name__ == "__main__":
    compile_pipenet_foreach_unicast_iteration()
    compile_pipenet_foreach_multicast_iteration()


# CHECK-INITIAL-NOT: ttl.if_src
# CHECK-INITIAL-NOT: ttl.if_dst
# CHECK-INITIAL-NOT: ttl.create_pipe
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL: #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 1, dstEndX = 0, dstEndY = 1, isMulticast = true>
# CHECK-INITIAL-SAME: #ttl.pipe_record<srcX = 6, srcY = 0, dstStartX = 6, dstStartY = 1, dstEndX = 6, dstEndY = 1, isMulticast = true>
# CHECK-INITIAL: ^bb0(%{{.*}}: !ttl.selected_pipe_src):
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL-NOT: ttl.if_src
# CHECK-INITIAL-NOT: ttl.if_dst
# CHECK-INITIAL-NOT: ttl.create_pipe
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL: #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 1, dstEndX = 0, dstEndY = 1, isMulticast = true>
# CHECK-INITIAL-SAME: #ttl.pipe_record<srcX = 6, srcY = 0, dstStartX = 6, dstStartY = 1, dstEndX = 6, dstEndY = 1, isMulticast = true>
# CHECK-INITIAL: ^bb0(%{{.*}}: !ttl.selected_pipe_dst):
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL-NOT: ttl.if_src
# CHECK-INITIAL-NOT: ttl.if_dst
# CHECK-INITIAL-NOT: ttl.create_pipe

# CHECK-CPP: for (
# CHECK-CPP-COUNT-1: noc_async_write(
# CHECK-CPP-NOT: noc_async_write(
# CHECK-CPP-COUNT-1: noc_async_write_multicast(
# CHECK-CPP-NOT: noc_async_write_multicast(
# CHECK-CPP-NOT: noc_async_write(
