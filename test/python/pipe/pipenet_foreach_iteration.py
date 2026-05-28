# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-LOOPS < %t.output

"""Compile-only coverage for compact PipeNet foreach callback lowering."""

import pytest
import torch
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)

PIPE_COUNT = 7


class BFloat16Tensor:
    dtype = torch.bfloat16


UNICAST_NET = ttl.PipeNet(
    [ttl.Pipe(src=(node, 0), dst=(node, 1)) for node in range(PIPE_COUNT)]
)

SINGLETON_MULTICAST_NET = ttl.PipeNet(
    [
        ttl.Pipe(src=(node, 0), dst=(slice(node, node + 1), 1))
        for node in range(PIPE_COUNT)
    ]
)


@ttl.operation(grid=(PIPE_COUNT, 2))
def compile_pipenet_foreach_iteration():
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

                def send(pipe):
                    ttl.copy(send_blk, pipe).wait()

                UNICAST_NET.if_src(send)
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

        UNICAST_NET.if_dst(recv)
        SINGLETON_MULTICAST_NET.if_dst(recv)


if __name__ == "__main__":
    compile_pipenet_foreach_iteration()


# CHECK-INITIAL-NOT: ttl.if_src
# CHECK-INITIAL-NOT: ttl.if_dst
# CHECK-INITIAL-NOT: ttl.create_pipe
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL-SAME: name "UNICAST_NET"
# CHECK-INITIAL: ^bb0(%{{.*}}: !ttl.selected_pipe_src):
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL-SAME: name "UNICAST_NET"
# CHECK-INITIAL: ^bb0(%{{.*}}: !ttl.selected_pipe_dst):
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL-SAME: name "SINGLETON_MULTICAST_NET"
# CHECK-INITIAL: ^bb0(%{{.*}}: !ttl.selected_pipe_dst):
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL-NOT: ttl.if_src
# CHECK-INITIAL-NOT: ttl.if_dst
# CHECK-INITIAL-NOT: ttl.create_pipe

# CHECK-CPP-COUNT-1: noc_async_write(
# CHECK-CPP: noc_async_write_multicast
# CHECK-CPP-COUNT-2: noc_inline_dw_write

# CHECK-LOOPS-COUNT-4: for (
# CHECK-LOOPS-NOT: for (
