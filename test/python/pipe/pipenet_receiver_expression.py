# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL < %t.initial.mlir

"""Compile-only coverage for indexed PipeNet receiver expressions.

Generated communication patterns often keep related PipeNets in Python
containers. The frontend must resolve those compile-time container lookups for
PipeNet predicates, callbacks, and DFB role scopes.
"""

import pytest
import torch
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)


class BFloat16Tensor:
    dtype = torch.bfloat16


PIPE_NET_GROUPS = {
    "tree": (
        ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))]),
        ttl.PipeNet([ttl.Pipe(src=(3, 0), dst=(2, 0))]),
    )
}


@ttl.operation(grid=(4, 1))
def compile_pipenet_receiver_expression():
    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if PIPE_NET_GROUPS["tree"][0].is_dst():
            with recv_dfb.wait() as _recv_blk:
                pass
        if PIPE_NET_GROUPS["tree"][1].is_dst():
            with recv_dfb.wait() as _recv_blk:
                pass

    @ttl.datamovement()
    def send_dm():
        if PIPE_NET_GROUPS["tree"][0].is_src():
            with send_dfb.reserve() as send_blk:
                PIPE_NET_GROUPS["tree"][0].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if PIPE_NET_GROUPS["tree"][1].is_src():
            with send_dfb.reserve() as send_blk:
                PIPE_NET_GROUPS["tree"][1].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )

    @ttl.datamovement()
    def recv_dm():
        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()

        PIPE_NET_GROUPS["tree"][0].if_dst(recv)
        PIPE_NET_GROUPS["tree"][1].if_dst(recv)


if __name__ == "__main__":
    compile_pipenet_receiver_expression()


# CHECK-INITIAL: ttl.is_dst {pipe_net_id = 0 : i64}
# CHECK-INITIAL: ttl.is_dst {pipe_net_id = 1 : i64}
# CHECK-INITIAL: ttl.is_src {pipe_net_id = 0 : i64}
# CHECK-INITIAL: ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0], ttl.pipe_net_roles = [0]}
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL-SAME: net 0
# CHECK-INITIAL: ttl.is_src {pipe_net_id = 1 : i64}
# CHECK-INITIAL: ttl.pipenet_scope attributes {ttl.pipe_net_ids = [1], ttl.pipe_net_roles = [0]}
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL-SAME: net 1
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL-SAME: net 0
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL-SAME: net 1
