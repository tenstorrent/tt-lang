# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.mlir %python %s > %t.out 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL < %t.mlir

"""Compile-only coverage for indexed PipeNet selection.

Generated communication patterns often keep related PipeNets in Python
containers. The frontend must resolve those compile-time container lookups for
PipeNet source and destination predicates, callbacks, and DFB role scopes.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402


class BFloat16Tensor:
    dtype = torch.bfloat16


GLOBAL_PIPE_NET_GROUPS = {
    "tree": (
        ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))]),
        ttl.PipeNet([ttl.Pipe(src=(3, 0), dst=(2, 0))]),
    )
}
CYCLIC_PIPE_NET_GROUP = [GLOBAL_PIPE_NET_GROUPS["tree"][0]]
CYCLIC_PIPE_NET_GROUP.append(CYCLIC_PIPE_NET_GROUP)


@ttl.operation(grid=(6, 1))
def compile_indexed_pipenet_selection():
    local_pipe_nets = [ttl.PipeNet([ttl.Pipe(src=(5, 0), dst=(4, 0))])]
    pipe_nets = [
        GLOBAL_PIPE_NET_GROUPS["tree"][0],
        GLOBAL_PIPE_NET_GROUPS["tree"][1],
        local_pipe_nets[0],
    ]

    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for pipe_index in range(len(pipe_nets)):
            net_index = pipe_index + 0
            if pipe_nets[net_index].is_dst():
                with recv_dfb.wait() as _recv_blk:
                    pass

    @ttl.datamovement()
    def send_dm():
        for pipe_index in range(len(pipe_nets)):
            net_index = pipe_index + 0
            if pipe_nets[net_index].is_src():
                with send_dfb.reserve() as send_blk:
                    pipe_nets[net_index].if_src(
                        lambda pipe: ttl.copy(send_blk, pipe).wait()
                    )

    @ttl.datamovement()
    def recv_dm():
        for pipe_index in range(len(pipe_nets)):
            net_index = pipe_index + 0
            pipe_nets[net_index].if_dst(
                lambda pipe: ttl.copy(pipe, recv_dfb.reserve()).wait()
            )


if __name__ == "__main__":
    compile_indexed_pipenet_selection()


# CHECK-INITIAL-NOT: scf.for
# CHECK-INITIAL: ttl.is_dst {pipe_net_id = 0 : i64}
# CHECK-INITIAL: ttl.is_dst {pipe_net_id = 1 : i64}
# CHECK-INITIAL: ttl.is_dst {pipe_net_id = 2 : i64}
# CHECK-INITIAL-NOT: scf.for
# CHECK-INITIAL: ttl.is_src {pipe_net_id = 0 : i64}
# CHECK-INITIAL: ttl.pipenet_scope attributes
# CHECK-INITIAL-SAME: ttl.pipe_net_ids = [0]
# CHECK-INITIAL-SAME: ttl.pipe_net_roles = [0]
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL: ttl.is_src {pipe_net_id = 1 : i64}
# CHECK-INITIAL: ttl.pipenet_scope attributes
# CHECK-INITIAL-SAME: ttl.pipe_net_ids = [1]
# CHECK-INITIAL-SAME: ttl.pipe_net_roles = [0]
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL: ttl.is_src {pipe_net_id = 2 : i64}
# CHECK-INITIAL: ttl.pipenet_scope attributes
# CHECK-INITIAL-SAME: ttl.pipe_net_ids = [2]
# CHECK-INITIAL-SAME: ttl.pipe_net_roles = [0]
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL: ttl.cb_reserve
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL: ttl.wait
# CHECK-INITIAL: ttl.cb_push
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL: ttl.cb_reserve
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL: ttl.wait
# CHECK-INITIAL: ttl.cb_push
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL: ttl.cb_reserve
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL: ttl.wait
# CHECK-INITIAL: ttl.cb_push
# CHECK-INITIAL-NOT: scf.for
